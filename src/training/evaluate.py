import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch_geometric.loader import DataLoader

from src.models.logwat import HeavyWebGNN
from src.utils.config import load_all
from src.utils.metrics import confusion_matrix_array, pair_overlap_count

cfgs = load_all()
dataset_cfg = cfgs['dataset']
hin_dir = dataset_cfg.get('hin_dir')
paths = dataset_cfg.get('paths', {})
DATA_PATH = os.path.join(hin_dir, paths.get('graphs_pkl', 'data/web_graphs.pkl'))
MODEL_PATH = os.path.join(hin_dir, paths.get('model_save', 'data/models_pretrained/best_web_gnn.pth'))
ORIGINAL_CSV = os.path.join(hin_dir, paths.get('augmented', 'data/augmented_web_attack.csv'))
RESULTS_CSV = Path(hin_dir) / 'results' / 'main_results.csv'
TEST_SPLIT_PATH = Path(hin_dir) / 'data' / 'test_split_indices.pkl'


def _resolve_family(source_uid, family_map):
    """Resolve family on synthetic SQLi rows from their source_uid."""
    if source_uid is None:
        return None
    source_uid = str(source_uid)
    if not source_uid.startswith('sqli_pool_'):
        return None
    try:
        suffix = int(source_uid.split('_')[-1])
    except ValueError:
        return None
    return family_map.get(suffix)


def _load_test_indices(graphs):
    """Load or create the same group-based test split used by training."""
    if TEST_SPLIT_PATH.exists():
        with open(TEST_SPLIT_PATH, 'rb') as f:
            loaded = pickle.load(f)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            _, test_idx = loaded
        else:
            test_idx = loaded
        return [int(i) for i in test_idx]

    groups = []
    for g in graphs:
        uid = None
        if hasattr(g, 'source_uid'):
            uid = getattr(g, 'source_uid')
        elif isinstance(g, dict) and 'source_uid' in g:
            uid = g['source_uid']
        elif hasattr(g, 'source'):
            uid = getattr(g, 'source')
        groups.append(uid)

    if any(x is not None for x in groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        _, test_idx = next(gss.split(graphs, groups=groups))
    else:
        _, test_idx = train_test_split(range(len(graphs)), test_size=0.15, random_state=42)

    test_idx = [int(i) for i in test_idx]
    with open(TEST_SPLIT_PATH, 'wb') as f:
        pickle.dump(test_idx, f)
    return test_idx


def _append_subset_rows(rows, subset_name, y_true_sel, y_pred_sel, source_filter=None, family_filter=None, family_lookup=None):
    if len(y_true_sel) == 0:
        return

    labels = sorted(set(y_true_sel) | set(y_pred_sel))
    target_names = {int(lbl): str(lbl) for lbl in labels}
    report = classification_report(y_true_sel, y_pred_sel, labels=labels, target_names=[target_names.get(int(lbl), str(lbl)) for lbl in labels],
                                  output_dict=True, zero_division=0)

    for metric_name, metrics in report.items():
        if not isinstance(metrics, dict):
            continue

        support = int(metrics.get('support', 0))
        if family_filter is not None and support < 3:
            precision_val = 'N/A (insufficient samples)'
            recall_val = 'N/A (insufficient samples)'
            f1_val = 'N/A (insufficient samples)'
        else:
            precision_val = metrics.get('precision')
            recall_val = metrics.get('recall')
            f1_val = metrics.get('f1-score')

        rows.append({
            'subset': subset_name,
            'source': source_filter if source_filter is not None else 'all',
            'family': family_filter if family_filter is not None else '',
            'label': metric_name,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1_val,
            'support': support,
        })

        if subset_name in {'SQLi tổng hợp', 'SQLi chỉ sqli_pool'} or family_filter is not None:
            print(f"[{subset_name}] n={support} | label={metric_name} | precision={precision_val} | recall={recall_val} | f1={f1_val}")


def _compute_report_rows(df_test, y_true_test, y_pred_test, family_lookup):
    rows = []

    _append_subset_rows(rows, 'SQLi tổng hợp',
                        [y_true_test[i] for i in range(len(y_true_test))],
                        [y_pred_test[i] for i in range(len(y_pred_test))],
                        source_filter=None, family_filter=None)

    sqli_pool_mask = df_test['source'].fillna('').astype(str).eq('sqli_pool').to_numpy()
    sqli_pool_idx = [i for i, keep in enumerate(sqli_pool_mask) if keep]
    if sqli_pool_idx:
        _append_subset_rows(rows, 'SQLi chỉ sqli_pool',
                            [y_true_test[i] for i in sqli_pool_idx],
                            [y_pred_test[i] for i in sqli_pool_idx],
                            source_filter='sqli_pool', family_filter=None)

    for family_name in sorted(set(family_lookup.values())):
        family_idx = []
        for i, row in df_test.reset_index(drop=True).iterrows():
            if str(row.get('source', '')).strip() != 'sqli_pool':
                continue
            if _resolve_family(row.get('source_uid'), family_lookup) == family_name:
                family_idx.append(i)
        if not family_idx:
            continue
        _append_subset_rows(rows, 'family_breakdown',
                            [y_true_test[i] for i in family_idx],
                            [y_pred_test[i] for i in family_idx],
                            source_filter='sqli_pool', family_filter=family_name)

    return rows


def evaluate(top_k=10, plot_cm=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(DATA_PATH, 'rb') as f:
        data_pkl = pickle.load(f)
    dataset = data_pkl['graphs']
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = HeavyWebGNN().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(data.y.cpu().numpy())

    test_idx = _load_test_indices(dataset)
    y_true_test = [y_true[i] for i in test_idx]
    y_pred_test = [y_pred[i] for i in test_idx]

    # Metrics on the stored test split only.
    target_names = ['Benign', 'SQLi', 'XSS']
    report = classification_report(y_true_test, y_pred_test, labels=[0, 1, 2], target_names=target_names,
                                  output_dict=True, zero_division=0)
    cm = confusion_matrix_array(y_true_test, y_pred_test)

    print("=== CLASSIFICATION REPORT (TEST SPLIT) ===")
    for k, v in report.items():
        if k in target_names:
            print(f"{k}: precision={v['precision']:.3f}, recall={v['recall']:.3f}, f1={v['f1-score']:.3f}")
    print("macro avg:", report.get('macro avg'))

    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    overlap = pair_overlap_count(cm, 1, 2)
    print(f"\n[!] SQLi <-> XSS confusions (symmetric): {overlap}")

    results_rows = []
    if os.path.exists(ORIGINAL_CSV):
        df_all = pd.read_csv(ORIGINAL_CSV)
        df_test = df_all.iloc[test_idx].reset_index(drop=True)

        family_lookup = {}
        pool_df = pd.read_csv(Path(hin_dir) / 'data' / 'sqli_payload_pool.csv') if (Path(hin_dir) / 'data' / 'sqli_payload_pool.csv').exists() else None
        if pool_df is not None:
            for idx, fam in enumerate(pool_df['family'].fillna('unknown').astype(str).tolist()):
                family_lookup[idx] = fam

        results_rows = _compute_report_rows(df_test, y_true_test, y_pred_test, family_lookup)

        RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results_rows).to_csv(RESULTS_CSV, index=False)
        print(f"\n[+] Wrote detailed classification reports to {RESULTS_CSV}")
        print(pd.DataFrame(results_rows).to_string(index=False))

        missed_sqli = []
        false_positives = []
        for i, (t, p) in enumerate(zip(y_true_test, y_pred_test)):
            if t == 1 and p == 0 and len(missed_sqli) < top_k:
                missed_sqli.append(df_test.iloc[i]['content'])
            if t == 0 and p == 1 and len(false_positives) < top_k:
                false_positives.append(df_test.iloc[i]['content'])
            if len(missed_sqli) >= top_k and len(false_positives) >= top_k:
                break

        print("\nTOP missed SQLi samples:")
        for s in missed_sqli:
            print("->", s)

        print("\nTOP false positives (Benign->SQLi):")
        for s in false_positives:
            print("->", s)

    else:
        print(f"[!] Original CSV not found at {ORIGINAL_CSV}; skipping payload examples.")


if __name__ == '__main__':
    evaluate()
