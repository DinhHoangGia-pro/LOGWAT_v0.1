import os
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader

from src.models.logwat import HeavyWebGNN
from src.utils.config import load_all
from src.utils.seed import set_seed

cfgs = load_all()
dataset_cfg = cfgs['dataset']
config_cfg = cfgs['config']

HIN_DIR = dataset_cfg.get('hin_dir')
paths = dataset_cfg.get('paths', {})
AUGMENTED_CSV = os.path.join(HIN_DIR, paths.get('augmented', 'data/augmented_web_attack.csv'))
DATA_PATH = os.path.join(HIN_DIR, paths.get('graphs_pkl', 'data/web_graphs.pkl'))
DEFAULT_SEED = int(config_cfg.get('seed', 42))
MODEL_SAVE_PATH = os.path.join(HIN_DIR, paths.get('model_save', f'data/models_pretrained/best_web_gnn_seed{DEFAULT_SEED}.pth'))
LOG_PATH = os.path.join(HIN_DIR, paths.get('logs_dir', 'logs'), 'training_history.log')
# Global dataset split is kept in data/test_split_indices.pkl.
# SQLi family-specific evaluation split must not overwrite it.
SQLI_FAMILY_TEST_INDEX_PATH = os.path.join(HIN_DIR, 'data', 'sqli_family_test_indices.pkl')
GLOBAL_SPLIT_PATH = os.path.join(HIN_DIR, 'data', 'test_split_indices.pkl')


def _load_global_split(num_graphs):
    """Load the frozen, class-balanced train/test split used for training
    validation and early stopping (same split evaluate.py reports against).
    This is intentionally NOT `_fixed_family_group_split`, which selects an
    SQLi-only (single-class) test subset meant for the separate family
    breakdown report, not for the main training loop's validation.

    `test_idx` is used exactly as frozen (never touched). `train_idx` is
    derived as "every graph index NOT in test_idx", not read verbatim from
    the pickle -- this way, any row appended to the dataset AFTER the split
    was frozen (e.g. train-only augmentation rows placed past the original
    index range) is automatically included in training instead of silently
    dropped, while the frozen test set stays exactly as-is.
    """
    with open(GLOBAL_SPLIT_PATH, 'rb') as f:
        split = pickle.load(f)
    test_idx = list(split['test_idx'])
    test_set = set(test_idx)
    train_idx = [i for i in range(num_graphs) if i not in test_set]
    return train_idx, test_idx


def _load_sqli_family_map():
    pool_path = os.path.join(HIN_DIR, 'data', 'sqli_payload_pool.csv')
    if not os.path.exists(pool_path):
        return {}, []

    pool_df = pd.read_csv(pool_path)
    family_map = {}
    for idx, fam in enumerate(pool_df['family'].fillna('unknown').astype(str).tolist()):
        family_map[f'sqli_pool_{idx}'] = fam
    ordered_families = [str(f) for f in pd.unique(pool_df['family'].fillna('unknown').astype(str))]
    return family_map, ordered_families


def _fixed_family_group_split(graphs):
    """Split unique sqli_pool_* payload IDs by family using a fixed deterministic rule.

    For each family, sort the unique payload IDs by increasing source_uid and put the
    first 2 payloads into test. For short families with 2-3 payloads, put 1 into test.
    All repeated graph rows for a given source_uid inherit the same split assignment.
    """
    family_map, ordered_families = _load_sqli_family_map()
    family_to_uids = defaultdict(list)

    for uid in sorted(family_map.keys(), key=lambda x: int(x.split('_')[-1])):
        family_to_uids[family_map[uid]].append(uid)

    test_uids = set()
    train_uids = set()
    for family in ordered_families:
        uids = family_to_uids.get(family, [])
        if not uids:
            continue

        if len(uids) <= 3:
            test_count = 1
            print(f"[WARN] family {family} chỉ có {len(uids)}<5 payload, coverage tối thiểu")
        else:
            test_count = 2

        test_uids.update(uids[:test_count])
        train_uids.update(uids[test_count:])

    if os.path.exists(AUGMENTED_CSV):
        aug_df = pd.read_csv(AUGMENTED_CSV)
        csic_rows = aug_df[aug_df['source'].fillna('').astype(str).eq('csic_original')].copy()
        csic_uids = sorted(csic_rows['source_uid'].dropna().astype(str).unique().tolist())
        if csic_uids:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            _, csic_test_idx = next(gss.split(csic_uids, groups=csic_uids))
            csic_test_uids = {csic_uids[i] for i in csic_test_idx}
            test_uids.update(csic_test_uids)
            train_uids.update(set(csic_uids) - csic_test_uids)

    train_idx = []
    test_idx = []
    for idx, g in enumerate(graphs):
        uid = None
        if hasattr(g, 'source_uid'):
            uid = getattr(g, 'source_uid')
        elif isinstance(g, dict) and 'source_uid' in g:
            uid = g['source_uid']
        if uid is None:
            train_idx.append(idx)
            continue
        uid = str(uid)
        if uid in test_uids:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    # Save the SQLi family split under a dedicated filename so that the
    # global dataset-level split in data/test_split_indices.pkl remains stable.
    with open(SQLI_FAMILY_TEST_INDEX_PATH, 'wb') as f:
        pickle.dump({'train_idx': train_idx, 'test_idx': test_idx}, f)

    return train_idx, test_idx


def train(num_epochs=None, batch_size=None, lr=None, target_metric='acc', seed=None):
    seed = seed if seed is not None else DEFAULT_SEED
    set_seed(seed)

    # fill defaults from config if not provided
    training_cfg = config_cfg.get('training', {})
    if num_epochs is None:
        num_epochs = training_cfg.get('epochs', 100)
    if batch_size is None:
        batch_size = training_cfg.get('batch_size', 64)
    if lr is None:
        lr = training_cfg.get('lr', 5e-4)

    early_stop_patience = int(training_cfg.get('early_stopping_patience', 10))
    early_stop_min_delta = float(training_cfg.get('early_stopping_min_delta', 0.0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] seed={seed} | model_save={MODEL_SAVE_PATH} | early_stop_patience={early_stop_patience} | early_stop_min_delta={early_stop_min_delta}")
    start_wall = time.time()
    with open(DATA_PATH, 'rb') as f:
        data_pkl = pickle.load(f)

    graphs = list(data_pkl['graphs'])
    train_idx, test_idx = _load_global_split(len(graphs))
    train_data = [graphs[i] for i in train_idx]
    test_data = [graphs[i] for i in test_idx]

    y_all = [int(g.y.item()) for g in graphs]
    train_bincount = [sum(1 for i in train_idx if y_all[i] == c) for c in range(3)]
    test_bincount = [sum(1 for i in test_idx if y_all[i] == c) for c in range(3)]
    print(f"[*] Using GLOBAL balanced split for training validation: "
          f"train={len(train_idx)}, test={len(test_idx)}, "
          f"bincount_train={train_bincount}, bincount_test={test_bincount}")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    y_train_for_weight = np.array([int(graphs[i].y.item()) for i in train_idx])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1, 2]),
        y=y_train_for_weight
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weight_line = f"[*] Computed class weights (balanced): {class_weights.tolist()}"
    print(class_weight_line)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'a') as _cw_log:
        _cw_log.write(class_weight_line + "\n")

    model = HeavyWebGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=training_cfg.get('weight_decay', 0.1))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stop_reason = 'reached max epoch'
    first_epoch_stats = None
    last_epoch_stats = None
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    with open(LOG_PATH, 'a') as log_f:
        log_f.write(f"\n--- NEW SESSION: {time.ctime()} ---\n")

        print(f"[DEBUG] ACTUAL EPOCH LOOP BOUND = {num_epochs}", flush=True)
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                loss.backward(); optimizer.step()
                total_loss += loss.item()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
                    correct += (pred == data.y).sum().item()

            epoch_loss = total_loss / max(1, len(train_loader))
            acc = correct / len(test_data)
            scheduler.step(acc)
            epoch_time = time.time() - start_time
            epoch_stats = {'epoch': epoch, 'loss': epoch_loss, 'val_acc': acc}
            if first_epoch_stats is None:
                first_epoch_stats = epoch_stats
            last_epoch_stats = epoch_stats

            log_str = f"Epoch {epoch:03d} | Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_time:.1f}s"
            print(log_str)
            log_f.write(log_str + "\n")

            if acc > best_acc + early_stop_min_delta:
                best_acc = acc
                best_epoch = epoch
                epochs_without_improvement = 0
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"   [V] Saved Best: {best_acc:.4f}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stop_patience:
                early_stop_reason = f"plateau: validation accuracy did not improve for {early_stop_patience} consecutive epochs"
                print(f"[*] Early stopping at epoch {epoch}: {early_stop_reason}")
                log_f.write(f"[*] Early stopping at epoch {epoch}: {early_stop_reason}\n")
                break

        total_time = time.time() - start_wall
        if first_epoch_stats is not None:
            print(f"[*] First epoch summary: epoch={first_epoch_stats['epoch']} loss={first_epoch_stats['loss']:.4f} val_acc={first_epoch_stats['val_acc']:.4f}")
            log_f.write(f"[*] First epoch summary: epoch={first_epoch_stats['epoch']} loss={first_epoch_stats['loss']:.4f} val_acc={first_epoch_stats['val_acc']:.4f}\n")
        if last_epoch_stats is not None:
            print(f"[*] Final epoch summary: epoch={last_epoch_stats['epoch']} loss={last_epoch_stats['loss']:.4f} val_acc={last_epoch_stats['val_acc']:.4f}")
            log_f.write(f"[*] Final epoch summary: epoch={last_epoch_stats['epoch']} loss={last_epoch_stats['loss']:.4f} val_acc={last_epoch_stats['val_acc']:.4f}\n")
        print(f"[*] TRAINING COMPLETE | epochs_run={last_epoch_stats['epoch'] if last_epoch_stats else 0} | wall_time={total_time:.1f}s | best_acc={best_acc:.4f} | best_epoch={best_epoch} | early_stop_reason={early_stop_reason} | model={MODEL_SAVE_PATH}")
        log_f.write(f"[*] TRAINING COMPLETE | epochs_run={last_epoch_stats['epoch'] if last_epoch_stats else 0} | wall_time={total_time:.1f}s | best_acc={best_acc:.4f} | best_epoch={best_epoch} | early_stop_reason={early_stop_reason} | model={MODEL_SAVE_PATH}\n")

    # SQLi family-breakdown split (for evaluate.py's family report only) is
    # computed as a separate post-training step, not used for early stopping
    # or best-checkpoint selection above.
    print('[*] Computing SQLi family-breakdown split (post-training, report-only)')
    fam_train_idx, fam_test_idx = _fixed_family_group_split(graphs)
    print(f'[*] SQLi family-breakdown split written to {SQLI_FAMILY_TEST_INDEX_PATH}: '
          f'train={len(fam_train_idx)}, test={len(fam_test_idx)}')


if __name__ == '__main__':
    train()
