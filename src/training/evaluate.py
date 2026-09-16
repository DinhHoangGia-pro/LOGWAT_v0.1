import os
import pickle
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from src.models.logwat import HeavyWebGNN
from src.utils.metrics import classification_report_dict, confusion_matrix_array, pair_overlap_count
from src.utils.config import load_all

cfgs = load_all()
dataset_cfg = cfgs['dataset']
hin_dir = dataset_cfg.get('hin_dir')
paths = dataset_cfg.get('paths', {})
DATA_PATH = os.path.join(hin_dir, paths.get('graphs_pkl', 'data/web_graphs.pkl'))
MODEL_PATH = os.path.join(hin_dir, paths.get('model_save', 'data/models_pretrained/best_web_gnn.pth'))
ORIGINAL_CSV = os.path.join(hin_dir, paths.get('augmented', 'data/augmented_web_attack.csv'))


def evaluate(top_k=10, plot_cm=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    with open(DATA_PATH, 'rb') as f:
        data_pkl = pickle.load(f)
    dataset = data_pkl['graphs']
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = HeavyWebGNN().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
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

    # Metrics
    target_names = ['Benign', 'SQLi', 'XSS']
    report = classification_report_dict(y_true, y_pred, target_names=target_names)
    cm = confusion_matrix_array(y_true, y_pred)

    # Print
    print("=== CLASSIFICATION REPORT ===")
    for k, v in report.items():
        if k in target_names:
            print(f"{k}: precision={v['precision']:.3f}, recall={v['recall']:.3f}, f1={v['f1-score']:.3f}")
    print("macro avg:", report.get('macro avg'))

    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    overlap = pair_overlap_count(cm, 1, 2)
    print(f"\n[!] SQLi <-> XSS confusions (symmetric): {overlap}")

    # Show example payloads for debugging similar to original script
    if os.path.exists(ORIGINAL_CSV):
        df = pd.read_csv(ORIGINAL_CSV)
        missed_sqli = []
        false_positives = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if t == 1 and p == 0 and len(missed_sqli) < top_k:
                missed_sqli.append(df.iloc[i]['content'])
            if t == 0 and p == 1 and len(false_positives) < top_k:
                false_positives.append(df.iloc[i]['content'])
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
