import pickle
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
GRAPHS_PKL = DATA_DIR / 'web_graphs.pkl'
SPLIT_PKL = DATA_DIR / 'test_split_indices.pkl'

LABELS = {0: 'Benign', 1: 'SQLi', 2: 'XSS'}


def main():
    with open(GRAPHS_PKL, 'rb') as f:
        graphs = pickle.load(f)['graphs']
    with open(SPLIT_PKL, 'rb') as f:
        split = pickle.load(f)

    train_idx = split['train_idx']
    test_idx = split['test_idx']

    def features_labels(idx_list):
        X, y = [], []
        for idx in idx_list:
            g = graphs[idx]
            X.append([g.num_nodes, g.num_edges])
            y.append(int(g.y.item()))
        return np.array(X), np.array(y)

    X_train, y_train = features_labels(train_idx)
    X_test, y_test = features_labels(test_idx)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Benign', 'SQLi', 'XSS'], digits=4)

    print("=" * 90)
    print("DECISION TREE BASELINE: chi dung [num_nodes, num_edges] cua graph (web_graphs.pkl MOI)")
    print("de xem kien truc/graph moi co vo tinh tao shortcut kich thuoc moi hay khong")
    print("=" * 90)
    print(f"\nTrain: n={len(train_idx)}, Test: n={len(test_idx)}")
    print(f"\nOverall accuracy (chi dung graph size): {acc:.4f}")
    print(f"\n{report}")

    with open(DATA_DIR / 'decision_tree_size_baseline.txt', 'w') as f:
        f.write("DECISION TREE BASELINE (num_nodes, num_edges) tren web_graphs.pkl MOI (sau semantic_edges fix)\n")
        f.write(f"Train: n={len(train_idx)}, Test: n={len(test_idx)}\n")
        f.write(f"Overall accuracy: {acc:.4f}\n\n")
        f.write(report + "\n")

    print("\nDa ghi: data/decision_tree_size_baseline.txt")


if __name__ == '__main__':
    main()
