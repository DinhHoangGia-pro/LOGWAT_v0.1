"""Two analyses on the external dataset for the deployed seed-42 checkpoint (no training):

 A. XSS precision: base-rate vs. over-prediction. PR-AUC of the XSS softmax probability, Bayes precision at assumed base rates,
    and the one-vs-rest threshold that reaches precision 50%.
 B. External Benign errors: direction of the errors and the structure of the misclassified rows (the "4th benign form":
    bare values without key/value, header or JSON delimiters), with examples.

Recomputes what docs/DATASET.md used to state for the pre-fix checkpoint; every number here comes from the checkpoint named in
the output header. Writes data/external/external_xss_benign_gap_analysis.txt.
"""
import hashlib
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from torch_geometric.loader import DataLoader

from scripts.evaluate_external_dataset import EXTERNAL_CLEAN_CSV, EXTERNAL_GRAPHS_PKL, MODEL_PATH
from src.models.logwat import HeavyWebGNN

OUT = Path(__file__).resolve().parents[1] / 'data' / 'external' / 'external_xss_benign_gap_analysis.txt'


def main():
    df = pd.read_csv(EXTERNAL_CLEAN_CSV)
    y = df['attack_type'].astype(int).values
    graphs = pickle.load(open(EXTERNAL_GRAPHS_PKL, 'rb'))['graphs']
    model = HeavyWebGNN(use_edge_attr=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    probs = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=256, shuffle=False):
            probs.append(torch.softmax(model(b.x, b.edge_index, b.batch, edge_attr=None), dim=1).numpy())
    P = np.concatenate(probs)
    pred = P.argmax(1)
    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]
    L = [f"checkpoint {MODEL_PATH.name} sha256[:12]={sha}, n={len(y)} (Benign={(y == 0).sum()}, SQLi={(y == 1).sum()}, XSS={(y == 2).sum()})", ""]

    # ---- A. XSS precision
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    L.append(f"confusion matrix (rows=true, cols=pred, Benign/SQLi/XSS):\n{cm}\n")
    yx = (y == 2).astype(int)
    px = P[:, 2]
    tp = int(((pred == 2) & (yx == 1)).sum()); fp = int(((pred == 2) & (yx == 0)).sum())
    fn = int(((pred != 2) & (yx == 1)).sum()); tn = int(((pred != 2) & (yx == 0)).sum())
    tpr, fpr, prec = tp / (tp + fn), fp / (fp + tn), tp / (tp + fp)
    prev = yx.mean()
    L.append("A. XSS precision at the 3-way argmax operating point")
    L.append(f"   TP={tp} FP={fp} FN={fn} TN={tn}  precision={prec:.4f} recall={tpr:.4f} FPR={fpr:.4f}")
    L.append(f"   PR-AUC (average precision of the XSS softmax probability) = {average_precision_score(yx, px):.4f}; no-skill baseline (prevalence) = {prev:.4f}")
    for pi, name in ((0.001, '0.1%'), (0.01, '1%'), (prev, f'{100 * prev:.2f}% (this set)')):
        bay = pi * tpr / (pi * tpr + (1 - pi) * fpr)
        need = pi * tpr / (1 - pi)          # FPR that gives precision 0.5 at this base rate
        L.append(f"   pi={name}: precision={100 * bay:.2f}% ; FPR needed for 50% precision <= {need:.6f} (measured {fpr:.4f} = {fpr / need:.1f}x higher)")
    p_, r_, th = precision_recall_curve(yx, px)
    ok = np.where(p_[:-1] >= 0.5)[0]
    if len(ok):
        k = ok[np.argmax(r_[:-1][ok])]
        L.append(f"   one-vs-rest threshold on the XSS probability reaching precision >= 50%: threshold={th[k]:.4f} -> precision={p_[k]:.4f}, recall={r_[k]:.4f} (argmax: recall {tpr:.4f})")
    else:
        L.append("   no threshold reaches precision >= 50%")
    L.append("")

    # ---- B. Benign errors
    ben = np.where(y == 0)[0]
    wrong = ben[pred[ben] != 0]
    to_sqli = int((pred[wrong] == 1).sum()); to_xss = int((pred[wrong] == 2).sum())
    L.append("B. External Benign rows misclassified")
    L.append(f"   {len(wrong)} of {len(ben)} ({100 * len(wrong) / len(ben):.1f}%): {to_sqli} -> SQLi, {to_xss} -> XSS")
    txt = df['content'].astype(str).values
    def share(mask_fn, idx):
        return int(sum(1 for i in idx if mask_fn(txt[i])))
    for name, fn_ in (("with '=' (query-string form)", lambda s: '=' in s), ("with '&' (query-string form)", lambda s: '&' in s),
                      ("with ':' (header-style form)", lambda s: ':' in s), ("with '{' or '}' (json form)", lambda s: '{' in s or '}' in s),
                      ("pure digit strings", lambda s: s.isdigit()), ("containing '@'", lambda s: '@' in s)):
        L.append(f"   {name:34s}: {share(fn_, wrong)} / {len(wrong)}  ({100 * share(fn_, wrong) / len(wrong):.1f}%)")
    no_delim = sum(1 for i in wrong if not re.search(r'[=&:{}]', txt[i]))
    L.append(f"   containing none of = & : {{ }}        : {no_delim} / {len(wrong)}  ({100 * no_delim / len(wrong):.1f}%)")
    for direction, name in ((1, 'SQLi'), (2, 'XSS')):
        idx = wrong[pred[wrong] == direction]
        L.append(f"   predicted {name}: {len(idx)} rows; '@' in {share(lambda s: '@' in s, idx)}, pure digits in {share(lambda s: s.isdigit(), idx)}, "
                 f"mean length {np.mean([len(txt[i]) for i in idx]):.1f}")
    rng = np.random.RandomState(0)
    L.append("   10 examples (seed 0, proportional to the two directions):")
    n_sql = round(10 * to_sqli / len(wrong))
    for direction, k in ((1, n_sql), (2, 10 - n_sql)):
        idx = wrong[pred[wrong] == direction]
        for i in rng.choice(idx, size=min(k, len(idx)), replace=False):
            L.append(f"     -> {'SQLi' if direction == 1 else 'XSS ':4s} len={len(txt[i]):3d}  {txt[i]!r}")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[+] wrote {OUT}")


if __name__ == '__main__':
    main()
