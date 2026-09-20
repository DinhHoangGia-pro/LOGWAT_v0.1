"""Editable vector source (SVG) of the new Fig. 5: confusion matrix of the deployed LOGWAT checkpoint on the EXTERNAL dataset.

Replaces the old Fig. 5 (a 3,000-request balanced matrix, TPR 98.2% / 97.5%) that matches neither the current data set nor the
current checkpoint. The frozen test split is not shown: the deployed checkpoint classifies all of its 4,215 requests correctly,
so its matrix is a perfect diagonal and carries no information. The errors are on HttpParamsDataset (n = 30,677).

Matrix (rows = true class, columns = predicted class; Benign, SQLi, XSS) is parsed from
data/external/external_xss_benign_gap_analysis.txt (checkpoint best_web_gnn_seed42.pth, sha256[:12] = 113a6f2cf513) and
cross-checked against results/external_dataset_evaluation.csv (per-class precision and recall) and against the numbers quoted in
the paper (5,877 of 19,293 Benign labelled as attacks: 4,750 XSS + 1,127 SQLi; accuracy 0.7829; XSS precision 0.089).
Cell colour is the share of the TRUE class (row-normalised), because the classes are very unbalanced (19,293 / 10,852 / 532);
each cell prints the raw count and that share.

Text is kept as <text> objects (svg.fonttype = 'none') so the SVG stays editable in Inkscape/Illustrator; no mathtext is used.
The PDF is exported by hand from Inkscape.

Usage: python scripts/make_fig5_new.py [output.svg]
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['svg.hashsalt'] = 'logwat-fig5'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent.parent
GAP = ROOT / 'data' / 'external' / 'external_xss_benign_gap_analysis.txt'
EVAL = ROOT / 'results' / 'external_dataset_evaluation.csv'
OUT = Path('/home/dhgia/Work/iot_mailware/hin_web_vulne/paper_LOGWAT/Fig/Fig.5_new.svg')

CLASSES = ['Benign', 'SQLi', 'XSS']
EXPECTED = np.array([[13416, 1127, 4750], [125, 10074, 653], [0, 5, 527]])  # as given for the figure
C_ERR = '#D55E00'


# ------------------------------------------------------------------------------------------------ data
def load():
    txt = GAP.read_text()
    assert 'best_web_gnn_seed42.pth' in txt and '113a6f2cf513' in txt and 'n=30677' in txt
    block = re.search(r'\[\[(.*?)\]\]', txt, flags=re.S).group(0)
    cm = np.array([[int(v) for v in row.split()] for row in re.findall(r'\[([\d\s]+)\]', block)])
    assert cm.shape == (3, 3) and (cm == EXPECTED).all(), cm

    n_true = cm.sum(axis=1)
    n_pred = cm.sum(axis=0)
    assert list(n_true) == [19293, 10852, 532] and cm.sum() == 30677
    recall = np.diag(cm) / n_true
    precision = np.diag(cm) / n_pred
    acc = np.trace(cm) / cm.sum()

    # cross-check against the evaluation CSV (method, class, precision, recall, f1, support)
    with open(EVAL, newline='') as f:
        rows = [r for r in csv.reader(f) if r and r[0] == 'gatv2' and r[1] in CLASSES]
    for r in rows:
        i = CLASSES.index(r[1])
        assert abs(float(r[2]) - precision[i]) < 1e-9 and abs(float(r[3]) - recall[i]) < 1e-9 and int(float(r[5])) == n_true[i], r
    assert len(rows) == 3
    # numbers quoted in the paper
    assert cm[0, 1] + cm[0, 2] == 5877 and cm[0, 2] == 4750 and cm[0, 1] == 1127
    assert round(acc, 4) == 0.7829 and round(precision[2], 3) == 0.089 and round(recall[0], 3) == 0.695
    return cm, n_true, precision, recall


# ------------------------------------------------------------------------------------------------ figure
def draw(out):
    cm, n_true, precision, recall = load()
    share = cm / n_true[:, None]

    fig, ax = plt.subplots(figsize=(5.1, 4.9))
    fig.subplots_adjust(left=0.235, right=0.975, top=0.855, bottom=0.295)
    # cells are drawn as vector rectangles (not imshow), so no raster <image> is embedded in the SVG
    cmap = plt.get_cmap('Blues')
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=cmap(share[i, j]), ec='none', lw=0, zorder=1))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_aspect('equal')

    for i in range(3):
        for j in range(3):
            dark = share[i, j] > 0.5
            col = 'white' if dark else 'black'
            ax.text(j, i - 0.09, f'{cm[i, j]:,}', ha='center', va='center', fontsize=12, fontweight='bold', color=col)
            ax.text(j, i + 0.21, f'{100 * share[i, j]:.1f}% of row', ha='center', va='center', fontsize=7.5, color=col)

    # frame the cells that make up the 5,877 wrongly flagged Benign requests
    ax.add_patch(Rectangle((0.5, -0.5), 2, 1, fill=False, ec=C_ERR, lw=2.4, zorder=5))
    n_bad = cm[0, 1] + cm[0, 2]
    ax.text(2.5, -0.62, f'{n_bad:,} of {n_true[0]:,} Benign requests ({100 * n_bad / n_true[0]:.1f}%)\nlabelled as attacks',
            ha='right', va='bottom', fontsize=8.5, color=C_ERR, fontweight='bold', linespacing=1.25)

    ax.set_xticks(range(3))
    ax.set_xticklabels([f'{c}\nprecision\n{precision[k]:.3f}' for k, c in enumerate(CLASSES)], fontsize=8.5,
                       linespacing=1.15)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f'{c}\nn = {n_true[k]:,}' for k, c in enumerate(CLASSES)], fontsize=8.5)
    ax.set_xlabel('Predicted class', fontsize=9, labelpad=6)
    ax.set_ylabel('True class', fontsize=9, labelpad=6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([0.5, 1.5], minor=True)      # inner grid lines only (no half-clipped outer lines)
    ax.set_yticks([0.5, 1.5], minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='minor', length=0)

    fig.text(0.5, 0.03, f'Deployed seed-42 checkpoint on HttpParamsDataset (n = {cm.sum():,}), no retraining.\n'
             'Cell colour = share of the true class.', ha='center', va='bottom', fontsize=7, color='#555555', style='italic',
             linespacing=1.3)

    fig.savefig(out, format='svg')
    return fig


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    fig = draw(out)
    print('wrote', out, f'({out.stat().st_size} bytes)')
    return fig


if __name__ == '__main__':
    main()
