"""Editable vector source (SVG) of the new Fig. 1: the four-stage LOGWAT workflow.

The old Fig. 1 had three phases plus an output and a node-attribute list that named the HTTP 'Method'; the text of the paper
(Sec. 3.1, 'the overall workflow consists of four integrated stages') and the code describe FOUR stages, and the node
features are computed from the token alone. Content of the blocks, and where each statement comes from:

  1  Request preprocessing and BIA-based augmentation
       URL-decode once (not to a fixed point), lower-case       src/preprocessing/tokenizer.py  (unquote(...).lower())
       BIA (M1 template poisoning, M2 keyword-noise insertion,
       M2b context-spacing insertion): applied once when the
       training data are built, no role at inference             paper Sec. 3.3 (src/bia/bia.py)
  2  Multi-relational graph construction
       one node per token of the flat token sequence             src/bag/graph_builder.py
       G = (V, E, R), R = {seq, skip, sem}
       E_seq: (i, i+1)   E_skip: (i, i+2), k = 2                  src/edges/sequential.py, src/edges/skip.py
       E_sem: the nine ordered keyword pairs, < 15 tokens apart,
       fragments split by noise tokens are merged                 src/edges/semantic.py
  3  Token feature encoding
       64-dimensional vector per node, from the token alone:
       statistics (length, relative position, entropy, digit and
       non-alphanumeric ratios, ...), 15 dangerous characters /
       sequences, 9 SQL + 7 web keyword indicators               src/bag/node_features.py  (8 + 15 + 9 + 7 values, zero-padded)
  4  Attention-based inductive learning
       2 GATv2 layers (8 heads, BatchNorm, ELU) -> jumping knowledge (concatenation) -> global max-pooling
       -> MLP (dropout 0.5) -> 3 classes                          src/models/layers.py, configs/model.yaml

Text is kept as <text> objects (svg.fonttype = 'none'); the few mathtext labels (E_seq ...) are rewritten as ONE <text> with
relative-positioned subscripts by make_fig6_new.harden_svg_text (same fix as Fig. 6). The PDF is exported by hand from Inkscape.

Usage: python scripts/make_fig1_new.py [output.svg]
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from make_fig6_new import harden_svg_text   # noqa: E402  (also sets the shared rcParams)

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['svg.hashsalt'] = 'logwat-fig1'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = Path('/home/dhgia/Work/iot_mailware/hin_web_vulne/paper_LOGWAT/Fig/Fig1_new.svg')

SEM_RED = '#D62728'
SEQ_GREY = '#B8B8B8'
SKIP_GREY = '#555555'
NAVY = '#2F4B6E'
BIA_ORANGE = '#D55E00'
OUT_GREEN = '#2E7D5B'

W, H = 6.6, 8.7
X0, X1 = 0.12, W - 0.12          # stage boxes span X0..X1
GAP = 0.31


def rbox(ax, x, y, w, h, fc, ec, lw=1.2, ls='-', r=0.07, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def down_arrow(ax, x, y_from, y_to, label=None):
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle='-|>', mutation_scale=13, color='#444444', lw=1.6,
                                 shrinkA=0, shrinkB=0, zorder=3))
    if label:
        ax.text(x + 0.14, (y_from + y_to) / 2, label, fontsize=7.4, color='#555555', va='center', ha='left', style='italic')


def stage(ax, top, height, num, title):
    rbox(ax, X0, top - height, X1 - X0, height, '#F2F6FA', NAVY, lw=1.3)
    ax.add_patch(Circle((X0 + 0.32, top - 0.30), 0.17, fc=NAVY, ec='none', zorder=4))
    ax.text(X0 + 0.32, top - 0.30, str(num), color='white', fontsize=11, fontweight='bold', ha='center', va='center', zorder=5)
    ax.text(X0 + 0.65, top - 0.30, title, fontsize=9.6, fontweight='bold', color='#1B2E45', va='center')
    return top - height


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    xc = W / 2
    tx = X0 + 0.65                                    # left text margin inside the stage boxes

    # ---- input
    y = H - 0.12
    rbox(ax, xc - 1.55, y - 0.34, 3.1, 0.34, '#FFFFFF', '#555555', lw=1.1, r=0.16)
    ax.text(xc, y - 0.17, 'HTTP request: one text field (content string)', fontsize=8.6, ha='center', va='center')
    y -= 0.34
    down_arrow(ax, xc, y, y - GAP)
    y -= GAP

    # ---- stage 1
    top = y
    h1 = 1.78
    y = stage(ax, top, h1, 1, 'Request preprocessing and BIA-based augmentation')
    sub_top, sub_h = top - 0.58, h1 - 0.70
    lw_ = 2.35
    rbox(ax, tx, sub_top - sub_h, lw_, sub_h, '#FFFFFF', '#5B7392', lw=1.0)
    ax.text(tx + 0.10, sub_top - 0.09, 'training and inference', fontsize=7.6, color=NAVY, style='italic', va='top')
    for k, line in enumerate(['raw content string', '→  URL-decode (single pass)', '→  lower-case']):
        ax.text(tx + 0.10, sub_top - 0.33 - 0.19 * k, line, fontsize=8, va='top')
    bx = tx + lw_ + 0.16
    bw = X1 - 0.12 - bx
    rbox(ax, bx, sub_top - sub_h, bw, sub_h, '#FFF4E8', BIA_ORANGE, lw=1.2, ls=(0, (4, 2.5)))
    ax.text(bx + 0.10, sub_top - 0.09, 'training data only: BIA augmentation', fontsize=7.6, color=BIA_ORANGE, style='italic',
            fontweight='bold', va='top')
    for k, line in enumerate(['template poisoning (M1)', 'keyword-noise insertion (M2)', 'context-spacing insertion (M2b)']):
        ax.text(bx + 0.10, sub_top - 0.32 - 0.18 * k, '•  ' + line, fontsize=7.8, va='top')
    ax.text(bx + 0.10, sub_top - sub_h + 0.08, 'applied once when the data are built;\nno role at inference', fontsize=6.9,
            color='#7A4A1A', va='bottom', linespacing=1.2)
    down_arrow(ax, xc, y, y - GAP, 'normalized content string')
    y -= GAP

    # ---- stage 2
    top = y
    h2 = 1.78
    y = stage(ax, top, h2, 2, 'Multi-relational graph construction')
    ax.text(tx, top - 0.56, 'One node per token of the flat token sequence (no field-level structure):', fontsize=8, va='top')
    ax.text(tx, top - 0.76, 'G = (V, E, R),   R = {seq, skip, sem}', fontsize=8.4, va='top', family='DejaVu Sans Mono')
    rows = [(SEQ_GREY, 2.0, '-', '$E_{seq}$', 'adjacent tokens (i, i+1)'),
            (SKIP_GREY, 1.2, (0, (3, 2)), '$E_{skip}$', 'tokens two apart (i, i+2)'),
            (SEM_RED, 2.8, '-', '$E_{sem}$', 'the nine ordered keyword pairs, fewer than 15 tokens apart\n'
                                              '(keywords split by comments or whitespace are merged first)')]
    for k, (col, lw, ls, lab, desc) in enumerate(rows):
        yy = top - 1.06 - 0.20 * k
        ax.add_line(Line2D([tx + 0.05, tx + 0.45], [yy, yy], color=col, lw=lw, ls=ls))
        ax.text(tx + 0.55, yy, lab, fontsize=8.6, va='center')
        ax.text(tx + 1.30, yy - (0.05 if '\n' in desc else 0), desc, fontsize=7.8, va='center', linespacing=1.25)
    down_arrow(ax, xc, y, y - GAP, 'graph G')
    y -= GAP

    # ---- stage 3
    top = y
    h3 = 1.30
    y = stage(ax, top, h3, 3, 'Token feature encoding')
    ax.text(tx, top - 0.56, '64-dimensional vector per node, computed from the token alone:', fontsize=8, va='top')
    for k, line in enumerate(['statistics: length, relative position, entropy, digit and non-alphanumeric ratios, …',
                              'indicators for 15 dangerous characters or character sequences',
                              'one-hot keyword indicators: 9 SQL and 7 web keywords']):
        ax.text(tx + 0.10, top - 0.79 - 0.18 * k, '•  ' + line, fontsize=7.9, va='top')
    down_arrow(ax, xc, y, y - GAP, 'graph with node features')
    y -= GAP

    # ---- stage 4
    top = y
    h4 = 1.22
    y = stage(ax, top, h4, 4, 'Attention-based inductive learning')
    chips = ['2 GATv2 layers\n(8 heads,\nBatchNorm + ELU)', 'jumping knowledge\n(concatenation of\nthe layer outputs)',
             'global\nmax-pooling', 'MLP\n(dropout 0.5)']
    cw, cg = 1.20, 0.22
    cx = tx
    cy_top = top - 0.55
    ch = 0.56
    for k, txt in enumerate(chips):
        x = cx + k * (cw + cg)
        rbox(ax, x, cy_top - ch, cw, ch, '#FFFFFF', '#5B7392', lw=1.1)
        ax.text(x + cw / 2, cy_top - ch / 2, txt, fontsize=7.6, ha='center', va='center', linespacing=1.15)
        if k < len(chips) - 1:
            ax.add_patch(FancyArrowPatch((x + cw + 0.02, cy_top - ch / 2), (x + cw + cg - 0.02, cy_top - ch / 2),
                                         arrowstyle='-|>', mutation_scale=9, color='#444444', lw=1.2, shrinkA=0, shrinkB=0))
    down_arrow(ax, xc, y, y - GAP, 'softmax')
    y -= GAP

    # ---- output
    rbox(ax, xc - 1.6, y - 0.56, 3.2, 0.56, '#EAF5EF', OUT_GREEN, lw=1.3, r=0.16)
    ax.text(xc, y - 0.15, 'Output: class probabilities', fontsize=8.2, ha='center', va='center', color='#1B4D38', fontweight='bold')
    for k, lab in enumerate(['Benign', 'SQLi', 'XSS']):
        px = xc - 1.05 + k * 1.05
        rbox(ax, px - 0.4, y - 0.50, 0.8, 0.21, '#FFFFFF', OUT_GREEN, lw=0.9, r=0.08, z=4)
        ax.text(px, y - 0.395, lab, fontsize=7.6, ha='center', va='center', zorder=5)
    y -= 0.56
    assert y > 0.05, f'content overflows the canvas by {0.05 - y:.2f} in'

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    fig.savefig(out, format='svg')
    n_fixed = harden_svg_text(out)
    print(f'[+] {n_fixed} mathtext labels rewritten as single <text> with relative-positioned subscripts')
    print(f'[+] wrote {out}  ({out.stat().st_size} bytes)   canvas {W} x {H} in   free space at the bottom: {y:.2f} in')


if __name__ == '__main__':
    main()
