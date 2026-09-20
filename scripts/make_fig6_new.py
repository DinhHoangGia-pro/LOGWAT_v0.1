"""Editable vector source (SVG) of the new Fig. 6: 'Case study: edge existence versus edge processing'.

Panel (a): simplified BAG of the fragmented payload (E_seq / E_skip / E_sem) with the GATv2 attention annotation.
Panel (b): SQLi logit margin per configuration and payload, 5 seeds (points) + mean (bar).

Numbers come from data/case_study_fig6.txt (means, predictions, attention) and data/case_study_fig6_perseed.json
(per-seed margins, scripts/case_study_fig6_perseed.py); the script asserts that the two agree before drawing.
Text is kept as <text> objects (svg.fonttype = 'none') so the SVG stays editable in Inkscape/Illustrator.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.lines import Line2D

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['svg.hashsalt'] = 'logwat-fig6'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['hatch.linewidth'] = 0.8

ROOT = Path(__file__).resolve().parent.parent
TXT = ROOT / 'data' / 'case_study_fig6.txt'
JSON = ROOT / 'data' / 'case_study_fig6_perseed.json'
OUT = Path('/home/dhgia/Work/iot_mailware/hin_web_vulne/paper_LOGWAT/Fig/Fig.6_new.svg')

SEM_RED = '#D62728'
SEQ_GREY = '#B8B8B8'
SKIP_GREY = '#555555'
C_FRAG = '#0072B2'   # fragmented payload (dark: still separable in greyscale)
C_INTACT = '#F0E442'  # intact-keyword payload (light)


# ---------------------------------------------------------------- data + cross-check against the .txt
def load():
    seeds = json.loads(JSON.read_text())
    txt = TXT.read_text()
    blocks = {p: b for p, b in zip(('P_intact', 'P_split'), re.split(r'^=== P_', txt, flags=re.M)[1:3])}
    checks = {}
    for pname, blk in (('P_intact', blocks['P_intact']), ('P_split', blocks['P_split'])):
        for m in re.finditer(r'edit (\w+)\s+(\w+)\s+pred = (\w+)\s+mean SQLi margin ([+-][\d.]+)', blk):
            checks[f"{pname}|{m.group(1)}|{m.group(2)}"] = (m.group(3), float(m.group(4)))
    for key, (pred, mean) in checks.items():
        if key not in seeds:
            continue
        got = float(np.mean(seeds[key]))
        assert abs(got - mean) < 0.006, (key, got, mean)
        assert pred.count('S') == sum(v > 0 for v in seeds[key]), (key, pred, seeds[key])
    att = re.search(r'degree-corrected.*?sem = ([\d.]+), other = ([\d.]+)', blocks['P_split'])
    split_head = re.search(r"edges \{'seq': (\d+), 'skip': (\d+), 'sem': (\d+)\} \(total (\d+)\)", blocks['P_split'])
    return seeds, checks, (float(att.group(1)), float(att.group(2))), tuple(int(x) for x in split_head.groups())


# ---------------------------------------------------------------- panel (a)
def token_box(ax, x, y, label, kind):
    w = 0.64 if kind != 'noise' else 0.76
    style = dict(frag=('#FBE3E4', SEM_RED, 1.6, '#7A1F1F'), other=('#FFFFFF', '#777777', 1.0, '#333333'),
                 noise=('#F1F1F1', '#C8C8C8', 1.0, '#999999'))[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.3), w, 0.6, boxstyle='round,pad=0.02,rounding_size=0.14',
                                fc=style[0], ec=style[1], lw=style[2], zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=7.6, family='DejaVu Sans Mono', color=style[3], zorder=4)


def arrow(ax, p, q, color, lw, ls='-', rad=0.0, style='-|>', ms=7, z=2, both=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='<|-|>' if both else style, mutation_scale=ms,
                                 connectionstyle=f'arc3,rad={rad}', color=color, lw=lw, ls=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def panel_a(ax, att, counts):
    seq, skip, sem, total = counts
    ax.set_xlim(-0.6, 13.6)
    ax.set_ylim(-3.6, 4.6)
    ax.axis('off')
    nodes = [('p=0;', 'other'), ('uni', 'frag'), ('/**/', 'noise'), ('on', 'frag'), ('all', 'other'), ('sel', 'frag'),
             ('/**/', 'noise'), ('ect', 'frag'), ('*', 'other'), ('fr', 'frag'), ('/**/', 'noise'), ('om', 'frag'),
             ('users', 'other')]
    xs = list(range(len(nodes)))
    y0 = 0.0
    # E_skip (k=2), drawn under the chain; E_seq between neighbours
    for i in range(len(nodes) - 2):
        arrow(ax, (xs[i], y0 - 0.32), (xs[i + 2], y0 - 0.32), '#777777', 0.7, ls=(0, (3, 2)), rad=0.5, ms=1, style='-')
    for i in range(len(nodes) - 1):
        arrow(ax, (xs[i] + 0.35, y0), (xs[i + 1] - 0.35, y0), SEQ_GREY, 1.4, ms=5, both=True)
    for x, (lab, kind) in zip(xs, nodes):
        token_box(ax, x, y0, lab, kind)
    # E_sem: on<->ect and ect<->om (the last fragment of each reconstructed keyword)
    on, ect, om = 3, 7, 11
    arrow(ax, (on, y0 + 0.36), (ect, y0 + 0.36), SEM_RED, 2.6, rad=-0.42, ms=10, both=True, z=5)
    arrow(ax, (ect, y0 + 0.36), (om, y0 + 0.36), SEM_RED, 2.6, rad=-0.42, ms=10, both=True, z=5)
    ax.text((on + ect) / 2, 2.4, '$E_{sem}$', color=SEM_RED, fontsize=9.5, ha='center', va='bottom', fontweight='bold')
    ax.text((ect + om) / 2, 2.4, '$E_{sem}$', color=SEM_RED, fontsize=9.5, ha='center', va='bottom', fontweight='bold')
    ax.text(7.6, 3.8, f"GATv2 layer-1 attention on $E_{{sem}}$ ≈ uniform ({att[0]:.3f} vs {att[1]:.3f} on other edges)",
            color=SEM_RED, fontsize=8.5, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=SEM_RED, lw=0.9))
    # reconstructed keywords: brackets under the chain
    yb = -1.6
    for (a, b, lab) in ((1, 3, 'union'), (5, 7, 'select'), (9, 11, 'from')):
        ax.plot([a, a, b, b], [yb + 0.14, yb, yb, yb + 0.14], color='#222222', lw=1.0, zorder=2)
        ax.text((a + b) / 2, yb - 0.1, f"{lab}", ha='center', va='top', fontsize=8.5, style='italic', color='#222222')
    ax.text(6.5, -2.7, f"The builder reconstructs the fragmented keywords and adds {sem} directed $E_{{sem}}$ edges among {total} "
                        f"({seq} $E_{{seq}}$, {skip} $E_{{skip}}$).",
            ha='center', va='center', fontsize=7.8, color='#333333')
    # legend (manual)
    handles = [Line2D([0], [0], color=SEQ_GREY, lw=1.8, label='$E_{seq}$'),
               Line2D([0], [0], color=SKIP_GREY, lw=1.0, ls=(0, (3, 2)), label='$E_{skip}$'),
               Line2D([0], [0], color=SEM_RED, lw=2.6, label='$E_{sem}$')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=8, frameon=False,
              handlelength=2.2, borderaxespad=0.0)
    ax.text(6.5, -3.35, 'schematic: each /**/ (4 tokens) is drawn as one node; chain abridged', ha='center', va='center',
            fontsize=6.5, color='#777777')


# ---------------------------------------------------------------- panel (b)
def panel_b(ax, seeds):
    groups = [  # (x centre, config label, (model key, edit))
        (0.0, '$E_{sem}$\npresent', ('gatv2', 'real')),
        (1.0, '$E_{sem}$\ndeleted', ('gatv2', 'drop_sem')),
        (2.0, '$E_{seq}$\ndeleted', ('gatv2', 'drop_seq')),
        (3.35, '$E_{sem}$\npresent', ('hgt', 'real')),
        (4.35, '$E_{sem}$\ndeleted', ('hgt', 'drop_sem')),
    ]
    bw = 0.36
    ax.axhspan(-9, 0, color='#EDEDED', zorder=0)
    ax.axhline(0, color='black', lw=1.1, zorder=1)
    rng = np.random.RandomState(0)
    for gx, _, (mk, cond) in groups:
        for off, pname, color, hatch in ((-bw / 2 - 0.01, 'P_split', C_FRAG, '///'), (bw / 2 + 0.01, 'P_intact', C_INTACT, 'xxx')):
            v = np.array(seeds[f"{pname}|{mk}|{cond}"])
            m = v.mean()
            x = gx + off
            ax.bar(x, m, width=bw, color=color, edgecolor='black', hatch=hatch, lw=0.8, zorder=2)
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=13, c='white', edgecolors='black', linewidths=0.8, zorder=4)
            k = int((v > 0).sum())
            top = v.max() if m > 0 else v.min()
            va, dy = ('bottom', 0.35) if m > 0 else ('top', -0.35)
            ax.text(x, top + dy, f"{m:+.2f}".replace('-', '−') + f"\n({k}/5)" if m > 0 else f"({k}/5)\n" + f"{m:+.2f}".replace('-', '−'),
                    ha='center', va=va, fontsize=6.8, zorder=5, linespacing=1.0)
    ax.set_xticks([g[0] for g in groups])
    ax.set_xticklabels([g[1] for g in groups], fontsize=8.5)
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(-8.2, 20)
    ax.set_ylabel('SQLi logit margin\n(SQLi − Benign)', fontsize=8.5)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', length=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    # group labels
    for x0, x1, lab in ((-0.45, 2.45, 'LOGWAT (GATv2, shared aggregation)'), (2.9, 4.8, 'HGT (typed edges)')):
        ax.annotate('', xy=(x0, -0.125), xytext=(x1, -0.125), xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
                    arrowprops=dict(arrowstyle='-', lw=1.0, color='#222222'), annotation_clip=False)
        ax.text((x0 + x1) / 2, -0.135, lab, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8.5, fontweight='bold')
    ax.text(5.57, 0.5, 'SQLi (correct)', ha='right', va='bottom', fontsize=7.5, style='italic', color='#333333')
    ax.text(5.57, -0.5, 'Benign (missed)', ha='right', va='top', fontsize=7.5, style='italic', color='#333333')
    handles = [Patch(fc=C_FRAG, ec='black', hatch='///', label='fragmented keywords (UNI/**/ON ALL SEL/**/ECT …)'),
               Patch(fc=C_INTACT, ec='black', hatch='xxx', label='intact keywords (UNION/**/SELECT/**/schema)'),
               Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', ms=5, label='one seed (42–46); bar = mean; (k/5) = seeds predicting SQLi')]
    ax.legend(handles=handles, loc='upper left', fontsize=7.3, frameon=True, framealpha=1.0, edgecolor='#BBBBBB', ncol=1,
              bbox_to_anchor=(0.0, 1.0))


def main():
    seeds, checks, att, counts = load()
    fig = plt.figure(figsize=(7.4, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 4.0], hspace=0.14, left=0.09, right=0.985, top=0.975, bottom=0.115)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])
    panel_a(axa, att, counts)
    panel_b(axb, seeds)
    fig.text(0.005, 0.975, '(a)', fontsize=11, fontweight='bold', va='top')
    fig.text(0.005, 0.655, '(b)', fontsize=11, fontweight='bold', va='top')
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    fig.savefig(out, format='svg')
    print(f"[+] wrote {out}  ({out.stat().st_size} bytes)")


if __name__ == '__main__':
    main()
