"""Editable vector source (SVG) of the new Fig. 2: behavioural attack graphs (BAG) of a Benign, an SQLi and an XSS request.

The old Fig. 2 drew HTTP-header nodes in panel (a) (LOGWAT's input is ONE content string, not HTTP fields) and an E_sem edge
SELECT -- sys.users that is not one of the nine keyword pairs of semantic_edges() (panel b). This version uses three example
content strings and takes the tokens and ALL edges from the project's own code, so that nothing in the drawing can disagree with
the builder:
  tokens    src.preprocessing.tokenizer.web_security_tokenizer   (percent-decode once, lower-case, flat token sequence)
  E_seq     src.edges.sequential.sequential_edges                (i <-> i+1)
  E_skip    src.edges.skip.skip_edges(k=2)                       (i <-> i+2, as in src.bag.graph_builder)
  E_sem     src.edges.semantic.semantic_edges(window=15)         (the nine ordered keyword pairs; both directions)
Every edge type is added in both directions by the builder, so E_seq and E_skip are drawn as plain lines and E_sem with an
arrowhead at each end. The layout is a zig-zag (even tokens on the upper level, odd tokens on the lower level), so the boxes of
long tokens never overlap and E_skip (tokens two apart) always joins two boxes of the same level.

Visual language shared with Fig. 6: E_seq solid light grey, E_skip dashed dark grey, E_sem heavy red; tokens that take part in an
E_sem edge are drawn with the pink/red box of Fig. 6.

Text is kept as <text> objects (svg.fonttype = 'none'); mathtext labels (E_seq ...) are rewritten as ONE <text> with
relative-positioned subscripts by make_fig6_new.harden_svg_text (same fix as Fig. 6). The PDF is exported by hand from Inkscape.

Usage: python scripts/make_fig2_new.py [output.svg]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.edges.semantic import semantic_edges, semantic_pairs           # noqa: E402
from src.edges.sequential import sequential_edges                       # noqa: E402
from src.edges.skip import skip_edges                                   # noqa: E402
from src.preprocessing.tokenizer import web_security_tokenizer          # noqa: E402
from make_fig6_new import harden_svg_text                               # noqa: E402  (also sets the shared rcParams)

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['svg.hashsalt'] = 'logwat-fig2'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = Path('/home/dhgia/Work/iot_mailware/hin_web_vulne/paper_LOGWAT/Fig/Fig2_new.svg')

SEM_RED = '#D62728'
SEQ_GREY = '#B8B8B8'
SKIP_GREY = '#555555'

EXAMPLES = [
    ('(a) Benign request', 'id=3&nombre=Queso+Manchego&precio=100&cantidad=25'),
    ('(b) SQL injection', 'UNION SELECT username, password FROM users--'),
    ('(c) Cross-site scripting', '<img src=x onerror=alert(1)>'),
]
# what the builder must produce for these strings (undirected token-index pairs of E_sem)
EXPECTED_SEM = [set(), {(0, 1), (1, 5)}, {(1, 5)}]
EXPECTED_N = [17, 9, 12]

W = 6.6                 # figure width in inches; every coordinate below is in inches
BOX_H = 0.24
LEVEL_DY = 0.60         # vertical distance between the upper and the lower level of the zig-zag
CHAR_W = 0.0585         # width of one DejaVu Sans Mono character at 7 pt, in inches


def build(text):
    """Tokens and the three edge lists, straight from the project's code."""
    tokens = web_security_tokenizer(text)
    seq = {tuple(sorted(e)) for e in sequential_edges(tokens)}
    skip = {tuple(sorted(e)) for e in skip_edges(tokens, k=2)}
    sem = {tuple(sorted(e)) for e in semantic_edges(tokens, window=15)}
    n = len(tokens)
    assert seq == {(i, i + 1) for i in range(n - 1)} and skip == {(i, i + 2) for i in range(n - 2)}
    for i, j in sem:                                                   # every E_sem edge is one of the nine pairs
        assert (tokens[i], tokens[j]) in semantic_pairs, (tokens[i], tokens[j])
    return tokens, sorted(seq), sorted(skip), sorted(sem)


def box_w(label):
    return max(0.22, CHAR_W * len(label) + 0.13)


def token_box(ax, x, y, label, hot):
    w = box_w(label)
    fc, ec, lw, tc = ('#FBE3E4', SEM_RED, 1.6, '#7A1F1F') if hot else ('#FFFFFF', '#777777', 1.0, '#333333')
    ax.add_patch(FancyBboxPatch((x - w / 2, y - BOX_H / 2), w, BOX_H, boxstyle='round,pad=0.0,rounding_size=0.05',
                                fc=fc, ec=ec, lw=lw, zorder=4))
    ax.text(x, y, label, ha='center', va='center', fontsize=7, family='DejaVu Sans Mono', color=tc, zorder=5)


def arc(ax, p, q, color, lw, bulge, side, ls='-', both=False, ms=9, z=3):
    """Circular arc from p to q that bulges by `bulge` inches upwards (side='up') or downwards (side='down'),
    for a left-to-right chord; rad = 2*bulge/chord."""
    chord = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5
    rad = 2 * bulge / chord * (1 if side == 'down' else -1)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='<|-|>' if both else '-', mutation_scale=ms,
                                 connectionstyle=f'arc3,rad={rad}', color=color, lw=lw, ls=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def panel(ax, y_top, title, raw, tokens, seq, skip, sem, x_left, x_right, sem_note):
    n = len(tokens)
    hot = {i for e in sem for i in e}
    pitch = (x_right - x_left) / (n - 1)
    y_up = y_top - 0.92
    y_low = y_up - LEVEL_DY
    xs = [x_left + i * pitch for i in range(n)]
    ys = [y_up if i % 2 == 0 else y_low for i in range(n)]
    for i in range(n - 2):                                              # same-level neighbours must not overlap
        assert 2 * pitch - (box_w(tokens[i]) + box_w(tokens[i + 2])) / 2 > 0.04, (tokens[i], tokens[i + 2], pitch)

    ax.text(0.06, y_top, title, fontsize=9.5, fontweight='bold', va='top')
    ax.text(0.06, y_top - 0.24, 'content string:', fontsize=7, color='#777777', va='top')
    ax.text(0.98, y_top - 0.24, raw, fontsize=7.3, family='DejaVu Sans Mono', color='#222222', va='top')

    for i, j in seq:                                                    # E_seq (both directions), centre to centre, under the boxes
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=SEQ_GREY, lw=1.6, zorder=1, solid_capstyle='butt')
    for i, j in skip:                                                   # E_skip: arcs that bulge away from the other level
        up = ys[i] == y_up
        y_edge = ys[i] + (BOX_H / 2 if up else -BOX_H / 2)
        arc(ax, (xs[i], y_edge), (xs[j], y_edge), SKIP_GREY, 0.9, 0.15, 'up' if up else 'down', ls=(0, (3, 2)), z=2)
    for i, j in sem:                                                    # E_sem, heavy red, arrowheads at both ends
        if ys[i] == ys[j]:                                              # same level (lower): wide arc under the skip arcs
            y_edge = ys[i] - BOX_H / 2
            arc(ax, (xs[i], y_edge), (xs[j], y_edge), SEM_RED, 2.5, 0.42, 'down', both=True, z=6)
        else:                                                           # adjacent tokens: arc to the left of the seq line
            arc(ax, (xs[i] - box_w(tokens[i]) / 2, ys[i]), (xs[j] - box_w(tokens[j]) / 2, ys[j]), SEM_RED, 2.5, 0.20,
                'down', both=True, z=6)
    for i in range(n):
        token_box(ax, xs[i], ys[i], tokens[i], i in hot)

    y_note = y_low - BOX_H / 2 - (0.62 if sem else 0.36)
    if sem:
        ax.text(W / 2, y_note, sem_note, ha='center', va='top', fontsize=8, color=SEM_RED, fontweight='bold')
    else:
        ax.text(W / 2, y_note, sem_note, ha='center', va='top', fontsize=8, color='#666666', style='italic')
    return y_note - 0.22


def main():
    built = [build(raw) for _, raw in EXAMPLES]
    for (tokens, seq, skip, sem), exp_sem, exp_n in zip(built, EXPECTED_SEM, EXPECTED_N):
        assert len(tokens) == exp_n and set(sem) == exp_sem, (tokens, sem)
    notes = ['no keyword pair occurs in the string, so the builder adds no $E_{sem}$ edge',
             '$E_{sem}$: union $\\leftrightarrow$ select and select $\\leftrightarrow$ from (2 of the 9 keyword pairs)',
             '$E_{sem}$: img $\\leftrightarrow$ onerror (1 of the 9 keyword pairs)']

    H = 8.4
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    # shared legend, same colours / line styles as Fig. 6
    y_leg = H - 0.16
    items = [(SEQ_GREY, 2.0, '-', '$E_{seq}$  adjacent tokens (i, i+1)', 0.10),
             (SKIP_GREY, 1.2, (0, (3, 2)), '$E_{skip}$  tokens two apart (i, i+2)', 2.30),
             (SEM_RED, 2.8, '-', '$E_{sem}$  keyword pair', 4.55)]
    for color, lw, ls, label, x in items:
        ax.add_line(Line2D([x, x + 0.34], [y_leg, y_leg], color=color, lw=lw, ls=ls))
        ax.text(x + 0.42, y_leg, label, fontsize=8, va='center')
    ax.text(W - 0.05, y_leg - 0.22, 'every edge is added in both directions', fontsize=6.8, color='#777777', ha='right', va='center')

    y = H - 0.55
    for k, ((title, raw), (tokens, seq, skip, sem), note) in enumerate(zip(EXAMPLES, built, notes)):
        y = panel(ax, y, title, raw, tokens, seq, skip, sem, 0.45 if k == 0 else 0.75, W - (0.45 if k == 0 else 0.75), note) - 0.08
    ax.text(W / 2, 0.10, 'Tokens are shown after the tokenizer\'s lower-casing; edges are the ones built by the graph builder for each string.',
            ha='center', va='bottom', fontsize=6.8, color='#777777', style='italic')
    assert y > 0.3, f'panels overflow the canvas ({y:.2f} in left)'

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    fig.savefig(out, format='svg')
    n_fixed = harden_svg_text(out)
    print(f'[+] {n_fixed} mathtext labels rewritten as single <text> with relative-positioned subscripts')
    print(f'[+] wrote {out}  ({out.stat().st_size} bytes)   canvas {W} x {H} in   free space below panel (c): {y:.2f} in')
    for (title, raw), (tokens, seq, skip, sem) in zip(EXAMPLES, built):
        print(f'    {title:28s} n={len(tokens):2d} seq={len(seq)} skip={len(skip)} sem={[(tokens[i], tokens[j]) for i, j in sem]}')


if __name__ == '__main__':
    main()
