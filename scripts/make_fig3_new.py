"""Editable vector source (SVG) of the new Fig. 3: the three dataset-construction mechanisms of BIA.

Same visual language as the old Fig3_BIA.pdf (four rounded blocks joined by curved arrows: benign space, malicious payload
space, engine, augmented dataset) but the content matches Section 3.3 (Mechanism 1 / 2 / 2b); there is no optimization step.

Every number is read from the data, not typed in:
  data/augmented_web_attack.csv           row counts by class and source
  data/sqli_payload_pool.csv              payloads per family
  data/XSS_dataset.csv                    external XSS corpus (Label == 1), unique payloads
  data/train_noise_augmentation_report.txt   Mechanism 2: sampled / skipped-without-keyword counts
  data/xss_context_augmentation_report.txt   Mechanism 2b: window used by the acceptance check
The example strings are real rows/payloads from the same files. Text stays <text> (svg.fonttype = 'none').
Usage: PYTHONPATH=. python -m scripts.make_fig3_new [out.svg]
"""
import re
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['svg.hashsalt'] = 'logwat-fig3'
matplotlib.rcParams['font.family'] = 'Liberation Serif'

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = Path('/home/dhgia/Work/iot_mailware/hin_web_vulne/paper_LOGWAT/Fig/Fig3_BIA_new.svg')
MONO = 'Liberation Mono'

BLUE = dict(fc='#E9EEFB', ec='#9AA5E8')
RED = dict(fc='#FDE7E7', ec='#E53935')
YEL = dict(fc='#F8FBE8', ec='#F2C46D')
GRN = dict(fc='#E5F7EA', ec='#1E8E3E')
ARROW = '#3F4CF2'
ORANGE = '#D9731A'


def n(x):
    return f"{x:,}"


def load():
    d = pd.read_csv(DATA / 'augmented_web_attack.csv', keep_default_na=False)
    src = d.groupby(['attack_type', 'source']).size()
    cls = d.groupby('attack_type').size()
    pool = pd.read_csv(DATA / 'sqli_payload_pool.csv')
    fam = pool['family'].value_counts()
    xss = pd.read_csv(DATA / 'XSS_dataset.csv')
    xss_unique = xss[xss['Label'] == 1]['Sentence'].dropna().astype(str).nunique()
    rep2 = (DATA / 'train_noise_augmentation_report.txt').read_text()
    m2 = re.search(r'sampled for noising: (\d+) \(skipped, no keyword match: (\d+)\)', rep2)
    rep2b = (DATA / 'xss_context_augmentation_report.txt').read_text()
    win = int(re.search(r'window=(\d+)', rep2b).group(1))
    v = dict(
        total=len(d), benign=int(cls[0]), sqli=int(cls[1]), xss=int(cls[2]),
        ben_csic=int(src[(0, '')]), ben_syn=int(src[(0, 'benign_short_synthetic')]),
        sqli_m1=int(src[(1, 'sqli_pool')] + src[(1, 'sqli_pool_extra')]), sqli_m2=int(src[(1, 'sqli_pool_noise')]),
        sqli_csic=int(src[(1, 'csic_original')]), xss_m1=int(src[(2, '')]), xss_m2b=int(src[(2, 'xss_pool_context')]),
        n_pool=len(pool), n_fam=len(fam), fam=fam, xss_unique=xss_unique,
        m2_sampled=int(m2.group(1)), m2_skipped=int(m2.group(2)), window=win,
    )
    assert v['sqli_m1'] + v['sqli_m2'] + v['sqli_csic'] == v['sqli'] and v['xss_m1'] + v['xss_m2b'] == v['xss']
    assert v['ben_csic'] + v['ben_syn'] == v['benign'] and v['benign'] + v['sqli'] + v['xss'] == v['total']
    # real example strings
    ben = d[(d.attack_type == 0) & (d.source == '')].content.iloc[0]
    v['ex_benign'] = ben.split('&B1=')[0]
    v['ex_sqli'] = pool[pool.family == 'UNION-based'].payload.iloc[0]
    xs = xss[xss['Label'] == 1]['Sentence'].astype(str)
    v['ex_xss'] = next(s for s in xs if s.startswith('<svg><meta onload='))
    noised = d[d.source == 'sqli_pool_noise']
    v['ex_noise'] = noised[noised.content.str.contains('F   ROM')].content.iloc[0]
    ctx_rows = d[d.source == 'xss_pool_context'].content
    v['ex_ctx'] = next(c for c in ctx_rows if ' style=' in c and c.index(' style=') < c.index(' id=') < c.index(' onload='))
    return v


def box(ax, x, y, w, h, style, lw=1.6, dashed=False, r=0.12, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=style['fc'], ec=style['ec'],
                                lw=lw, ls=(0, (3, 2)) if dashed else '-', zorder=z))


def db_icon(ax, x, y, w, h, color):
    """Stacked-cylinder database icon, lower-left corner at (x, y)."""
    eh = h * 0.22
    for k in range(3):
        yy = y + k * (h - eh) / 2.6
        ax.add_patch(Rectangle((x, yy + eh / 2), w, (h - eh) / 2.6, fc=color, ec='none', zorder=5))
        ax.add_patch(Ellipse((x + w / 2, yy + eh / 2), w, eh, fc=color, ec='white', lw=0.8, zorder=6))
    ax.add_patch(Ellipse((x + w / 2, y + h - eh / 2), w, eh, fc=color, ec='white', lw=0.8, zorder=7))


def gear_icon(ax, cx, cy, r, color):
    import numpy as np
    pts = []
    for k in range(16):
        a = 2 * np.pi * k / 16
        rr = r if k % 2 == 0 else r * 0.78
        pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec='none', zorder=5))
    ax.add_patch(Ellipse((cx, cy), r * 0.9, r * 0.9, fc=YEL['fc'], ec='none', zorder=6))


def curve(ax, p, q, rad):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14, connectionstyle=f'arc3,rad={rad}',
                                 color=ARROW, lw=2.0, zorder=8, shrinkA=0, shrinkB=0))


CHECKS = []   # (text artist, container x-range) -> verified after drawing so nothing overflows its block


def t(ax, x, y, s, size=9, weight='normal', color='#1a1a1a', ha='left', style='normal', family=None, va='center', z=10, within=None):
    kw = dict(fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style, zorder=z)
    if family:
        kw['family'] = family
    a = ax.text(x, y, s, **kw)
    if within:
        CHECKS.append((a, within))
    return a


def example(ax, x, y, w, lines, size=7.6, within=None):
    h = 0.17 * len(lines) + 0.10
    box(ax, x, y, w, h, dict(fc='#FBF6E3', ec='#E8B860'), lw=0.9, dashed=True, r=0.06, z=3)
    for k, ln in enumerate(lines):
        t(ax, x + 0.07, y + h - 0.05 - 0.17 * (k + 0.5), ln, size=size, family=MONO, color='#333333', within=(x, x + w))
    return h


def main():
    v = load()
    W, H = 8.2, 5.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    NB = ' '   # keep runs of spaces visible in SVG text

    # ---------------- block 1: benign data space
    x1, w1 = 0.04, 2.20
    c1 = (x1, x1 + w1)
    box(ax, x1, 2.68, w1, 2.28, BLUE)
    t(ax, x1 + 0.13, 4.70, 'Benign Data Space', 11.5, 'bold', within=c1)
    t(ax, x1 + 0.13, 4.46, '(T: frames)', 9, style='italic', color='#444444', within=c1)
    db_icon(ax, x1 + w1 - 0.52, 4.38, 0.36, 0.42, '#2C4FA8')
    t(ax, x1 + 0.13, 4.06, 'CSIC 2010 request frames', 9, within=c1)
    t(ax, x1 + 0.13, 3.87, 'not used as Benign-class rows', 9, within=c1)
    t(ax, x1 + 0.13, 3.50, 'e.g. a real request content:', 8.4, style='italic', color='#444444', within=c1)
    eb = v['ex_benign']
    example(ax, x1 + 0.10, 2.86, w1 - 0.20, [eb[:27], eb[27:] + '&…'], within=c1)

    # ---------------- block 2: malicious payload space
    box(ax, x1, 0.04, w1, 2.50, RED)
    t(ax, x1 + 0.13, 2.28, 'Malicious Payload Space', 11.5, 'bold', within=c1)
    t(ax, x1 + 0.13, 2.05, '(P: two separate sources)', 9, style='italic', color='#444444', within=c1)
    short = {'Boolean-based': 'Boolean', 'UNION-based': 'UNION', 'Time-based': 'Time', 'Error-based': 'Error', 'Stacked/other': 'Stacked/other'}
    f = v['fam']
    t(ax, x1 + 0.13, 1.83, f"SQLi pool: {v['n_pool']} payloads, {v['n_fam']} families", 9, 'bold', '#7A1F1F', within=c1)
    t(ax, x1 + 0.13, 1.66, f"{short['Boolean-based']} {f['Boolean-based']} · {short['UNION-based']} {f['UNION-based']} · {short['Time-based']} {f['Time-based']}", 7.8, color='#333333', within=c1)
    t(ax, x1 + 0.13, 1.51, f"{short['Error-based']} {f['Error-based']} · {short['Stacked/other']} {f['Stacked/other']}", 7.8, color='#333333', within=c1)
    p1 = v['ex_sqli']
    cut = p1.index('password')
    example(ax, x1 + 0.10, 0.94, w1 - 0.20, [p1[:cut].rstrip(), p1[cut:]], within=c1)
    t(ax, x1 + 0.13, 0.81, 'External XSS corpus', 9, 'bold', '#7A1F1F', within=c1)
    t(ax, x1 + 0.13, 0.665, f"{n(v['xss_unique'])} unique payloads", 7.8, color='#333333', within=c1)
    px = v['ex_xss']
    k = px.index('alert(1)>') + len('alert(1)>')
    example(ax, x1 + 0.10, 0.09, w1 - 0.20, [px[:k], px[k:]], within=c1)

    # ---------------- block 3: the three mechanisms
    x2, w2, y2, h2 = 2.68, 3.42, 0.30, 4.46
    c2 = (x2 + 0.10, x2 + w2 - 0.10)
    box(ax, x2, y2, w2, h2, YEL)
    gear_icon(ax, x2 + 0.33, y2 + h2 - 0.34, 0.19, '#E8A33D')
    t(ax, x2 + 0.62, y2 + h2 - 0.27, 'BIA: THREE MECHANISMS', 11.5, 'bold', ORANGE, within=c2)
    t(ax, x2 + 0.62, y2 + h2 - 0.50, 'applied once, when the training data are built', 8.6, style='italic', color='#444444', within=c2)
    box(ax, x2 + 0.10, y2 + 0.12, w2 - 0.20, h2 - 0.80, dict(fc='#FBEFEF', ec='#444444'), lw=1.0, dashed=True, r=0.08, z=2)
    xx = x2 + 0.22
    y = y2 + h2 - 0.84
    d = 0.168
    T = 8.5
    t(ax, xx, y, '1. Mechanism 1 – Template poisoning', 9.2, 'bold', within=c2); y -= d
    t(ax, xx + 0.12, y, 'SQLi and XSS: frame content ← payload, both cycled', T, within=c2); y -= d
    t(ax, xx + 0.12, y, 'T[i mod |T|], P[i mod |P|]; no optimization step', T, within=c2); y -= d
    t(ax, xx + 0.12, y, f"→ {n(v['sqli_m1'] + v['xss_m1'])} rows (SQLi {n(v['sqli_m1'])} · XSS {n(v['xss_m1'])})", T, color='#7A1F1F', within=c2); y -= d * 1.9

    t(ax, xx, y, '2. Mechanism 2 – Whitespace-noise insertion', 9.2, 'bold', within=c2); y -= d
    t(ax, xx + 0.12, y, 'SQLi: copy a row, insert 1–3 whitespace characters', T, within=c2); y -= d
    t(ax, xx + 0.12, y, 'inside one of 29 fixed keywords; no match → skipped', T, within=c2); y -= d
    t(ax, xx + 0.12, y, f"→ +{n(v['sqli_m2'])} rows ({v['m2_skipped']} of {n(v['m2_sampled'])} sampled rows skipped)", T, color='#7A1F1F', within=c2); y_res = y
    ne = '…' + v['ex_noise'][v['ex_noise'].index('password'):]
    ne = ne.replace(' ', NB)
    eh = 0.17 * 1 + 0.10
    example(ax, xx + 0.12, y_res - 0.14 - eh, w2 - 0.64, [ne], size=7.6, within=c2)
    y = y_res - 0.14 - eh - 0.34

    t(ax, xx, y, '3. Mechanism 2b – Benign-attribute insertion', 9.2, 'bold', within=c2); y -= d
    t(ax, xx + 0.12, y, 'XSS: copy a row, insert 1–2 benign HTML attributes', T, within=c2); y -= d
    t(ax, xx + 0.12, y, f"between a keyword pair; gap < {v['window']} (E_sem window)", T, within=c2); y -= d
    t(ax, xx + 0.12, y, f"→ +{n(v['xss_m2b'])} rows", T, color='#7A1F1F', within=c2); y_res = y
    ctx = v['ex_ctx']
    a_ = ctx.index(' style=')
    b_ = ctx.index(' id=')
    end = ctx.index('>', ctx.index('onload')) + 1
    example(ax, xx + 0.12, y_res - 0.14 - (0.17 * 2 + 0.10), w2 - 0.64, [ctx[:a_] + ' style=…', ctx[b_ + 1:end] + '…'], size=7.6, within=c2)

    # ---------------- block 4: augmented dataset
    x3, w3, y3, h3 = 6.50, 1.66, 0.62, 3.80
    c3 = (x3, x3 + w3)
    box(ax, x3, y3, w3, h3, GRN)
    t(ax, x3 + w3 / 2, y3 + h3 - 0.26, 'Augmented Dataset', 11.0, 'bold', ha='center', within=c3)
    t(ax, x3 + w3 / 2, y3 + h3 - 0.50, f"{n(v['total'])} rows", 10.5, 'bold', '#14602B', ha='center', within=c3)
    yy = y3 + h3 - 0.88
    rows = [
        ('Benign', v['benign'], [f"{n(v['ben_csic'])} CSIC frames", f"+ {n(v['ben_syn'])} synthetic", 'header/JSON (not BIA)']),
        ('SQLi', v['sqli'], [f"{n(v['sqli_m1'])} (M1) + {n(v['sqli_m2'])} (M2)", f"+ {v['sqli_csic']} unmodified CSIC"]),
        ('XSS', v['xss'], [f"{n(v['xss_m1'])} (M1) + {v['xss_m2b']} (M2b)"]),
    ]
    for name, cnt, sub in rows:
        t(ax, x3 + 0.12, yy, f"{name}: {n(cnt)}", 9.6, 'bold', within=c3)
        yy -= 0.18
        for s_ in sub:
            t(ax, x3 + 0.12, yy, s_, 7.9, color='#333333', within=c3)
            yy -= 0.15
        yy -= 0.14
    shares = ' / '.join(f"{100 * v[k]/v['total']:.1f}" for k in ('benign', 'sqli', 'xss'))
    t(ax, x3 + w3 / 2, y3 + 0.34, 'class sizes are unequal:', 7.9, style='italic', color='#444444', ha='center', within=c3)
    t(ax, x3 + w3 / 2, y3 + 0.17, f"{shares} %", 7.9, style='italic', color='#444444', ha='center', within=c3)

    # ---------------- arrows
    curve(ax, (x1 + w1 + 0.02, 3.80), (x2 - 0.02, 3.58), -0.30)
    curve(ax, (x1 + w1 + 0.02, 1.30), (x2 - 0.02, 1.55), 0.30)
    curve(ax, (x2 + w2 + 0.02, 2.55), (x3 - 0.02, 2.55), -0.35)

    # every text must stay inside its block
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for a, (lo, hi) in CHECKS:
        bb = a.get_window_extent(fig.canvas.get_renderer())
        (x0, _), (x1_, _) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        if x0 < lo - 0.005 or x1_ > hi + 0.005:
            bad.append((a.get_text()[:50], round(x0, 2), round(x1_, 2), lo, round(hi, 2)))
    print('text overflow check:', 'OK, all', len(CHECKS), 'labels inside their blocks' if not bad else f'{len(bad)} OVERFLOW')
    for b in bad:
        print('   ', b)

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    fig.savefig(out, format='svg')
    print(f"[+] wrote {out} ({out.stat().st_size} bytes)")
    for k in ('total', 'benign', 'sqli', 'xss', 'ben_csic', 'ben_syn', 'sqli_m1', 'sqli_m2', 'sqli_csic', 'xss_m1', 'xss_m2b',
              'n_pool', 'xss_unique', 'm2_sampled', 'm2_skipped', 'window'):
        print(f"  {k}: {v[k]}")
    print('  families:', v['fam'].to_dict())
    for k in ('ex_benign', 'ex_sqli', 'ex_xss', 'ex_noise', 'ex_ctx'):
        print(f"  {k}: {v[k]!r}")


if __name__ == '__main__':
    main()
