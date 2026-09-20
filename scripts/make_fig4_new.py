"""Editable vector source (SVG) of the new Fig. 4: 'Convergence of the deployed seed-42 checkpoint'.

Replaces the old 100-epoch curve (accuracy 84% -> 97.84%, loss 0.3856 -> 0.0551), which does not match the current
training configuration. Data come from logs/training_history.log, session '--- NEW SESSION: Fri Sep 18 17:17:46 2026 ---'
(retrain #6, the run behind data/models_pretrained/best_web_gnn_seed42.pth): 19 epochs, best epoch 9, LR halved at epoch 15,
early stop at epoch 19.

Three stacked panels with a shared epoch axis:
  top    accuracy on the frozen test split, y-axis zoomed to 0.998-1.000 (the curve does not climb: the split is saturated
         from epoch 1), with a right-hand axis in misclassified requests (1 request = 1/4215 = 0.000237)
  middle class-weighted training loss (as logged, 4 decimals; 0.0000 means < 0.00005)
  bottom learning rate (5e-4, halved once by ReduceLROnPlateau)
Vertical lines mark the best epoch (9, the checkpoint that is kept), the LR reduction (15) and the early stop (19).

The 'validation accuracy' wording of the log lines is inherited from the training script: there is no validation set, the
monitored accuracy IS the test-split accuracy (paper Sec. 4.1), so the figure calls it test-split accuracy.

Text is kept as <text> objects (svg.fonttype = 'none') so the SVG stays editable in Inkscape/Illustrator. No mathtext is used
(plain strings only), so the subscript-layout problem of Fig. 6 does not arise. The PDF is exported by hand from Inkscape.

Usage: python scripts/make_fig4_new.py [output.svg]
"""
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['svg.hashsalt'] = 'logwat-fig4'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / 'logs' / 'training_history.log'
SESSION = '--- NEW SESSION: Fri Sep 18 17:17:46 2026 ---'
OUT = Path('/home/dhgia/Work/iot_mailware/hin_web_vulne/paper_LOGWAT/Fig/Fig.4_new.svg')

N_TEST = 4215  # frozen test split (1550 Benign + 1214 SQLi + 1451 XSS)

C_ACC = '#0072B2'
C_LOSS = '#8C564B'
C_LR = '#444444'
C_BEST = '#009E73'
C_LRDROP = '#D55E00'
C_STOP = '#555555'


# ------------------------------------------------------------------------------------------------ data
def load():
    """Parse the 19 'Epoch NNN | Loss | Acc | LR | Time' lines of the target session and cross-check the summary lines."""
    lines = LOG.read_text().splitlines()
    start = lines.index(SESSION)
    block = []
    for ln in lines[start + 1:]:
        if ln.startswith('--- NEW SESSION'):
            break
        block.append(ln)
    rows = []
    for ln in block:
        m = re.match(r'Epoch (\d+) \| Loss: ([\d.]+) \| Acc: ([\d.]+) \| LR: ([\d.]+) \|', ln)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    summary = next(ln for ln in block if 'TRAINING COMPLETE' in ln)
    epochs_run = int(re.search(r'epochs_run=(\d+)', summary).group(1))
    best_epoch = int(re.search(r'best_epoch=(\d+)', summary).group(1))
    best_acc = float(re.search(r'best_acc=([\d.]+)', summary).group(1))
    stop_epoch = int(re.search(r'Early stopping at epoch (\d+)', '\n'.join(block)).group(1))
    assert 'best_web_gnn_seed42.pth' in summary, summary

    ep = np.array([r[0] for r in rows])
    loss = np.array([r[1] for r in rows])
    acc = np.array([r[2] for r in rows])
    lr = np.array([r[3] for r in rows])

    # consistency checks against the log's own summary and the numbers quoted in paper Sec. 4.2
    assert list(ep) == list(range(1, 20)) and epochs_run == 19 and stop_epoch == 19, (ep, epochs_run, stop_epoch)
    assert best_epoch == 9 and acc[best_epoch - 1] == best_acc == acc.max() == 1.0, (best_epoch, best_acc)
    assert int(np.argmax(acc)) + 1 == best_epoch                       # 1.0000 is reached only at epoch 9
    changes = [int(ep[i]) for i in range(1, len(lr)) if lr[i] != lr[i - 1]]
    assert changes == [15] and lr[0] == 0.0005 and lr[-1] == 0.00025, changes  # one halving, at epoch 15
    assert stop_epoch - best_epoch == 10                               # patience 10
    assert acc[0] == 0.9993 and loss[0] == 0.0159 and loss[1] == 0.0021
    assert acc.min() == 0.9988 and int(ep[np.argmin(acc)]) == 5        # at most five misclassified requests
    assert np.isclose(loss[4], 0.0023) and np.isclose(loss[9], 0.0041)
    assert all(v == 0.0 for v in loss[12:18])                          # 0.0000 from epoch 13 to 18
    # accuracy is quantised in units of 1/4215: every logged value must equal a whole number of errors
    errors = np.rint((1 - acc) * N_TEST).astype(int)
    assert np.allclose(np.round(1 - errors / N_TEST, 4), acc), (acc, errors)
    return ep, loss, acc, lr, errors, best_epoch, int(changes[0]), stop_epoch


# ------------------------------------------------------------------------------------------------ figure
def draw(out):
    ep, loss, acc, lr, errors, best_ep, lr_ep, stop_ep = load()

    fig = plt.figure(figsize=(5.4, 5.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.3, 1.7, 0.9], hspace=0.10, left=0.15, right=0.86, top=0.965, bottom=0.085)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ---- (top) accuracy, zoomed
    ax1.plot(ep, acc, '-o', color=C_ACC, lw=1.4, ms=4, zorder=3)
    ax1.plot([best_ep], [acc[best_ep - 1]], marker='*', ms=13, color=C_BEST, mec='black', mew=0.6, zorder=5,
             linestyle='none')
    ax1.set_ylim(0.9979, 1.0003)
    ax1.set_yticks([0.998, 0.9985, 0.999, 0.9995, 1.000])
    ax1.set_yticklabels(['0.9980', '0.9985', '0.9990', '0.9995', '1.0000'])
    ax1.set_ylabel('Test-split accuracy\n(y-axis zoomed to 0.998-1.000)', fontsize=8)
    ax1.grid(axis='y', color='#DDDDDD', lw=0.5, zorder=0)
    sec = ax1.secondary_yaxis('right', functions=(lambda a: (1 - a) * N_TEST, lambda n: 1 - n / N_TEST))
    sec.set_yticks([0, 1, 2, 3, 4, 5])
    sec.set_ylabel(f'Misclassified requests (of {N_TEST:,})', fontsize=8)
    sec.tick_params(labelsize=8)
    ax1.annotate(f'epoch {best_ep}: 1.0000\n(0 errors)', xy=(best_ep, 1.0), xytext=(best_ep - 2.1, 1.00003 - 0.00006),
                 fontsize=7.5, ha='right', va='top', color='black',
                 arrowprops=dict(arrowstyle='-', lw=0.6, color='black', shrinkA=0, shrinkB=4))
    ax1.text(0.015, 0.03, 'seed 42, deployed checkpoint', transform=ax1.transAxes, fontsize=7, ha='left', va='bottom',
             color='#666666', style='italic')

    # ---- (middle) loss
    ax2.plot(ep, loss, '-o', color=C_LOSS, lw=1.4, ms=3.6, zorder=3)
    ax2.set_ylim(-0.0008, 0.0175)
    ax2.set_yticks([0, 0.005, 0.010, 0.015])
    ax2.set_yticklabels(['0', '0.005', '0.010', '0.015'])
    ax2.set_ylabel('Training loss\n(class-weighted)', fontsize=8)
    ax2.grid(axis='y', color='#DDDDDD', lw=0.5, zorder=0)
    ax2.annotate(f'{loss[0]:.4f}', xy=(1, loss[0]), xytext=(1.5, loss[0]), fontsize=7.5, va='center', ha='left')
    ax2.annotate(f'{loss[4]:.4f}', xy=(5, loss[4]), xytext=(5, loss[4] + 0.0016), fontsize=7.5, ha='center', va='bottom')
    ax2.annotate(f'{loss[9]:.4f}', xy=(10, loss[9]), xytext=(10, loss[9] + 0.0016), fontsize=7.5, ha='center', va='bottom')
    ax2.text(13.0, 0.0019, '0.0000\nepochs 13-18', fontsize=7, ha='center', va='bottom',
             color='#666666', style='italic')

    # ---- (bottom) learning rate
    ax3.step(ep, lr, where='mid', color=C_LR, lw=1.6, zorder=3)
    ax3.plot(ep, lr, 'o', color=C_LR, ms=3, zorder=3)
    ax3.set_ylim(0.0001, 0.00068)
    ax3.set_yticks([0.00025, 0.0005])
    ax3.set_yticklabels(['2.5e-4', '5e-4'])
    ax3.set_ylabel('LR', fontsize=8)
    ax3.grid(axis='y', color='#DDDDDD', lw=0.5, zorder=0)
    ax3.set_xlabel('Epoch', fontsize=8.5)

    # ---- event markers on all three panels
    events = [(best_ep, C_BEST, '-', f'best epoch ({best_ep}), kept'),
              (lr_ep, C_LRDROP, '--', f'LR halved ({lr_ep})'),
              (stop_ep, C_STOP, ':', f'early stop ({stop_ep})')]
    for x, col, ls, _ in events:
        for ax in (ax1, ax2, ax3):
            ax.axvline(x, color=col, ls=ls, lw=1.3, zorder=2, alpha=0.95)
    lbl_y = 0.9981
    for x, col, ls, txt in events:
        ax1.text(x - 0.28, lbl_y, txt, rotation=90, fontsize=7.5, ha='right', va='bottom', color=col, zorder=6,
                 fontweight='bold')

    ax1.set_xlim(0.4, 19.6)
    ax1.set_xticks(range(1, 20, 2))
    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=8, length=3)
        for sp in ('top',):
            ax.spines[sp].set_visible(False)
    for ax in (ax1, ax2):
        plt.setp(ax.get_xticklabels(), visible=False)
    sec.spines['top'].set_visible(False)

    fig.savefig(out, format='svg')
    return fig


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    fig = draw(out)
    print('wrote', out, f'({out.stat().st_size} bytes)')
    return fig


if __name__ == '__main__':
    main()
