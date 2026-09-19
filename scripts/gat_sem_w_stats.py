"""Per-cell breakdown and external trade-off tests for results/gat_sem_w_5seed.csv (uncorrected, n=5 vs 5).

Writes results/gat_sem_w_tests.txt.
"""
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

from scripts.hgt_investigation_common import ROOT


def main():
    d = pd.read_csv(ROOT / 'results' / 'gat_sem_w_5seed.csv')
    cells = [c for c in d.columns if c.startswith('cell:')]
    L = ['per-cell correct, summed over 5 seeds (/50):',
         d.groupby('model', sort=False)[cells].sum().rename(columns=lambda c: c[5:]).T.to_string()]
    g, s, p = (d[d.model == m] for m in ('gatv2', 'gat_semW', 'gat_placebo'))
    for col in ('ext_wF1', 'ext_benign_recall'):
        L.append(f"{col}: semW {s[col].mean():.3f}±{s[col].std():.3f} vs gatv2 {g[col].mean():.3f}±{g[col].std():.3f}: "
                 f"Welch p={ttest_ind(s[col], g[col], equal_var=False).pvalue:.3f}, MWU p={mannwhitneyu(s[col], g[col]).pvalue:.3f}"
                 f" | vs placebo {p[col].mean():.3f}: Welch p={ttest_ind(s[col], p[col], equal_var=False).pvalue:.3f}")
    L.append(f"ext SQLi/XSS recall semW {s.ext_sqli_recall.mean():.3f} {s.ext_xss_recall.mean():.3f} | "
             f"gatv2 {g.ext_sqli_recall.mean():.3f} {g.ext_xss_recall.mean():.3f}")
    print("\n".join(L))
    (ROOT / 'results' / 'gat_sem_w_tests.txt').write_text("\n".join(L) + "\n")


if __name__ == '__main__':
    main()
