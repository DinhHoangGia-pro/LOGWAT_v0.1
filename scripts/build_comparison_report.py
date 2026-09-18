import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'

PRE_MAIN = Path('/tmp/main_results_PRE_semantic_fix.csv')
PRE_HELDOUT = Path('/tmp/heldout_matrix_full_PRE_semantic_fix.csv')
POST_MAIN = RESULTS_DIR / 'main_results.csv'
POST_HELDOUT = RESULTS_DIR / 'heldout_matrix_full.csv'
DT_BASELINE_TXT = DATA_DIR / 'decision_tree_size_baseline.txt'

OUT_CSV = RESULTS_DIR / 'COMPARISON_before_after_semantic_fix.csv'


def main():
    rows = []

    # --- (a) main_results.csv: test split goc (chi lay macro avg cua SQLi tong hop) ---
    pre_main = pd.read_csv(PRE_MAIN)
    post_main = pd.read_csv(POST_MAIN)

    pre_macro = pre_main[(pre_main['subset'] == 'SQLi tổng hợp') & (pre_main['label'] == 'macro avg')].iloc[0]
    post_macro = post_main[(post_main['subset'] == 'SQLi tổng hợp') & (post_main['label'] == 'macro avg')].iloc[0]

    rows.append({
        'section': 'a_test_split', 'metric': 'macro_precision',
        'before_PRE_semantic_fix': pre_macro['precision'], 'after_POST_semantic_fix': post_macro['precision'],
    })
    rows.append({
        'section': 'a_test_split', 'metric': 'macro_recall',
        'before_PRE_semantic_fix': pre_macro['recall'], 'after_POST_semantic_fix': post_macro['recall'],
    })
    rows.append({
        'section': 'a_test_split', 'metric': 'macro_f1',
        'before_PRE_semantic_fix': pre_macro['f1_score'], 'after_POST_semantic_fix': post_macro['f1_score'],
    })

    for label, name in [(0, 'Benign'), (1, 'SQLi'), (2, 'XSS')]:
        pre_row = pre_main[(pre_main['subset'] == 'SQLi tổng hợp') & (pre_main['label'] == str(label))]
        post_row = post_main[(post_main['subset'] == 'SQLi tổng hợp') & (post_main['label'] == str(label))]
        if pre_row.empty:
            pre_row = pre_main[(pre_main['subset'] == 'SQLi tổng hợp') & (pre_main['label'] == label)]
        if post_row.empty:
            post_row = post_main[(post_main['subset'] == 'SQLi tổng hợp') & (post_main['label'] == label)]
        if not pre_row.empty and not post_row.empty:
            rows.append({
                'section': 'a_test_split', 'metric': f'{name}_f1',
                'before_PRE_semantic_fix': pre_row.iloc[0]['f1_score'],
                'after_POST_semantic_fix': post_row.iloc[0]['f1_score'],
            })

    # --- (b) heldout_matrix_full.csv: 9 dong day du ---
    pre_held = pd.read_csv(PRE_HELDOUT)
    post_held = pd.read_csv(POST_HELDOUT)
    for _, pre_r in pre_held.iterrows():
        post_r = post_held[(post_held['class_name'] == pre_r['class_name']) & (post_held['technique'] == pre_r['technique'])].iloc[0]
        rows.append({
            'section': 'b_heldout_matrix',
            'metric': f"{pre_r['class_name']}_{pre_r['technique']}_gatv2_accuracy",
            'before_PRE_semantic_fix': pre_r['gatv2_accuracy'],
            'after_POST_semantic_fix': post_r['gatv2_accuracy'],
        })

    # --- (c) decision tree size-only baseline (chi co SAU fix, vi can graph moi) ---
    dt_acc = None
    if DT_BASELINE_TXT.exists():
        for line in DT_BASELINE_TXT.read_text().splitlines():
            if line.startswith('Overall accuracy'):
                dt_acc = line.split(':')[1].strip()
    rows.append({
        'section': 'c_decision_tree_size_baseline',
        'metric': 'overall_accuracy_num_nodes_num_edges_only',
        'before_PRE_semantic_fix': 'khong_do_truoc_fix',
        'after_POST_semantic_fix': dt_acc,
    })

    # --- (d) case_mixing tokenizer .lower() check ---
    rows.append({
        'section': 'd_case_mixing_validity',
        'metric': 'tokenizer_calls_lower',
        'before_PRE_semantic_fix': 'True (tokenizer.py:13, khong doi giua truoc/sau fix)',
        'after_POST_semantic_fix': 'True (tokenizer.py:13, khong doi giua truoc/sau fix)',
    })
    rows.append({
        'section': 'd_case_mixing_validity',
        'metric': 'ket_luan',
        'before_PRE_semantic_fix': 'case_mixing KHONG phai thu thach that vi tokenizer tu dong lowercase truoc khi so khop',
        'after_POST_semantic_fix': 'accuracy=1.0 o ca truoc va sau fix khong co y nghia kiem dinh cho ky thuat nay',
    })

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['section', 'metric', 'before_PRE_semantic_fix', 'after_POST_semantic_fix'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Da ghi: {OUT_CSV}")
    print()
    with open(OUT_CSV) as f:
        print(f.read())


if __name__ == '__main__':
    main()
