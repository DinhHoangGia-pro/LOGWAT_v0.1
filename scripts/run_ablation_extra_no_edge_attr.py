"""Extra ablation configs, isolating E_skip/E_sem removal WITHOUT edge_attr,
so their effect isn't confounded with the edge_attr mechanism tested in
configs (2)-(4) of scripts/run_ablation_edge_attr_seq_skip_sem.py.

  (5) no-skip, no edge_attr -- directly comparable to (1): only difference
      is E_skip removed.
  (6) no-sem,  no edge_attr -- directly comparable to (1): only difference
      is E_sem removed.

Appends 2 rows to the existing results/ablation_edge_attr_seq_skip_sem.csv
(does not touch rows 1-4). Reuses all the helper functions from the main
ablation driver instead of duplicating them.
"""
import shutil

import pandas as pd

from scripts.run_ablation_edge_attr_seq_skip_sem import (
    ABLATION_CSV, CONFIGS, DATA_DIR, LOGS_DIR, MODEL_PATH, RESULTS_DIR,
    evaluate_config, heldout_config, rebuild_graphs, summarize, train_config,
)

EXTRA_CONFIGS = [
    {'key': '5_no_skip_no_edge_attr', 'use_seq': True, 'use_skip': False, 'use_sem': True, 'use_edge_attr': False, 'retrain': True},
    {'key': '6_no_sem_no_edge_attr', 'use_seq': True, 'use_skip': True, 'use_sem': False, 'use_edge_attr': False, 'retrain': True},
]


def main():
    new_rows = []

    for cfg in EXTRA_CONFIGS:
        key = cfg['key']
        print(f"\n{'=' * 80}\nCONFIG {key}: use_seq={cfg['use_seq']} use_skip={cfg['use_skip']} "
              f"use_sem={cfg['use_sem']} use_edge_attr={cfg['use_edge_attr']}\n{'=' * 80}")

        model_out = DATA_DIR / 'models_pretrained' / f'best_web_gnn_seed42_ablation_{key}.pth'
        log_out = LOGS_DIR / f'training_history_ablation_{key}.log'
        main_out = RESULTS_DIR / f'main_results_ablation_{key}.csv'
        heldout_out = RESULTS_DIR / f'heldout_matrix_full_ablation_{key}.csv'

        rebuild_graphs(cfg)
        train_config(cfg)
        shutil.copy(MODEL_PATH, model_out)
        shutil.copy(LOGS_DIR / 'training_history.log', log_out)
        evaluate_config(cfg)
        shutil.copy(RESULTS_DIR / 'main_results.csv', main_out)
        heldout_config(cfg, model_out, heldout_out)

        new_rows.append(summarize(key, main_out, heldout_out))

    existing = pd.read_csv(ABLATION_CSV)
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(ABLATION_CSV, index=False)
    print(f"\n\nAppended {len(new_rows)} rows to {ABLATION_CSV}")
    print(combined.to_string(index=False))

    # restore canonical paths back to config (1) state, same as the main driver
    print("\nRestoring canonical paths to config (1) state...")
    cfg1 = CONFIGS[0]
    rebuild_graphs(cfg1)
    shutil.copy(DATA_DIR / 'models_pretrained' / f'best_web_gnn_seed42_ablation_{cfg1["key"]}.pth', MODEL_PATH)
    shutil.copy(RESULTS_DIR / f'main_results_ablation_{cfg1["key"]}.csv', RESULTS_DIR / 'main_results.csv')
    shutil.copy(RESULTS_DIR / f'heldout_matrix_full_ablation_{cfg1["key"]}.csv', RESULTS_DIR / 'heldout_matrix_full.csv')
    print("Restored.")


if __name__ == '__main__':
    main()
