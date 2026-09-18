"""Ablation driver: 4 configurations of {seq, skip, sem} edges x edge_attr.

  (1) full, no edge_attr      -- the current/already-trained state (retrain
                                  #5: 5 rounds of train-only augmentation +
                                  balanced class weighting). NOT retrained
                                  here, just re-evaluated for a fresh,
                                  directly-comparable number.
  (2) full, with edge_attr    -- adds relation-type (seq/skip/sem) one-hot
                                  edge_attr into GATv2Conv's edge_dim, to
                                  test whether letting attention explicitly
                                  condition on relation type raises E_sem's
                                  learned attention weight (data/
                                  attention_weights_comment_split.txt showed
                                  it undervalued: alpha_Esem_mean=0.154 vs
                                  alpha_other_mean=0.202, ratio 0.76).
  (3) no-skip, with edge_attr -- only seq+sem edges.
  (4) no-sem,  with edge_attr -- only seq+skip edges (E_sem removed
                                  entirely, to measure its contribution).

Each of (2)/(3)/(4): rebuild web_graphs.pkl with the matching toggles,
train from scratch (same dataset as retrain #5 -- 5 augmentation rounds
already baked into data/augmented_web_attack.csv, same class-weighted
loss), evaluate on the frozen test split AND the 90-sample held-out
matrix. Every checkpoint/log/result is saved under a config-specific
name; the canonical data/models_pretrained/best_web_gnn_seed42.pth,
logs/training_history.log, results/main_results.csv and
results/heldout_matrix_full.csv are backed up before being overwritten
by each config's run and restored to the (1) state at the end, so the
"live" pipeline state is left as it was before this ablation.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from src.bag.graph_builder import build_web_graphs
from src.training import train as train_mod
from src.training import evaluate as evaluate_mod
from src.bag.graph_builder import build_single_graph
from src.models.logwat import HeavyWebGNN

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
LOGS_DIR = ROOT / 'logs'
MODEL_PATH = DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed42.pth'
LOG_PATH = LOGS_DIR / 'training_history.log'
MAIN_RESULTS_PATH = RESULTS_DIR / 'main_results.csv'
HELDOUT_RESULTS_PATH = RESULTS_DIR / 'heldout_matrix_full.csv'
ABLATION_CSV = RESULTS_DIR / 'ablation_edge_attr_seq_skip_sem.csv'
ATTENTION_REPORT = DATA_DIR / 'attention_weights_comment_split_config2.txt'

COMMENT_SPLIT_CONTENT = 'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--'
E_SEM_EDGES = {(9, 16), (16, 9), (16, 23), (23, 16)}

CONFIGS = [
    {'key': '1_full_no_edge_attr', 'use_seq': True, 'use_skip': True, 'use_sem': True, 'use_edge_attr': False, 'retrain': False},
    {'key': '2_full_edge_attr', 'use_seq': True, 'use_skip': True, 'use_sem': True, 'use_edge_attr': True, 'retrain': True},
    {'key': '3_no_skip_edge_attr', 'use_seq': True, 'use_skip': False, 'use_sem': True, 'use_edge_attr': True, 'retrain': True},
    {'key': '4_no_sem_edge_attr', 'use_seq': True, 'use_skip': True, 'use_sem': False, 'use_edge_attr': True, 'retrain': True},
]


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, **kw)
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return r.stdout


def rebuild_graphs(cfg):
    build_web_graphs(use_seq=cfg['use_seq'], use_skip=cfg['use_skip'], use_sem=cfg['use_sem'],
                      use_edge_attr=cfg['use_edge_attr'])


def train_config(cfg):
    # give this run's log a clean slate to grep epoch counts from just this run
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    train_mod.train()


def evaluate_config(cfg):
    evaluate_mod.evaluate()


def heldout_config(cfg, model_path, result_path):
    cmd = [sys.executable, 'scripts/build_heldout_matrix_eval.py',
           '--model_path', str(model_path), '--result_path', str(result_path)]
    if not cfg['use_skip']:
        cmd.append('--no_skip')
    if not cfg['use_sem']:
        cmd.append('--no_sem')
    if cfg['use_edge_attr']:
        cmd.append('--use_edge_attr')
    run(cmd)


def extract_esem_attention(model_path, use_edge_attr):
    model = HeavyWebGNN(use_edge_attr=use_edge_attr)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    g = build_single_graph(COMMENT_SPLIT_CONTENT, use_edge_attr=use_edge_attr)
    with torch.no_grad():
        if use_edge_attr:
            out, (ei_att, alpha) = model.backbone.conv1(g.x, g.edge_index, g.edge_attr, return_attention_weights=True)
        else:
            out, (ei_att, alpha) = model.backbone.conv1(g.x, g.edge_index, return_attention_weights=True)

    alpha_mean_per_edge = alpha.mean(dim=1)
    src, dst = ei_att[0].tolist(), ei_att[1].tolist()
    esem, other, self_loop = [], [], []
    for e in range(len(src)):
        i, j = src[e], dst[e]
        a = alpha_mean_per_edge[e].item()
        if (i, j) in E_SEM_EDGES:
            esem.append(a)
        elif i == j:
            self_loop.append(a)
        else:
            other.append(a)

    esem_mean = sum(esem) / len(esem) if esem else float('nan')
    other_mean = sum(other) / len(other) if other else float('nan')
    lines = [
        f"model_path={model_path}",
        f"use_edge_attr={use_edge_attr}",
        f"n_esem_found={len(esem)} n_other={len(other)} n_self_loop={len(self_loop)}",
        f"alpha_Esem_mean={esem_mean:.6f}",
        f"alpha_other_mean={other_mean:.6f}",
        f"ratio={esem_mean / other_mean:.6f}" if other_mean else "ratio=n/a",
        f"(reference, config 1 / retrain #5, no edge_attr: alpha_Esem_mean=0.154079 alpha_other_mean=0.202362 ratio=0.7614)",
    ]
    report = "\n".join(lines)
    print(report)
    ATTENTION_REPORT.write_text(report + "\n")
    return esem_mean, other_mean


def summarize(key, main_results_path, heldout_path):
    main_df = pd.read_csv(main_results_path)
    macro = main_df[(main_df['subset'] == 'SQLi tổng hợp') & (main_df['label'] == 'macro avg')].iloc[0]
    held_df = pd.read_csv(heldout_path)
    cells_correct = int((held_df['gatv2_accuracy'] == 1.0).sum())
    row = {'config': key, 'test_split_macro_f1': macro['f1_score'], 'heldout_cells_correct': f'{cells_correct}/9'}
    for _, r in held_df.iterrows():
        row[f"{r['class_name']}_{r['technique']}"] = r['gatv2_accuracy']
    return row


def main():
    ablation_rows = []

    for cfg in CONFIGS:
        key = cfg['key']
        print(f"\n{'=' * 80}\nCONFIG {key}: use_seq={cfg['use_seq']} use_skip={cfg['use_skip']} "
              f"use_sem={cfg['use_sem']} use_edge_attr={cfg['use_edge_attr']} retrain={cfg['retrain']}\n{'=' * 80}")

        model_out = DATA_DIR / 'models_pretrained' / f'best_web_gnn_seed42_ablation_{key}.pth'
        log_out = LOGS_DIR / f'training_history_ablation_{key}.log'
        main_out = RESULTS_DIR / f'main_results_ablation_{key}.csv'
        heldout_out = RESULTS_DIR / f'heldout_matrix_full_ablation_{key}.csv'

        if cfg['retrain']:
            rebuild_graphs(cfg)
            train_config(cfg)
            shutil.copy(MODEL_PATH, model_out)
            shutil.copy(LOG_PATH, log_out)
            evaluate_config(cfg)
            shutil.copy(MAIN_RESULTS_PATH, main_out)
            heldout_config(cfg, model_out, heldout_out)
        else:
            # (1) reuse the already-trained current state; re-run evaluate +
            # held-out fresh (cheap) for a directly comparable number,
            # without retraining.
            if not model_out.exists():
                shutil.copy(MODEL_PATH, model_out)
            evaluate_config(cfg)
            shutil.copy(MAIN_RESULTS_PATH, main_out)
            heldout_config(cfg, MODEL_PATH, heldout_out)

        if key == '2_full_edge_attr':
            extract_esem_attention(model_out, use_edge_attr=True)

        ablation_rows.append(summarize(key, main_out, heldout_out))

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(ABLATION_CSV, index=False)
    print(f"\n\nWrote {ABLATION_CSV}")
    print(ablation_df.to_string(index=False))

    # Restore the canonical fixed-path files back to config (1)'s state, so
    # the "live" pipeline (data/web_graphs.pkl, the checkpoint, the results
    # CSVs) is left exactly as it was before this ablation ran.
    print("\nRestoring canonical paths to config (1) state...")
    cfg1 = CONFIGS[0]
    rebuild_graphs(cfg1)
    shutil.copy(DATA_DIR / 'models_pretrained' / f'best_web_gnn_seed42_ablation_{cfg1["key"]}.pth', MODEL_PATH)
    shutil.copy(RESULTS_DIR / f'main_results_ablation_{cfg1["key"]}.csv', MAIN_RESULTS_PATH)
    shutil.copy(RESULTS_DIR / f'heldout_matrix_full_ablation_{cfg1["key"]}.csv', HELDOUT_RESULTS_PATH)
    print("Restored.")


if __name__ == '__main__':
    main()
