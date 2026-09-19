"""Train an HGT control variant (src/models/hgt_controls.py) through the SAME `train()` loop,
split, class weights, LR and early stopping as every other #16 baseline.

  python -m scripts.train_hgt_control --model hgt_collapsed --seeds 42 43 44 45 46
  python -m scripts.train_hgt_control --model hgt_random    --seeds 42 43 44 45 46

Writes data/models_pretrained/hgtctl_<variant>_seed<seed>.pth and logs/<model>_training.log.
Skips a (model, seed) whose checkpoint already exists, so an interrupted run can be resumed.
"""
import argparse
import json
from pathlib import Path

from src.models.hgt_controls import CONTROL_NAMES, HGTControl
from src.training.train import train

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
GRAPHS_PKL = DATA_DIR / 'web_graphs.pkl'
MODEL_DIR = DATA_DIR / 'models_pretrained'
LOGS_DIR = ROOT / 'logs'


def ckpt_path(name, seed):
    return MODEL_DIR / f'hgtctl_{name}_seed{seed}.pth'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=sorted(CONTROL_NAMES))
    parser.add_argument('--seeds', nargs='*', type=int, default=[42])
    args = parser.parse_args()

    mode = CONTROL_NAMES[args.model]
    for seed in args.seeds:
        if ckpt_path(args.model, seed).exists():
            print(f"[=] {args.model} seed {seed}: checkpoint exists, skipping", flush=True)
            continue
        summary = train(seed=seed, model_factory=lambda use_edge_attr: HGTControl(mode),
                        data_path=str(GRAPHS_PKL), model_save_path=str(ckpt_path(args.model, seed)),
                        log_path=str(LOGS_DIR / f'{args.model}_training.log'),
                        run_tag=f" | model={args.model} seed={seed} lr=default(5e-4) data={GRAPHS_PKL.name}",
                        write_family_split=False)
        print(json.dumps(summary, default=str), flush=True)


if __name__ == '__main__':
    main()
