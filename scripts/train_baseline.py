"""Train one #16 baseline through the SAME loop as GATv2 (src.training.train.train):
same frozen split (test_split_indices.pkl; train = every index not in test), same
class-weighted CrossEntropy, AdamW(weight_decay=0.1), ReduceLROnPlateau, batch_size=64,
max 100 epochs, early-stopping patience 10 on test-split accuracy -- exactly the
protocol the deployed GATv2 checkpoint was selected under. Only the architecture
(and, for HGT / sequence models, the input .pkl it needs) differs.

  python -m scripts.train_baseline --model gcn --seed 42
  python -m scripts.train_baseline --model hgt --seed 42 --lr 1e-3     # per-baseline LR override, logged

Writes  data/models_pretrained/baseline_<model>_seed<seed>.pth
        logs/<model>_training.log   (appended; one session header per run)
"""
import argparse
import json
from pathlib import Path

from src.baselines.sequence_data import SEQ_PKL, VOCAB_JSON
from src.models.baselines_graph import GRAPH_BASELINES
from src.models.baselines_seq import SEQ_BASELINES
from src.models.logwat import HeavyWebGNN
from src.training.train import train

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
LOGS_DIR = ROOT / 'logs'
MODEL_DIR = DATA_DIR / 'models_pretrained'

GRAPHS_PKL = DATA_DIR / 'web_graphs.pkl'
GRAPHS_EDGE_ATTR_PKL = DATA_DIR / 'web_graphs_edge_attr.pkl'


def graph_spec(name):
    cls = GRAPH_BASELINES[name]
    data_path = GRAPHS_EDGE_ATTR_PKL if cls.needs_edge_attr else GRAPHS_PKL
    return (lambda use_edge_attr: cls()), data_path


def sequence_spec(name):
    cls = SEQ_BASELINES[name]
    with open(VOCAB_JSON) as f:
        vocab_size = json.load(f)['vocab_size']
    return (lambda use_edge_attr: cls(vocab_size)), SEQ_PKL


def baseline_spec(name):
    """-> (model_factory(use_edge_attr), data_pkl_path)"""
    if name == 'gatv2':  # evaluation-only reference: the deployed model, never trained here
        return (lambda use_edge_attr: HeavyWebGNN(use_edge_attr=use_edge_attr)), GRAPHS_PKL
    if name in GRAPH_BASELINES:
        return graph_spec(name)
    if name in SEQ_BASELINES:
        return sequence_spec(name)
    raise KeyError(f"unknown baseline {name!r}; choices: {sorted(GRAPH_BASELINES) + sorted(SEQ_BASELINES)}")


def ckpt_path(name, seed):
    if name == 'gatv2':  # deployed seed-42 checkpoint + the 5-seed run's seed-43..46 checkpoints
        return MODEL_DIR / f'best_web_gnn_seed{seed}.pth'
    return MODEL_DIR / f'baseline_{name}_seed{seed}.pth'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=None, help="override; default = configs/config.yaml (GATv2's 5e-4)")
    args = parser.parse_args()

    factory, data_path = baseline_spec(args.model)
    tag = f" | model={args.model} seed={args.seed} lr={args.lr if args.lr is not None else 'default(5e-4)'} data={Path(data_path).name}"
    summary = train(seed=args.seed, lr=args.lr, model_factory=factory, data_path=str(data_path),
                    model_save_path=str(ckpt_path(args.model, args.seed)),
                    log_path=str(LOGS_DIR / f'{args.model}_training.log'),
                    run_tag=tag, write_family_split=False)
    print(json.dumps(summary, default=str))


if __name__ == '__main__':
    main()
