"""Train the minimal-intervention GATv2 variants (src/models/gat_sem_w.py) through the SAME `train()` loop,
frozen split, LR, class weights and early stopping as every other baseline.

  python -m scripts.train_gat_sem_w --selftest                       # equivalence checks vs stock GATv2Conv, no training
  python -m scripts.train_gat_sem_w --model gat_semW   --seeds 42 43 44 45 46
  python -m scripts.train_gat_sem_w --model gat_placebo --seeds 42 43 44 45 46

Trains on data/web_graphs_edge_attr.pkl (x / edge_index identical to web_graphs.pkl; edge_attr only feeds the
flagged-edge mask, never attention). Writes data/models_pretrained/gatalt_<model>_seed<seed>.pth and
logs/<model>_training.log. Skips a (model, seed) whose checkpoint exists.
"""
import argparse
import json
from pathlib import Path

import torch
from torch_geometric.nn import GATv2Conv

from src.models.gat_sem_w import ARMS, GATAltMsgNet, GATv2ConvAltMsg
from src.training.train import train

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
GRAPHS_EA = DATA_DIR / 'web_graphs_edge_attr.pkl'
MODEL_DIR = DATA_DIR / 'models_pretrained'
LOGS_DIR = ROOT / 'logs'


def ckpt_path(name, seed):
    return MODEL_DIR / f'gatalt_{name}_seed{seed}.pth'


def selftest():
    """The layer must equal stock GATv2Conv when nothing is routed through W_alt, and when W_alt is tied to W_l."""
    torch.manual_seed(0)
    n, e = 40, 160
    x = torch.randn(n, 64)
    ei = torch.randint(0, n, (2, e))
    mask = (torch.rand(e) < 0.3).float()
    stock = GATv2Conv(64, 32, heads=8).eval()
    alt = GATv2ConvAltMsg(64, 32, 8).eval()
    alt.load_state_dict(stock.state_dict(), strict=False)          # everything but lin_alt copied
    with torch.no_grad():
        ref, (ei_s, a_s) = stock(x, ei, return_attention_weights=True)
        out_none = alt(x, ei, torch.zeros(e))
        alt.lin_alt.weight.copy_(alt.lin_l.weight); alt.lin_alt.bias.copy_(alt.lin_l.bias)
        out_tied, (ei_a, a_a) = alt(x, ei, mask, return_attention_weights=True)
        alt.lin_alt.reset_parameters()
        out_diff = alt(x, ei, mask)
    print(f"mask all-zero  vs stock: max|diff| = {(out_none - ref).abs().max():.2e}")
    print(f"W_alt := W_l   vs stock: max|diff| = {(out_tied - ref).abs().max():.2e}")
    print(f"attention identical to stock (edge order + alpha): {torch.equal(ei_s, ei_a)} / max|d alpha| = {(a_s - a_a).abs().max():.2e}")
    print(f"independent W_alt DOES change the output on flagged edges: max|diff| = {(out_diff - ref).abs().max():.3f}")
    ok = (out_none - ref).abs().max() < 1e-5 and (out_tied - ref).abs().max() < 1e-5 and (a_s - a_a).abs().max() < 1e-6 \
        and (out_diff - ref).abs().max() > 1e-2
    print("SELFTEST", "PASS" if ok else "FAIL")
    n_stock = sum(p.numel() for p in stock.parameters())
    print(f"params per conv: stock {n_stock} -> alt-message {sum(p.numel() for p in alt.parameters())}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--model', choices=sorted(ARMS))
    parser.add_argument('--seeds', nargs='*', type=int, default=[42])
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    arm = ARMS[args.model]
    for seed in args.seeds:
        if ckpt_path(args.model, seed).exists():
            print(f"[=] {args.model} seed {seed}: checkpoint exists, skipping", flush=True)
            continue
        summary = train(seed=seed, model_factory=lambda use_edge_attr: GATAltMsgNet(arm), data_path=str(GRAPHS_EA),
                        model_save_path=str(ckpt_path(args.model, seed)),
                        log_path=str(LOGS_DIR / f'{args.model}_training.log'),
                        run_tag=f" | model={args.model} seed={seed} lr=default(5e-4) data={GRAPHS_EA.name}",
                        write_family_split=False)
        print(json.dumps(summary, default=str), flush=True)


if __name__ == '__main__':
    main()
