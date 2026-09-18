"""Minimal, honest interpretability comparison (Reviewer #3) between GATv2's
existing attention-weight extraction (scripts/inspect_attention_weights.py
-- edge-level attention, mappable to a relation type: E_seq/E_skip/E_sem)
and a simple attention-rollout proxy for the RoBERTa/CodeBERT baselines.

Method (deliberately simple, per spec -- not the full recursive Abnar &
Zuidema rollout): for each of the 3 example payloads below, average the
model's per-layer, per-head attention FROM the [CLS]/<s> token (position 0,
the token whose final hidden state actually feeds the classifier) TO every
input token, across all layers and heads. This gives one importance score
per input token -- which tokens the classifier's own attention actually
looked at -- with no claim of relation-level structure.

Examples (fixed, not cherry-picked post-hoc):
  1. sqli_clear: unobfuscated UNION-based SQLi (keyword tokens intact)
  2. sqli_comment_split: EXACT SAME content as
     scripts/inspect_attention_weights.py's GATv2 probe, for direct
     side-by-side comparison with that script's finding
  3. xss_svg_onload: unobfuscated svg/onload XSS

Writes, per (model, example): a bar chart PNG
(results/interpretability/<model>_<example>.png) and a text line in
results/interpretability/attention_rollout_report.txt with the top-5
highest-attention tokens.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification

from src.baselines.transformer_common import load_finetuned

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results' / 'interpretability'

METHODS = {
    'RoBERTa': DATA_DIR / 'models_pretrained' / 'roberta_baseline_seed42',
    'CodeBERT': DATA_DIR / 'models_pretrained' / 'codebert_baseline_seed42',
}

EXAMPLES = {
    'sqli_clear': "1' UNION SELECT username, password FROM users--",
    # exact string from scripts/inspect_attention_weights.py's CONTENT
    'sqli_comment_split': 'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--',
    'xss_svg_onload': '<svg onload=alert(1)>',
}


def cls_attention_rollout(model, tokenizer, content, max_length, device):
    enc = tokenizer(content, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0].tolist())
    # attentions: tuple of [1, num_heads, seq, seq] per layer
    all_layers = torch.stack(out.attentions, dim=0)  # [num_layers, 1, num_heads, seq, seq]
    cls_to_all = all_layers[:, 0, :, 0, :]  # [num_layers, num_heads, seq] -- CLS as query
    scores = cls_to_all.mean(dim=(0, 1)).cpu().tolist()  # [seq], averaged over layers+heads
    return tokens, scores


def plot_scores(method, example_name, tokens, scores, out_path):
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.35), 3))
    ax.bar(range(len(tokens)), scores, color='#4C72B0')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=75, ha='right', fontsize=7)
    ax.set_ylabel('mean attention from [CLS]')
    ax.set_title(f'{method} attention-rollout: {example_name}')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = []

    for method, output_dir in METHODS.items():
        if not output_dir.exists():
            print(f"[!] {method} checkpoint not found at {output_dir}, skipping")
            continue
        model, tokenizer, max_length, device = load_finetuned(output_dir)
        # sdpa (the default attention backend) doesn't support
        # output_attentions=True -- reload with eager attention for this
        # script only; predict_batch()/finetune_transformer() don't need
        # attentions and keep the faster sdpa default.
        model = AutoModelForSequenceClassification.from_pretrained(
            output_dir, attn_implementation='eager').to(device)
        model.eval()

        for example_name, content in EXAMPLES.items():
            tokens, scores = cls_attention_rollout(model, tokenizer, content, max_length, device)
            ranked = sorted(zip(tokens, scores), key=lambda x: -x[1])
            top5 = ', '.join(f"{t}={s:.4f}" for t, s in ranked[:5] if t not in ('<s>', '</s>', '<pad>'))
            line = f"[{method}][{example_name}] content={content!r} top-attended-tokens: {top5}"
            print(line)
            report_lines.append(line)

            png_path = RESULTS_DIR / f"{method.lower()}_{example_name}.png"
            plot_scores(method, example_name, tokens, scores, png_path)
            print(f"  saved {png_path}")

    report_path = RESULTS_DIR / 'attention_rollout_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines) + '\n')
    print(f"\nWrote {report_path}")


if __name__ == '__main__':
    main()
