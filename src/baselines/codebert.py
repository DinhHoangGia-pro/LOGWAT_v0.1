"""CodeBERT baseline (Reviewer #3) -- fine-tunes `microsoft/codebert-base`
for 3-class web-attack classification on raw `content` text. See
`src/baselines/transformer_common.py` for the shared training/eval logic.
"""
from pathlib import Path

from src.baselines.transformer_common import finetune_transformer, load_finetuned, predict_batch

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = 'microsoft/codebert-base'
OUTPUT_DIR = ROOT / 'data' / 'models_pretrained' / 'codebert_baseline_seed42'
LOG_PATH = ROOT / 'logs' / 'codebert_training.log'
TRAIN_CSV = ROOT / 'data' / 'transformer_baseline' / 'train.csv'
TEST_CSV = ROOT / 'data' / 'transformer_baseline' / 'test.csv'


def finetune_codebert(seed=42, **kwargs):
    return finetune_transformer(
        model_name=MODEL_NAME, output_dir=OUTPUT_DIR, log_path=LOG_PATH,
        train_csv=TRAIN_CSV, test_csv=TEST_CSV, seed=seed, **kwargs)


def codebert_predict(contents, output_dir=OUTPUT_DIR, batch_size=32):
    """contents: list of raw text strings. Returns (pred_labels, probs)."""
    model, tokenizer, max_length, device = load_finetuned(output_dir)
    return predict_batch(model, tokenizer, contents, max_length, device, batch_size=batch_size)
