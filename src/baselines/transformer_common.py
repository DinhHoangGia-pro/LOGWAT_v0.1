"""Shared fine-tuning/prediction logic for the RoBERTa and CodeBERT
baselines (Reviewer #3). `src/baselines/roberta.py` and `codebert.py` are
thin wrappers around this module supplying only their HuggingFace model
name and output paths -- the two baselines are otherwise identical in
architecture (`AutoModelForSequenceClassification`, num_labels=3), training
procedure, and evaluation, so duplicating this logic per-model would just
be drift risk with no benefit.

Operates on raw `content` text (data/transformer_baseline/{train,test}.csv,
from scripts/prepare_transformer_baseline_data.py), NOT the BAG/graph
pipeline GATv2 uses.

Model selection uses a validation split carved out of TRAIN only (stratified,
`val_fraction` of train, seed-derived, never overlapping the frozen test
set) -- GATv2's training loop selects its best checkpoint by test-set
accuracy each epoch, which is a known methodological weakness of this
project (see docs/REPRODUCIBILITY.md); these baselines deliberately avoid
repeating it so the test-set numbers reported for them are on genuinely
unseen data throughout training and model selection.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.seed import set_seed

CLASS_NAMES = ['Benign', 'SQLi', 'XSS']


class TextClsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def compute_max_length(tokenizer, contents, percentile=95):
    """95th-percentile token length over the given contents, per-tokenizer
    (not assumed shared between RoBERTa/CodeBERT, even though both currently
    use a RoBERTa-style BPE tokenizer) -- printed so the choice is visible,
    not silently baked in."""
    enc = tokenizer(list(contents), truncation=False, padding=False)
    lens = np.array([len(x) for x in enc['input_ids']])
    p = int(np.ceil(np.percentile(lens, percentile)))
    # round up to the nearest multiple of 8 for GPU efficiency -- covers
    # slightly MORE than the raw percentile, never less, so this is not a
    # more aggressive truncation than requested.
    max_length = int(np.ceil(p / 8) * 8)
    print(f"[*] token length: p50={np.percentile(lens, 50):.0f} p90={np.percentile(lens, 90):.0f} "
          f"p{percentile}={p} max={lens.max()} -> max_length={max_length} (p{percentile} rounded up to mult-of-8)")
    return max_length


def _evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            n_batches += 1
            preds = out.logits.argmax(dim=1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    return total_loss / max(1, n_batches), correct / max(1, total)


def finetune_transformer(model_name, output_dir, log_path,
                          train_csv, test_csv,
                          batch_size=32, lr=2e-5, num_epochs=5,
                          seed=42, val_fraction=0.05,
                          early_stop_patience=2, early_stop_min_delta=0.0,
                          max_length_percentile=95):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)  # loaded only for shape/sanity print, NOT used for model selection here

    tr_df, val_df = train_test_split(
        train_df, test_size=val_fraction, random_state=seed,
        stratify=train_df['attack_type'])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = compute_max_length(tokenizer, train_df['content'].astype(str).tolist(),
                                     percentile=max_length_percentile)

    def make_dataset(df):
        enc = tokenizer(df['content'].astype(str).tolist(), padding='max_length',
                         truncation=True, max_length=max_length, return_tensors='pt')
        labels = torch.tensor(df['attack_type'].astype(int).tolist(), dtype=torch.long)
        return TextClsDataset(enc, labels)

    train_ds = make_dataset(tr_df)
    val_ds = make_dataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"[*] model={model_name} seed={seed} n_train={len(tr_df)} n_val={len(val_df)} "
          f"n_test(reported later)={len(test_df)} batch_size={batch_size} lr={lr} "
          f"num_epochs={num_epochs} max_length={max_length} early_stop_patience={early_stop_patience}")

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    early_stop_reason = 'reached max epoch'
    last_epoch = 0

    with open(log_path, 'a') as log_f:
        log_f.write(f"\n--- NEW SESSION: {time.ctime()} | model={model_name} ---\n")

        for epoch in range(1, num_epochs + 1):
            model.train()
            t0 = time.time()
            total_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                out = model(**batch)
                out.loss.backward()
                optimizer.step()
                total_loss += out.loss.item()
                n_batches += 1
            train_loss = total_loss / max(1, n_batches)

            val_loss, val_acc = _evaluate(model, val_loader, device)
            epoch_time = time.time() - t0
            last_epoch = epoch

            log_str = (f"Epoch {epoch:03d} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                       f"ValAcc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                       f"Time: {epoch_time:.1f}s")
            print(log_str)
            log_f.write(log_str + "\n")
            log_f.flush()

            if val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop_patience:
                    early_stop_reason = f'val loss did not improve for {early_stop_patience} consecutive epochs'
                    break

        summary = (f"[*] TRAINING COMPLETE | model={model_name} | epochs_run={last_epoch} | "
                   f"best_epoch={best_epoch} | best_val_loss={best_val_loss:.4f} | "
                   f"early_stop_reason={early_stop_reason} | saved={output_dir}")
        print(summary)
        log_f.write(summary + "\n")

    with open(Path(output_dir) / 'max_length.txt', 'w') as f:
        f.write(str(max_length))

    return {'output_dir': str(output_dir), 'max_length': max_length,
            'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
            'epochs_run': last_epoch, 'early_stop_reason': early_stop_reason}


def load_finetuned(output_dir, device=None):
    """Load a fine-tuned checkpoint saved by `finetune_transformer()`,
    including the `max_length` it was trained with (saved alongside the
    weights so callers never have to hardcode or guess it)."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    model.eval()
    max_length = int((Path(output_dir) / 'max_length.txt').read_text().strip())
    return model, tokenizer, max_length, device


def predict_batch(model, tokenizer, contents, max_length, device, batch_size=32):
    """Returns (pred_labels, probs) for a list of raw content strings,
    batched, no gradient tracking."""
    import torch.nn.functional as F
    preds = []
    probs = []
    with torch.no_grad():
        for i in range(0, len(contents), batch_size):
            chunk = [str(c) for c in contents[i:i + batch_size]]
            enc = tokenizer(chunk, padding='max_length', truncation=True,
                             max_length=max_length, return_tensors='pt').to(device)
            logits = model(**enc).logits
            p = F.softmax(logits, dim=1)
            preds.extend(int(x) for x in logits.argmax(dim=1).tolist())
            probs.extend(p.tolist())
    return preds, probs
