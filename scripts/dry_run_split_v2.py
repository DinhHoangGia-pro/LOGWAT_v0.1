#!/usr/bin/env python3
import os, pickle
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from collections import defaultdict
from datetime import datetime
import pandas as pd

ROOT = Path.cwd()
AUG = ROOT / 'data' / 'augmented_web_attack.csv'
PKL = ROOT / 'data' / 'web_graphs.pkl'
OUT = ROOT / 'data' / 'split_verification_report_v2.txt'

# Load augmented DF and report duplication stats
if not AUG.exists():
    raise SystemExit('Augmented CSV not found: '+str(AUG))
df = pd.read_csv(AUG)

# call existing report function if available
try:
    from src.preprocessing.normalization import report_duplication_stats
    report_duplication_stats(df, out_path=str(OUT.with_suffix('.duplication_tmp.txt')))
except Exception:
    # ignore
    pass

lines = []
lines.append(f"Split verification v2 - {datetime.utcnow().isoformat()}Z")
lines.append(f"Augmented file: {AUG}")

# total unique source_uid overall and per class
if 'source_uid' in df.columns:
    total_unique = df['source_uid'].nunique()
    lines.append(f"Total unique source_uid in augmented CSV: {total_unique}")
    for c in sorted(df['attack_type'].unique()):
        cnt = df[df['attack_type']==c]['source_uid'].nunique()
        lines.append(f" Class {c}: unique source_uid = {cnt}")
else:
    lines.append('No source_uid column in augmented CSV')

# ensure graphs pickle exists; rebuild instruction
if not PKL.exists():
    lines.append('\nweb_graphs.pkl not found; cannot run graph-level GroupShuffleSplit.\nPlease run scripts/build_graphs.py first to regenerate graphs.')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    raise SystemExit('Missing graphs pickle')

with open(PKL,'rb') as f:
    data = pickle.load(f)
graphs = list(data.get('graphs', []))

uids = []
labels = []
for g in graphs:
    uid = getattr(g, 'source_uid', None)
    uid = str(uid) if uid is not None else None
    uids.append(uid)
    lab = None
    try:
        lab = int(g.y.item())
    except Exception:
        lab = None
    labels.append(lab)

unique_uids = set(x for x in uids if x is not None)
lines.append(f"Total unique source_uid in graphs: {len(unique_uids)}")

# unique source_uid per class
per_class_uids = defaultdict(set)
for uid, lab in zip(uids, labels):
    if uid is not None and lab is not None:
        per_class_uids[lab].add(uid)
for c in sorted(per_class_uids.keys()):
    lines.append(f" Graphs Class {c}: unique source_uid = {len(per_class_uids[c])}")

# GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(graphs, groups=uids))
train_uids = set(uids[i] for i in train_idx if uids[i] is not None)
test_uids = set(uids[i] for i in test_idx if uids[i] is not None)
lines.append(f"\nTrain unique source_uid: {len(train_uids)}")
lines.append(f"Test unique source_uid: {len(test_uids)}")
intersection = train_uids & test_uids
lines.append(f"Intersection empty: {len(intersection)==0}")
if len(intersection)>0:
    lines.append(f"Intersection sample: {list(intersection)[:10]}")

# number of unique source_uid in TEST per class
test_class_counts = defaultdict(int)
for i in test_idx:
    lab = labels[i]
    if lab is not None and uids[i] is not None:
        test_class_counts[lab] += 1
# but user requested number of unique source_uid in TEST per class
test_class_uids = defaultdict(set)
for i in test_idx:
    lab = labels[i]
    if lab is not None and uids[i] is not None:
        test_class_uids[lab].add(uids[i])
lines.append('\nUnique source_uid in TEST, per class:')
for c in sorted(test_class_uids.keys()):
    lines.append(f" Class {c}: {len(test_class_uids[c])}")

# write to OUT (append, but create new file)
with open(OUT,'w',encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print('\n'.join(lines))
print('\nWrote split verification v2 to', OUT)
