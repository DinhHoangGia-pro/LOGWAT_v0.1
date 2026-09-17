#!/usr/bin/env python3
import pickle, os
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

H = os.getcwd()
DATA_PKL = os.path.join(H, 'data', 'web_graphs.pkl')
AUG = os.path.join(H, 'data', 'augmented_web_attack.csv')
LABELED = os.path.join(H, 'data', 'labeled_csic_database.csv')
OUT = os.path.join(H, 'data', 'split_verification_report.txt')
SQLI_OUT = os.path.join(H, 'data', 'sqli_unique_samples.txt')

if not os.path.exists(DATA_PKL):
    raise SystemExit('graphs pickle not found: '+DATA_PKL)
if not os.path.exists(AUG):
    raise SystemExit('augmented CSV not found: '+AUG)
if not os.path.exists(LABELED):
    raise SystemExit('labeled CSV not found: '+LABELED)

with open(DATA_PKL,'rb') as f:
    data = pickle.load(f)
graphs = list(data.get('graphs', []))

# extract source_uid and labels
uids = []
labels = []
for g in graphs:
    uid = getattr(g, 'source_uid', None)
    if uid is None:
        try:
            uid = g['source_uid'] if isinstance(g, dict) and 'source_uid' in g else None
        except Exception:
            uid = None
    u = str(uid) if uid is not None else None
    uids.append(u)
    lab = None
    try:
        lab = int(g.y.item()) if hasattr(g, 'y') else (int(g['y']) if isinstance(g, dict) and 'y' in g else None)
    except Exception:
        lab = None
    labels.append(lab)

# group split
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(graphs, groups=[x for x in uids]))
train_uids = set(uids[i] for i in train_idx if uids[i] is not None)
test_uids = set(uids[i] for i in test_idx if uids[i] is not None)

# load augmented CSV into df
df = pd.read_csv(AUG)
# ensure source_uid is string
if 'source_uid' in df.columns:
    df['source_uid'] = df['source_uid'].astype(str)
else:
    raise SystemExit('augmented CSV missing source_uid column')

# 1. number of unique source_uid in TEST per class
classes = [0,1,2]
test_uid_counts_per_class = {}
for c in classes:
    uids_in_class = set(df[df['attack_type']==c]['source_uid'].unique())
    test_uids_in_class = test_uids & uids_in_class
    test_uid_counts_per_class[c] = len(test_uids_in_class)

# 2. write contents of all unique SQLi source_uid (representative line per uid)
sqli_uids = sorted(df[df['attack_type']==1]['source_uid'].unique())
with open(SQLI_OUT,'w',encoding='utf-8') as sf:
    for uid in sqli_uids:
        row = df[(df['source_uid']==uid) & (df['attack_type']==1)].iloc[0]
        content = str(row.get('content','')).replace('\n',' ')
        sf.write(f"{uid}\t{content}\n")

# 3. Explain 2406 origin: compute len(df_normal) from labeled CSV after cleaning/unknown removal
labeled = pd.read_csv(LABELED)
# clean similar to normalization.clean_content: drop NaN/short and duplicates
labeled['content'] = labeled['content'].astype(str)
labeled = labeled[~labeled['content'].str.lower().isin(['nan',''])]
labeled = labeled[labeled['content'].str.len()>1]
# count per class
orig_counts = labeled['attack_type'].value_counts().to_dict()
orig_benign = int(orig_counts.get(0,0))
# n_benign used in balance_dataset = max(1, len(df_normal)//2)
n_benign = max(1, orig_benign//2)

# Compose report lines
now = datetime.utcnow().isoformat()+'Z'
lines = []
lines.append(f"Split supplementary analysis - {now}")
lines.append(f"1) Unique source_uid in TEST, per class (unique source_uid counts):")
for c in classes:
    lines.append(f" Class {c}: {test_uid_counts_per_class[c]}")
lines.append("")
lines.append(f"2) Wrote representative content for {len(sqli_uids)} unique SQLi source_uid to: {SQLI_OUT}")
lines.append("")
lines.append("3) Where 2406 comes from:")
lines.append(f" Original labeled benign rows (after cleaning/unknown removal): {orig_benign}")
lines.append(f" n_benign used in balance_dataset = max(1, len(df_normal)//2) = {n_benign}")
lines.append(f" Percentage of original benign used as base: {n_benign/orig_benign:.4f} (≈ {n_benign/orig_benign*100:.1f}%)")
lines.append("")
lines.append("Additional notes:")
lines.append(f" Total unique SQLi source_uid in augmented dataset: {len(sqli_uids)}")
lines.append('\n--- End of supplementary analysis ---\n')

# print and append
print('\n'.join(lines))
with open(OUT,'a',encoding='utf-8') as f:
    f.write('\n'.join(lines)+"\n")
print('\nAppended supplementary analysis to', OUT)
