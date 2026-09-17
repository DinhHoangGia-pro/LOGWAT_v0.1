#!/usr/bin/env python3
import pickle, os
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit

H = os.getcwd()
DATA_PKL = os.path.join(H, 'data', 'web_graphs.pkl')
OUT = os.path.join(H, 'data', 'split_verification_report.txt')
if not os.path.exists(DATA_PKL):
    raise SystemExit('graphs pickle not found: '+DATA_PKL)
with open(DATA_PKL,'rb') as f:
    data = pickle.load(f)
graphs = list(data.get('graphs', []))

# extract source_uid and labels
uids = []
labels = []
for g in graphs:
    uid = getattr(g, 'source_uid', None)
    # allow string/numeric
    if uid is None:
        # try dict-like
        try:
            uid = g['source_uid'] if isinstance(g, dict) and 'source_uid' in g else None
        except Exception:
            uid = None
    # normalize to str for counting
    u = str(uid) if uid is not None else None
    uids.append(u)
    # label
    lab = None
    try:
        lab = int(g.y.item()) if hasattr(g, 'y') else (int(g['y']) if isinstance(g, dict) and 'y' in g else None)
    except Exception:
        lab = None
    labels.append(lab)

# unique source_uid overall
unique_uids = set(x for x in uids if x is not None)

# group split
groups = [x for x in uids]
if not any(x is not None for x in groups):
    raise SystemExit('No source_uid present on graphs; cannot GroupShuffleSplit')

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(graphs, groups=groups))
train_uids = set(uids[i] for i in train_idx if uids[i] is not None)
test_uids = set(uids[i] for i in test_idx if uids[i] is not None)

# assertions
intersection = train_uids & test_uids
assert_pass = len(intersection) == 0

# per-class counts (by graph rows)
per_class = defaultdict(lambda: {'train':0,'test':0})
for i in train_idx:
    lab = labels[i]
    per_class[lab]['train'] += 1
for i in test_idx:
    lab = labels[i]
    per_class[lab]['test'] += 1

# compute approx ratio for classes
ratio_info = {}
for lab, cnts in per_class.items():
    t = cnts['train']
    s = cnts['test']
    total = t + s if (t+s)>0 else 1
    ratio = t/total
    ratio_info[lab] = (t, s, ratio)

# count duplicated source_uids overall and per-class
uid_counts = Counter(x for x in uids if x is not None)
dup_uids_overall = sum(1 for k,v in uid_counts.items() if v>1)
# per-class: for graphs of a class, count source_uid frequency>1
per_class_dup = {}
for lab in set(labels):
    # gather uids for graphs with this label
    lab_uids = [uids[i] for i,l in enumerate(labels) if l==lab and uids[i] is not None]
    c = Counter(lab_uids)
    per_class_dup[lab] = sum(1 for k,v in c.items() if v>1)

# prepare report
now = datetime.utcnow().isoformat()+'Z'
lines = []
lines.append(f"Split verification report - {now}")
lines.append(f"Total graphs: {len(graphs)}")
lines.append(f"Total unique source_uid in dataset: {len(unique_uids)}")
lines.append(f"Train unique source_uid: {len(train_uids)}")
lines.append(f"Test unique source_uid: {len(test_uids)}")
lines.append(f"Intersection empty: {assert_pass}")
if not assert_pass:
    lines.append(f"Intersection example (first 10): {list(intersection)[:10]}")
lines.append("")
lines.append("Per-class row counts and train/test ratio:")
for lab, (t,s,ratio) in ratio_info.items():
    lines.append(f" Class {lab}: train_rows={t}, test_rows={s}, train_ratio={ratio:.4f}")
lines.append("")
lines.append(f"Total source_uid with >1 copy (overall): {dup_uids_overall}")
lines.append("Per-class source_uid with >1 copies:")
for lab, count in per_class_dup.items():
    lines.append(f" Class {lab}: source_uids_with_dup={count}")
lines.append('\n--- End of split report ---\n')

# print and append to file
print('\n'.join(lines))
with open(OUT,'a',encoding='utf-8') as f:
    f.write('\n'.join(lines)+"\n")
print('\nAppended report to', OUT)
