#!/usr/bin/env python3
import os, pickle
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
import pandas as pd

ROOT = os.getcwd()
AUG = os.path.join(ROOT, 'data', 'augmented_web_attack.csv')
PKL = os.path.join(ROOT, 'data', 'web_graphs.pkl')
POOL = os.path.join(ROOT, 'data', 'sqli_payload_pool.csv')
OUT = os.path.join(ROOT, 'data', 'sqli_test_coverage.txt')

if not os.path.exists(AUG) or not os.path.exists(PKL):
    raise SystemExit('Missing augmented CSV or graphs pickle')

df = pd.read_csv(AUG)
with open(PKL,'rb') as f:
    data = pickle.load(f)
graphs = list(data.get('graphs', []))

# build uids and labels from graphs
uids = []
labels = []
for g in graphs:
    uid = getattr(g, 'source_uid', None)
    uid = str(uid) if uid is not None else None
    uids.append(uid)
    try:
        labels.append(int(g.y.item()))
    except Exception:
        labels.append(None)

# split
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(graphs, groups=uids))
test_uids = set(uids[i] for i in test_idx if uids[i] is not None)

# from augmented df, get SQLi source_uids present
sqli_df = df[df['attack_type'] == 1]
sqli_uids = set(sqli_df['source_uid'].astype(str).unique())
# test SQLi uids = intersection
test_sqli_uids = sorted(list(test_uids & sqli_uids))

# classify each test sqli uid as csic_original or sqli_pool
pool_families = {}
if os.path.exists(POOL):
    p_df = pd.read_csv(POOL)
    payloads = p_df['payload'].dropna().astype(str).tolist()
    families = p_df['family'].astype(str).tolist()
else:
    payloads = []
    families = []

results = []
for uid in test_sqli_uids:
    rows = sqli_df[sqli_df['source_uid'].astype(str)==uid]
    source_val = None
    if 'source' in sqli_df.columns:
        if len(rows)>0:
            source_val = rows.iloc[0].get('source', None)
    if source_val is None:
        source_val = 'sqli_pool' if uid.startswith('sqli_pool_') else ('csic_original' if uid.isdigit() or uid.isnumeric() else 'unknown')
    fam = None
    if source_val == 'sqli_pool':
        # extract index
        try:
            idx = int(uid.split('sqli_pool_')[-1])
            fam = families[idx] if idx < len(families) else None
        except Exception:
            fam = None
    results.append((uid, source_val, fam))

# counts
cnt_csic = sum(1 for _,src,_ in results if src=='csic_original')
cnt_pool = sum(1 for _,src,_ in results if src=='sqli_pool')
other = sum(1 for _,src,_ in results if src not in ('csic_original','sqli_pool'))

# family coverage among pool ones
fam_counts = Counter([fam for _,src,fam in results if src=='sqli_pool' and fam is not None])

lines = []
lines.append(f"SQLi test coverage - {pd.Timestamp.utcnow().isoformat()}Z")
lines.append(f"Total SQLi source_uids in TEST: {len(test_sqli_uids)}")
lines.append(f" - csic_original: {cnt_csic}")
lines.append(f" - sqli_pool: {cnt_pool}")
lines.append(f" - other/unknown: {other}")
lines.append('')
lines.append('sqli_pool family coverage in TEST:')
for fam,ct in fam_counts.items():
    lines.append(f" - {fam}: {ct}")
lines.append('')
lines.append('Detailed list (uid, source, family):')
for uid,src,fam in results:
    lines.append(f"{uid}\t{src}\t{fam}")

with open(OUT,'w',encoding='utf-8') as f:
    f.write('\n'.join(lines)+"\n")
print('\n'.join(lines))
print('\nWrote report to', OUT)
