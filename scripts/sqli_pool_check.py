#!/usr/bin/env python3
import os, random
from urllib.parse import unquote
import pandas as pd
from collections import Counter

PWD = os.getcwd()
POOL = os.path.join(PWD, 'data', 'sqli_payload_pool.csv')
AUG = os.path.join(PWD, 'data', 'augmented_web_attack.csv')
OUT = os.path.join(PWD, 'data', 'sqli_pool_integrity_check.txt')

if not os.path.exists(POOL):
    raise SystemExit('Payload pool missing: '+POOL)
if not os.path.exists(AUG):
    raise SystemExit('Augmented CSV missing: '+AUG)

p_df = pd.read_csv(POOL)
payloads = p_df['payload'].dropna().astype(str).tolist()
unique_payloads = len(set(payloads))
lines = []
lines.append(f"Payload pool rows: {len(p_df)}; unique payload texts: {unique_payloads}")
# family counts
family_counts = p_df['family'].value_counts().to_dict()
lines.append("Family counts:")
for fam, cnt in family_counts.items():
    lines.append(f" - {fam}: {cnt}")

# load augmented
aug = pd.read_csv(AUG)
# find synthetic sqli rows
if 'source' in aug.columns:
    synth = aug[aug['source']=='sqli_pool']
else:
    synth = aug[aug['source_uid'].astype(str).str.startswith('sqli_pool')]

total_synth = len(synth)
lines.append(f"\nTotal synthetic SQLi rows (source='sqli_pool' or source_uid prefix): {total_synth}")

# sample 10 random rows
sample_n = 10
random.seed(42)
if total_synth == 0:
    lines.append('No synthetic SQLi rows found. Bailing out.')
else:
    sampled = synth.sample(n=min(sample_n, total_synth), random_state=42)
    preserved = 0
    lines.append('\nSample integrity check (payload present verbatim after URL-decode):')
    for idx, row in sampled.iterrows():
        content = unquote(str(row.get('content','')))
        suid = str(row.get('source_uid',''))
        # parse index
        payload_text = None
        if suid.startswith('sqli_pool_'):
            try:
                i = int(suid.split('sqli_pool_')[-1])
                payload_text = payloads[i]
            except Exception:
                # fallback: try to find any payload present
                for p in payloads:
                    if p in content:
                        payload_text = p
                        break
        else:
            for p in payloads:
                if p in content:
                    payload_text = p
                    break
        ok = False
        if payload_text is not None and payload_text in content:
            ok = True
            preserved += 1
        lines.append(f" source_uid={suid} | payload='{payload_text}' | preserved={ok}")
    lines.append(f"\nPreserved payloads in sample: {preserved}/{len(sampled)}")

# distribution per source_uid
count_by_uid = synth['source_uid'].astype(str).value_counts().to_dict()
lines.append('\nDistribution of synthetic rows per payload (source_uid -> count):')
for uid, cnt in sorted(count_by_uid.items(), key=lambda x: -x[1])[:50]:
    lines.append(f" - {uid}: {cnt}")

# check for imbalance (>50% any payload)
imbalanced = [uid for uid,cnt in count_by_uid.items() if cnt > 0.5 * total_synth]
if imbalanced:
    lines.append('\nImbalance detected: payloads exceeding 50% of synthetic rows:')
    for uid in imbalanced:
        lines.append(f" - {uid}: {count_by_uid[uid]}")
else:
    lines.append('\nNo payload exceeds 50% of synthetic rows (OK)')

# write and print
with open(OUT,'w',encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('\nWrote integrity report to', OUT)
