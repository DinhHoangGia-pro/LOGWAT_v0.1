#!/usr/bin/env python3
import pandas as pd
import os

aug = os.path.join('data','augmented_web_attack.csv')
df = pd.read_csv(aug)
# select synthetic sqli
synth = df[(df['attack_type']==1) & (df.get('source')=='sqli_pool')]
# if empty, infer by source_uid prefix
if len(synth)==0 and 'source_uid' in df.columns:
    synth = df[(df['attack_type']==1) & df['source_uid'].astype(str).str.startswith('sqli_pool_')]
out = os.path.join('data','sqli_unique_samples.txt')
with open(out,'w',encoding='utf-8') as f:
    for uid, g in synth.groupby('source_uid'):
        content = str(g.iloc[0]['content']).replace('\n',' ')
        f.write(f"{uid}\t{content}\n")
print('Wrote', out, 'with', len(synth["source_uid"].unique()), 'unique synthetic payload samples')
