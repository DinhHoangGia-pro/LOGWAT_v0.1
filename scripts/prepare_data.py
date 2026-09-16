"""Entry point to prepare and build the balanced/augmented dataset.

This script runs the sequence:
 1. Label raw CSIC dataset -> labeled file
 2. Clean labeled dataset
 3. Rebuild / augment balanced dataset (XSS poisoning + balancing)

It re-uses logic consolidated in `src.preprocessing.normalization`.
"""
import argparse
import os
import pandas as pd
from src.preprocessing.normalization import label_dataframe, clean_dataframe
from src.utils.config import load_all


cfgs = load_all()
paths_cfg = cfgs['dataset'].get('paths', {})



def rebuild_and_augment(df_labeled, xss_source_df, target_size=10000):
    """Rebuild balanced dataset from cleaned labeled CSIC and XSS source."""
    # split classes
    df_normal = df_labeled[df_labeled['attack_type'] == 0].reset_index(drop=True)
    df_sqli = df_labeled[df_labeled['attack_type'] == 1].reset_index(drop=True)

    # extract xss payloads
    payload_col = 'Sentence' if 'Sentence' in xss_source_df.columns else xss_source_df.columns[4]
    xss_payloads = xss_source_df[xss_source_df['Label'] == 1][payload_col].dropna().astype(str).tolist()
    if len(xss_payloads) == 0:
        raise RuntimeError('No XSS payloads found in XSS source')

    # determine target per class
    target = max(len(df_normal) // 2, len(df_sqli), target_size)

    # benign: half of normal
    df_benign = df_normal.sample(n=max(1, len(df_normal)//2), random_state=42)
    df_benign_final = pd.concat([df_benign] * (target // len(df_benign) + 1)).iloc[:target]

    # sqli: replicate
    if len(df_sqli) == 0:
        df_sqli_final = pd.DataFrame(columns=df_labeled.columns)
    else:
        df_sqli_final = pd.concat([df_sqli] * (target // len(df_sqli) + 1)).iloc[:target]

    # xss: poison frames from remaining normal records
    frames = df_normal.drop(df_benign.index)
    if len(frames) == 0:
        frames = df_normal

    poisoned = []
    for i in range(target):
        template = frames.iloc[i % len(frames)].to_dict()
        template['content'] = xss_payloads[i % len(xss_payloads)]
        template['attack_type'] = 2
        poisoned.append(template)
    df_xss_final = pd.DataFrame(poisoned)

    final = pd.concat([df_benign_final, df_sqli_final, df_xss_final]).sample(frac=1, random_state=42).reset_index(drop=True)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csic_raw', default=None)
    parser.add_argument('--labeled_out', default=None)
    parser.add_argument('--xss_source', default=None)
    parser.add_argument('--augmented_out', default=None)
    parser.add_argument('--target', type=int, default=10000)
    args = parser.parse_args()

    # fill defaults from config if not provided
    csic_raw = args.csic_raw or os.path.join(cfgs['dataset'].get('hin_dir'), paths_cfg.get('csic_raw'))
    labeled_out = args.labeled_out or os.path.join(cfgs['dataset'].get('hin_dir'), paths_cfg.get('labeled'))
    xss_source = args.xss_source or os.path.join(cfgs['dataset'].get('hin_dir'), paths_cfg.get('xss_source'))
    augmented_out = args.augmented_out or os.path.join(cfgs['dataset'].get('hin_dir'), paths_cfg.get('augmented'))

    # Step 1: label raw CSIC
    if not os.path.exists(csic_raw):
        raise FileNotFoundError(f'CSIC raw file not found: {csic_raw}')

    df_raw = pd.read_csv(csic_raw)
    df_labeled = label_dataframe(df_raw)
    df_labeled.to_csv(labeled_out, index=False)
    print(f'Wrote labeled CSIC to: {labeled_out}')

    # Step 2: clean labeled CSIC
    df_labeled = pd.read_csv(labeled_out)
    df_clean = clean_dataframe(df_labeled, content_col='content')
    # overwrite labeled_out with cleaned+labeled to keep downstream scripts compatible
    df_clean.to_csv(labeled_out, index=False)
    print(f'Wrote cleaned labeled CSIC to: {labeled_out}')

    # Step 3: rebuild/augment
    if not os.path.exists(xss_source):
        raise FileNotFoundError(f'XSS source not found: {xss_source}')

    df_xss_src = pd.read_csv(xss_source)
    final = rebuild_and_augment(df_clean, df_xss_src, target_size=args.target)
    final.to_csv(augmented_out, index=False)
    print(f'Wrote augmented balanced dataset to: {augmented_out}')


if __name__ == '__main__':
    main()

