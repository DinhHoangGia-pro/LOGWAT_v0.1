"""Labeling and cleaning utilities for dataset preparation.

This module consolidates labeling logic (multi-class assignment)
and cleaning/normalization used across the old scripts.
"""
import re
from urllib.parse import unquote
import csv
import os
from pathlib import Path
from datetime import datetime
import pandas as pd


def web_attack_classifier(row):
    """Assign multi-class labels to a row.

    Labels: 0=Normal, 1=SQLi, 2=XSS, 3=Other
    """
    try:
        if str(row.get('classification', '')).lower() == 'normal' or row.get('classification') == 0:
            return 0
    except Exception:
        pass

    target_data = (str(row) + " " + str(row.get('content', ''))).lower()

    xss_patterns = [r"<script.*?>", r"alert\(", r"onerror=", r"onload=", r"javascript:",
                    r"document\.cookie", r"<img.*?src=", r"onmouseover=", r"eval\(", r"window\.location"]
    if any(re.search(p, target_data) for p in xss_patterns):
        return 2

    sqli_patterns = [r"union.*select", r"select.*from", r"insert.*into", r"drop.*table",
                     r"or\s+\d+=\d+", r"sleep\(", r"benchmark\(", r"information_schema",
                     r"admin'--", r"order\s+by"]
    if any(re.search(p, target_data) for p in sqli_patterns):
        return 1

    other_patterns = [r"\.\./\.\./", r"etc/passwd", r"%0d%0a", r"boot\.ini"]
    if any(re.search(p, target_data) for p in other_patterns):
        return 3

    return 1


def clean_text(value):
    """Normalize a single content value; returns None for invalid/short values."""
    if pd.isna(value) or str(value).lower() == 'nan':
        return None
    text = unquote(str(value))
    text = text.strip()
    if len(text) < 2:
        return None
    return text


def clean_dataframe(df, content_col='content'):
    """Clean a DataFrame in-place (returns cleaned copy).

    - URL-decode content
    - Drop NaN/short entries
    - Remove rows where content is string 'nan'
    - Drop duplicates by content
    """
    df = df.copy()
    if content_col not in df.columns:
        return df

    df[content_col] = df[content_col].apply(clean_text)
    df = df.dropna(subset=[content_col])
    df = df[df[content_col].astype(str).str.lower() != 'nan']
    df = df[df[content_col].astype(str).str.len() > 1]
    df = df.drop_duplicates(subset=[content_col])
    return df


def label_dataframe(df):
    """Apply `web_attack_classifier` to produce `attack_type` column."""
    df = df.copy()
    # ensure columns normalized (fix known typos)
    df.columns = [c.replace('lenght', 'length') for c in df.columns]
    df['attack_type'] = df.apply(web_attack_classifier, axis=1)
    return df


# --- New, stable API requested by the user ---
def classify_request(content: str, classification=None) -> int:
    """Classify a single request content string into labels.

    Labels: 0=Normal, 1=SQLi, 2=XSS, 3=Other, -1=Unknown

    IMPORTANT: `attack_type` here is *inferred by regex heuristics* (Table 1-like
    rules), and is NOT the original CSIC 2010 ground-truth for specific SQLi/XSS
    attacks. This function deliberately returns -1 for requests that do not match
    any heuristic so they can be handled explicitly (and excluded from the
    augmented dataset) rather than being implicitly labeled as SQLi.

    If `classification` indicates original CSIC label 'normal' (or 0), return 0
    immediately. Otherwise inspect `content` with regexes.
    """
    try:
        if str(classification).lower() == 'normal' or classification == 0:
            return 0
    except Exception:
        pass

    if content is None:
        return -1
    s = str(content).lower()
    if s.strip() == '' or s.strip() == 'nan':
        return -1

    xss_patterns = [r"<script.*?>", r"alert\(", r"onerror=", r"onload=", r"javascript:",
                    r"document\.cookie", r"<img.*?src=", r"onmouseover=", r"eval\(", r"window\.location"]
    if any(re.search(p, s) for p in xss_patterns):
        return 2

    sqli_patterns = [r"union.*select", r"select.*from", r"insert.*into", r"drop.*table",
                     r"or\s+\d+=\d+", r"sleep\(", r"benchmark\(", r"information_schema",
                     r"admin'--", r"order\s+by"]
    if any(re.search(p, s) for p in sqli_patterns):
        return 1

    other_patterns = [r"\.\./\.\./", r"etc/passwd", r"%0d%0a", r"boot\.ini"]
    if any(re.search(p, s) for p in other_patterns):
        return 3

    # default -> Unknown (do not assume SQLi)
    return -1


def clean_content(text):
    """Normalize a content string: URL-unquote and reject very short/empty values.

    Returns cleaned string or None if the value should be discarded.
    """
    # reuse existing semantics of `clean_text`
    if pd.isna(text) or str(text).lower() == 'nan':
        return None
    value = unquote(str(text))
    value = value.strip()
    if len(value) < 2:
        return None
    return value


def label_dataset(csic_raw_path, output_path):
    """Read raw CSIC CSV from `csic_raw_path`, label and write to `output_path`.

    Returns the labeled DataFrame.
    """
    df = pd.read_csv(csic_raw_path)
    # normalize columns (fix known typos)
    df.columns = [c.replace('lenght', 'length') for c in df.columns]

    # clean content column in-place
    if 'content' in df.columns:
        df['content'] = df['content'].apply(clean_content)
        df = df.dropna(subset=['content']).drop_duplicates(subset=['content']).reset_index(drop=True)

    # classify using the classifier that has access to original `classification` column
    if 'classification' in df.columns:
        df['attack_type'] = df.apply(lambda r: classify_request(r['content'], r.get('classification')), axis=1)
    else:
        df['attack_type'] = df['content'].apply(lambda c: classify_request(c, None))
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    return df


def balance_dataset(labeled_path, xss_source_path, output_path) -> pd.DataFrame:
    """Balance and augment a labeled CSIC dataset using an XSS source CSV.

    Implements the same logic as the old `rebuild_data()`:
    - Clean labeled dataset
    - Determine target size: max(len(normal)//2, len(sqli), 10000)
    - Build Benign (half-normal, duplicated), SQLi (duplicated), XSS (poison frames)
    - Shuffle and write to `output_path`
    """
    print("--- Rebuilding balanced dataset ---")
    df = pd.read_csv(labeled_path)
    if 'content' in df.columns:
        df['content'] = df['content'].apply(clean_content)
        df = df.dropna(subset=['content']).drop_duplicates(subset=['content']).reset_index(drop=True)

    # add a deterministic source id for each original row so duplicated copies
    # can be grouped and kept together during train/test splits (use
    # GroupShuffleSplit on `source_uid` in training later).
    df = df.reset_index(drop=True)
    df['source_uid'] = df.index.astype(str)

    # remove Unknown (-1) labels produced by the heuristic classifier; these
    # represent requests that did not match any regex and should not be
    # implicitly labeled as SQLi/XSS. Count them for the report.
    unknown_count = 0
    if 'attack_type' in df.columns:
        unknown_count = int((df['attack_type'] == -1).sum())
        if unknown_count > 0:
            print(f"[!] Removing {unknown_count} rows with Unknown attack_type (-1)")
        df = df[df['attack_type'] != -1].reset_index(drop=True)

    df_normal = df[df['attack_type'] == 0].reset_index(drop=True)
    df_sqli = df[df['attack_type'] == 1].reset_index(drop=True)

    # read xss source
    df_xss_source = pd.read_csv(xss_source_path)
    payload_col = 'Sentence' if 'Sentence' in df_xss_source.columns else df_xss_source.columns[4]
    xss_payloads = df_xss_source[df_xss_source['Label'] == 1][payload_col].dropna().astype(str).unique().tolist()
    if len(xss_payloads) == 0:
        raise RuntimeError('No XSS payloads found in XSS source')

    target_size = max(len(df_normal) // 2, len(df_sqli), 10000)
    print(f"[*] Target per-class size: {target_size}")

    # BENIGN: half of normal
    n_benign = max(1, len(df_normal) // 2)
    df_benign = df_normal.sample(n=n_benign, random_state=42)
    df_benign_final = pd.concat([df_benign] * (target_size // len(df_benign) + 1)).iloc[:target_size]

    # SQLi: build synthetic SQLi pool using external payload list, and keep
    # original CSIC SQLi rows separately marked as 'csic_original'.
    payload_pool_path = Path.cwd() / 'data' / 'sqli_payload_pool.csv'
    sqli_payloads = []
    if payload_pool_path.exists():
        try:
            p_df = pd.read_csv(payload_pool_path)
            if 'payload' in p_df.columns:
                sqli_payloads = p_df['payload'].dropna().astype(str).tolist()
        except Exception:
            sqli_payloads = []
    if len(sqli_payloads) == 0:
        # fallback to replicating originals if no pool available
        if len(df_sqli) == 0:
            df_sqli_final = pd.DataFrame(columns=df.columns)
        else:
            df_sqli_final = pd.concat([df_sqli] * (target_size // len(df_sqli) + 1)).iloc[:target_size]
    else:
        # keep original CSIC SQLi rows as a small separate subset
        df_sqli_original = df_sqli.copy()
        if len(df_sqli_original) > 0:
            df_sqli_original['source'] = 'csic_original'

        # create synthetic SQLi by poisoning benign frames with payloads
        frames_for_sqli = df_normal.drop(df_benign.index) if len(df_normal) > 0 else df_normal
        if len(frames_for_sqli) == 0:
            frames_for_sqli = df_normal
        synthetic = []
        for i in range(target_size):
            template = frames_for_sqli.iloc[i % len(frames_for_sqli)].to_dict()
            payload = sqli_payloads[i % len(sqli_payloads)]
            template['content'] = payload
            template['attack_type'] = 1
            template['source'] = 'sqli_pool'
            # assign a source_uid per payload so duplicates of same payload
            # can be grouped during splitting
            template['source_uid'] = f"sqli_pool_{i % len(sqli_payloads)}"
            synthetic.append(template)
        df_sqli_final = pd.DataFrame(synthetic)
        # append originals to final as well (so evaluations can separate them)
        if len(df_sqli_original) > 0:
            # ensure original source_uid preserved; set source column on originals
            df_sqli_original = df_sqli_original.reset_index(drop=True)
            df_sqli_final = pd.concat([df_sqli_final, df_sqli_original], ignore_index=True)

    # XSS: poison frames from remaining normal
    frames = df_normal.drop(df_benign.index) if len(df_normal) > 0 else df_normal
    if len(frames) == 0:
        frames = df_normal
    poisoned = []
    for i in range(target_size):
        template = frames.iloc[i % len(frames)].to_dict()
        template['content'] = xss_payloads[i % len(xss_payloads)]
        template['attack_type'] = 2
        poisoned.append(template)
    df_xss_final = pd.DataFrame(poisoned)

    final_df = pd.concat([df_benign_final, df_sqli_final, df_xss_final])
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    try:
        report_duplication_stats(final_df, unknown_removed=unknown_count)
    except Exception:
        pass
    print(f"Wrote balanced dataset to: {output_path}")
    print(final_df['attack_type'].value_counts())
    return final_df


def report_duplication_stats(df: pd.DataFrame, out_path: str = None, unknown_removed: int = 0):
    """Write a concise duplication report to `data/duplication_report.txt`.

    The report contains: total rows, unique contents, and per-class duplication rates.
    If `out_path` is None the function writes to `data/duplication_report.txt` in
    the current working directory (project root).
    """
    if out_path is None:
        out_path = Path.cwd() / 'data' / 'duplication_report.txt'
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(df)
    unique_content = int(df['content'].nunique()) if 'content' in df.columns else total
    duplicates = total - unique_content

    lines = []
    lines.append(f"Duplication report - {datetime.utcnow().isoformat()}Z")
    lines.append(f"Total rows (after augmentation): {total}")
    lines.append(f"Unique content values: {unique_content}")
    lines.append(f"Duplicate rows: {duplicates} (rate: {duplicates/total:.4f})")
    lines.append(f"Unknown rows removed prior to augmentation: {unknown_removed}")
    lines.append("")

    # per-class stats
    if 'attack_type' in df.columns:
        for t in sorted(df['attack_type'].unique()):
            sub = df[df['attack_type'] == t]
            cnt = len(sub)
            uniq = int(sub['content'].nunique()) if 'content' in sub.columns else cnt
            dup = cnt - uniq
            lines.append(f"Class {t}: rows={cnt}, unique_content={uniq}, duplicates={dup}, dup_rate={(dup/cnt if cnt>0 else 0):.4f}")

    lines.append("\n--- End of report ---\n")

    with open(out_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

