"""Labeling and cleaning utilities for dataset preparation.

This module consolidates labeling logic (multi-class assignment)
and cleaning/normalization used across the old scripts.
"""
import re
from urllib.parse import unquote
import csv
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

    Labels: 0=Normal, 1=SQLi, 2=XSS, 3=Other
    If `classification` indicates original CSIC label 'normal' (or 0), return 0 immediately.
    Otherwise inspect `content` with regexes.
    """
    try:
        if str(classification).lower() == 'normal' or classification == 0:
            return 0
    except Exception:
        pass

    if content is None:
        return 1
    s = str(content).lower()
    if s.strip() == '' or s.strip() == 'nan':
        return 1

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

    # default to SQLi/anomalous as original scripts did
    return 1


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

    # SQLi: replicate existing
    if len(df_sqli) == 0:
        df_sqli_final = pd.DataFrame(columns=df.columns)
    else:
        df_sqli_final = pd.concat([df_sqli] * (target_size // len(df_sqli) + 1)).iloc[:target_size]

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
    print(f"Wrote balanced dataset to: {output_path}")
    print(final_df['attack_type'].value_counts())
    return final_df

