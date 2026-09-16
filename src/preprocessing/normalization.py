"""Labeling and cleaning utilities for dataset preparation.

This module consolidates labeling logic (multi-class assignment)
and cleaning/normalization used across the old scripts.
"""
import re
from urllib.parse import unquote
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

