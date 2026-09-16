"""Tokenization and text utilities used across preprocessing.

This module provides the project's tokenizer and a small entropy
utility extracted from `scripts/build_web_graphs.py`.
"""
import re
import math
from urllib.parse import unquote


def web_security_tokenizer(text):
    """Tokenize web-related content preserving punctuation tokens."""
    text = unquote(str(text)).lower()
    text = re.sub(r"(['\"#;%\(\)\-\+\/\*<>=\[\]\{\},\.@])", r" \1 ", text)
    return re.findall(r"[\w']+|[^\w\s]", text)


def get_entropy(text):
    """Shannon entropy of the string (base-2). Returns 0 for empty input."""
    if not text or len(text) == 0:
        return 0
    probs = [float(text.count(c)) / len(text) for c in set(text)]
    return -sum([p * math.log(p, 2) for p in probs])


def tokenize(text):
    """Simple whitespace tokenizer kept for compatibility."""
    return web_security_tokenizer(text)
