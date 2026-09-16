"""Configuration loader for LOGWAT.

Loads YAML files from the package `configs/` directory.
Requirements: PyYAML (`pip install pyyaml`).
"""
import os
from pathlib import Path

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML is required to load configs. Install with `pip install pyyaml`")


LOGWAT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = LOGWAT_ROOT / 'configs'


def load_yaml(name):
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_all():
    return {
        'config': load_yaml('config.yaml'),
        'dataset': load_yaml('dataset.yaml'),
        'model': load_yaml('model.yaml')
    }
