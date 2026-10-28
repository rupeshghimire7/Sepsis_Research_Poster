"""Load config.yaml"""
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parents[1] / "config.yaml"


def load(path=None):
    with open(path or DEFAULT) as f:
        return yaml.safe_load(f)
