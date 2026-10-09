"""File access for MIMIC-IV / MIMIC-IV-ED (demo or full). Accepts .csv.gz or .csv."""
from pathlib import Path

import pandas as pd


def _path(base, module, table):
    for ext in (".csv.gz", ".csv"):
        p = Path(base) / module / f"{table}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{module}/{table}.csv[.gz] not found under {base}")


def read(base, module, table, usecols=None, dates=(), filter_fn=None, chunksize=5_000_000, dtype=None):
    """Read a table. `filter_fn(df)->mask` is applied per chunk so the full
    labevents/chartevents tables (100M+ rows) never sit in memory at once."""
    p = _path(base, module, table)
    kw = dict(usecols=usecols, low_memory=False, dtype=dtype)
    if filter_fn is None:
        df = pd.read_csv(p, **kw)
    else:
        parts = [c[filter_fn(c)] for c in pd.read_csv(p, chunksize=chunksize, **kw)]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
    for c in dates:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def exists(base, module, table):
    try:
        _path(base, module, table)
        return True
    except FileNotFoundError:
        return False


def hours(td):
    return td.dt.total_seconds() / 3600.0
