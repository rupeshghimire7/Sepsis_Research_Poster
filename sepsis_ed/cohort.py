"""Cohort and cleaning: one row per adult ED stay with at least one usable triage vital.

Steps (protocol, "Who is included"):
  1. every edstays row joined one-to-one to triage on stay_id
  2. age >= min_age, age = anchor_age + (visit year - anchor_year)
  3. drop stays with no triage vital at all (after cleaning); partly missing stays are kept
The count after each step is returned for the cohort flow figure.
"""
import numpy as np
import pandas as pd

from .io import read

STAY_COLS = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "gender", "race", "arrival_transport",
             "disposition"]
TRIAGE_COLS = ["stay_id", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity",
               "chiefcomplaint"]


def clean_temperature(t, cfg):
    """MIMIC triage temperature is Fahrenheit, with some values typed in Celsius. Celsius range is kept,
    Fahrenheit range is converted, everything else becomes missing. Returns °C."""
    t = pd.to_numeric(t, errors="coerce")
    c_lo, c_hi = cfg["cleaning"]["temperature"]["celsius_range"]
    f_lo, f_hi = cfg["cleaning"]["temperature"]["fahrenheit_range"]
    return pd.Series(np.where(t.between(c_lo, c_hi), t, np.where(t.between(f_lo, f_hi), (t - 32) * 5 / 9, np.nan)),
                     index=t.index)


def clean_triage(tr, cfg):
    """Unit-fix temperature, set implausible vitals to missing, keep pain 0-10 and acuity 1-5 only."""
    tr = tr.copy()
    tr["temperature"] = clean_temperature(tr.temperature, cfg)
    for v, (lo, hi) in cfg["cleaning"]["ranges"].items():
        x = pd.to_numeric(tr[v], errors="coerce")
        tr[v] = x.where(x.between(lo, hi))
    lo, hi = cfg["cleaning"]["pain_range"]
    pain = pd.to_numeric(tr.pain, errors="coerce")  # free text such as "unable" or "8-9" -> missing
    tr["pain"] = pain.where(pain.between(lo, hi))
    lo, hi = cfg["cleaning"]["acuity_range"]
    acuity = pd.to_numeric(tr.acuity, errors="coerce")
    tr["acuity"] = acuity.where(acuity.between(lo, hi) & (acuity == acuity.round()))
    return tr


def build_cohort(mimic, ed, cfg):
    st = read(ed, "ed", "edstays", STAY_COLS, ["intime", "outtime"])
    tr = read(ed, "ed", "triage", TRIAGE_COLS)
    pt = read(mimic, "hosp", "patients", ["subject_id", "anchor_age", "anchor_year", "anchor_year_group"])
    if tr.stay_id.duplicated().any():
        raise ValueError("triage has more than one row per stay_id")

    flow = []

    def log(step, d):
        flow.append({"step": step, "stays": int(len(d)), "patients": int(d.subject_id.nunique()),
                     "removed": int(flow[-1]["stays"] - len(d)) if flow else 0})

    log("ED stays", st)
    df = st.merge(clean_triage(tr, cfg), on="stay_id", how="inner")
    log("with a triage record", df)
    df = df.merge(pt, on="subject_id", how="inner")
    log("with a patient record (age known)", df)
    df["age"] = df.anchor_age + (df.intime.dt.year - df.anchor_year)
    df = df[df.age >= cfg["cohort"]["min_age"]]
    log(f"aged {cfg['cohort']['min_age']} or over", df)
    df = df[df[cfg["cohort"]["vitals"]].notna().any(axis=1)]
    log("at least one triage vital recorded", df)
    return df.drop(columns=["anchor_age", "anchor_year"]).reset_index(drop=True), flow
