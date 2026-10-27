"""Triage-time features only: everything here is known when the patient is triaged.

Three nested sets (protocol, "Feature engineering"):
  A  vitals: six cleaned vitals, derived vitals, clinical flags, missing flags, age, sex
  B  A + pain, chief complaint groups, arrival mode and time, history before this arrival  (primary model)
  C  B + nurse triage acuity
History uses only records closed before this arrival (ED outtime / hospital dischtime < intime).
"""
import numpy as np
import pandas as pd

from .io import read
from .labels import icd_match

VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
ARRIVAL = ["AMBULANCE", "WALK IN", "HELICOPTER", "OTHER", "UNKNOWN"]


# ---------------------------------------------------------------- set A
def vital_features(df, cfg):
    f, out = cfg["flags"], pd.DataFrame(index=df.index)
    out["age"] = df.age
    out["male"] = (df.gender == "M").astype(int)
    for v in VITALS:
        out[v] = df[v]
    out["map"] = (df.sbp + 2 * df.dbp) / 3
    out["shock_index"] = df.heartrate / df.sbp
    out["pulse_pressure"] = df.sbp - df.dbp
    # comparisons with NaN are False, so a missing vital never raises a flag (its missing flag does)
    out["fever"] = (df.temperature >= f["fever_c"]).astype(int)
    out["low_temp"] = (df.temperature < f["low_temp_c"]).astype(int)
    out["hr_high"] = (df.heartrate > f["hr_high"]).astype(int)
    out["rr_high"] = (df.resprate > f["rr_high"]).astype(int)
    out["rr_qsofa"] = (df.resprate >= f["rr_qsofa"]).astype(int)
    out["sbp_low"] = (df.sbp <= f["sbp_low"]).astype(int)
    out["spo2_low"] = (df.o2sat < f["spo2_low"]).astype(int)
    out["n_abnormal_vitals"] = (out.fever | out.low_temp) + out.hr_high + out.rr_high + out.sbp_low + out.spo2_low
    for v in VITALS:
        out[f"miss_{v}"] = df[v].isna().astype(int)
    out["n_missing_vitals"] = df[VITALS].isna().sum(axis=1)
    return out


# ---------------------------------------------------------------- set B (context)
def context_features(df, cfg):
    out = pd.DataFrame(index=df.index)
    out["pain"] = df.pain
    out["miss_pain"] = df.pain.isna().astype(int)

    cc = df.chiefcomplaint.fillna("").str.lower()
    for k, pat in cfg["complaint_groups"].items():
        out[k] = cc.str.contains(pat, regex=True).astype(int)
    out["cc_other"] = (out[list(cfg["complaint_groups"])].sum(axis=1) == 0).astype(int)

    at = df.arrival_transport.fillna("UNKNOWN").str.upper()
    for k in ARRIVAL:
        out[f"arrive_{k.lower().replace(' ', '_')}"] = (at == k).astype(int)

    # MIMIC date shifting keeps time of day and day of week
    night_from, night_to = cfg["flags"]["night_hours"]
    out["arrival_hour"] = df.intime.dt.hour
    out["weekend"] = (df.intime.dt.day_of_week >= 5).astype(int)
    out["night"] = ((df.intime.dt.hour >= night_from) | (df.intime.dt.hour < night_to)).astype(int)
    return out


def _seconds(t):
    return t.astype("datetime64[s]").astype("int64").to_numpy()


def count_in_window(q_subject, q_time, e_subject, e_time, days):
    """For each query, events of the same subject with time in [q_time - days, q_time).
    Sorted composite keys (subject, seconds) + searchsorted, so no subject self-join."""
    ok = e_time.notna()
    if not ok.any():
        return np.zeros(len(q_subject), dtype=int)
    e_subject, e_time = e_subject[ok], e_time[ok]
    span = days * 86400
    base = min(_seconds(q_time).min(), _seconds(e_time).min()) - span
    scale = np.int64(10 ** 10)  # > 300 years of seconds, so subjects never overlap
    ek = np.sort(e_subject.to_numpy(np.int64) * scale + (_seconds(e_time) - int(base)))
    qk = q_subject.to_numpy(np.int64) * scale + (_seconds(q_time) - int(base))
    return np.searchsorted(ek, qk, "left") - np.searchsorted(ek, qk - span, "left")


def ever_before(q_subject, q_time, e_subject, e_time):
    """1 if the subject has any event strictly before q_time."""
    ok = e_time.notna()
    if not ok.any():
        return np.zeros(len(q_subject), dtype=int)
    first = pd.Series(e_time[ok].to_numpy(), index=e_subject[ok].to_numpy()).groupby(level=0).min()
    return (first.reindex(q_subject.to_numpy()).to_numpy() < q_time.to_numpy()).astype(int)


def history_features(df, mimic, ed, cfg):
    out = pd.DataFrame(index=df.index)
    days = cfg["history"]["lookback_days"]
    subj, t = df.subject_id, df.intime

    stays = read(ed, "ed", "edstays", ["subject_id", "outtime"], ["outtime"])
    adm = read(mimic, "hosp", "admissions", ["subject_id", "hadm_id", "dischtime"], ["dischtime"])
    out["ed_visits_365d"] = count_in_window(subj, t, stays.subject_id, stays.outtime, days)
    out["admissions_365d"] = count_in_window(subj, t, adm.subject_id, adm.dischtime, days)

    dx = read(mimic, "hosp", "diagnoses_icd", ["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    dx = dx.merge(adm, on="hadm_id")

    def coded(icd10, icd9):
        hit = dx[icd_match(dx.icd_code, dx.icd_version, icd10, icd9)]
        return ever_before(subj, t, hit.subject_id, hit.dischtime)

    sep = cfg["outcomes"]["sepsis_icd"]
    out["prior_sepsis_admission"] = coded(sep["icd10"], sep["icd9"])
    for name, codes in cfg["history"]["comorbidities"].items():
        out[f"hx_{name}"] = coded(codes.get("icd10", []), codes.get("icd9", []))
    return out


# ---------------------------------------------------------------- assemble
def build_features(cohort, mimic, ed, cfg):
    """Returns (features DataFrame aligned to cohort, {set name: feature list})."""
    a = vital_features(cohort, cfg)
    b = pd.concat([context_features(cohort, cfg), history_features(cohort, mimic, ed, cfg)], axis=1)
    c = pd.DataFrame({"acuity": cohort.acuity, "miss_acuity": cohort.acuity.isna().astype(int)})
    sets = {"A": list(a), "B": list(a) + list(b), "C": list(a) + list(b) + list(c)}
    return pd.concat([a, b, c], axis=1), sets


def missingness_table(df, cols, outcome):
    """% missing per variable, overall and by outcome (aggregate only)."""
    rows = {"overall": df[cols].isna().mean()}
    for val, g in df.groupby(outcome):
        rows[f"{outcome}={val}"] = g[cols].isna().mean()
    return (pd.DataFrame(rows) * 100).round(2).rename_axis("variable")
