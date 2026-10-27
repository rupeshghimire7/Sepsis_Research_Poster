"""Outcomes for each ED stay.

sepsis_icd  (main)       a sepsis diagnosis code on the ED stay or on the hospital admission that followed
critical    (secondary)  died in hospital or in the ED, or reached ICU within 12 h of arrival
"""
import pandas as pd

from .io import exists, hours, read

# ---------------------------------------------------------------- ICD codes
ICD10 = ("A40", "A41", "R652")
ICD9 = ("038", "99591", "99592", "78552")


def _prefix_mask(code, patterns):
    """Prefix match; "LO-HI" is a range compared on the first len(LO) characters (e.g. "C00-C97")."""
    m = pd.Series(False, index=code.index)
    for p in patterns:
        p = str(p).replace(".", "").upper()
        if "-" in p:
            lo, hi = p.split("-")
            head = code.str[:len(lo)]
            m |= (code.str.len() >= len(lo)) & (head >= lo) & (head <= hi)
        else:
            m |= code.str.startswith(p)
    return m


def icd_match(code, ver, icd10, icd9):
    code = code.astype(str).str.replace(".", "", regex=False).str.strip().str.upper()
    ver = pd.to_numeric(ver, errors="coerce")
    return ((ver == 10) & _prefix_mask(code, icd10)) | ((ver == 9) & _prefix_mask(code, icd9))


# ---------------------------------------------------------------- per-stay outcomes
# Each takes `stays` (one row per ED stay: stay_id, subject_id, hadm_id, intime, ...) and returns
# a Series indexed like `stays`.

def sepsis_icd_label(mimic, ed, stays, icd10=ICD10, icd9=ICD9):
    """Primary outcome: a sepsis code on the ED stay or on its linked hospital admission."""
    dx = read(mimic, "hosp", "diagnoses_icd", ["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    hadm_pos = set(dx.loc[icd_match(dx.icd_code, dx.icd_version, icd10, icd9), "hadm_id"])
    ed_pos = set()
    if exists(ed, "ed", "diagnosis"):
        edx = read(ed, "ed", "diagnosis", ["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})
        ed_pos = set(edx.loc[icd_match(edx.icd_code, edx.icd_version, icd10, icd9), "stay_id"])
    return (stays.hadm_id.isin(hadm_pos) | stays.stay_id.isin(ed_pos)).astype(int)


def critical_label(mimic, stays, icu_within_h=12):
    """Secondary outcome: in-hospital death on the linked admission, death in the ED, or an ICU stay
    starting within `icu_within_h` hours of ED arrival (linked by subject_id + time, since some ICU
    stays have no ED hadm_id)."""
    adm = read(mimic, "hosp", "admissions", ["hadm_id", "hospital_expire_flag"])
    died = stays.hadm_id.isin(set(adm.loc[adm.hospital_expire_flag == 1, "hadm_id"]))
    died |= stays.disposition.fillna("").str.upper().eq("EXPIRED")

    icu = read(mimic, "icu", "icustays", ["subject_id", "intime"], ["intime"]).dropna()
    icu = icu.rename(columns={"intime": "icu_intime"}).sort_values("icu_intime")
    q = stays[["subject_id", "intime"]].reset_index().sort_values("intime")
    nxt = pd.merge_asof(q, icu, left_on="intime", right_on="icu_intime", by="subject_id", direction="forward")
    icu_soon = (hours(nxt.icu_intime - nxt.intime) <= icu_within_h).set_axis(nxt["index"]).reindex(stays.index)
    return (died | icu_soon.fillna(False)).astype(int)
