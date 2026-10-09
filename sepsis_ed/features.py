"""Triage-time features only: everything here is known when the patient is
triaged, so the model can be used at the ED front door without label leakage."""
import numpy as np
import pandas as pd

from .io import read

COMPLAINT_FLAGS = {
    "cc_fever": r"fever|febrile|chills|rigors",
    "cc_respiratory": r"cough|sob|dyspnea|short(?:ness)? of breath|pneumonia|hypoxi",
    "cc_urinary": r"uti|dysuria|urinary|hematuria|flank",
    "cc_abdominal": r"abd|abdominal|n/v|vomit|nausea|diarrhea",
    "cc_neuro": r"altered|ams|confus|letharg|unresponsive|mental status|weak",
    "cc_skin": r"cellulitis|wound|abscess|ulcer|infection|redness|swelling",
    "cc_hypotension": r"hypotens|low bp|syncope|dizz",
    "cc_transfer": r"transfer",
}
VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]


def build_features(mimic, ed):
    st = read(ed, "ed", "edstays", ["subject_id", "stay_id", "intime", "gender", "arrival_transport"], ["intime"])
    tr = read(ed, "ed", "triage", ["stay_id"] + VITALS + ["chiefcomplaint"])
    pt = read(mimic, "hosp", "patients", ["subject_id", "anchor_age", "anchor_year"])
    adm = read(mimic, "hosp", "admissions", ["subject_id", "admittime"], ["admittime"])

    df = st.merge(tr, on="stay_id", how="left").merge(pt, on="subject_id", how="left")
    df["age"] = df.anchor_age + (df.intime.dt.year - df.anchor_year)
    df["male"] = (df.gender == "M").astype(int)
    for v in VITALS:
        df[v] = pd.to_numeric(df[v], errors="coerce")

    # plausibility clipping (triage data has typos, e.g. temp 9.8 / HR 1000)
    df.loc[~df.temperature.between(85, 110), "temperature"] = np.nan  # °F
    df.loc[~df.heartrate.between(20, 250), "heartrate"] = np.nan
    df.loc[~df.resprate.between(4, 70), "resprate"] = np.nan
    df.loc[~df.o2sat.between(50, 100), "o2sat"] = np.nan
    df.loc[~df.sbp.between(40, 300), "sbp"] = np.nan
    df.loc[~df.dbp.between(15, 200), "dbp"] = np.nan
    df.loc[~df.pain.between(0, 10), "pain"] = np.nan

    df["map"] = (df.sbp + 2 * df.dbp) / 3
    df["shock_index"] = df.heartrate / df.sbp
    df["sirs_partial"] = ((df.temperature > 100.4) | (df.temperature < 96.8)).astype(int) + \
        (df.heartrate > 90).astype(int) + (df.resprate > 20).astype(int)
    df["qsofa_partial"] = (df.resprate >= 22).astype(int) + (df.sbp <= 100).astype(int)

    cc = df.chiefcomplaint.fillna("").str.lower()
    for k, pat in COMPLAINT_FLAGS.items():
        df[k] = cc.str.contains(pat, regex=True).astype(int)

    at = df.arrival_transport.fillna("UNKNOWN").str.upper()
    for k in ["AMBULANCE", "WALK IN", "HELICOPTER", "OTHER"]:
        df[f"arrive_{k.lower().replace(' ', '_')}"] = (at == k).astype(int)

    # prior hospital admissions strictly before this ED arrival
    prior = df[["stay_id", "subject_id", "intime"]].merge(adm, on="subject_id")
    prior = prior[prior.admittime < prior.intime].groupby("stay_id").size()
    df["prior_admissions"] = df.stay_id.map(prior).fillna(0).astype(int)

    feats = ["age", "male", *VITALS, "map", "shock_index", "sirs_partial", "qsofa_partial",
             *COMPLAINT_FLAGS, *[c for c in df if c.startswith("arrive_")], "prior_admissions"]
    return df[["stay_id", "subject_id"] + feats], feats
