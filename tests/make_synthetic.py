"""Generate a fake dataset with the MIMIC-IV / MIMIC-IV-ED demo schema, so the
pipeline can be tested without credentialed data. NOT real patients."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
root = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000  # ED stays
for d in ("mimic/hosp", "mimic/icu", "ed/ed"):
    (root / d).mkdir(parents=True, exist_ok=True)

# patients: about 1.3 stays each; a few are under 18 to exercise the age filter
n_pat = int(N * 0.75)
P = pd.DataFrame({"subject_id": 10000 + np.arange(n_pat), "gender": rng.choice(["M", "F"], n_pat),
                  "anchor_age": rng.integers(16, 91, n_pat), "anchor_year": 2150,
                  "anchor_year_group": rng.choice(["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"], n_pat),
                  "race": rng.choice(["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO", "ASIAN", "OTHER"], n_pat,
                                     p=[.6, .2, .1, .05, .05]),
                  "chronic": rng.random(n_pat) < .3})
INFECTION_CC = ["Fever, Cough", "Altered mental status", "Dysuria", "Fever", "SOB", "Weakness", "Cellulitis"]
OTHER_CC = ["Abd pain", "Laceration", "Chest pain", "Headache", "s/p Fall", "Back pain", "N/V", "Dizziness"]

rows = {k: [] for k in ["edstays", "triage", "vitalsign", "pyxis", "edx", "admissions",
                        "rx", "micro", "lab", "chart", "input", "dx", "icu"]}
for i in range(N):
    p = P.iloc[int(rng.integers(0, n_pat))]
    sid, stay, hadm = int(p.subject_id), 30000 + i, 200000 + i
    t0 = pd.Timestamp("2150-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)), hours=int(rng.integers(0, 24)))
    logit = -3.6 + 0.035 * (p.anchor_age - 50) + 0.8 * p.chronic + rng.normal(0, .5)
    septic = rng.random() < 1 / (1 + np.exp(-logit))
    admitted = septic or rng.random() < 0.35
    rows["edstays"].append([sid, hadm if admitted else np.nan, stay, t0, t0 + pd.Timedelta("6h"), p.gender, p.race,
                            rng.choice(["AMBULANCE", "WALK IN", "OTHER"], p=[.6, .35, .05] if septic else [.3, .65, .05]),
                            "ADMITTED" if admitted else "HOME"])
    hr = rng.normal(100 if septic else 86, 16); sbp = rng.normal(118 if septic else 132, 20)
    infection_cc = rng.random() < (.55 if septic else .12)
    rows["triage"].append([sid, stay, rng.normal(99.6 if septic else 98.3, 1.2), hr, rng.normal(20 if septic else 17, 3.5),
                           rng.normal(95.5 if septic else 97.8, 2.5), sbp, sbp * .6, str(int(rng.integers(0, 11))),
                           int(rng.choice([1, 2, 3], p=[.15, .55, .3]) if septic
                               else rng.choice([1, 2, 3, 4, 5], p=[.03, .3, .5, .15, .02])),
                           rng.choice(INFECTION_CC if infection_cc else OTHER_CC)])
    if rng.random() < .03:  # no vitals at all -> excluded from the cohort
        rows["triage"][-1][2:8] = [np.nan] * 6
    elif rng.random() < .05:  # temperature typed in Celsius
        rows["triage"][-1][2] = (rows["triage"][-1][2] - 32) * 5 / 9
    rows["vitalsign"].append([sid, stay, t0 + pd.Timedelta("1h"), sbp - (25 if septic else 0), sbp * .55])
    if admitted:
        died = rng.random() < (.15 if septic else .01)
        rows["admissions"].append([sid, hadm, t0 + pd.Timedelta("5h"), t0 + pd.Timedelta(days=4), int(died)])
        rows["lab"].append([sid, hadm, 51265, t0 + pd.Timedelta("2h"), 80 if septic else 250])
        rows["lab"].append([sid, hadm, 50912, t0 + pd.Timedelta("2h"), 2.4 if septic else 0.9])
        rows["lab"].append([sid, hadm, 50821, t0 + pd.Timedelta("3h"), 70 if septic else 95])
        rows["lab"].append([sid, hadm, 50816, t0 + pd.Timedelta("3h"), 40])
        # sepsis is coded for most, not all, septic admissions, and occasionally for others
        if rng.random() < (.8 if septic else .005):
            rows["dx"].append([sid, hadm, 1, "A419", 10])
    if septic or rng.random() < .1:  # infection workup
        rows["micro"].append([sid, hadm, t0.normalize(), t0 + pd.Timedelta("1h"), "BLOOD CULTURE"])
        rows["pyxis"].append([sid, stay, t0 + pd.Timedelta("2h"), "CefTRIAXone", "009182"])
        rows["rx"].append([sid, hadm, t0 + pd.Timedelta("8h"), "Vancomycin", "IV", "043952 009163"])
    if p.chronic and rng.random() < .5:  # an earlier admission with a long-term condition coded at discharge
        rows["admissions"].append([sid, 600000 + i, t0 - pd.Timedelta(days=100), t0 - pd.Timedelta(days=95), 0])
        rows["dx"].append([sid, 600000 + i, 1, rng.choice(["E119", "I509", "N184", "J449"]), 10])
    if septic or rng.random() < .03:
        if rng.random() < .5:
            rows["icu"].append([sid, hadm, 40000 + i, t0 + pd.Timedelta("7h"), t0 + pd.Timedelta("50h")])
    if septic and rng.random() < .5:
        rows["input"].append([sid, hadm, 40000 + i, t0 + pd.Timedelta("7h"), t0 + pd.Timedelta("20h"), 221906, 0.08,
                              "mcg/kg/min", 75.0])
        for it, v in [(220739, 3), (223900, 4), (223901, 6)]:
            rows["chart"].append([sid, hadm, 40000 + i, t0 + pd.Timedelta("8h"), it, v])

W = lambda df, p: df.to_csv(root / p, index=False, compression="gzip")
W(pd.DataFrame(rows["edstays"], columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "gender", "race",
                                         "arrival_transport", "disposition"]), "ed/ed/edstays.csv.gz")
W(pd.DataFrame(rows["triage"], columns=["subject_id", "stay_id", "temperature", "heartrate", "resprate", "o2sat",
                                        "sbp", "dbp", "pain", "acuity", "chiefcomplaint"]), "ed/ed/triage.csv.gz")
W(pd.DataFrame(rows["vitalsign"], columns=["subject_id", "stay_id", "charttime", "sbp", "dbp"]), "ed/ed/vitalsign.csv.gz")
W(pd.DataFrame(rows["pyxis"], columns=["subject_id", "stay_id", "charttime", "name", "gsn"]), "ed/ed/pyxis.csv.gz")
W(pd.DataFrame(columns=["subject_id", "stay_id", "seq_num", "icd_code", "icd_version", "icd_title"]), "ed/ed/diagnosis.csv.gz")
W(P[["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"]], "mimic/hosp/patients.csv.gz")
W(pd.DataFrame(rows["admissions"], columns=["subject_id", "hadm_id", "admittime", "dischtime",
                                              "hospital_expire_flag"]), "mimic/hosp/admissions.csv.gz")
W(pd.DataFrame(rows["rx"], columns=["subject_id", "hadm_id", "starttime", "drug", "route", "gsn"]), "mimic/hosp/prescriptions.csv.gz")
W(pd.DataFrame(rows["micro"], columns=["subject_id", "hadm_id", "chartdate", "charttime", "spec_type_desc"]),
  "mimic/hosp/microbiologyevents.csv.gz")
W(pd.DataFrame(rows["lab"], columns=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]), "mimic/hosp/labevents.csv.gz")
W(pd.DataFrame(rows["dx"], columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]),
  "mimic/hosp/diagnoses_icd.csv.gz")
W(pd.DataFrame(rows["chart"], columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"]),
  "mimic/icu/chartevents.csv.gz")
W(pd.DataFrame(rows["input"], columns=["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "rate", "rateuom", "patientweight"]),
  "mimic/icu/inputevents.csv.gz")
W(pd.DataFrame(rows["icu"], columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime"]), "mimic/icu/icustays.csv.gz")
print(f"wrote synthetic data to {root}")
