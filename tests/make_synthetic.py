"""Generate a fake dataset with the MIMIC-IV / MIMIC-IV-ED demo schema, so the
pipeline can be tested without credentialed data. NOT real patients."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
root = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
for d in ("mimic/hosp", "mimic/icu", "ed/ed"):
    (root / d).mkdir(parents=True, exist_ok=True)

N = 400
rows = {k: [] for k in ["edstays", "triage", "vitalsign", "pyxis", "edx", "patients", "admissions",
                        "rx", "micro", "lab", "chart", "input", "dx"]}
for i in range(N):
    sid, stay, hadm = 10000 + i, 30000 + i, 20000 + i
    t0 = pd.Timestamp("2150-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)), hours=int(rng.integers(0, 24)))
    septic = rng.random() < 0.15
    admitted = septic or rng.random() < 0.4
    rows["patients"].append([sid, "M" if rng.random() < .5 else "F", int(rng.integers(18, 91)), 2150])
    rows["edstays"].append([sid, hadm if admitted else np.nan, stay, t0, t0 + pd.Timedelta("6h"),
                            rows["patients"][-1][1], rng.choice(["AMBULANCE", "WALK IN", "OTHER"]),
                            "ADMITTED" if admitted else "HOME"])
    hr = rng.normal(112 if septic else 85, 12); sbp = rng.normal(98 if septic else 130, 15)
    rows["triage"].append([sid, stay, rng.normal(101.5 if septic else 98.4, 1), hr, rng.normal(24 if septic else 17, 3),
                           rng.normal(93 if septic else 98, 2), sbp, sbp * .6, str(int(rng.integers(0, 10))),
                           int(rng.integers(1, 3) if septic else rng.integers(2, 5)),
                           rng.choice(["Fever, Cough", "Altered mental status", "Dysuria"] if septic
                                      else ["Abd pain", "Laceration", "Chest pain", "Fever"])])
    rows["vitalsign"].append([sid, stay, t0 + pd.Timedelta("1h"), sbp - (25 if septic else 0), sbp * .55])
    if admitted:
        rows["admissions"].append([sid, hadm, t0 + pd.Timedelta("5h")])
        rows["lab"].append([sid, hadm, 51265, t0 + pd.Timedelta("2h"), 80 if septic else 250])
        rows["lab"].append([sid, hadm, 50912, t0 + pd.Timedelta("2h"), 2.4 if septic else 0.9])
        rows["lab"].append([sid, hadm, 50821, t0 + pd.Timedelta("3h"), 70 if septic else 95])
        rows["lab"].append([sid, hadm, 50816, t0 + pd.Timedelta("3h"), 40])
    if septic or rng.random() < .1:  # infection workup
        rows["micro"].append([sid, hadm, t0.normalize(), t0 + pd.Timedelta("1h"), "BLOOD CULTURE"])
        rows["pyxis"].append([sid, stay, t0 + pd.Timedelta("2h"), "CefTRIAXone", "009182"])
        rows["rx"].append([sid, hadm, t0 + pd.Timedelta("8h"), "Vancomycin", "IV", "043952 009163"])
    if septic:
        rows["dx"].append([sid, hadm, 1, "A419", 10])
        if rng.random() < .5:
            rows["input"].append([sid, hadm, 40000 + i, t0 + pd.Timedelta("7h"), t0 + pd.Timedelta("20h"), 221906, 0.08, "mcg/kg/min", 75.0])
            for it, v in [(220739, 3), (223900, 4), (223901, 6)]:
                rows["chart"].append([sid, hadm, 40000 + i, t0 + pd.Timedelta("8h"), it, v])

W = lambda df, p: df.to_csv(root / p, index=False, compression="gzip")
W(pd.DataFrame(rows["edstays"], columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "gender",
                                         "arrival_transport", "disposition"]), "ed/ed/edstays.csv.gz")
W(pd.DataFrame(rows["triage"], columns=["subject_id", "stay_id", "temperature", "heartrate", "resprate", "o2sat",
                                        "sbp", "dbp", "pain", "acuity", "chiefcomplaint"]), "ed/ed/triage.csv.gz")
W(pd.DataFrame(rows["vitalsign"], columns=["subject_id", "stay_id", "charttime", "sbp", "dbp"]), "ed/ed/vitalsign.csv.gz")
W(pd.DataFrame(rows["pyxis"], columns=["subject_id", "stay_id", "charttime", "name", "gsn"]), "ed/ed/pyxis.csv.gz")
W(pd.DataFrame(columns=["subject_id", "stay_id", "seq_num", "icd_code", "icd_version", "icd_title"]), "ed/ed/diagnosis.csv.gz")
W(pd.DataFrame(rows["patients"], columns=["subject_id", "gender", "anchor_age", "anchor_year"]), "mimic/hosp/patients.csv.gz")
W(pd.DataFrame(rows["admissions"], columns=["subject_id", "hadm_id", "admittime"]), "mimic/hosp/admissions.csv.gz")
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
print(f"wrote synthetic data to {root}")
