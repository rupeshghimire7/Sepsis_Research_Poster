"""Tests for the Sepsis-3 label (kept for later) and the ICD code matching."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sepsis_ed import sepsis3 as L  # noqa: E402
from sepsis_ed.labels import ICD9, ICD10, icd_match  # noqa: E402

T0 = pd.Timestamp("2150-01-01 00:00")


def h(x):
    return T0 + pd.Timedelta(hours=x)


def write(base, module, table, rows, columns):
    p = Path(base) / module
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(p / f"{table}.csv", index=False)


# ---------------------------------------------------------------- helpers
def test_band_high_is_bad():
    # bilirubin cuts: 1.2, 2, 6, 12 -> scores 1..4, thresholds inclusive
    x = pd.Series([1.0, 1.2, 1.9, 2.0, 6.0, 11.9, 12.0])
    assert L._band(x, [1.2, 2, 6, 12]).tolist() == [0, 1, 1, 2, 3, 3, 4]


def test_band_low_is_bad():
    # platelets cuts: <150, <100, <50, <20, thresholds exclusive
    x = pd.Series([200, 150, 149, 100, 99, 50, 49, 19])
    assert L._band(x, [150, 100, 50, 20], high_is_bad=False).tolist() == [0, 0, 1, 1, 2, 2, 3, 4]


def test_gsn_match_handles_multi_codes_padding_and_missing():
    gsn = pd.Series(["009182", "043952 009163", "9182", None, "000000"])
    # 009163 is not in the list; 9182 is zero-padded to 009182
    assert L._gsn_match(gsn).tolist() == [True, False, True, False, False]


def test_is_sepsis_icd():
    code = pd.Series(["A419", "A41.9", "R6521", "0389", "99591", "78552", "A419", "J189", "99590"])
    ver = pd.Series([10, 10, 10, 9, 9, 9, 9, 10, 9])
    # A419 under ICD-9 is not sepsis; 99590 is not in the list
    assert icd_match(code, ver, ICD10, ICD9).tolist() == [True, True, True, True, True, True, False, False, False]


# ---------------------------------------------------------------- suspected infection
def test_suspected_infection_windows():
    abx = pd.DataFrame({"subject_id": [1, 2, 3, 4, 5], "abx_time": [h(100)] * 5})
    cx = pd.DataFrame({"subject_id": [1, 2, 3, 4, 5],
                       "culture_time": [h(100 - 72), h(100 - 73), h(100 + 24), h(100 + 25), h(100)]})
    si = L.suspected_infection(abx, cx).set_index("subject_id").si_time
    assert si[1] == h(28)    # culture 72h before abx -> culture time
    assert 2 not in si       # 73h before -> no pair
    assert si[3] == h(100)   # culture 24h after abx -> abx time
    assert 4 not in si       # 25h after -> no pair
    assert si[5] == h(100)   # simultaneous -> culture time
    assert L.suspected_infection(abx[abx.subject_id == 1], cx[cx.subject_id == 2]).empty  # different patients


# ---------------------------------------------------------------- vasopressors
def test_norepinephrine_equivalent():
    ie = pd.DataFrame({
        "itemid": [221906, 221289, 221906, 221749, 221662, 222315, 222315, 221653, 221906],
        "rate": [0.2, 0.05, 16.0, 1.0, 10.0, 0.04, 2.4, 5.0, 8.0],
        "rateuom": ["mcg/kg/min", "mcg/kg/min", "mcg/min", "mcg/kg/min", "mcg/kg/min",
                    "units/min", "units/hour", "mcg/kg/min", "mcg/min"],
        "patientweight": [70, 70, 80, 70, 70, 70, 70, 70, np.nan],
    })
    ne = L.norepinephrine_equivalent(ie)
    expected = [0.2, 0.05, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1]  # last row: missing weight -> 80 kg
    np.testing.assert_allclose(ne.values, expected)


# ---------------------------------------------------------------- SOFA aggregation
def test_sepsis3_events_window_and_component_max():
    si = pd.DataFrame({"subject_id": [1, 2, 3], "si_time": [h(0)] * 3})
    obs = pd.DataFrame([
        # subject 1: coag max 2 + renal 1 -> 3 (lower coag score ignored)
        (1, h(-10), "coag", 1), (1, h(5), "coag", 2), (1, h(20), "renal", 1),
        # subject 2: only in-window obs score 1; out-of-window obs ignored
        (2, h(-49), "coag", 4), (2, h(25), "liver", 4), (2, h(-48), "renal", 1),
        # subject 3: two components at the edges of the window -> 2
        (3, h(-48), "cns", 1), (3, h(24), "cardio", 1),
    ], columns=["subject_id", "time", "component", "score"])
    ev = L.sepsis3_events(si, obs).set_index("subject_id").sofa
    assert ev.to_dict() == {1: 3, 3: 2}


# ---------------------------------------------------------------- file-backed functions
def test_antibiotics_filters_non_systemic(tmp_path):
    mimic, ed = tmp_path / "mimic", tmp_path / "ed"
    write(mimic, "hosp", "prescriptions", [
        (1, h(1), "CefTRIAXone", "IV", ""),             # name match
        (2, h(1), "Mystery drug", "PO", "009182"),      # GSN match only
        (3, h(1), "Erythromycin Ophth Oint", "OU", ""),  # topical form -> excluded
        (4, h(1), "Vancomycin", "TP", ""),              # topical route -> excluded
        (5, h(1), "Acetaminophen", "PO", ""),           # not an antibiotic
    ], ["subject_id", "starttime", "drug", "route", "gsn"])
    write(ed, "ed", "pyxis", [(6, h(2), "Zosyn", ""), (7, h(2), "Ondansetron", "")],
          ["subject_id", "charttime", "name", "gsn"])
    abx = L.antibiotics(mimic, ed)
    assert sorted(abx.subject_id) == [1, 2, 6]


def test_build_labels_end_to_end(tmp_path):
    mimic, ed = tmp_path / "mimic", tmp_path / "ed"
    # stays: 1 septic early, 2 infection but SOFA 1, 3 septic but after horizon, 4 discharged home
    write(ed, "ed", "edstays", [
        (1, 101, 1001, h(0), h(6)),
        (2, 102, 1002, h(0), h(6)),
        (3, 103, 1003, h(0), h(6)),
        (4, None, 1004, h(0), h(6)),
    ], ["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    write(mimic, "hosp", "microbiologyevents", [
        (1, h(0).normalize(), h(1)), (2, h(0).normalize(), h(1)), (3, h(0).normalize(), h(60)),
    ], ["subject_id", "chartdate", "charttime"])
    write(mimic, "hosp", "prescriptions", [
        (1, h(2), "CefTRIAXone", "IV", ""), (2, h(2), "CefTRIAXone", "IV", ""), (3, h(61), "Zosyn", "IV", ""),
    ], ["subject_id", "starttime", "drug", "route", "gsn"])
    write(mimic, "hosp", "labevents", [
        (1, 51265, h(2), 80),    # platelets -> coag 2
        (2, 51265, h(2), 120),   # platelets -> coag 1
        (3, 51265, h(62), 40),   # platelets -> coag 3
        (4, 51265, h(2), 30),    # no infection -> irrelevant
    ], ["subject_id", "itemid", "charttime", "valuenum"])
    write(mimic, "hosp", "diagnoses_icd", [(103, "A419", 10), (102, "J189", 10)],
          ["hadm_id", "icd_code", "icd_version"])

    lab = L.build_labels(mimic, ed, horizon_h=48).set_index("stay_id")

    assert lab.sepsis3.to_dict() == {1001: 1, 1002: 0, 1003: 0, 1004: 0}
    assert lab.loc[1001, "si_time"] == h(1)
    assert lab.loc[1001, "sofa"] == 2
    assert lab.loc[1001, "onset_hours_from_arrival"] == pytest.approx(1.0)
    assert pd.isna(lab.loc[1002, "si_time"])
    assert lab.sepsis_icd.to_dict() == {1001: 0, 1002: 0, 1003: 1, 1004: 0}
    assert lab.admitted.to_dict() == {1001: 1, 1002: 1, 1003: 1, 1004: 0}

    # a longer horizon picks up stay 3's onset at 60h
    lab72 = L.build_labels(mimic, ed, horizon_h=72).set_index("stay_id")
    assert lab72.loc[1003, "sepsis3"] == 1
    assert lab72.loc[1003, "onset_hours_from_arrival"] == pytest.approx(60.0)


def test_pruning_does_not_change_label(tmp_path, monkeypatch):
    mimic, ed = tmp_path / "mimic", tmp_path / "ed"
    # 1: two stays; SI 1h after the first (septic, with a lab from before arrival), SI 1h before the second
    # (counts for neither). 2: SI exactly 48h after arrival; 3: SI at 49h; 9: not in the cohort
    stays = pd.DataFrame({"stay_id": [1001, 1002, 2001, 3001], "subject_id": [1, 1, 2, 3],
                          "intime": [h(0), h(100), h(0), h(0)]})
    write(mimic, "hosp", "microbiologyevents", [
        (1, h(0).normalize(), h(1)), (1, h(0).normalize(), h(99)), (2, h(0).normalize(), h(48)),
        (3, h(0).normalize(), h(49)), (9, h(0).normalize(), h(1)),
    ], ["subject_id", "chartdate", "charttime"])
    write(mimic, "hosp", "prescriptions", [
        (1, h(2), "CefTRIAXone", "IV", ""), (1, h(99.5), "Zosyn", "IV", ""), (2, h(50), "Zosyn", "IV", ""),
        (3, h(50), "Zosyn", "IV", ""), (9, h(2), "Zosyn", "IV", ""),
    ], ["subject_id", "starttime", "drug", "route", "gsn"])
    write(mimic, "hosp", "labevents", [
        (1, 51265, h(2), 80), (1, 50912, h(-10), 2.5), (1, 51265, h(100), 30), (2, 51265, h(47), 80),
        (1, 50912, h(-24 * 30), 2.2), (1, 50912, h(-24 * 3), 0.9),  # renal 2 a month before; 3 days is too recent
        (3, 51265, h(50), 40), (9, 51265, h(2), 30),
    ], ["subject_id", "itemid", "charttime", "valuenum"])

    pruned = L.sepsis3_label(mimic, ed, stays, horizon_h=48)
    monkeypatch.setattr(L, "near_arrival", lambda si, stays, horizon_h: si)
    unpruned = L.sepsis3_label(mimic, ed, stays, horizon_h=48)

    pd.testing.assert_frame_equal(pruned, unpruned)
    lab = pruned.set_index(stays.stay_id)
    assert lab.sepsis3.to_dict() == {1001: 1, 1002: 0, 2001: 1, 3001: 0}
    assert lab.loc[1001, "sofa"] == 4  # coag 2 + renal 2 (creatinine drawn before ED arrival)
    assert lab.loc[2001, "sofa"] == 2
    assert lab.loc[1001, ["coag", "renal", "resp", "liver", "cardio", "cns"]].tolist() == [2, 2, 0, 0, 0, 0]
    assert lab.loc[1001, "renal_prior"] == 2 and pd.isna(lab.loc[1001, "liver_prior"])
    assert pd.isna(lab.loc[2001, "renal_prior"]) and pd.isna(lab.loc[1002, "renal"])


# ---------------------------------------------------------------- Sepsis-3 check
def test_agreement_counts_shares_and_kappa():
    from sepsis_ed.sepsis3_check import agreement
    icd = [1] * 20 + [0] * 80
    s3 = [1] * 15 + [0] * 5 + [1] * 15 + [0] * 65
    a = agreement(icd, s3, min_cell=10).set_index("measure").value
    assert (a["stays: both"], a["stays: ICD code only"], a["stays: Sepsis-3 only"], a["stays: neither"]) == \
        (15, "<10", 15, 65)
    assert a["ICD cases meeting Sepsis-3 %"] == 75.0
    assert a["Sepsis-3 cases with an ICD sepsis code %"] == 50.0
    # po = 0.80, pe = 0.2*0.3 + 0.8*0.7 = 0.62 -> kappa = 0.18/0.38
    assert a["Cohen's kappa"] == round(0.18 / 0.38, 3)


def test_sofa_breakdown():
    from sepsis_ed.sepsis3_check import sofa_breakdown
    zero = dict(resp=0, coag=0, liver=0, cardio=0, cns=0, renal=0, coag_prior=np.nan, liver_prior=np.nan,
                renal_prior=np.nan)
    rows = ([dict(zero, sepsis3=1, sofa=2, renal=2, renal_prior=2)] * 10      # chronic renal -> drops out
            + [dict(zero, sepsis3=1, sofa=2, renal=1, cardio=1, renal_prior=0)] * 10  # acute -> stays
            + [dict(zero, sepsis3=1, sofa=4, coag=2, cns=2)] * 10              # no earlier result -> stays
            + [dict(zero, sepsis3=0, sofa=np.nan)] * 5)
    lab = pd.DataFrame(rows)
    icd = [1] * 10 + [0] * 20 + [0] * 5
    critical = [0] * 10 + [1] * 20 + [0] * 5
    b = sofa_breakdown(lab, icd, critical, min_cell=10).set_index(["section", "item"])
    assert b.loc[("component scores >= 1: all Sepsis-3 cases", "renal"), "pct"] == round(100 * 20 / 30, 1)
    assert b.loc[("SOFA exactly 2", "all Sepsis-3 cases"), "cases"] == 20
    assert b.loc[("SOFA exactly 2", "lab components only (coag, liver, renal)"), "pct"] == 50.0
    assert b.loc[("renal points already present 7-365 days before", "earlier result scored as high"), "pct"] == 50.0
    still = b.loc["still SOFA >= 2 without chronic lab points"]
    assert still.loc["all Sepsis-3 cases", "cases"] == 20 and still.loc["both", "pct"] == 0.0
    assert still.loc["critical outcome: still positive", "pct"] == 100.0
    assert still.loc["critical outcome: no longer positive", "pct"] == 0.0


def test_onset_is_cumulative():
    from sepsis_ed.sepsis3_check import onset
    o = onset([1] * 10 + [5] * 10 + [30] * 10 + [np.nan], min_cell=10).set_index("within_hours")
    assert o.loc[3, "cases"] == 10 and o.loc[6, "cases"] == 20 and o.loc[48, "cases"] == 30
    assert o.loc[24, "share_of_sepsis3_cases_pct"] == round(100 * 20 / 30, 1)
