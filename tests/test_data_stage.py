"""Unit tests for cohort, outcomes, features and split. Run: pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sepsis_ed import config  # noqa: E402
from sepsis_ed.cohort import build_cohort, clean_temperature, clean_triage  # noqa: E402
from sepsis_ed.features import context_features, count_in_window, ever_before, vital_features  # noqa: E402
from sepsis_ed.labels import critical_label, icd_match  # noqa: E402
from sepsis_ed.split import CheckFailed, check_no_forbidden, check_patients_disjoint, split_patients  # noqa: E402

CFG = config.load()
T0 = pd.Timestamp("2150-01-01 00:00")


def h(x):
    return T0 + pd.Timedelta(hours=x)


def write(base, module, table, rows, columns):
    p = Path(base) / module
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(p / f"{table}.csv", index=False)


# ---------------------------------------------------------------- cleaning
def test_clean_temperature_units():
    t = pd.Series([98.6, 37.0, 104.0, 9.8, 120.0, None, "abc"])
    c = clean_temperature(t, CFG)
    np.testing.assert_allclose(c[:3], [37.0, 37.0, 40.0])
    assert c[3:].isna().all()


def test_clean_triage_ranges_pain_acuity():
    tr = pd.DataFrame({"temperature": [86.0, 98.6], "heartrate": [300, 80], "resprate": [3, 18],
                       "o2sat": [101, 97], "sbp": [120, 45], "dbp": [80, 10],
                       "pain": ["unable", "7"], "acuity": [2.5, 3]})
    c = clean_triage(tr, CFG)
    assert np.isnan(c.temperature[0])          # 86 F = 30 C, converted then outside 32-43
    assert c.temperature[1] == pytest.approx(37.0)
    assert c[["heartrate", "resprate", "o2sat"]].iloc[0].isna().all()
    assert c[["sbp", "dbp"]].iloc[1].isna().all()
    assert np.isnan(c.pain[0]) and c.pain[1] == 7
    assert np.isnan(c.acuity[0]) and c.acuity[1] == 3


def test_build_cohort_flow(tmp_path):
    mimic, ed = tmp_path / "mimic", tmp_path / "ed"
    write(ed, "ed", "edstays", [
        (1, 101, 1001, h(0), h(5), "F", "WHITE", "AMBULANCE", "ADMITTED"),  # kept
        (2, None, 1002, h(0), h(5), "M", "ASIAN", "WALK IN", "HOME"),      # age 17 -> dropped
        (3, None, 1003, h(0), h(5), "M", "ASIAN", "WALK IN", "HOME"),      # no vitals -> dropped
        (4, None, 1004, h(0), h(5), "M", "ASIAN", "WALK IN", "HOME"),      # no triage row -> dropped
    ], ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "gender", "race", "arrival_transport", "disposition"])
    write(ed, "ed", "triage", [
        (1001, 98.6, 80, 16, 98, 120, 80, "3", 3, "Fever"),
        (1002, 98.6, 80, 16, 98, 120, 80, "3", 3, "Fall"),
        (1003, None, None, None, None, None, None, "3", 3, "Rash"),
    ], ["stay_id", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"])
    write(mimic, "hosp", "patients", [(1, 40, 2150, "2011 - 2013"), (2, 17, 2150, "2011 - 2013"),
                                      (3, 40, 2150, "2011 - 2013"), (4, 40, 2150, "2011 - 2013")],
          ["subject_id", "anchor_age", "anchor_year", "anchor_year_group"])
    df, flow = build_cohort(mimic, ed, CFG)
    assert df.stay_id.tolist() == [1001]
    assert [f["stays"] for f in flow] == [4, 3, 3, 2, 1]
    assert [f["removed"] for f in flow] == [0, 1, 0, 1, 1]


# ---------------------------------------------------------------- codes and outcomes
def test_icd_match_prefixes_and_ranges():
    code = pd.Series(["C509", "D000", "1749", "2090", "V4511", "E119", "E149", "J45.909"])
    ver = pd.Series([10, 10, 9, 9, 9, 10, 10, 10])
    assert icd_match(code, ver, ["C00-C97"], ["140-208"]).tolist() == [True, False, True, False] + [False] * 4
    assert icd_match(code, ver, ["E08-E13", "J40-J47"], ["V4511"]).tolist() == \
        [False, False, False, False, True, True, False, True]


def test_critical_label(tmp_path):
    mimic = tmp_path / "mimic"
    stays = pd.DataFrame({
        "stay_id": [1, 2, 3, 4, 5, 6], "subject_id": [1, 2, 3, 4, 5, 6],
        "hadm_id": [101, 102, 103, 104, None, None],
        "intime": [h(0)] * 6,
        "disposition": ["ADMITTED"] * 4 + ["EXPIRED", "HOME"],
    })
    write(mimic, "hosp", "admissions", [(101, 0), (102, 0), (103, 1), (104, 0)], ["hadm_id", "hospital_expire_flag"])
    write(mimic, "icu", "icustays", [(1, h(11)), (2, h(13)), (4, h(-5)), (4, h(30))], ["subject_id", "intime"])
    # 1 ICU at 11h -> 1; 2 ICU at 13h -> 0; 3 died in hospital -> 1; 4 ICU before / after window -> 0;
    # 5 died in the ED -> 1; 6 nothing -> 0
    assert critical_label(mimic, stays, 12).tolist() == [1, 0, 1, 0, 1, 0]


# ---------------------------------------------------------------- features
def test_count_in_window_respects_time_and_subject():
    q_subject, q_time = pd.Series([1, 2]), pd.Series([h(0), h(0)])
    e_subject = pd.Series([1, 1, 1, 1, 2, 1])
    e_time = pd.Series([h(-400 * 24), h(-100 * 24), h(-1), h(1), h(-2), pd.NaT])
    # subject 1: only -100d and -1h fall in [t-365d, t); the +1h event is in the future
    assert count_in_window(q_subject, q_time, e_subject, e_time, 365).tolist() == [2, 1]
    # no event from subject 1 leaks into subject 2 even at the window edge
    assert count_in_window(pd.Series([2]), pd.Series([h(0)]), pd.Series([1]), pd.Series([h(-1)]), 365).tolist() == [0]


def test_ever_before_is_strict():
    q = pd.Series([1, 2, 3]), pd.Series([h(0)] * 3)
    e = pd.Series([1, 2]), pd.Series([h(-1), h(0)])
    assert ever_before(*q, *e).tolist() == [1, 0, 0]
    assert ever_before(*q, pd.Series([], dtype=int), pd.Series([], dtype="datetime64[ns]")).tolist() == [0, 0, 0]


def test_vital_and_context_features():
    df = pd.DataFrame({"age": [70, 30], "gender": ["M", "F"], "temperature": [38.5, np.nan],
                       "heartrate": [110, 70], "resprate": [24, 14], "o2sat": [90, 99], "sbp": [95, 120],
                       "dbp": [50, 80], "pain": [np.nan, 4], "chiefcomplaint": ["Fever, SOB", "Ankle injury"],
                       "arrival_transport": ["AMBULANCE", None], "intime": [h(23), h(12)]})
    a = vital_features(df, CFG)
    assert a.loc[0, ["fever", "hr_high", "rr_high", "rr_qsofa", "sbp_low", "spo2_low"]].tolist() == [1] * 6
    assert a.loc[0, "n_abnormal_vitals"] == 5
    assert a.loc[1, "n_abnormal_vitals"] == 0 and a.loc[1, "miss_temperature"] == 1
    assert a.loc[0, "shock_index"] == pytest.approx(110 / 95)
    b = context_features(df, CFG)
    assert b.loc[0, ["cc_fever", "cc_sob", "cc_other", "arrive_ambulance", "night"]].tolist() == [1, 1, 0, 1, 1]
    assert b.loc[1, ["cc_injury", "cc_fever", "arrive_unknown", "night", "miss_pain"]].tolist() == [1, 0, 1, 0, 0]


# ---------------------------------------------------------------- split and checks
def test_split_keeps_patients_together():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"subject_id": rng.integers(0, 300, 2000), "y": rng.random(2000) < 0.1})
    df["split"] = split_patients(df, "y", CFG)
    check_patients_disjoint(df)
    share = df.drop_duplicates("subject_id").split.value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.70, abs=0.03)
    df.loc[df.index[0], "split"] = "test" if df.split.iloc[0] != "test" else "train"
    with pytest.raises(CheckFailed):
        check_patients_disjoint(df)


def test_forbidden_columns():
    check_no_forbidden(["age", "heartrate"], CFG, ["sepsis_icd"])
    for bad in (["age", "disposition"], ["sepsis_icd"], ["race"], ["lab_creatinine"]):
        with pytest.raises(CheckFailed):
            check_no_forbidden(bad, CFG, ["sepsis_icd"])
