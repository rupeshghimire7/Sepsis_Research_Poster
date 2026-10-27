"""Tests for metrics, baselines, cut-offs and the full two-step pipeline on synthetic data."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sepsis_ed import config  # noqa: E402
from sepsis_ed.baselines import baseline_scores, news_score  # noqa: E402
from sepsis_ed.evaluate import (Ranker, calibration_slope_intercept, net_benefit, patient_weights,  # noqa: E402
                                suppress, threshold_metrics)
from sepsis_ed.model import reference_cutoff  # noqa: E402

CFG = config.load()


def test_ranker_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.07).astype(int)
    s = np.round(rng.normal(y, 1), 1)  # rounding creates many ties
    w = rng.integers(0, 4, 5000).astype(float)
    r = Ranker(y, s)
    assert r.auroc() == pytest.approx(roc_auc_score(y, s))
    assert r.auroc(w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))
    assert r.auprc(w) == pytest.approx(average_precision_score(y, s, sample_weight=w))
    assert np.isnan(Ranker(np.zeros(5), np.arange(5)).auroc())


def test_threshold_metrics():
    y = np.array([1, 1, 0, 0, 0, 0])
    a = np.array([1, 0, 1, 0, 0, 0])
    m = threshold_metrics(y, a)
    assert (m["sensitivity"], m["specificity"], m["ppv"], m["npv"]) == (0.5, 0.75, 0.5, 0.75)
    assert m["alert_rate"] == pytest.approx(2 / 6) and m["reviews_per_case"] == 2


def test_calibration_of_well_calibrated_risks():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.3, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    c = calibration_slope_intercept(y, p)
    assert c["calibration_slope"] == pytest.approx(1, abs=0.05)
    assert c["calibration_intercept"] == pytest.approx(0, abs=0.03)
    over = calibration_slope_intercept(y, np.clip(p * 2, 0, 0.99))  # risks doubled -> negative intercept
    assert over["calibration_intercept"] < -0.3


def test_net_benefit():
    y = np.array([1, 0, 0, 0])
    nb = net_benefit(y, np.array([0.9, 0.5, 0.1, 0.1]), [0.2]).iloc[0]
    assert nb.net_benefit == pytest.approx(1 / 4 - 1 / 4 * 0.25)  # 1 TP, 1 FP at odds 0.25
    assert nb.treat_all == pytest.approx(0.25 - 0.75 * 0.25)


def test_patient_weights_keep_visits_together():
    groups = np.array([7, 7, 7, 3, 3, 9])
    for w in patient_weights(groups, 50, 0):
        assert w[0] == w[1] == w[2] and w[3] == w[4]
        assert w[[0, 3, 5]].sum() == 3  # three patients drawn


def test_news_bands_and_baselines():
    df = pd.DataFrame({"resprate": [8, 18, 26], "o2sat": [91, 97, 94], "sbp": [90, 120, 225],
                       "heartrate": [40, 80, 131], "temperature": [35.0, 37.0, 39.1], "cc_confusion": [1, 0, 0],
                       "acuity": [1, 3, np.nan]})
    assert news_score(df).tolist() == [3 + 3 + 3 + 3 + 3 + 3, 0, 3 + 1 + 3 + 3 + 2]
    b = baseline_scores(df, CFG)
    assert b["Triage acuity"][1].tolist() == [True, False, False]
    assert b["qSOFA (triage)"][0].tolist() == [2, 0, 1]  # RR 8 is not >= 22


def test_reference_cutoff_reaches_target_sensitivity():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.1).astype(int)
    s = rng.normal(y, 1)
    t = reference_cutoff(y, s, 0.86)
    assert (s[y == 1] >= t).mean() >= 0.86
    assert (s[y == 1] > t).mean() < 0.86  # and is the highest such threshold


def test_suppress():
    assert suppress(0, 10) == 0 and suppress(9, 10) == "<10" and suppress(10, 10) == 10


# ---------------------------------------------------------------- end to end
def test_pipeline_end_to_end(tmp_path):
    """Both steps on small synthetic data with a fast config; then the test lock blocks retraining."""
    py = sys.executable
    subprocess.run([py, str(ROOT / "tests" / "make_synthetic.py"), str(tmp_path / "syn"), "6000"], check=True,
                   capture_output=True)
    cfg = config.load()
    m = cfg["modelling"]
    m["cv_folds"] = 3
    m["logistic"]["n_iter"] = m["random_forest"]["n_iter"] = m["xgboost"]["n_iter"] = 2
    m["random_forest"]["n_estimators"] = [50]
    cfg["evaluation"]["n_bootstrap"] = 30
    cfg["checks"]["min_cases_per_split"] = 10
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    base = [py, str(ROOT / "run.py"), "--config", str(tmp_path / "cfg.yaml"), "--mimic-dir",
            str(tmp_path / "syn" / "mimic"), "--ed-dir", str(tmp_path / "syn" / "ed"), "--out", str(out)]

    dev = subprocess.run(base, capture_output=True, text=True, cwd=ROOT)
    assert dev.returncode == 0, dev.stderr[-2000:]
    assert (out / "models" / "bundle.joblib").exists() and not (out / "table2_performance.csv").exists()

    audit = subprocess.run(base + ["--audit"], capture_output=True, text=True, cwd=ROOT)
    assert audit.returncode == 0, audit.stderr[-2000:]
    for f in ("audit_univariate.csv", "audit_complaints.csv", "audit_without_flagged.csv",
              "audit_rules_validation.csv"):
        assert (out / f).exists(), f
    assert not (out / "test_lock.json").exists() and not (out / "table2_performance.csv").exists()

    test = subprocess.run(base + ["--open-test"], capture_output=True, text=True, cwd=ROOT)
    assert test.returncode == 0, test.stderr[-2000:]
    for f in ("table1_characteristics.csv", "table2_performance.csv", "table2_summary.csv", "table2_comparisons.csv",
              "table3_traffic_light.csv", "table4_fairness.csv", "table_feature_sets.csv", "table_shap_top.csv",
              "table_importance_crosscheck.csv", "fig_roc_pr.png", "fig_calibration.png", "fig_decision_curve.png",
              "fig_shap_summary.png", "fig_shap_dependence.png", "findings.md", "test_lock.json"):
        assert (out / f).exists(), f
    for f in ("findings.md", "fig_rules_matched.png", "fig_rules_matched.pdf", "fig_cohort_flow.png",
              "fig_shap_top.png", "table2_summary.csv"):
        assert (out / "results" / f).exists(), f
    assert not (out / "results" / "fig_shap_summary.png").exists()  # one dot per visit: stays local
    perf = pd.read_csv(out / "table2_performance.csv")
    auroc = perf[(perf.method == "xgboost") & (perf.metric == "auroc") & (perf.outcome == "sepsis_icd")].iloc[0]
    assert 0.5 < auroc.estimate < 1 and auroc.lo <= auroc.estimate <= auroc.hi

    again = subprocess.run(base, capture_output=True, text=True, cwd=ROOT)
    assert again.returncode != 0 and "test set was opened" in again.stderr
