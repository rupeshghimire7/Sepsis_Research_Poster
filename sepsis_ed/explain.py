"""Explanations: SHAP values for the XGBoost model, checked against permutation importance and logistic
regression coefficients, and the top three reasons shown for each patient in the prototype."""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

LABELS = {
    "age": "Age", "male": "Male sex", "temperature": "Temperature", "heartrate": "Heart rate",
    "resprate": "Respiratory rate", "o2sat": "Oxygen saturation", "sbp": "Systolic BP", "dbp": "Diastolic BP",
    "map": "Mean arterial pressure", "shock_index": "Shock index", "pulse_pressure": "Pulse pressure",
    "fever": "Fever ≥ 38 °C", "low_temp": "Temperature < 36 °C", "hr_high": "Heart rate > 90",
    "rr_high": "Respiratory rate > 20", "rr_qsofa": "Respiratory rate ≥ 22", "sbp_low": "Systolic BP ≤ 100",
    "spo2_low": "SpO2 < 92%", "n_abnormal_vitals": "Number of abnormal vitals",
    "n_missing_vitals": "Number of missing vitals", "pain": "Pain score", "acuity": "Triage acuity (ESI)",
    "arrival_hour": "Hour of arrival", "weekend": "Weekend arrival", "night": "Night arrival",
    "ed_visits_365d": "ED visits, past year", "admissions_365d": "Admissions, past year",
    "prior_sepsis_admission": "Previous sepsis admission",
}


def label(f):
    if f in LABELS:
        return LABELS[f]
    for prefix, text in (("miss_", "Missing: "), ("cc_", "Complaint: "), ("arrive_", "Arrival: "),
                         ("hx_", "History: ")):
        if f.startswith(prefix):
            rest = f[len(prefix):]
            return text + (LABELS.get(rest, rest) if prefix == "miss_" else rest).replace("_", " ")
    return f.replace("_", " ")


def contributions(fm, X):
    """Exact TreeSHAP contributions (log-odds) from XGBoost: (values n x p, base value)."""
    booster = fm.estimator.get_booster()
    it = (0, fm.estimator.best_iteration + 1) if getattr(fm.estimator, "best_iteration", None) is not None else (0, 0)
    c = booster.predict(xgb.DMatrix(X[fm.features], missing=np.nan), pred_contribs=True, iteration_range=it)
    return c[:, :-1], c[:, -1]


def shap_top(fm, X, sv, top_n):
    """Mean |SHAP| ranking with the direction of effect (Spearman sign of value vs SHAP)."""
    rows = []
    for j, f in enumerate(fm.features):
        x = X[f].to_numpy(float)
        ok = ~np.isnan(x)
        varies = ok.sum() > 2 and np.std(x[ok]) > 0 and np.std(sv[ok, j]) > 0
        rho = spearmanr(x[ok], sv[ok, j]).statistic if varies else np.nan
        rows.append({"feature": f, "label": label(f), "mean_abs_shap": np.abs(sv[:, j]).mean(),
                     "direction": "higher value → higher risk" if rho > 0 else
                     "higher value → lower risk" if rho < 0 else "mixed"})
    t = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    return t, t.head(top_n)


def crosscheck(xgb_fm, lr_fm, X, y, sv, cfg):
    """Rank features three ways; agreement across methods is a stronger claim than SHAP alone."""
    n = min(cfg["explain"]["permutation_rows"], len(X))
    sub = X.sample(n, random_state=cfg["seed"]) if n < len(X) else X
    pi = permutation_importance(xgb_fm.estimator, sub[xgb_fm.features], y[X.index.get_indexer(sub.index)],
                                scoring="average_precision", n_repeats=5, random_state=cfg["seed"],
                                n_jobs=cfg["modelling"]["n_jobs"])
    coef = lr_fm.estimator[-1].coef_[0]  # on standardised inputs, so magnitudes are comparable
    lr = pd.Series(np.abs(coef), index=lr_fm.features)
    t = pd.DataFrame({"feature": xgb_fm.features,
                      "shap_mean_abs": np.abs(sv).mean(axis=0),
                      "permutation_auprc_drop": pi.importances_mean,
                      "lr_abs_std_coef": lr.reindex(xgb_fm.features).to_numpy()})
    for c in ("shap_mean_abs", "permutation_auprc_drop", "lr_abs_std_coef"):
        t[c.split("_")[0] + "_rank"] = t[c].rank(ascending=False, method="min")
    t["label"] = t.feature.map(label)
    t = t.sort_values("shap_rank").reset_index(drop=True)
    top = t.head(cfg["explain"]["top_n"])
    agree = {"spearman_shap_vs_permutation_top": spearmanr(top.shap_rank, top.permutation_rank).statistic,
             "spearman_shap_vs_lr_top": spearmanr(top.shap_rank, top.lr_rank).statistic}
    return t, agree


def reasons(fm, row, n=3):
    """Top-n contributions for one stay (row: one-row DataFrame with fm.features)."""
    sv, _ = contributions(fm, row)
    order = np.argsort(-np.abs(sv[0]))[:n]
    out = []
    for j in order:
        f = fm.features[j]
        v = row[f].iloc[0]
        out.append({"feature": f, "label": label(f), "value": None if pd.isna(v) else float(v),
                    "contribution": float(sv[0, j]), "effect": "raises risk" if sv[0, j] > 0 else "lowers risk"})
    return out
