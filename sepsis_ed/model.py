"""Step 1: tune logistic regression, random forest and XGBoost on the training split, refit, then calibrate
the risks and set the traffic-light cut-offs on the validation split. The test split is never used here.

- Tuning: random search, StratifiedGroupKFold by patient, scored on AUPRC.
- No class weights, oversampling or SMOTE: imbalance is handled by cut-offs and PR reporting.
- LR and RF get training-set medians (missing flags are already features); XGBoost gets the gaps.
- Ranking (AUROC, tiers) uses the model score; displayed risk is the isotonic-calibrated score.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODELS = ("logistic", "random_forest", "xgboost")


@dataclass
class FittedModel:
    name: str
    outcome: str
    feature_set: str
    features: list
    estimator: object
    params: dict
    calibrator: object = None
    cutoffs: dict = field(default_factory=dict)  # on the model score: high, moderate, reference

    def score(self, df):
        return self.estimator.predict_proba(df[self.features])[:, 1]

    def risk(self, df):
        return self.calibrator.predict(self.score(df))

    def tier(self, df):
        s = self.score(df)
        return np.where(s >= self.cutoffs["high"], "high", np.where(s >= self.cutoffs["moderate"], "moderate", "low"))


# ---------------------------------------------------------------- estimators
def search_space(name, cfg):
    m = cfg["modelling"][name]
    if name == "logistic":
        return {"C": loguniform(*m["C"])}
    if name == "random_forest":
        return {k: m[k] for k in ("n_estimators", "min_samples_leaf", "max_features", "max_samples")}
    span = lambda lo, hi: uniform(lo, hi - lo)  # noqa: E731
    return {"max_depth": randint(m["max_depth"][0], m["max_depth"][1] + 1),
            "learning_rate": span(*m["learning_rate"]), "subsample": span(*m["subsample"]),
            "colsample_bytree": span(*m["colsample_bytree"]), "min_child_weight": m["min_child_weight"]}


def make_estimator(name, params, cfg, n_rounds=None):
    seed, n_jobs = cfg["seed"], cfg["modelling"]["n_jobs"]
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=params["C"], max_iter=5000))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(**params, n_jobs=n_jobs, random_state=seed))
    x = cfg["modelling"]["xgboost"]
    return XGBClassifier(**params, n_estimators=n_rounds or x["max_rounds"],
                         early_stopping_rounds=None if n_rounds else x["early_stopping_rounds"],
                         eval_metric="aucpr", tree_method="hist", n_jobs=n_jobs, random_state=seed)


def fit_estimator(name, est, X, y, X_eval=None, y_eval=None):
    if name == "xgboost" and est.get_params()["early_stopping_rounds"]:
        return est.fit(X, y, eval_set=[(X_eval, y_eval)], verbose=False)
    return est.fit(X, y)


def tune(name, X, y, groups, cfg, log=print):
    """Random search with patient-grouped CV on AUPRC. Returns (best params, results table)."""
    folds = list(StratifiedGroupKFold(cfg["modelling"]["cv_folds"], shuffle=True, random_state=cfg["seed"])
                 .split(X, y, groups))
    cands = list(ParameterSampler(search_space(name, cfg), cfg["modelling"][name]["n_iter"],
                                  random_state=cfg["seed"]))
    rows = []
    for i, params in enumerate(cands):
        scores = []
        for tr, te in folds:
            est = fit_estimator(name, make_estimator(name, params, cfg), X.iloc[tr], y[tr], X.iloc[te], y[te])
            scores.append(average_precision_score(y[te], est.predict_proba(X.iloc[te])[:, 1]))
        rows.append({**params, "cv_auprc": np.mean(scores), "cv_auprc_sd": np.std(scores)})
        log(f"    {name} {i + 1}/{len(cands)}  cv AUPRC {np.mean(scores):.3f}")
    res = pd.DataFrame(rows)
    best = cands[int(res.cv_auprc.idxmax())]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}, \
        res.sort_values("cv_auprc", ascending=False)


# ---------------------------------------------------------------- calibration and cut-offs
def reference_cutoff(y, s, sensitivity):
    """Highest score threshold whose sensitivity is at least `sensitivity`."""
    pos = np.sort(s[y == 1])[::-1]
    return pos[max(int(np.ceil(sensitivity * len(pos))) - 1, 0)]


def calibrate(fm, valid, cfg):
    """Isotonic calibration and traffic-light cut-offs, both from the validation split."""
    y, s = valid[fm.outcome].to_numpy(), fm.score(valid)
    fm.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)
    tl = cfg["traffic_light"]
    fm.cutoffs = {"high": float(np.quantile(s, 1 - tl["high_top_fraction"])),
                  "moderate": float(np.quantile(s, 1 - tl["high_top_fraction"] - tl["moderate_next_fraction"])),
                  "reference": float(reference_cutoff(y, s, tl["reference_sensitivity"]))}
    return fm


def fit_final(name, params, outcome, set_name, feats, train, valid, cfg):
    """Refit on the whole training split (XGBoost early-stops on validation), then calibrate."""
    est = fit_estimator(name, make_estimator(name, params, cfg), train[feats], train[outcome].to_numpy(),
                        valid[feats], valid[outcome].to_numpy())
    if name == "xgboost":
        params = {**params, "n_rounds": int(est.best_iteration) + 1}
    return calibrate(FittedModel(name, outcome, set_name, list(feats), est, params), valid, cfg)


# ---------------------------------------------------------------- phase 6-7 driver
def tuning_sample(train, max_stays, seed):
    """Whole patients, drawn at random, until about max_stays stays (keeps RF tuning to minutes, not hours).
    The chosen setting is still refit on the full training split."""
    if not max_stays or len(train) <= max_stays:
        return train
    pats = train.subject_id.drop_duplicates().sample(frac=1, random_state=seed)
    sizes = train.groupby("subject_id").size().reindex(pats)
    keep = sizes.index[sizes.cumsum().to_numpy() <= max_stays]
    return train[train.subject_id.isin(keep)]


def develop(table, sets, cfg, log=print):
    """Tune and fit every model the evaluation needs. Returns the bundle saved to out/models/."""
    m, primary = cfg["modelling"], cfg["outcomes"]["primary"]
    train, valid = table[table.split == "train"], table[table.split == "valid"]
    main = m["feature_set"]
    feats = sets[main]
    bundle = {"config": cfg, "feature_sets": sets, "primary": {}, "feature_set_models": {}, "secondary": {},
              "tuning": {}}

    for name in MODELS:
        tr = tuning_sample(train, m[name].get("tune_max_stays"), cfg["seed"])
        log(f"  tuning {name} ({m[name]['n_iter']} settings x {m['cv_folds']} folds on {len(tr):,} stays)")
        params, res = tune(name, tr[feats], tr[primary].to_numpy(), tr.subject_id.to_numpy(), cfg, log)
        bundle["tuning"][name] = res
        bundle["primary"][name] = fit_final(name, params, primary, main, feats, train, valid, cfg)

    # other feature sets and the secondary outcome reuse the tuned XGBoost settings
    xgb_params = {k: v for k, v in bundle["primary"]["xgboost"].params.items() if k != "n_rounds"}
    for s in m["compare_feature_sets"]:
        bundle["feature_set_models"][s] = (bundle["primary"]["xgboost"] if s == main else
                                           fit_final("xgboost", xgb_params, primary, s, sets[s], train, valid, cfg))
    if m["secondary_outcome"]:
        sec = cfg["outcomes"]["secondary"]
        bundle["secondary"]["xgboost"] = fit_final("xgboost", xgb_params, sec, main, feats, train, valid, cfg)
    return bundle


def validation_report(bundle, valid):
    """Development-only numbers on the validation split (calibration was fitted here, so Brier is optimistic)."""
    rows = []
    fitted = [(f"{k} (set {v.feature_set})", v) for k, v in bundle["primary"].items()]
    fitted += [(f"xgboost (set {k})", v) for k, v in bundle["feature_set_models"].items()
               if v is not bundle["primary"]["xgboost"]]
    fitted += [(f"xgboost (set {v.feature_set})", v) for v in bundle["secondary"].values()]
    for label, fm in fitted:
        y, s = valid[fm.outcome].to_numpy(), fm.score(valid)
        rows.append({"model": label, "outcome": fm.outcome, "auroc": roc_auc_score(y, s),
                     "auprc": average_precision_score(y, s), "prevalence": y.mean(), **fm.params})
    return pd.DataFrame(rows)
