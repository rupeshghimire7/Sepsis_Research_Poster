"""Patient-grouped, stratified cross-validation of two baselines."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "logistic": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                      LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)),
    "gbm": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                  class_weight="balanced", random_state=0),
}


def cross_validate(df, feats, target, n_splits=5, seed=0):
    y, X, groups = df[target].values, df[feats], df.subject_id.values
    n_pos = int(y.sum())
    k = min(n_splits, n_pos, int((y == 0).sum()))
    if k < 2:
        return None, f"only {n_pos} positive stay(s) for '{target}' — not enough to cross-validate"

    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = pd.DataFrame({"stay_id": df.stay_id, "y": y})
    metrics = {}
    for name, make in MODELS.items():
        p = np.full(len(y), np.nan)
        for tr, te in cv.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            p[te] = make().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        ok = ~np.isnan(p)
        oof[name] = p
        metrics[name] = {"auroc": roc_auc_score(y[ok], p[ok]), "auprc": average_precision_score(y[ok], p[ok]),
                         "brier": brier_score_loss(y[ok], p[ok]), "prevalence": y[ok].mean(), "folds": k}
    note = f"{k}-fold grouped CV, {n_pos} positives / {len(y)} stays"
    if n_pos < 30:
        note += " — too few positives for stable estimates; treat as a pipeline smoke test"
    return (pd.DataFrame(metrics).T, oof), note
