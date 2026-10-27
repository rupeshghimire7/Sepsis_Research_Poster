"""Metric primitives and the patient-level bootstrap.

A bootstrap resample of patients is expressed as integer weights per stay (how many times its patient was
drawn), so every metric takes `w` and nothing is copied. Scores are sorted once per Ranker; each resample
is then O(n) via bincount, which keeps 1,000 resamples of ~64k test stays fast.
"""
import numpy as np
import pandas as pd


class Ranker:
    """Weighted AUROC / AUPRC for fixed labels and scores. Ties are handled like scikit-learn."""

    def __init__(self, y, s):
        order = np.argsort(s, kind="mergesort")
        self.order = order
        self.y = np.asarray(y, float)[order]
        _, self.grp = np.unique(np.asarray(s)[order], return_inverse=True)  # ascending tie groups
        self.n_grp = self.grp.max() + 1 if len(self.grp) else 0

    def _sums(self, w):
        w = np.ones(len(self.order)) if w is None else np.asarray(w, float)[self.order]
        wp = np.bincount(self.grp, w * self.y, self.n_grp)
        wn = np.bincount(self.grp, w * (1 - self.y), self.n_grp)
        return wp, wn

    def auroc(self, w=None):
        wp, wn = self._sums(w)
        P, N = wp.sum(), wn.sum()
        if P == 0 or N == 0:
            return np.nan
        below = np.cumsum(wn) - wn
        return float(np.sum(wp * (below + 0.5 * wn)) / (P * N))

    def auprc(self, w=None):
        wp, wn = self._sums(w)
        wp, wn = wp[::-1], wn[::-1]  # descending thresholds
        P = wp.sum()
        if P == 0:
            return np.nan
        tp, fp = np.cumsum(wp), np.cumsum(wn)
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        return float(np.sum(prec * wp) / P)


def threshold_metrics(y, alert, w=None):
    w = np.ones(len(y)) if w is None else w
    y, a = np.asarray(y, bool), np.asarray(alert, bool)
    tp, fp = w[a & y].sum(), w[a & ~y].sum()
    fn, tn = w[~a & y].sum(), w[~a & ~y].sum()
    div = lambda n, d: n / d if d > 0 else np.nan  # noqa: E731
    ppv = div(tp, tp + fp)
    return {"alert_rate": div(tp + fp, tp + fp + fn + tn), "sensitivity": div(tp, tp + fn),
            "specificity": div(tn, tn + fp), "fpr": div(fp, tn + fp), "ppv": ppv, "npv": div(tn, tn + fn),
            "reviews_per_case": div(1, ppv) if ppv else np.nan}


def brier(y, p, w=None):
    return float(np.average((p - y) ** 2, weights=w))


def calibration_slope_intercept(y, p, w=None, iters=25):
    """Slope: logistic regression of y on logit(p). Intercept: calibration-in-the-large, logit(p) as offset."""
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        beta = beta + np.linalg.solve(H + 1e-9 * np.eye(2), X.T @ (w * (y - mu)))
    a = 0.0
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(a + lp)))
        a += np.sum(w * (y - mu)) / max(np.sum(w * mu * (1 - mu)), 1e-12)
    return {"calibration_slope": float(beta[1]), "calibration_intercept": float(a)}


def net_benefit(y, p, thresholds):
    y = np.asarray(y, bool)
    n, prev = len(y), y.mean()
    rows = []
    for t in thresholds:
        a = p >= t
        nb = (a & y).sum() / n - (a & ~y).sum() / n * t / (1 - t)
        rows.append({"threshold": t, "net_benefit": nb, "treat_all": prev - (1 - prev) * t / (1 - t)})
    return pd.DataFrame(rows)


def calibration_bins(y, p, bins):
    q = pd.qcut(p, bins, duplicates="drop")
    return (pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin", observed=True)
            .agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True))


# ---------------------------------------------------------------- bootstrap
def patient_weights(groups, n_boot, seed):
    """Yields one integer weight per stay for each patient-level resample."""
    codes, uniq = pd.factorize(np.asarray(groups))
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        yield counts[codes].astype(float)


def bootstrap(fn, groups, n_boot, seed, alpha=0.05):
    """fn(w) -> {key: value}. Returns DataFrame [key, estimate, lo, hi] (estimate at w=None)."""
    point = fn(None)
    draws = pd.DataFrame([fn(w) for w in patient_weights(groups, n_boot, seed)])
    lo = draws.quantile(alpha / 2)
    hi = draws.quantile(1 - alpha / 2)
    return pd.DataFrame({"key": list(point), "estimate": list(point.values()),
                         "lo": [lo.get(k, np.nan) for k in point], "hi": [hi.get(k, np.nan) for k in point]})


def suppress(n, min_cell):
    """Counts below min_cell (but above 0) are reported as '<min_cell'."""
    return f"<{min_cell}" if 0 < n < min_cell else int(n)
