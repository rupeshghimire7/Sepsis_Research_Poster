"""Rule-based comparators computed from the same triage data ("triage versions", not full bedside scores).

SIRS   2+ of: temperature > 38 or < 36 °C, HR > 90, RR > 20                       (no white cell count)
qSOFA  2+ of: RR >= 22, SBP <= 100, confusion in the chief complaint             (no formal mental state)
NEWS   NEWS2 point bands for RR, SpO2 (scale 1), SBP, HR, temperature + 3 for confusion  (no O2 flag)
Acuity the triage nurse's ESI level; higher urgency = higher score
A missing vital scores 0 points.
"""
import numpy as np
import pandas as pd


def _points(x, bands):
    """bands: list of (lower bound inclusive, points), ascending. NaN -> 0."""
    out = np.zeros(len(x))
    for lo, pts in bands:
        out = np.where(x >= lo, pts, out)
    return np.where(np.isnan(x), 0, out)


def news_score(df):
    rr, spo2, sbp, hr, t = (df[c].to_numpy(float) for c in ("resprate", "o2sat", "sbp", "heartrate", "temperature"))
    return (_points(rr, [(-np.inf, 3), (9, 1), (12, 0), (21, 2), (25, 3)])
            + _points(spo2, [(-np.inf, 3), (92, 2), (94, 1), (96, 0)])
            + _points(sbp, [(-np.inf, 3), (91, 2), (101, 1), (111, 0), (220, 3)])
            + _points(hr, [(-np.inf, 3), (41, 1), (51, 0), (91, 1), (111, 2), (131, 3)])
            + _points(t, [(-np.inf, 3), (35.05, 1), (36.05, 0), (38.05, 1), (39.05, 2)])
            + 3 * df.cc_confusion.to_numpy())


def baseline_scores(df, cfg):
    """Returns {name: (score, alert)} with numpy arrays aligned to df."""
    t, hr, rr, sbp = (df[c] for c in ("temperature", "heartrate", "resprate", "sbp"))
    sirs = (((t > 38) | (t < 36)).astype(int) + (hr > 90).astype(int) + (rr > 20).astype(int)).to_numpy()
    qsofa = ((rr >= 22).astype(int) + (sbp <= 100).astype(int) + df.cc_confusion).to_numpy()
    news = news_score(df)
    acuity = df.acuity.to_numpy(float)
    b = cfg["baselines"]
    return {
        "SIRS (triage)": (sirs, sirs >= 2),
        "qSOFA (triage)": (qsofa, qsofa >= 2),
        "Modified NEWS (triage)": (news, news >= b["news_alert_min"]),
        "Triage acuity": (np.nan_to_num(6 - acuity, nan=0), acuity <= b["acuity_alert_max"]),
    }


def score_table(df, cfg):
    return pd.DataFrame({k: v[0] for k, v in baseline_scores(df, cfg).items()}, index=df.index)
