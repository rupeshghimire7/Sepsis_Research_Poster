"""Patient-level 70/15/15 split and the quality checks run on every pipeline run.

train: fitting and CV tuning. valid: calibration and traffic-light cut-offs. test: opened once at the end.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


class CheckFailed(RuntimeError):
    pass


def split_patients(df, outcome, cfg):
    """Assign every stay to train/valid/test by patient, stratified on 'patient ever had the outcome'."""
    s, seed = cfg["split"], cfg["seed"]
    pat = df.groupby("subject_id")[outcome].max()

    def strat(y):  # stratify only when each class can be spread over the parts
        return y if y.value_counts().min() >= 3 else None

    rest, test = train_test_split(pat, test_size=s["test"], stratify=strat(pat), random_state=seed)
    train, valid = train_test_split(rest, test_size=s["valid"] / (s["train"] + s["valid"]),
                                    stratify=strat(rest), random_state=seed)
    assign = pd.concat([pd.Series("train", train.index), pd.Series("valid", valid.index),
                        pd.Series("test", test.index)])
    return df.subject_id.map(assign).rename("split")


def estimated_year(df, anchor_year):
    """Real calendar year of each visit, accurate to about +/- 1 year.

    MIMIC shifts each patient's dates by a fixed amount. anchor_year_group is the real 3-year window containing
    the patient's (shifted) anchor_year, so real year ~ window middle + years since anchor_year."""
    start = df.anchor_year_group.str[:4].astype(int)
    return start + 1 + df.intime.dt.year - df.subject_id.map(anchor_year)


def temporal_split(df, anchor_year, outcome, cfg):
    """train/valid: visits before the cut-off year (split by patient). test: visits in or after it.
    Later visits of patients already seen before the cut-off are 'excluded' unless returning_patients: include."""
    t, seed = cfg["temporal"], cfg["seed"]
    if t["returning_patients"] not in ("exclude", "include"):
        raise ValueError("temporal.returning_patients must be 'exclude' or 'include'")
    late = (estimated_year(df, anchor_year) >= t["cutoff_year"]).to_numpy()
    new_patient = pd.Series(late).groupby(df.subject_id.to_numpy()).transform("all").to_numpy()

    pat = df[~late].groupby("subject_id")[outcome].max()
    strat = pat if pat.value_counts().min() >= 3 else None
    train, valid = train_test_split(pat, test_size=t["valid"], stratify=strat, random_state=seed)
    dev = pd.concat([pd.Series("train", train.index), pd.Series("valid", valid.index)])

    split = pd.Series("excluded", index=df.index, name="split")
    split[~late] = df.subject_id[~late].map(dev)
    split[late & (new_patient | (t["returning_patients"] == "include"))] = "test"
    return split


def check_patients_disjoint(df):
    n = df.groupby("subject_id").split.nunique()
    if (n > 1).any():
        raise CheckFailed(f"{int((n > 1).sum())} patient(s) appear in more than one split")


def check_no_forbidden(features, cfg, outcome_cols):
    c = cfg["checks"]
    banned = set(c["forbidden_columns"]) | set(outcome_cols)
    bad = [f for f in features if f in banned or f.startswith(tuple(c["forbidden_prefixes"]))]
    if bad:
        raise CheckFailed(f"forbidden column(s) in the feature matrix: {bad}")


def prevalence_by_split(df, outcomes):
    rows = []
    for name, g in [("all", df)] + list(df.groupby("split")):
        for o in outcomes:
            rows.append({"split": name, "outcome": o, "stays": len(g), "patients": g.subject_id.nunique(),
                         "cases": int(g[o].sum()), "prevalence": g[o].mean()})
    return pd.DataFrame(rows)


def enough_cases(prev, outcome, cfg):
    p = prev[(prev.outcome == outcome) & (prev.split != "all")]
    need = cfg["checks"]["min_cases_per_split"]
    short = p[p.cases < need]
    return short.empty, (f"every split has >= {need} '{outcome}' cases" if short.empty else
                         f"too few '{outcome}' cases for modelling (need >= {need} per split): "
                         + ", ".join(f"{r.split}={r.cases}" for r in short.itertuples()))


def check_prevalence(prev, outcome, cfg):
    p = prev[prev.outcome == outcome].set_index("split").prevalence
    gap = ((p.drop("all") - p["all"]).abs() / p["all"]).max() if p["all"] > 0 else 0.0
    if gap > cfg["checks"]["max_relative_prevalence_gap"]:
        raise CheckFailed(f"'{outcome}' prevalence differs across splits by {gap:.0%} (relative): "
                          + ", ".join(f"{k}={v:.2%}" for k, v in p.items()))
