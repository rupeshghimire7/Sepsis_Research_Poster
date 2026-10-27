"""Checks before opening the test set: signs of leakage, and a preview of model vs rules on validation.

The test split is dropped first, so running this does not open the test set.
- audit_univariate.csv     AUROC of each feature on its own (one feature near 0.9 would be a leakage alarm)
- audit_complaints.csv     chief complaints that may already name the diagnosis, by outcome
- audit_without_flagged.csv  primary model on validation with and without those stays
- audit_rules_validation.csv model vs SIRS / qSOFA / NEWS / acuity on validation, incl. matched alert rates
All outputs are aggregate; counts below evaluation.min_cell are suppressed.
"""
import numpy as np
import pandas as pd

from .baselines import baseline_scores
from .evaluate import Ranker, bootstrap, suppress, threshold_metrics
from .io import read


def univariate(df, feats, sets, outcome):
    """AUROC of each feature alone (missing rows left out). Above 0.5 = higher values, more cases."""
    first = {}
    for s in sorted(sets):
        for f in sets[s]:
            first.setdefault(f, s)
    rows = []
    for f in feats:
        ok = df[f].notna().to_numpy()
        y = df[outcome].to_numpy()[ok]
        auc = Ranker(y, df[f].to_numpy(float)[ok]).auroc()
        rows.append({"feature": f, "set": first[f], "auroc": auc, "strength": abs(auc - 0.5) + 0.5,
                     "pct_missing": 100 * (1 - ok.mean())})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).round(4)


def complaint_flags(cc, patterns):
    cc = cc.fillna("").str.lower()
    return pd.DataFrame({k: cc.str.contains(p, regex=True) for k, p in patterns.items()}, index=cc.index)


def complaint_table(flags, y, exclude, min_cell):
    flags = flags.assign(**{"any excluded group (" + ", ".join(exclude) + ")": flags[exclude].any(axis=1)})
    total = int(y.sum())
    rows = []
    for k in flags:
        m = flags[k].to_numpy()
        n, cases = int(m.sum()), int(y[m].sum())
        hidden = 0 < cases < min_cell or 0 < n < min_cell
        rows.append({"complaint_group": k, "stays": suppress(n, min_cell), "cases": suppress(cases, min_cell),
                     "case_rate_pct": None if hidden or n == 0 else round(100 * cases / n, 1),
                     "share_of_all_cases_pct": None if hidden else round(100 * cases / total, 2)})
    return pd.DataFrame(rows)


def without_flagged(valid, flagged, fm):
    y, s = valid[fm.outcome].to_numpy(), fm.score(valid)
    rows = []
    for label, keep in (("all validation stays", np.ones(len(y), bool)),
                        ("excluding diagnosis-like complaints", ~flagged)):
        r = Ranker(y[keep], s[keep])
        rows.append({"subset": label, "stays": int(keep.sum()), "prevalence": y[keep].mean(),
                     "auroc": r.auroc(), "auprc": r.auprc()})
    return pd.DataFrame(rows).round(4)


def rules_preview(valid, bundle, cfg, n_boot):
    """Model vs each rule on validation: ranking, rule at its own threshold, model at the same alert rate."""
    y = valid[cfg["outcomes"]["primary"]].to_numpy()
    S = {m: fm.score(valid) for m, fm in bundle["primary"].items()}
    R = {m: Ranker(y, s) for m, s in S.items()}
    rules = baseline_scores(valid, cfg)
    RR = {r: Ranker(y, sc) for r, (sc, _) in rules.items()}
    xgb = S["xgboost"]
    matched = {r: xgb >= np.quantile(xgb, 1 - a.mean()) for r, (_, a) in rules.items()}

    def fn(w):
        d = {}
        for m in S:
            d[f"{m}|ranking|auroc"], d[f"{m}|ranking|auprc"] = R[m].auroc(w), R[m].auprc(w)
        for r, (_, a) in rules.items():
            d[f"{r}|ranking|auroc"], d[f"{r}|ranking|auprc"] = RR[r].auroc(w), RR[r].auprc(w)
            rule, model = threshold_metrics(y, a, w), threshold_metrics(y, matched[r], w)
            for k in ("alert_rate", "sensitivity", "ppv", "reviews_per_case"):
                d[f"{r}|rule alert|{k}"] = rule[k]
            for k in ("sensitivity", "ppv", "reviews_per_case"):
                d[f"{r}|xgboost at same alert rate|{k}"] = model[k]
            d[f"{r}|xgboost minus rule, same alert rate|sensitivity"] = model["sensitivity"] - rule["sensitivity"]
            d[f"{r}|xgboost minus rule|auroc"] = d["xgboost|ranking|auroc"] - d[f"{r}|ranking|auroc"]
        return d

    res = bootstrap(fn, valid.subject_id.to_numpy(), n_boot, cfg["seed"])
    res[["method", "comparison", "metric"]] = res.key.str.split("|", expand=True)
    return res[["method", "comparison", "metric", "estimate", "lo", "hi"]].round(4)


def run_audit(table, bundle, ed, out, cfg, log=print):
    a, primary = cfg["audit"], cfg["outcomes"]["primary"]
    dev = table[table.split.isin(["train", "valid"])].copy()  # the test split never enters
    valid = dev[dev.split == "valid"]
    min_cell = cfg["evaluation"]["min_cell"]
    sets = bundle["feature_sets"]

    uni = univariate(valid, sets["C"], sets, primary)
    uni.to_csv(out / "audit_univariate.csv", index=False)

    tri = read(ed, "ed", "triage", ["stay_id", "chiefcomplaint"])
    cc = dev[["stay_id"]].merge(tri, on="stay_id", how="left").chiefcomplaint
    cc.index = dev.index
    flags = complaint_flags(cc, a["complaint_flags"])
    comp = complaint_table(flags, dev[primary].to_numpy(), a["exclude_for_sensitivity"], min_cell)
    comp.to_csv(out / "audit_complaints.csv", index=False)

    flagged = flags.loc[valid.index, a["exclude_for_sensitivity"]].any(axis=1).to_numpy()
    wf = without_flagged(valid, flagged, bundle["primary"]["xgboost"])
    wf.to_csv(out / "audit_without_flagged.csv", index=False)

    log(f"  model vs rules on validation ({a['n_bootstrap']} patient bootstrap resamples)")
    rp = rules_preview(valid, bundle, cfg, a["n_bootstrap"])
    rp.to_csv(out / "audit_rules_validation.csv", index=False)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        log("\nStrongest single features (validation AUROC, missing rows left out):")
        log(uni.head(12).to_string(index=False))
        log("\nDiagnosis-like chief complaints (train + validation):")
        log(comp.to_string(index=False))
        log("\nPrimary model on validation, with and without those stays:")
        log(wf.to_string(index=False))
        log("\nModel vs rules on validation (estimate [95% CI]):")
        show = rp.assign(value=rp.apply(lambda r: f"{r.estimate:.3f} [{r.lo:.3f}, {r.hi:.3f}]", axis=1))
        log(show[["method", "comparison", "metric", "value"]].to_string(index=False))
