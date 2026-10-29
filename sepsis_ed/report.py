"""Step 2: score the saved models on the test split, explain them, check fairness, and write the tables,
figures and findings. Everything written here is aggregate; counts below 10 are shown as "<10".
The shareable subset is copied to results/ for the public repository."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from . import figures
from .baselines import baseline_scores
from .evaluate import (Ranker, bootstrap, brier, calibration_bins, calibration_slope_intercept, net_benefit,
                       suppress, threshold_metrics)
from .explain import contributions, crosscheck, label, shap_top

OPS = {"top 5%": "high", "top 20%": "moderate"}  # operating point -> cut-off name


def _split_key(df):
    parts = df.key.str.split("|", expand=True)
    parts.columns = ["section", "method", "operating_point", "metric"]
    return pd.concat([parts, df.drop(columns="key")], axis=1)


def _fmt(est, lo, hi, pct=False):
    if pd.isna(est):
        return ""
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.3f}")
    return f"{f(est)} ({f(lo)} to {f(hi)})"


# ---------------------------------------------------------------- table 1
def table1(table, cfg):
    primary, mc = cfg["outcomes"]["primary"], cfg["evaluation"]["min_cell"]
    groups = [("Overall", table), ("Sepsis", table[table[primary] == 1]), ("No sepsis", table[table[primary] == 0])]
    rows = []

    def add(name, fn):
        rows.append({"characteristic": name, **{g: fn(d) for g, d in groups}})

    def n_pct(mask_fn):
        return lambda d: (lambda n: f"{suppress(n, mc)} ({n / len(d):.1%})" if n >= mc or n == 0
                          else f"<{mc}")(int(mask_fn(d).sum())) if len(d) else ""

    def med(col):
        return lambda d: (f"{d[col].median():.1f} [{d[col].quantile(.25):.1f}–{d[col].quantile(.75):.1f}]"
                          if d[col].notna().any() else "")

    add("ED stays, n", lambda d: str(suppress(len(d), mc)))
    add("Patients, n (a patient with several visits can count in both outcome columns)",
        lambda d: str(suppress(d.subject_id.nunique(), mc)))
    add("Age, median [IQR]", med("age"))
    add("Female, n (%)", n_pct(lambda d: d.male == 0))
    for a in ("ambulance", "walk_in", "other", "helicopter", "unknown"):
        add(f"Arrival: {a.replace('_', ' ')}, n (%)", n_pct(lambda d, a=a: d[f"arrive_{a}"] == 1))
    for k in range(1, 6):
        add(f"Triage acuity {k}, n (%)", n_pct(lambda d, k=k: d.acuity == k))
    for v in ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"):
        add(f"{label(v)}, median [IQR]", med(v))
    add("Any triage vital missing, n (%)", n_pct(lambda d: d.n_missing_vitals > 0))
    add("ED visits in past year, median [IQR]", med("ed_visits_365d"))
    add("Previous sepsis admission, n (%)", n_pct(lambda d: d.prior_sepsis_admission == 1))
    for c in [c for c in table if c.startswith("hx_")]:
        add(f"{label(c)}, n (%)", n_pct(lambda d, c=c: d[c] == 1))
    for g in race_groups(table.race, cfg).unique():
        add(f"Race: {g}, n (%)", n_pct(lambda d, g=g: race_groups(d.race, cfg) == g))
    add("Critical deterioration, n (%)", n_pct(lambda d: d.critical == 1))
    add("Admitted, n (%)", n_pct(lambda d: d.admitted == 1))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- subgroups
def race_groups(race, cfg):
    r = race.fillna("").str.upper()
    out = pd.Series("Other/unknown", index=race.index)
    for prefix, name in cfg["fairness"]["race_groups"].items():
        out[r.str.startswith(prefix)] = name
    return out


def subgroups(test, cfg):
    edges = cfg["fairness"]["age_bands"] + [np.inf]
    age_labels = [f"{lo}–{hi - 1}" if np.isfinite(hi) else f"{lo}+" for lo, hi in zip(edges[:-1], edges[1:])]
    arrival = np.select([test.arrive_ambulance == 1, test.arrive_walk_in == 1], ["Ambulance", "Walk-in"],
                        "Other/unknown")
    return {
        "Sex": np.where(test.male == 1, "Male", "Female"),
        "Age": pd.cut(test.age, edges, right=False, labels=age_labels).astype(str).to_numpy(),
        "Race": race_groups(test.race, cfg).to_numpy(),
        "Arrival": arrival,
        "Vitals": np.where(test.n_missing_vitals == 0, "All recorded", "One or more missing"),
        "Period": test.anchor_year_group.fillna("unknown").to_numpy(),
    }


def fairness_audit(test, fm, cfg):
    primary, mc = fm.outcome, cfg["evaluation"]["min_cell"]
    y, s, p = test[primary].to_numpy(), fm.score(test), fm.risk(test)
    high = s >= fm.cutoffs["high"]
    cells, meta = {}, []
    for attr, vals in subgroups(test, cfg).items():
        for g in sorted(pd.unique(vals), key=lambda v: (v.startswith("Other"), v)):
            m = vals == g
            n, cases = int(m.sum()), int(y[m].sum())
            ok = n >= mc and cases >= mc and (n - cases) >= mc
            meta.append({"attribute": attr, "group": g, "stays": suppress(n, mc), "cases": suppress(cases, mc),
                         "suppressed": not ok})
            if ok:
                cells[(attr, g)] = (m, Ranker(y[m], s[m]))

    def fn(w):
        d = {}
        for (attr, g), (m, r) in cells.items():
            wm = None if w is None else w[m]
            tm = threshold_metrics(y[m], high[m], wm)
            obs, exp = np.average(y[m], weights=wm), np.average(p[m], weights=wm)
            k = f"{attr}|{g}"
            d.update({f"{k}|auroc": r.auroc(wm), f"{k}|sensitivity_high": tm["sensitivity"],
                      f"{k}|fpr_high": tm["fpr"], f"{k}|ppv_high": tm["ppv"], f"{k}|observed_rate": obs,
                      f"{k}|expected_rate": exp, f"{k}|observed_to_expected": obs / exp if exp else np.nan})
        return d

    ev = cfg["evaluation"]
    b = bootstrap(fn, test.subject_id.to_numpy(), ev["n_bootstrap"], cfg["seed"]) if cells else pd.DataFrame(
        columns=["key", "estimate", "lo", "hi"])
    out = pd.DataFrame(meta)
    if len(b):
        parts = b.key.str.split("|", expand=True)
        b = b.assign(attribute=parts[0], group=parts[1], metric=parts[2])
        wide = b.pivot_table(index=["attribute", "group"], columns="metric", values=["estimate", "lo", "hi"])
        wide.columns = [m if s == "estimate" else f"{m}_{s}" for s, m in wide.columns]
        out = out.merge(wide.reset_index(), on=["attribute", "group"], how="left")
    else:
        out["auroc"] = np.nan
    return out


# ---------------------------------------------------------------- test evaluation
def evaluate_test(table, bundle, out, cfg, log=print):
    cfg_e, mc = cfg["evaluation"], cfg["evaluation"]["min_cell"]
    test = table[table.split == "test"].reset_index(drop=True)
    primary = cfg["outcomes"]["primary"]
    y = test[primary].to_numpy()
    models = bundle["primary"]
    S = {m: fm.score(test) for m, fm in models.items()}
    P = {m: fm.risk(test) for m, fm in models.items()}
    R = {m: Ranker(y, S[m]) for m in models}
    alerts = {m: {op: S[m] >= fm.cutoffs[c] for op, c in OPS.items()} for m, fm in models.items()}
    xgb = models["xgboost"]
    ref_alert = S["xgboost"] >= xgb.cutoffs["reference"]

    rules = baseline_scores(test, cfg)
    RR = {r: Ranker(y, sc) for r, (sc, _) in rules.items()}
    matched = {}
    for r, (_, alert) in rules.items():  # model flags the same share of arrivals as the rule
        f = alert.mean()
        matched[r] = (f, S["xgboost"] >= np.quantile(S["xgboost"], 1 - f) if f > 0 else np.zeros(len(y), bool))

    fs = {k: v for k, v in bundle["feature_set_models"].items()}
    FS = {k: Ranker(y, v.score(test)) for k, v in fs.items()}
    sec = bundle["secondary"].get("xgboost")
    if sec is not None:
        y2, s2 = test[sec.outcome].to_numpy(), sec.score(test)
        R2, p2 = Ranker(y2, s2), sec.risk(test)

    def fn(w):
        d = {}
        for m in models:
            d[f"performance|{m}|all|auroc"] = R[m].auroc(w)
            d[f"performance|{m}|all|auprc"] = R[m].auprc(w)
            d[f"performance|{m}|all|brier"] = brier(y, P[m], w)
            for k, v in calibration_slope_intercept(y, P[m], w, iters=10).items():
                d[f"performance|{m}|all|{k}"] = v
            for op, a in alerts[m].items():
                for k, v in threshold_metrics(y, a, w).items():
                    d[f"performance|{m}|{op}|{k}"] = v
        for k, v in threshold_metrics(y, ref_alert, w).items():
            d[f"performance|xgboost|sensitivity {cfg['traffic_light']['reference_sensitivity']:.2f} "
              f"(validation cut-off)|{k}"] = v
        for r, (_, a) in rules.items():
            d[f"performance|{r}|all|auroc"] = RR[r].auroc(w)
            d[f"performance|{r}|all|auprc"] = RR[r].auprc(w)
            for k, v in threshold_metrics(y, a, w).items():
                d[f"performance|{r}|rule alert|{k}"] = v
        for m in ("logistic", "random_forest"):
            d[f"comparison|xgboost minus {m}|all|auroc"] = d["performance|xgboost|all|auroc"] - \
                d[f"performance|{m}|all|auroc"]
            d[f"comparison|xgboost minus {m}|all|auprc"] = d["performance|xgboost|all|auprc"] - \
                d[f"performance|{m}|all|auprc"]
        for r, (f, ma) in matched.items():
            sm = threshold_metrics(y, ma, w)["sensitivity"]
            sr = d[f"performance|{r}|rule alert|sensitivity"]
            d[f"performance|xgboost|same alert rate as {r}|sensitivity"] = sm
            d[f"comparison|xgboost minus {r}|matched alert rate {f:.1%}|sensitivity"] = sm - sr
            d[f"comparison|xgboost minus {r}|all|auroc"] = d["performance|xgboost|all|auroc"] - d[
                f"performance|{r}|all|auroc"]
        for k, r in FS.items():
            d[f"feature_sets|xgboost set {k}|all|auroc"] = r.auroc(w)
            d[f"feature_sets|xgboost set {k}|all|auprc"] = r.auprc(w)
        names = sorted(FS)
        for a, b in zip(names[:-1], names[1:]):
            d[f"feature_sets|set {b} minus set {a}|all|auroc"] = d[f"feature_sets|xgboost set {b}|all|auroc"] - \
                d[f"feature_sets|xgboost set {a}|all|auroc"]
        if sec is not None:
            d[f"secondary|xgboost|all|auroc"] = R2.auroc(w)
            d[f"secondary|xgboost|all|auprc"] = R2.auprc(w)
            d[f"secondary|xgboost|all|brier"] = brier(y2, p2, w)
        return d

    log(f"  bootstrapping {cfg_e['n_bootstrap']} patient resamples of the test set")
    res = _split_key(bootstrap(fn, test.subject_id.to_numpy(), cfg_e["n_bootstrap"], cfg["seed"]))
    res.insert(0, "outcome", np.where(res.section == "secondary", cfg["outcomes"]["secondary"], primary))
    perf = res[res.section.isin(["performance", "secondary"])].drop(columns="section")
    perf.to_csv(out / "table2_performance.csv", index=False)
    res[res.section == "comparison"].drop(columns="section").to_csv(out / "table2_comparisons.csv", index=False)
    res[res.section == "feature_sets"].drop(columns="section").to_csv(out / "table_feature_sets.csv", index=False)
    summary = performance_summary(perf[perf.outcome == primary], list(models), list(rules))
    summary.to_csv(out / "table2_summary.csv", index=False)

    # traffic light (XGBoost)
    tier = xgb.tier(test)
    tl = []
    for t in ("low", "moderate", "high"):
        m = tier == t
        n, c = int(m.sum()), int(y[m].sum())
        hide = 0 < c < mc
        tl.append({"tier": t, "arrivals": suppress(n, mc), "share_of_arrivals": m.mean(),
                   "cases": suppress(c, mc), "sepsis_rate": np.nan if hide else c / max(n, 1),
                   "share_of_cases": np.nan if hide else c / max(y.sum(), 1),
                   "mean_predicted_risk": P["xgboost"][m].mean() if n else np.nan})
    tl = pd.DataFrame(tl).round(4)
    tl.to_csv(out / "table3_traffic_light.csv", index=False)

    # figures: ROC/PR, calibration, decision curve, traffic light
    curves = {}
    for m in ("xgboost", "logistic", "random_forest"):
        fpr, tpr, _ = roc_curve(y, S[m])
        prec, rec, _ = precision_recall_curve(y, S[m])
        curves[m] = (fpr, tpr, rec, prec, R[m].auroc(), R[m].auprc())
    rule_pts = {}
    for r, (_, a) in rules.items():
        tm = threshold_metrics(y, a)
        rule_pts[r] = (tm["fpr"], tm["sensitivity"], tm["ppv"])
    figures.roc_pr(curves, rule_pts, y.mean(), out / "fig_roc_pr.png")
    cal = {m: calibration_bins(y, P[m], cfg_e["calibration_bins"]) for m in curves}
    pd.concat([d.assign(model=m) for m, d in cal.items()]).to_csv(out / "fig_calibration.csv", index=False)
    figures.calibration(cal, out / "fig_calibration.png")
    lo, hi = cfg_e["decision_thresholds"]
    th = np.round(np.arange(lo, hi + 1e-9, 0.005), 4)
    nb = {m: net_benefit(y, P[m], th) for m in curves}
    pd.concat([d.assign(model=m) for m, d in nb.items()]).to_csv(out / "fig_decision_curve.csv", index=False)
    figures.decision_curve(nb, out / "fig_decision_curve.png")
    if not tl[["share_of_cases"]].isna().any().any():
        figures.traffic_light(tl, out / "fig_traffic_light.png")
    figures.rules_matched(_matched_rows(perf, matched), out / "fig_rules_matched.png")
    _cohort_figure(table, out, cfg)

    # explainability
    log("  SHAP and importance cross-check")
    sv, _ = contributions(xgb, test)
    full, top = shap_top(xgb, test[xgb.features], sv, cfg["explain"]["top_n"])
    top.to_csv(out / "table_shap_top.csv", index=False)
    figures.shap_bars(top, out / "fig_shap_top.png")
    figures.shap_summary(sv, test[xgb.features], top, out / "fig_shap_summary.png")
    dep = [f for f in cfg["explain"]["dependence"] if f in xgb.features]
    figures.shap_dependence(sv, test[xgb.features], dep[:4], {f: label(f) for f in dep}, out / "fig_shap_dependence.png")
    cc, agree = crosscheck(xgb, models["logistic"], test, y, sv, cfg)
    cc.to_csv(out / "table_importance_crosscheck.csv", index=False)

    # fairness
    log("  fairness audit")
    fair = fairness_audit(test, xgb, cfg)
    fair.to_csv(out / "table4_fairness.csv", index=False)
    overall = {"auroc": R["xgboost"].auroc(), "sensitivity_high": threshold_metrics(y, alerts["xgboost"]["top 5%"])[
        "sensitivity"]}
    if fair.auroc.notna().any():
        figures.fairness(fair, overall, out / "fig_fairness.png")

    findings(out, perf, res, tl, top, fair, agree, y, cfg, list(rules))
    publish(out, cfg)


PUBLIC = ["findings.md", "cohort_flow.json", "table1_characteristics.csv", "table2_summary.csv",
          "table2_comparisons.csv", "table2_performance.csv", "table3_traffic_light.csv", "table4_fairness.csv",
          "table_feature_sets.csv", "table_shap_top.csv", "table_importance_crosscheck.csv",
          "sepsis3_agreement.csv", "sepsis3_onset.csv", "sepsis3_sofa.csv", "sepsis3_test.csv"]
PUBLIC_FIGS = ["fig_cohort_flow", "fig_rules_matched", "fig_roc_pr", "fig_calibration", "fig_decision_curve",
               "fig_traffic_light", "fig_shap_top", "fig_fairness"]


def publish(out, cfg):
    """Copy the aggregate results to results/. SHAP scatter plots (one dot per visit) stay local."""
    dest = Path(cfg["paths"].get("results_dir", "results"))
    if not dest.is_absolute():
        dest = Path(__file__).resolve().parents[1] / dest
    dest.mkdir(parents=True, exist_ok=True)
    for f in PUBLIC + [f"{g}.{ext}" for g in PUBLIC_FIGS for ext in ("png", "pdf")]:
        if (out / f).exists():
            shutil.copy2(out / f, dest / f)


def _matched_rows(perf, matched):
    g = lambda m, op: perf[(perf.method == m) & (perf.operating_point == op) & (perf.metric == "sensitivity")].iloc[0]  # noqa
    rows = []
    for r, (f, _) in matched.items():
        a, b = g(r, "rule alert"), g("xgboost", f"same alert rate as {r}")
        rows.append({"name": r, "alert_rate": f, "rule": (a.estimate, a.lo, a.hi),
                     "model": (b.estimate, b.lo, b.hi)})
    return rows


def _cohort_figure(table, out, cfg):
    flow = json.loads((out / "cohort_flow.json").read_text())
    primary = cfg["outcomes"]["primary"]
    reasons = {"with a triage record": "no triage record", "with a patient record (age known)": "no patient record",
               "aged 18 or over": "under 18", "at least one triage vital recorded": "no triage vital signs"}
    first, last = flow["flow"][0], flow["flow"][-1]
    excluded = [f"{s['removed']:,} {reasons.get(s['step'], s['step'])}" for s in flow["flow"][1:] if s["removed"]]
    cases = int(table[primary].sum())
    splits = []
    for name, label in (("train", "Training 70%"), ("valid", "Validation 15%"), ("test", "Test 15%")):
        d = table[table.split == name]
        splits.append(f"{label}\n{d.subject_id.nunique():,} patients\n{len(d):,} visits\n"
                      f"{int(d[primary].sum()):,} sepsis")
    figures.cohort_flow(f"{first['stays']:,} ED visits\n{first['patients']:,} patients", excluded,
                        f"{last['stays']:,} visits analysed · {last['patients']:,} patients\n"
                        f"{cases:,} with sepsis ({cases / last['stays']:.1%})", splits, out / "fig_cohort_flow.png")


def performance_summary(perf, models, rules):
    get = lambda m, op, k: perf[(perf.method == m) & (perf.operating_point == op) & (perf.metric == k)].iloc[0]  # noqa
    rows = []
    for m in models + rules:
        op = "top 5%" if m in models else "rule alert"
        a, p = get(m, "all", "auroc"), get(m, "all", "auprc")
        sens, ppv, ar = get(m, op, "sensitivity"), get(m, op, "ppv"), get(m, op, "alert_rate")
        rows.append({"method": m, "operating_point": op, "auroc": _fmt(a.estimate, a.lo, a.hi),
                     "auprc": _fmt(p.estimate, p.lo, p.hi), "alerts_per_100": round(ar.estimate * 100, 1),
                     "sensitivity": _fmt(sens.estimate, sens.lo, sens.hi, True),
                     "cases_per_100_alerts": round(ppv.estimate * 100, 1),
                     "reviews_per_case": round(1 / ppv.estimate, 1) if ppv.estimate else np.nan})
    return pd.DataFrame(rows)


def findings(out, perf, res, tl, top, fair, agree, y, cfg, rules):
    """The protocol's 'sentence to complete' for each research question, filled in."""
    flow = json.loads((out / "cohort_flow.json").read_text())
    g = lambda d, m, op, k: d[(d.method == m) & (d.operating_point == op) & (d.metric == k)].iloc[0]  # noqa
    primary = cfg["outcomes"]["primary"]
    p = perf[perf.outcome == primary]
    a, ap = g(p, "xgboost", "all", "auroc"), g(p, "xgboost", "all", "auprc")
    cmp = res[res.section == "comparison"]
    lr = cmp[(cmp.method == "xgboost minus logistic") & (cmp.metric == "auroc")].iloc[0]
    fs = res[res.section == "feature_sets"]
    fa = fs[(fs.method == "xgboost set A") & (fs.metric == "auroc")]
    fb = fs[(fs.method == "xgboost set B") & (fs.metric == "auroc")]
    cal = g(p, "xgboost", "all", "calibration_slope")
    br = g(p, "xgboost", "all", "brier")
    hi = tl.set_index("tier").loc["high"]
    ppv_hi = g(p, "xgboost", "top 5%", "ppv").estimate
    f = fair[fair.sensitivity_high.notna()] if "sensitivity_high" in fair else fair.iloc[0:0]
    last = flow["flow"][-1]
    lines = [
        "# Findings (test set, 95% CI from patient bootstrap)",
        "",
        f"- **Who was studied?** Of {last['stays']:,} adult ED visits, {flow['outcomes'][primary]:,} "
        f"({flow['outcomes'][primary] / last['stays']:.1%}) had sepsis recorded.",
        f"- **RQ1.** Using triage data alone, XGBoost reached AUROC {_fmt(a.estimate, a.lo, a.hi)} and AUPRC "
        f"{_fmt(ap.estimate, ap.lo, ap.hi)}, against a sepsis rate of {y.mean():.1%}.",
        f"- **RQ2 (models).** XGBoost {'exceeded' if lr.estimate >= 0 else 'trailed'} logistic regression by "
        f"{abs(lr.estimate):.3f} AUROC (95% CI for XGBoost minus LR {lr.lo:.3f} to {lr.hi:.3f}"
        f"{'; includes 0' if lr.lo <= 0 <= lr.hi else ''}).",
    ]
    for r in rules:
        c = cmp[(cmp.method == f"xgboost minus {r}") & (cmp.metric == "sensitivity")].iloc[0]
        rs = g(p, r, "rule alert", "sensitivity").estimate
        lines.append(f"- **RQ2, {figures.RULE_NAMES.get(r, r)}.** Flagging the same {c.operating_point.split()[-1]} of arrivals, the model "
                     f"found {rs + c.estimate:.0%} of sepsis cases and the rule {rs:.0%} (difference "
                     f"{c.estimate * 100:.1f} percentage points, 95% CI {c.lo * 100:.1f} to {c.hi * 100:.1f}).")
    if len(fa) and len(fb):
        lines.append(f"- **Context.** Adding complaint, arrival and history to vitals changed AUROC from "
                     f"{fa.estimate.iloc[0]:.3f} to {fb.estimate.iloc[0]:.3f}.")
    lines += [
        f"- **Calibration.** After calibration the slope was {cal.estimate:.2f} "
        f"({cal.lo:.2f} to {cal.hi:.2f}) and the Brier score {br.estimate:.4f}.",
        f"- **RQ3.** The strongest drivers were {top.label.iloc[0]}, {top.label.iloc[1]} and {top.label.iloc[2]} "
        f"(SHAP vs permutation rank agreement ρ = {agree['spearman_shap_vs_permutation_top']:.2f}).",
    ]
    if len(f):
        name = lambda r: f"{r.attribute.lower()} {r.group}"  # noqa: E731
        lo_a, hi_a = f.loc[f.auroc.idxmin()], f.loc[f.auroc.idxmax()]
        oe = f.dropna(subset=["observed_to_expected"])
        lo_o, hi_o = oe.loc[oe.observed_to_expected.idxmin()], oe.loc[oe.observed_to_expected.idxmax()]
        lo_s = f.loc[f.sensitivity_high.idxmin()]
        lines.append(f"- **RQ4.** AUROC ranged from {lo_a.auroc:.3f} ({name(lo_a)}) to {hi_a.auroc:.3f} "
                     f"({name(hi_a)}). Observed/expected cases ranged from {lo_o.observed_to_expected:.2f} "
                     f"({name(lo_o)}) to {hi_o.observed_to_expected:.2f} ({name(hi_o)}). The high-risk tier caught "
                     f"the smallest share of cases for {name(lo_s)} ({lo_s.sensitivity_high:.0%}). Groups suppressed "
                     f"(< {cfg['evaluation']['min_cell']} cases): {int(fair.suppressed.sum())}.")
    if not pd.isna(hi.share_of_cases):
        lines.append(f"- **RQ5.** The high-risk tier was {hi.share_of_arrivals:.0%} of arrivals and held "
                     f"{hi.share_of_cases:.0%} of sepsis cases, at {1 / ppv_hi:.1f} reviews per case found "
                     f"({ppv_hi * 100:.0f} cases per 100 alerts).")
    (out / "findings.md").write_text("\n".join(lines) + "\n")
