"""Sepsis-3 check: does the model, trained on sepsis diagnosis codes, still beat the triage rules
when sepsis is defined clinically (Sepsis-3)?

- sepsis3_agreement.csv  ICD-coded sepsis vs Sepsis-3 on every cohort stay
- sepsis3_onset.csv      how soon after ED arrival Sepsis-3 starts (suspected-infection time)
- sepsis3_sofa.csv       which SOFA components make stays positive, and how many lab points were already
                         present 7-365 days before the visit (the label assumes a baseline SOFA of 0)
- sepsis3_test.csv       test split, Sepsis-3 as the outcome: AUROC / AUPRC of models and rules, and the
                         model vs each rule at the same alert rate (same layout as audit_rules_validation.csv)
The patient-level label is cached in out/ (git-ignored), because building it reads chartevents.
"""
import copy

import numpy as np
import pandas as pd

from .audit import rules_preview
from .evaluate import suppress
from .sepsis3 import COMPONENTS, LAB_COMPONENTS, PRIOR, sepsis3_label

ONSET_HOURS = [3, 6, 12, 24, 48]


def load_label(table, mimic, ed, out, horizon_h, log=print):
    """Sepsis-3 label for every stay in `table`, in the same order. Cached per horizon."""
    path = out / f"sepsis3_label_{horizon_h}h.parquet"
    stays = table[["stay_id", "subject_id", "intime"]].reset_index(drop=True)
    if path.exists():
        lab = pd.read_parquet(path)
        if lab.stay_id.equals(stays.stay_id) and set(COMPONENTS + PRIOR) <= set(lab):
            log(f"  using cached {path}")
            return lab
    log("  building the Sepsis-3 label (reads chartevents and labevents; slow on the full release)")
    lab = pd.concat([stays[["stay_id"]], sepsis3_label(mimic, ed, stays, horizon_h)], axis=1)
    lab.to_parquet(path, index=False)
    return lab


def agreement(icd, s3, min_cell):
    icd, s3 = np.asarray(icd, bool), np.asarray(s3, bool)
    cells = {"both": int((icd & s3).sum()), "ICD code only": int((icd & ~s3).sum()),
             "Sepsis-3 only": int((~icd & s3).sum()), "neither": int((~icd & ~s3).sum())}
    po = (cells["both"] + cells["neither"]) / len(icd)
    pe = icd.mean() * s3.mean() + (1 - icd.mean()) * (1 - s3.mean())
    hidden = 0 < cells["both"] < min_cell
    share = lambda d: None if hidden or d == 0 else round(100 * cells["both"] / d, 1)  # noqa: E731
    rows = [{"measure": f"stays: {k}", "value": suppress(v, min_cell)} for k, v in cells.items()]
    rows += [{"measure": "ICD sepsis prevalence %", "value": round(100 * icd.mean(), 2)},
             {"measure": "Sepsis-3 prevalence %", "value": round(100 * s3.mean(), 2)},
             {"measure": "ICD cases meeting Sepsis-3 %", "value": share(icd.sum())},
             {"measure": "Sepsis-3 cases with an ICD sepsis code %", "value": share(s3.sum())},
             {"measure": "Cohen's kappa", "value": round((po - pe) / (1 - pe), 3) if pe < 1 else None}]
    return pd.DataFrame(rows, dtype=object)  # keeps counts as integers


def onset(hours, min_cell):
    """Cumulative share of Sepsis-3 cases whose onset is within k hours of ED arrival."""
    h = pd.Series(hours, dtype=float).dropna()
    return pd.DataFrame([{"within_hours": k, "cases": suppress(int((h <= k).sum()), min_cell),
                          "share_of_sepsis3_cases_pct": round(100 * (h <= k).mean(), 1) if len(h) else None}
                         for k in ONSET_HOURS])


def sofa_breakdown(lab, icd, critical, min_cell, top_combos=10):
    """SOFA components among Sepsis-3 cases. One row per count: `cases` out of `of`, as `pct`.
    A lab point counts as chronic when the most recent result 7-365 days before arrival scored as high."""
    s3 = lab.sepsis3.to_numpy() == 1
    p = lab[s3].reset_index(drop=True)
    icd, critical = np.asarray(icd, bool)[s3], np.asarray(critical, bool)[s3]
    rows = []

    def add(section, item, mask, of):
        n, d = int(np.sum(mask)), int(np.sum(of))
        rows.append({"section": section, "item": item, "cases": suppress(n, min_cell), "of": suppress(d, min_cell),
                     "pct": None if 0 < n < min_cell or d == 0 else round(100 * n / d, 1)})

    every = np.ones(len(p), bool)
    for group, g in (("all Sepsis-3 cases", every), ("Sepsis-3 only (no ICD code)", ~icd), ("both", icd)):
        for c in COMPONENTS:
            add(f"component scores >= 1: {group}", c, g & (p[c] > 0), g)
    sofa2 = (p.sofa == 2).to_numpy()
    add("SOFA exactly 2", "all Sepsis-3 cases", sofa2, every)
    labs_only = (p[["resp", "cardio", "cns"]].sum(axis=1) == 0).to_numpy()
    add("SOFA exactly 2", "lab components only (coag, liver, renal)", sofa2 & labs_only, sofa2)
    combo = p[COMPONENTS].apply(lambda r: " + ".join(f"{c} {int(v)}" for c, v in r.items() if v > 0), axis=1)
    for k in combo[sofa2].value_counts().index[:top_combos]:
        add("SOFA exactly 2: combinations", k, sofa2 & (combo == k).to_numpy(), sofa2)
    add("cardio", "MAP < 70 only (1 point)", p.cardio == 1, every)
    add("cardio", "vasopressors (2-4 points)", p.cardio >= 2, every)

    adj = p[COMPONENTS].copy()
    for c in LAB_COMPONENTS:
        has, prior = (p[c] > 0).to_numpy(), p[f"{c}_prior"]
        chronic = (prior >= p[c]).to_numpy()  # missing prior -> not chronic
        add(f"{c} points already present 7-365 days before", "earlier result available", has & prior.notna(), has)
        add(f"{c} points already present 7-365 days before", "earlier result scored as high", has & chronic,
            has & prior.notna())
        adj.loc[chronic, c] = 0
    still = (adj.sum(axis=1) >= 2).to_numpy()
    sec = "still SOFA >= 2 without chronic lab points"
    for group, g in (("all Sepsis-3 cases", every), ("Sepsis-3 only (no ICD code)", ~icd), ("both", icd)):
        add(sec, group, g & still, g)
    add(sec, "critical outcome: still positive", still & critical, still)
    add(sec, "critical outcome: no longer positive", ~still & critical, ~still)
    return pd.DataFrame(rows, dtype=object)  # keeps counts as integers


def run_sepsis3_check(table, bundle, mimic, ed, out, cfg, log=print):
    min_cell, icd = cfg["evaluation"]["min_cell"], cfg["outcomes"]["primary"]
    lab = load_label(table, mimic, ed, out, cfg["outcomes"]["sepsis3"]["horizon_hours"], log)
    table = table.reset_index(drop=True).assign(sepsis3=lab.sepsis3.to_numpy())

    agr = agreement(table[icd], table.sepsis3, min_cell)
    agr.to_csv(out / "sepsis3_agreement.csv", index=False)
    ons = onset(lab.onset_hours_from_arrival[lab.sepsis3 == 1], min_cell)
    ons.to_csv(out / "sepsis3_onset.csv", index=False)
    sofa = sofa_breakdown(lab, table[icd], table[cfg["outcomes"]["secondary"]], min_cell)
    sofa.to_csv(out / "sepsis3_sofa.csv", index=False)

    test = table[table.split == "test"]
    cfg3 = copy.deepcopy(cfg)
    cfg3["outcomes"]["primary"] = "sepsis3"
    n_boot = cfg["evaluation"]["n_bootstrap"]
    log(f"  test split: {suppress(int(test.sepsis3.sum()), min_cell)} Sepsis-3 cases; "
        f"{n_boot} patient bootstrap resamples")
    res = rules_preview(test, bundle, cfg3, n_boot)
    res.to_csv(out / "sepsis3_test.csv", index=False)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        log("\nICD-coded sepsis vs Sepsis-3 (all cohort stays):")
        log(agr.to_string(index=False))
        log("\nSepsis-3 onset after ED arrival:")
        log(ons.to_string(index=False))
        log("\nSOFA components among Sepsis-3 cases:")
        log(sofa.to_string(index=False))
        log("\nTest split, Sepsis-3 as the outcome (estimate [95% CI]):")
        keep = (res.metric == "auroc") & (res.comparison == "ranking") | (res.metric == "sensitivity")
        show = res[keep].assign(value=res.apply(lambda r: f"{r.estimate:.3f} [{r.lo:.3f}, {r.hi:.3f}]", axis=1))
        log(show[["method", "comparison", "metric", "value"]].to_string(index=False))
