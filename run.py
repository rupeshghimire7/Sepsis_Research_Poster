"""Run the analysis in two steps, so the test split is used only once.

python run.py --mimic-dir data_real/mimic-iv --ed-dir data_real/mimic-iv-ed
    Step 1: build the cohort, outcomes and features, split by patient, then train and calibrate the models
    on the training and validation splits.
python run.py --audit
    Optional: leakage checks and a model-vs-rules preview on validation. Does not touch the test split.
python run.py --open-test
    Step 2, once: score the saved models on the test split and write tables, figures and findings.
    After this, step 1 refuses to retrain.

python run.py --sepsis3-check
    Optional, after step 2: score the locked models against a clinical Sepsis-3 label (nothing is retrained).
    Needs the extra tables listed in sepsis_ed/sepsis3.py.

python run.py --temporal   then, once:   python run.py --temporal --open-test
    Temporal validation: reuse the main analysis table, train on visits before temporal.cutoff_year and test on
    later visits. Everything (models, lock, outputs) lives in out/temporal/; results go to results/temporal/.

Other flags: --data-only stops after the data stage; --sepsis3 adds the Sepsis-3 label to the analysis table;
--reopen-test overrides the test lock (report it if you use it).
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from sepsis_ed import config
from sepsis_ed.cohort import build_cohort
from sepsis_ed.features import build_features, missingness_table
from sepsis_ed.labels import critical_label, sepsis_icd_label
from sepsis_ed.model import develop, validation_report
from sepsis_ed.io import read
from sepsis_ed.split import (CheckFailed, check_no_forbidden, check_patients_disjoint, check_prevalence,
                             enough_cases, prevalence_by_split, split_patients, temporal_split)

KEEP = ["stay_id", "subject_id", "hadm_id", "intime", "race", "anchor_year_group", "admitted"]  # audit only


def data_stage(cfg, mimic, ed, out, sepsis3):
    oc = cfg["outcomes"]
    primary, secondary = oc["primary"], oc["secondary"]

    df, flow = build_cohort(mimic, ed, cfg)
    df["admitted"] = df.hadm_id.notna().astype(int)
    df["sepsis_icd"] = sepsis_icd_label(mimic, ed, df, oc["sepsis_icd"]["icd10"], oc["sepsis_icd"]["icd9"])
    df["critical"] = critical_label(mimic, df, oc["critical"]["icu_within_hours"])
    outcomes, s3_cols = ["sepsis_icd", "critical"], []
    if sepsis3 or oc["sepsis3"]["enabled"]:
        from sepsis_ed.sepsis3 import sepsis3_label
        s3 = sepsis3_label(mimic, ed, df, oc["sepsis3"]["horizon_hours"])
        s3_cols = list(s3)
        df = pd.concat([df, s3], axis=1)
        outcomes.append("sepsis3")

    X, sets = build_features(df, mimic, ed, cfg)
    table = pd.concat([df[KEEP + outcomes + [c for c in s3_cols if c not in outcomes]], X], axis=1)

    table["split"] = split_patients(table, primary, cfg)
    prev = prevalence_by_split(table, outcomes)
    ok, cases_msg = enough_cases(prev, primary, cfg)
    check_patients_disjoint(table)
    for feats in sets.values():
        check_no_forbidden(feats, cfg, outcomes + s3_cols)
    if ok:
        for o in (primary, secondary):
            check_prevalence(prev, o, cfg)

    table.to_parquet(out / "analysis_table.parquet", index=False)
    table[["stay_id", "subject_id", "split"]].to_csv(out / "splits.csv", index=False)
    prev.to_csv(out / "prevalence_by_split.csv", index=False)
    missingness_table(table, sets["C"], primary).to_csv(out / "missingness.csv")
    (out / "feature_sets.json").write_text(json.dumps(sets, indent=2))
    summary = {"flow": flow, "outcomes": {o: int(table[o].sum()) for o in outcomes},
               "splits": table.groupby("split").subject_id.nunique().to_dict()}
    (out / "cohort_flow.json").write_text(json.dumps(summary, indent=2))

    print(pd.DataFrame(flow).to_string(index=False))
    print()
    print(prev.assign(prevalence=prev.prevalence.map("{:.2%}".format)).to_string(index=False))
    print(f"\nfeatures: A={len(sets['A'])}  B={len(sets['B'])}  C={len(sets['C'])}")
    print("checks passed: patients disjoint across splits, no forbidden columns"
          + (", prevalence balanced" if ok else ""))
    return table, sets, ok, cases_msg


def temporal_stage(cfg, mimic, main_out, out):
    """Re-split the main analysis table by time. No prevalence-balance check: drift over time is part of what
    temporal validation measures, so it is reported (prevalence_by_split.csv) instead of blocked."""
    src = main_out / "analysis_table.parquet"
    if not src.exists():
        sys.exit(f"No {src}: run the main data stage first (python run.py --data-only).")
    primary, secondary = cfg["outcomes"]["primary"], cfg["outcomes"]["secondary"]
    table = pd.read_parquet(src).drop(columns="split")
    anchor = read(mimic, "hosp", "patients", ["subject_id", "anchor_year"]).set_index("subject_id").anchor_year
    table["split"] = temporal_split(table, anchor, primary, cfg)
    n_excl = table.split.eq("excluded").sum()
    table = table[table.split != "excluded"].reset_index(drop=True)

    dev = table[table.split.isin(["train", "valid"])]
    check_patients_disjoint(dev if cfg["temporal"]["returning_patients"] == "include" else table)
    sets = json.loads((main_out / "feature_sets.json").read_text())
    outcomes = [o for o in (primary, secondary, "sepsis3") if o in table]
    prev = prevalence_by_split(table, outcomes)
    ok, cases_msg = enough_cases(prev, primary, cfg)

    table.to_parquet(out / "analysis_table.parquet", index=False)
    table[["stay_id", "subject_id", "split"]].to_csv(out / "splits.csv", index=False)
    prev.to_csv(out / "prevalence_by_split.csv", index=False)
    (out / "feature_sets.json").write_text(json.dumps(sets, indent=2))
    flow = json.loads((main_out / "cohort_flow.json").read_text())
    flow["temporal"] = {**cfg["temporal"], "excluded_returning_visits": int(n_excl)}
    flow["splits"] = table.groupby("split").subject_id.nunique().to_dict()
    (out / "cohort_flow.json").write_text(json.dumps(flow, indent=2))

    print(f"cut-off {cfg['temporal']['cutoff_year']}, returning patients: {cfg['temporal']['returning_patients']}"
          f" ({n_excl:,} later visits of earlier patients excluded)")
    print(prev.assign(prevalence=prev.prevalence.map("{:.2%}".format)).to_string(index=False))
    return table, sets, ok, cases_msg


def temporal_labels(cfg):
    c = cfg["temporal"]["cutoff_year"]
    test = "patients first seen" if cfg["temporal"]["returning_patients"] == "exclude" else "visits"
    cfg["split_labels"] = {"train": f"Training (before {c})", "valid": f"Validation (before {c})",
                           "test": f"Test ({test} {c} or later)"}
    cfg["test_label"] = f"temporal test set: {test} {c} or later, models trained on earlier visits"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="defaults to config.yaml next to this file")
    ap.add_argument("--mimic-dir", help="folder containing hosp/ and icu/")
    ap.add_argument("--ed-dir", help="folder containing ed/")
    ap.add_argument("--out")
    ap.add_argument("--sepsis3", action="store_true", help="also build the Sepsis-3 label (reads chartevents)")
    ap.add_argument("--data-only", action="store_true", help="stop after the data stage")
    ap.add_argument("--audit", action="store_true", help="leakage checks + rules preview on train/validation")
    ap.add_argument("--sepsis3-check", action="store_true",
                    help="locked models vs triage rules with Sepsis-3 as the outcome (after --open-test)")
    ap.add_argument("--open-test", action="store_true", help="evaluate the saved models on the test split")
    ap.add_argument("--reopen-test", action="store_true", help="override the test lock (report that you did)")
    ap.add_argument("--temporal", action="store_true",
                    help="temporal validation in out/temporal/ (train on earlier visits, test on later ones)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    mimic = a.mimic_dir or cfg["paths"]["mimic_dir"]
    ed = a.ed_dir or cfg["paths"]["ed_dir"]
    main_out = Path(cfg["paths"]["out_dir"])
    out = Path(a.out or main_out)
    if a.out:  # only the main run fills the repository's results/ folder
        cfg["paths"]["results_dir"] = str((out / "results").resolve())
    if a.temporal:
        if a.audit or a.sepsis3_check or a.sepsis3:
            sys.exit("--temporal works with step 1 and --open-test only.")
        main_out, out = out, out / "temporal"
        if not a.out:
            cfg["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]) / "temporal")
        temporal_labels(cfg)
    out.mkdir(parents=True, exist_ok=True)
    bundle_path, lock_path = out / "models" / "bundle.joblib", out / "test_lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else None

    if a.audit:
        if not bundle_path.exists():
            sys.exit("No saved models: run `python run.py` first.")
        from sepsis_ed.audit import run_audit
        print("Audit on train + validation (the test split is not read)")
        t0 = time.time()
        run_audit(pd.read_parquet(out / "analysis_table.parquet"), joblib.load(bundle_path), ed, out, cfg)
        print(f"\ndone in {time.time() - t0:.0f}s; files: {out}/audit_*.csv")
        return

    if a.sepsis3_check:
        if not (bundle_path.exists() and lock and lock["bundle_sha256"] == sha256(bundle_path)):
            sys.exit("Run `python run.py --open-test` first: the Sepsis-3 check uses the locked models.")
        from sepsis_ed.report import publish
        from sepsis_ed.sepsis3_check import run_sepsis3_check
        print("Sepsis-3 check with the locked models (nothing is retrained)")
        t0 = time.time()
        run_sepsis3_check(pd.read_parquet(out / "analysis_table.parquet"), joblib.load(bundle_path),
                          mimic, ed, out, cfg)
        publish(out, cfg)
        print(f"\ndone in {time.time() - t0:.0f}s; files: {out}/sepsis3_*.csv")
        return

    if a.open_test:
        if not bundle_path.exists():
            sys.exit("No saved models: run `python run.py` first.")
        digest = sha256(bundle_path)
        if lock and lock["bundle_sha256"] != digest and not a.reopen_test:
            sys.exit(f"CHECK FAILED: the test set was opened on {lock['opened_at']} with different models. "
                     "Re-opening it with retrained models is tuning on the test set. Use --reopen-test only if "
                     "you will report it.")
        table = pd.read_parquet(out / "analysis_table.parquet")
        bundle = joblib.load(bundle_path)
        from sepsis_ed.report import evaluate_test, table1  # heavy imports only when needed
        print("Evaluating locked models on the test split")
        t0 = time.time()
        table1(table, cfg).to_csv(out / "table1_characteristics.csv", index=False)
        evaluate_test(table, bundle, out, cfg)
        opened = lock["opened_at"] if lock and lock["bundle_sha256"] == digest else datetime.now().isoformat(
            timespec="seconds")
        history = (lock or {}).get("history", []) + [{"at": datetime.now().isoformat(timespec="seconds"),
                                                       "bundle_sha256": digest}]
        lock_path.write_text(json.dumps({"opened_at": opened, "bundle_sha256": digest, "history": history},
                                        indent=2))
        print(f"done in {time.time() - t0:.0f}s; see {out}/findings.md")
        return

    if lock and not a.data_only and not a.reopen_test:
        sys.exit(f"CHECK FAILED: the test set was opened on {lock['opened_at']}. Retraining now would be tuning "
                 "after seeing test results. Use --data-only, or --reopen-test if you will report it.")

    if a.temporal:
        table, sets, ok, cases_msg = temporal_stage(cfg, mimic, main_out, out)
    else:
        table, sets, ok, cases_msg = data_stage(cfg, mimic, ed, out, a.sepsis3)
    if a.data_only:
        return
    if not ok:
        print(f"\nStopping after the data stage: {cases_msg}.")
        return

    print("\nDevelopment (train + validation only)")
    t0 = time.time()
    bundle = develop(table, sets, cfg)
    for name, res in bundle["tuning"].items():
        res.to_csv(out / f"tuning_{name}.csv", index=False)
    rep = validation_report(bundle, table[table.split == "valid"])
    rep.to_csv(out / "validation_performance.csv", index=False)
    bundle_path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, bundle_path)
    print(rep[["model", "outcome", "auroc", "auprc", "prevalence"]].round(3).to_string(index=False))
    print(f"\nmodels saved to {bundle_path} ({time.time() - t0:.0f}s). "
          "Next, once, when every modelling choice is final: python run.py "
          + ("--temporal " if a.temporal else "") + "--open-test")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as e:
        sys.exit(f"CHECK FAILED: {e}")
