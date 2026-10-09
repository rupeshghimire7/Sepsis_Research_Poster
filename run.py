"""End-to-end: Sepsis-3 labels -> triage features -> cross-validated baselines.

python run.py --mimic-dir data/mimic-iv-clinical-database-demo-2.2 \
              --ed-dir data/mimic-iv-ed-demo-2.2 --out out --horizon 48
"""
import argparse
from pathlib import Path

from sepsis_ed.features import build_features
from sepsis_ed.labels import build_labels
from sepsis_ed.model import cross_validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-dir", required=False, help="folder containing hosp/ and icu/")
    ap.add_argument("--ed-dir", required=False, help="folder containing ed/")
    ap.add_argument("--out", default="out")
    ap.add_argument("--horizon", type=float, default=48, help="hours after ED arrival for sepsis onset")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    mimic_dir = a.mimic_dir or 'data/mimic-iv'
    ed_dir = a.ed_dir or 'data/mimic-iv-ed'

    labels = build_labels(mimic_dir, ed_dir, a.horizon)
    labels.to_csv(out / "labels.csv", index=False)
    feats_df, feats = build_features(mimic_dir, ed_dir)
    df = feats_df.merge(labels.drop(columns="subject_id"), on="stay_id")
    df.to_csv(out / "dataset.csv", index=False)

    n = len(df)
    print(f"ED stays: {n} | patients: {df.subject_id.nunique()} | admitted: {df.admitted.sum()}")
    print(f"Sepsis-3 within {a.horizon:g}h: {df.sepsis3.sum()} ({df.sepsis3.mean():.1%})"
          f" | sepsis ICD: {df.sepsis_icd.sum()} ({df.sepsis_icd.mean():.1%})"
          f" | both: {(df.sepsis3 & df.sepsis_icd).sum()}")

    for target in ("sepsis3", "sepsis_icd"):
        res, note = cross_validate(df, feats, target)
        print(f"\n[{target}] {note}")
        if res:
            metrics, oof = res
            print(metrics.round(3).to_string())
            metrics.to_csv(out / f"metrics_{target}.csv")
            oof.to_csv(out / f"oof_predictions_{target}.csv", index=False)


if __name__ == "__main__":
    main()
