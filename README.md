# ED sepsis risk from triage — MIMIC-IV-ED + MIMIC-IV

Predict, **at ED triage**, whether a patient will meet Sepsis-3 criteria within
`--horizon` hours (default 48h) of arrival. Labels come from MIMIC-IV
(hosp + icu); features come only from MIMIC-IV-ED triage + demographics.

## Layout expected

```
data/
  mimic-iv-clinical-database-demo-2.2/   # --mimic-dir
    hosp/  patients admissions prescriptions microbiologyevents labevents diagnoses_icd
    icu/   chartevents inputevents         (optional but improves SOFA)
  mimic-iv-ed-demo-2.2/                  # --ed-dir
    ed/    edstays triage vitalsign pyxis diagnosis
```
`.csv.gz` or `.csv` both work. The same code runs on the full credentialed
release; labevents/chartevents are read in chunks and filtered by itemid.

```bash
pip install pandas scikit-learn
python run.py --mimic-dir data/mimic-iv-clinical-database-demo-2.2 \
              --ed-dir   data/mimic-iv-ed-demo-2.2 --out out
```
Outputs in `out/`: `labels.csv`, `dataset.csv`, `metrics_<target>.csv`,
`oof_predictions_<target>.csv`.

Smoke test without real data: `python tests/make_synthetic.py syn && python run.py --mimic-dir syn/mimic --ed-dir syn/ed`

## Label: Sepsis-3 onset (`sepsis3`)

Linkage is by `subject_id` + time, not only `hadm_id`, because ED cultures and
ED Pyxis antibiotics often have no `hadm_id`. MIMIC date-shifting is
consistent per patient across modules, so times are comparable.

1. **Suspected infection time (t_si)** — systemic antibiotic (hosp
   `prescriptions` + ED `pyxis`, matched by GSN code *or* drug name) paired with
   any culture (`microbiologyevents` + ICU culture orders in `chartevents`):
   culture ≤72h before antibiotic → t_si = culture time; culture ≤24h after
   antibiotic → t_si = antibiotic time.
2. **SOFA** — worst value per component in [t_si − 48h, t_si + 24h], summed,
   baseline assumed 0:
   coag (platelets 51265), liver (bilirubin 50885), renal (creatinine 50912),
   respiratory (PaO2 50821 ÷ FiO2 50816/223835 within 4h), cardiovascular
   (MAP from ED vitals + 220052/220181/225312; vasopressors from inputevents as
   norepinephrine-equivalent dose),
   CNS (GCS 220739+223900+223901).
3. **Onset** = earliest t_si with SOFA ≥ 2; stay is positive if onset ∈
   [ED intime, intime + horizon].

Secondary label `sepsis_icd`: A40*, A41*, R65.2* (ICD-10) or 038*, 995.91/92,
785.52 (ICD-9) on the ED stay or linked admission — useful as a sensitivity
check, known to under-count sepsis relative to Sepsis-3.

### Known simplifications
- Window-worst SOFA, not hourly rolling SOFA; baseline SOFA assumed 0 (chronic
  organ dysfunction, e.g. CKD, will create false positives).
- No urine output in renal; respiratory 3–4 not gated on ventilation.
- Cardiovascular: any vasopressor → 3, NE-equivalent > 0.1 mcg/kg/min → 4
  (Goradia 2021 factors), rather than per-drug SOFA dose bands.
- GCS only exists for ICU stays; ED has none.
- ED stays not admitted have little downstream data → effectively labelled 0.
  Consider restricting to admitted stays as a sensitivity analysis.
- Triage BP also feeds the label's cardiovascular score — intended (sepsis
  present on arrival is still a case to flag), but note it when reporting.

## Borrowed from MIMIC-Sepsis (Huang et al., BHI 2025, MIT licence)
github.com/yongh7/MIMIC-sepsis is an ICU-anchored benchmark: it keeps only ICU
stays that already meet Sepsis-3 and predicts outcomes (mortality, LOS, septic
shock, vasopressor need). It has no non-septic patients and no ED stays, so it
cannot supply our ED-level 1/0 label directly. Reused pieces:
- 126 First DataBank GSN antibiotic codes (originally AI_Clinician).
- 27 ICU culture-order itemids from `chartevents`.
- Vasopressor unit handling for NE-equivalents (factors swapped for the
  Goradia 2021 consensus; the repo's phenylephrine ×0.45 and vasopressin ×5
  overstate dose).
Not reused: its onset loop (iterates antibiotics in file order, not time
order, so "first" onset isn't guaranteed earliest), its SOFA CV score (returns
2 when MAP < 65 before checking pressors, and scores missing components as 0
silently), and the PostgreSQL requirement. Its `init_preprocess.py` also
imports a `utils` module that isn't in the repo.

Natural second stage: for ED patients we flag who go on to ICU sepsis, its
4-hour trajectory format and septic-shock / vasopressor tasks fit as a
downstream deterioration model.

## Features (triage-time only, no leakage)
Age, sex, temp, HR, RR, SpO2, SBP, DBP, pain, ESI acuity, MAP, shock index,
partial SIRS and partial qSOFA (no mental status at triage), chief-complaint
keyword flags, arrival mode, count of prior admissions before this arrival.

## Model
Logistic regression (median impute + missingness indicators, scaled, balanced)
and histogram GBM, evaluated with `StratifiedGroupKFold` grouped by
`subject_id` (a patient never appears in train and test). Reports AUROC,
AUPRC, Brier, prevalence.

## Demo-data caveat
The demo has 100 patients and only a few hundred ED stays, so expect a
handful of Sepsis-3 positives. Treat demo metrics as a pipeline check only;
the code prints a warning when positives < 30.

## Next steps once full access lands
- Temporal (by anchor_year_group) hold-out instead of random CV.
- Calibration plots and decision-curve analysis.
- Compare against qSOFA / NEWS2 computed at triage as clinical baselines.
- Free-text chief complaint embeddings instead of keyword flags.
