# SepsisSense extension protocol: language-model representations of the triage chief complaint

| | |
|---|---|
| **Version / date** | 0.1 draft for team review, 28 November 2026 |
| **Status** | Pre-specified. Commit before any chief-complaint experiment and before any v2 test-set evaluation |
| **Parent study** | SepsisSense v1 (Paper 1), commit `b24375e` |
| **Frozen comparator** | `out/models/bundle.joblib`, SHA-256 `7dcb8a1e…6654d51` (from `out/test_lock.json`); temporal comparator `out/temporal/models/` |
| **Investigators** | `[TO INSERT]` |
| **Reporting** | TRIPOD+AI [11], TRIPOD-LLM [12] |

Open items for the team are marked **[TEAM TO AGREE]**. After commit, every change goes in the deviation log (Section 10).

---

## 1. Rationale

SepsisSense v1 predicts sepsis from triage data alone (MIMIC-IV-ED linked to MIMIC-IV [3–4]; 409,216 adult visits, 2.5% sepsis). On the locked patient-level test set (n = 61,690), XGBoost reached AUROC 0.925 and AUPRC 0.360, against 0.81 for the best rule (modified NEWS [2]). It held up against a clinical Sepsis-3 [1] label (AUROC 0.85 vs 0.59–0.68 for rules) and in temporal validation (AUROC 0.920). Two weak spots remain: AUROC was lowest at age 65–79 (0.892), and the high-risk tier caught only 30% of septic patients aged 18–44, compared with 56% overall.

In v1 the chief complaint enters only as 15 regex keyword groups (`config.yaml: complaint_groups`). Free text has added signal in earlier triage work [5–8], and embeddings group similar complaints well [9], including at the hospital behind MIMIC [5]. Older septic patients often present atypically (weakness, falls, confusion). Consciousness is not a structured field in MIMIC-IV-ED, so the complaint is the only triage-time source for it. Young adults tend to stay compensated on vital signs, which leaves the complaint carrying more of the early signal.

**Gap.** No study has compared keyword, bag-of-words, LLM-extracted and embedding versions of the same complaint field against the same frozen model, with an ICD label and a Sepsis-3 label, and tested whether the text closes subgroup gaps identified in advance.

## 2. Hypotheses

| ID | Hypothesis | Type |
|---|---|---|
| H1 | Arm C improves AUPRC over Arm A on the test set | **Primary** |
| H2 | The gain is larger at age ≥65 than at <65 | Secondary |
| H3 | Arm C raises the share of septic patients aged 18–44 placed in the high-risk tier | Secondary |
| H4 | The gain holds against the Sepsis-3 label | Secondary |
| H5 | The gain holds in temporal validation | Secondary |
| H6 | Arm C does not widen gaps across age band, sex or race | Safety |
| E1 | Performance of Arms B, D, E | Exploratory |

## 3. Data, cohort and outcomes (all unchanged from v1)

- **Cohort and splits.** The 409,216 v1 visits and the stored patient-level split in `out/splits.csv` (70/15/15). A missing complaint becomes `"[none]"` plus an indicator.
- **Primary outcome.** A sepsis code on the ED stay or the linked admission: ICD-10 A40, A41, R65.20, R65.21; ICD-9 038, 995.91, 995.92, 785.52.
- **Sepsis-3 label** (check only, `sepsis_ed/sepsis3.py`). Suspected infection means a culture up to 72 h before an antibiotic or up to 24 h after it. The window-worst SOFA from 48 h before to 24 h after must be ≥2, and the suspected-infection time must fall within 48 h of ED arrival.
- **Temporal validation.** Visit year is estimated as the midpoint of `anchor_year_group` (±1 year). The test period is patients first seen in 2017 or later (n = 87,069), and patients seen earlier are excluded.

## 4. Text processing

Source: `triage.chiefcomplaint`, which is short, upper case and full of abbreviations, often lists several complaints, and contains de-identification tokens (`___`).

1. Lower-case, remove `___`, strip punctuation except `,` and `/`, and collapse whitespace.
2. Expand abbreviations from a dictionary built **from training strings only** (`sepsis_ed/text/abbrev_v1.csv`; e.g. SOB, AMS, N/V, s/p, abd).
3. Keep negation in the text.
4. Compute every representation **once per unique normalised string**, cache it under a key of string, method and version, and join it back to visits.

Raw text stays in `checks.forbidden_columns`. Only the derived features reach a model.

## 5. Comparison arms

All retrained arms use v1's non-text features (set B minus the `cc_*` groups), XGBoost [10] with the v1 search (25 random draws, 5-fold patient-grouped CV, scored on AUPRC), early stopping on validation, isotonic calibration and traffic-light cut-offs fitted on validation. Only the complaint representation changes.

| Arm | Complaint representation |
|---|---|
| **A** | v1 keyword groups. Frozen v1 model, not retrained |
| **B** | TF-IDF on word 1–2-grams and character 3–5-grams, then truncated SVD (k ∈ {32, 64, 128}, chosen on validation) |
| **C** | LLM-extracted fields (below), **primary candidate** |
| **D** | Frozen sentence embeddings (Bio_ClinicalBERT [13], SapBERT [14], MedCPT [15], one general encoder), mean-pooled, then PCA (k ∈ {16, 32, 64}). Encoder and k chosen on validation AUPRC |
| **E** | C + D |

Arms B–E replace the keyword groups. A sensitivity analysis adds C to them instead.

**Arm C schema.** `infection_source` (respiratory / urinary / abdominal / skin-soft tissue / CNS / device-line / other / none), plus 0/1 fields: `fever_or_rigors`, `altered_mental_status`, `weakness_fall_unwell`, `syncope`, `dyspnoea`, `poor_intake_or_vomiting`, `hypotension_reported`, `immunosuppression`, `referred_abnormal_result`, `transfer`. A separate flag, `outcome_term_present` (sepsis, septic, r/o sepsis), is used only in the leakage analysis and is never a feature.

**Arm C settings.** A local open-weights instruction model (2–3 candidates, about 7–14B parameters) runs at temperature 0 with a fixed seed and JSON-schema-constrained output. A string that fails validation is retried once, then all its fields are set to 0 and it is flagged. The model is chosen by macro-F1 on the annotation development set only. Prompt, schema, model and weights hash are frozen in `config.yaml` under a new `text:` block.

## 6. Annotation substudy

- **Sample.** 600 unique training-split strings: the 200 most frequent plus 400 at random from the rest. 200 are used for prompt development and 400 are held out.
- **Labelling.** Two clinicians label independently, blinded to outcome and LLM output, and a third adjudicates. Agreement is reported as Cohen's κ per field.
- **Acceptance.** A field enters Arm C only if its held-out F1 is ≥0.80 **[TEAM TO AGREE]**. Dropped fields are reported. The v1 keyword groups are scored on the same labels where they map to a field.

## 7. Leakage controls

1. Dictionary, vocabulary, SVD/PCA, encoder, prompt and model selection are fitted on train or validation only. A synthetic-data test (`tests/`) asserts this.
2. Test strings are never inspected during development.
3. Complaints that name the outcome are legitimate triage information and stay in the primary analysis. As a sensitivity analysis, the analysis is re-run with the v1 audit exclusions (`audit.exclude_for_sensitivity`: sepsis, bacteremia, blood culture), and the share of the gain attributable to those complaints is reported.
4. H4 guards against the model learning coding behaviour from the ICD label instead of clinical signal.
5. `python run.py --audit` is re-run for each arm.

## 8. Evaluation and decision rule

Metrics and code are the same as in v1 (`sepsis_ed/evaluate.py`).

- **Primary analysis.** Arm C minus Arm A ΔAUPRC on the test set, with 95% CI from 1,000 patient-level paired bootstrap resamples. ΔAUROC is reported the same way.
- **Also reported.** Cases found at equal flag rate against SIRS, qSOFA, modified NEWS ≥5 and ESI 1–2. High tier (top 5%) and high + moderate tiers (top 20%) case capture. Calibration slope, intercept and curve [16]. Decision curves over thresholds of 1–20% [17]. Fairness across the v1 age bands (18–44, 45–64, 65–79, 80+), sex and race. Text can carry differences in how staff describe groups [18].
- **Multiplicity.** H2–H5 are tested hierarchically in order. Each counts as confirmatory only if all earlier ones held. H6 is always reported. E1 is exploratory.
- **Adoption rule.** Arm C becomes SepsisSense v2 only if (1) the ΔAUPRC CI is above 0, (2) calibration slope is between 0.90 and 1.10, (3) no fairness subgroup has a ΔAUROC CI entirely below 0, and (4) the Sepsis-3 ΔAUROC point estimate is ≥0. If D or E beats C, it is reported as a ceiling and not adopted.
- **Power.** Before any v2 test evaluation, a simulation on validation estimates the smallest detectable ΔAUPRC. The expected value is 0.01–0.02.
- **Explainability.** SHAP cross-checked against permutation importance, as in v1. Arm C fields are shown by name. For B, D and E, text SHAP values are summed into a single "chief complaint" contribution.

## 9. Test-set use and governance

The v1 test set was opened once for Paper 1. The `open_test` history in `out/test_lock.json` shows a second run on the same bundle hash, and Paper 1 should report it. For v2:

1. Commit this protocol and record the hash.
2. Do all development on train and validation, then freeze the config, prompt, schema, dictionary, model hashes and cache hash, and tag `v2.0-rc`.
3. Write all v2 outputs to `out/v2/` with its own `test_lock.json`, as `--temporal` already does. Run `--open-test` once for all arms together. Anything after that is reported as post hoc.
4. For temporal validation, re-fit all text processing on pre-2017 data and open the temporal test set once.

**Data.** All LLM and encoder inference runs locally, as the PhysioNet DUA and its LLM guidance require [19]. The string cache stays inside the approved environment and out of git. Code, prompt, schema and dictionary are released, with no MIMIC text or row-level outputs. Reported counts below 10 are suppressed. The app falls back to the v1 keyword groups for an uncached complaint unless the local LLM is available, and it shows which path it used. It is a research prototype, not for clinical use.

## 10. Deviation log

| Date | Section | Change | Reason | Before/after test opened |
|---|---|---|---|---|
| | | | | |

---

## References

1. Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):801–810.
2. Royal College of Physicians. *National Early Warning Score (NEWS) 2*. London: RCP; 2017.
3. Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Sci Data*. 2023;10:1.
4. Johnson A, et al. MIMIC-IV-ED (version 2.2). PhysioNet. 2023. doi:10.13026/5ntk-km72
5. Horng S, et al. Creating an automated trigger for sepsis clinical decision support at emergency department triage using machine learning. *PLoS One*. 2017;12(4):e0174708.
6. Brann F, et al. Sepsis prediction at emergency department triage using natural language processing. *JMIR AI*. 2024;3:e49784.
7. Mishra L, et al. Automated sepsis prediction from unstructured electronic health records using natural language processing. *BMJ Health Care Inform*. 2025;32(1):e101354.
8. Sax DR, et al. Automated analysis of unstructured clinical assessments improves emergency department triage performance. *JACEP Open*. 2023;4(4):e13003.
9. Chang D, Hong WS, Taylor RA. Generating contextual embeddings for emergency department chief complaints. *JAMIA Open*. 2020;3(2):160–166.
10. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. *Proc KDD*. 2016:785–794.
11. Collins GS, et al. TRIPOD+AI statement. *BMJ*. 2024;385:e078378.
12. Gallifant J, et al. The TRIPOD-LLM reporting guideline. *Nat Med*. 2025;31(1):60–69.
13. Alsentzer E, et al. Publicly available clinical BERT embeddings. *Proc ClinicalNLP Workshop*. 2019:72–78.
14. Liu F, et al. Self-alignment pretraining for biomedical entity representations. *Proc NAACL-HLT*. 2021:4228–4238.
15. Jin Q, et al. MedCPT: contrastive pre-trained transformers for zero-shot biomedical information retrieval. *Bioinformatics*. 2023;39(11):btad651.
16. Van Calster B, et al. Calibration: the Achilles heel of predictive analytics. *BMC Med*. 2019;17:230.
17. Vickers AJ, Elkin EB. Decision curve analysis. *Med Decis Making*. 2006;26(6):565–574.
18. Obermeyer Z, et al. Dissecting racial bias in an algorithm used to manage the health of populations. *Science*. 2019;366(6464):447–453.
19. PhysioNet. Use of MIMIC data with large language models and online services. 24 September 2025. https://physionet.org/news/post/llm-responsible-use/
