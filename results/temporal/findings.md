# Findings (temporal test set: patients first seen 2017 or later, models trained on earlier visits, 95% CI from patient bootstrap)

- **Who was studied?** Of 87,069 adult ED visits in the temporal test set, 2,434 (2.8%) had sepsis recorded.
- **RQ1.** Using triage data alone, XGBoost reached AUROC 0.920 (0.915 to 0.925) and AUPRC 0.364 (0.344 to 0.386), against a sepsis rate of 2.8%.
- **RQ2 (models).** XGBoost exceeded logistic regression by 0.003 AUROC (95% CI for XGBoost minus LR 0.001 to 0.005).
- **RQ2, SIRS ≥ 2.** Flagging the same 5.6% of arrivals, the model found 58% of sepsis cases and the rule 26% (difference 32.3 percentage points, 95% CI 30.1 to 34.6).
- **RQ2, qSOFA ≥ 2.** Flagging the same 0.6% of arrivals, the model found 14% of sepsis cases and the rule 7% (difference 7.4 percentage points, 95% CI 5.9 to 8.9).
- **RQ2, NEWS ≥ 5.** Flagging the same 2.7% of arrivals, the model found 40% of sepsis cases and the rule 22% (difference 18.1 percentage points, 95% CI 15.9 to 20.1).
- **RQ2, Acuity ESI 1–2.** Flagging the same 40.0% of arrivals, the model found 96% of sepsis cases and the rule 71% (difference 24.9 percentage points, 95% CI 23.0 to 26.8).
- **Context.** Adding complaint, arrival and history to vitals changed AUROC from 0.885 to 0.920.
- **Calibration.** After calibration the slope was 0.91 (0.87 to 0.95) and the Brier score 0.0215.
- **RQ3.** The strongest drivers were Age, Shock index and Arrival: walk in (SHAP vs permutation rank agreement ρ = 0.70).
- **RQ4.** AUROC ranged from 0.860 (age 80+) to 0.940 (race Asian). Observed/expected cases ranged from 0.94 (vitals One or more missing) to 1.41 (age 18–44). The high-risk tier caught the smallest share of cases for age 18–44 (33%). Groups suppressed (< 10 cases): 0.
- **RQ5.** The high-risk tier was 6% of arrivals and held 59% of sepsis cases, at 3.6 reviews per case found (28 cases per 100 alerts).
