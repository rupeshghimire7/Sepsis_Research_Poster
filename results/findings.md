# Findings (test set, 95% CI from patient bootstrap)

- **Who was studied?** Of 409,216 adult ED visits, 10,092 (2.5%) had sepsis recorded.
- **RQ1.** Using triage data alone, XGBoost reached AUROC 0.925 (0.919 to 0.932) and AUPRC 0.360 (0.335 to 0.388), against a sepsis rate of 2.4%.
- **RQ2 (models).** XGBoost exceeded logistic regression by 0.006 AUROC (95% CI for XGBoost minus LR 0.004 to 0.009).
- **RQ2, SIRS ≥ 2.** Flagging the same 5.3% of arrivals, the model found 58% of sepsis cases and the rule 27% (difference 31.7 percentage points, 95% CI 28.7 to 34.7).
- **RQ2, qSOFA ≥ 2.** Flagging the same 0.7% of arrivals, the model found 18% of sepsis cases and the rule 9% (difference 8.8 percentage points, 95% CI 6.4 to 11.1).
- **RQ2, NEWS ≥ 5.** Flagging the same 2.9% of arrivals, the model found 44% of sepsis cases and the rule 27% (difference 17.0 percentage points, 95% CI 14.2 to 19.7).
- **RQ2, Acuity ESI 1–2.** Flagging the same 37.7% of arrivals, the model found 96% of sepsis cases and the rule 72% (difference 24.1 percentage points, 95% CI 21.7 to 26.6).
- **Context.** Adding complaint, arrival and history to vitals changed AUROC from 0.887 to 0.925.
- **Calibration.** After calibration the slope was 1.02 (0.98 to 1.06) and the Brier score 0.0189.
- **RQ3.** The strongest drivers were Age, Shock index and Arrival: walk in (SHAP vs permutation rank agreement ρ = 0.76).
- **RQ4.** AUROC ranged from 0.892 (age 65–79) to 0.940 (age 18–44). Observed/expected cases ranged from 0.79 (race Hispanic/Latino) to 1.19 (age 18–44). The high-risk tier caught the smallest share of cases for age 18–44 (30%). Groups suppressed (< 10 cases): 0.
- **RQ5.** The high-risk tier was 5% of arrivals and held 56% of sepsis cases, at 3.6 reviews per case found (28 cases per 100 alerts).
