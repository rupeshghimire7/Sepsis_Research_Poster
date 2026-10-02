# Early Risk Stratification for Sepsis and Clinical Deterioration in NSW Emergency Departments
### A Machine Learning Research Proposal Building on the SAFE-WAIT Program

---

## Abstract

**Background:** Sepsis remains a leading cause of preventable in-hospital death, and each hour of delayed antibiotic treatment has been associated with roughly a 4–8% rise in mortality [14]. Emergency department (ED) waiting rooms are a critical blind spot, since deterioration often begins before a clinician has reviewed the patient. NSW Health's SAFE-WAIT program addresses this gap with a real-time XGBoost model that assigns a traffic-light sepsis risk score during triage. A retrospective evaluation of 108,401 Western Sydney ED presentations found the model correctly flagged 86.5% of patients who went on to develop septic shock [2], and moderate- and high-risk patients received antibiotics roughly 59 and 117 minutes sooner, respectively, than under the standard SIRS-based alert [2]. Backed by almost $500,000 of a $4.9 million NSW Government translational research investment, the program is expanding to five Western Sydney hospitals from 2027 [3,4].

**Aim:** To design and internally evaluate a machine learning pipeline for early sepsis/deterioration risk stratification that mirrors the SAFE-WAIT approach using open data, benchmarks it against traditional screening tools (SIRS/qSOFA/NEWS2), and situates it within NSW population and health-system data.

**Methods:** Using the Health Gym synthetic ICU sepsis dataset (GAN-generated from MIMIC-III, 2,164 patients) [6,7] alongside NSW Bureau of Health Information and HealthStats NSW indicators for epidemiological and system-load context [8,9], triage-relevant features (vitals, labs, comorbidities, demographics) will be engineered and used to train and compare logistic regression, random forest, and XGBoost classifiers via stratified cross-validation. Discrimination (AUROC/AUPRC), calibration, subgroup fairness, and SHAP-based interpretability will be assessed and benchmarked against published clinical-score performance.

**Expected outcomes:** Consistent with published meta-analyses, ensemble tree-based models — particularly XGBoost and random forest — are expected to outperform logistic regression and conventional clinical scores [11,12,13], while SHAP-based explanations should preserve the interpretability needed for triage-nurse decision support [21,22].

**Significance:** This project translates a live NSW Health research priority into a reproducible, ethically grounded ML pipeline, demonstrating how open synthetic data can prototype clinically deployable sepsis risk tools in a way consistent with the NSW AI Assurance Framework and national fairness-reporting expectations.

*(~290 words)*

---

## 1. Background and Significance

Sepsis is a dysregulated host response to infection that can progress to organ failure and death within hours. International estimates place the annual burden at tens of millions of cases and several million deaths, with delayed recognition being one of the strongest modifiable risk factors for mortality [14]. In Australia, the Australian Commission on Safety and Quality in Health Care (ACSQHC) introduced a national Sepsis Clinical Care Standard in 2022, and NSW has run the SEPSIS KILLS program since 2011, which measurably improved time-to-antibiotics and reduced sepsis-related mortality across the state [10,18].

Despite this, the ED waiting room remains a structural weak point: a nurse doing initial triage in a department seeing tens of thousands of presentations a year has only a few minutes to notice which of 40–60 waiting patients is quietly deteriorating [1]. This is precisely the problem NSW Health's **SAFE-WAIT** (Sepsis risk Artificial intelligence algorithm For Emergency department WAITing rooms) program was built to solve, and its recent translational funding signals that early risk stratification is now an explicit NSW Health priority rather than a purely academic exercise [3,4].

## 2. Literature Review

### 2.1 The NSW precedent: SAFE-WAIT

SAFE-WAIT is an XGBoost model trained on four years of Western Sydney ED data (SMEDSA initiative, 2017–2019) that generates a low/moderate/high sepsis-risk score from triage-nurse inputs and pre-existing comorbidities [2]. It has been piloted at Westmead Hospital since 2022 and formally evaluated against the existing SIRS-based electronic alert. In a cohort of 108,401 presentations between July 2022 and June 2024, the model assigned a risk category to 104,904 patients; 4.9% were later confirmed septic and 3% progressed to septic shock. SAFE-WAIT flagged 86.5% of the septic-shock group as high risk at triage, with a median time from triage to shock onset of around 92 minutes — underscoring how little time clinicians have to act [2]. Moderate- and high-risk patients received antibiotics significantly faster (β = −58.8 and −116.99 minutes respectively, p < 0.001), and high-risk patients were seen by a physician modestly sooner [2]. The project — formally "SAFE-WAIT SPACE to Sepsis Care" — has since received close to $500,000 of a $4.9 million NSW Health Translational Research Grants Scheme investment and will be evaluated across five EDs (Westmead, Auburn, Blacktown, Mount Druitt, Nepean) from 2027 [3,4,5]. This gives the present project an unusually strong real-world benchmark: any prototype model should be interpreted relative to SAFE-WAIT's demonstrated performance, not just to textbook AUROC figures.

### 2.2 Machine learning approaches to sepsis and deterioration prediction

A large and fast-growing literature has evaluated ML for sepsis prediction using electronic health record (EHR) and ICU data. A 2025 scoping review found ML-model AUROC values in sepsis prediction spanning roughly 0.5–0.9 across studies, with XGBoost repeatedly identified as a strong performer, though real-time deployments can carry a high false-alert burden if not carefully thresholded [11]. Two large systematic reviews/meta-analyses reinforce this: one covering 50 studies and 104 mortality-prediction models found pooled discrimination (C-index) of ~0.80, with random forest and XGBoost preferred over other algorithms for their accuracy given similar input variables, and identified lactate as the single most predictive — yet most frequently *omitted* — variable in traditional clinical scores [12]. A second, covering 23 studies and over 4.3 million ICU patients, found XGBoost and random forest achieving the strongest accuracy for predicting sepsis onset (up to 0.957 and 0.911 respectively in held-out test sets) [13]. Commonly used predictors across this literature converge on a similar core set: respiratory rate, temperature, systolic blood pressure, heart rate, white cell count, and lactate [11,13,21].

Beyond tabular gradient-boosted trees, deep learning approaches (LSTM, GRU, temporal convolutional networks) have also been applied, particularly to the PhysioNet/Computing in Cardiology 2019 Challenge, which defined a public benchmark task of predicting Sepsis-3 onset six hours in advance from 40,336 ICU stays across three US hospital systems [14,15]. This body of work is a useful methodological reference even though it is ICU-focused rather than ED-triage-focused.

More directly relevant is a smaller but growing set of studies applying interpretable ML specifically at ED triage. A 2025 *Scientific Reports* study built an interpretable gradient-boosting sepsis-risk model for ED triage data and used decision-curve analysis to show it delivered the greatest net clinical benefit at low probability thresholds — exactly the operating point needed for early, cautious flagging [21]. Related work on ED admission prediction has shown that population- and patient-level SHAP explanations can be layered onto triage models to support clinician trust without sacrificing discrimination [22]. This directly informs the interpretability component of the proposed pipeline.

### 2.3 Public and synthetic data resources

Because real ED/ICU data cannot be freely shared, this project relies on openly available **synthetic** data, primarily the **Health Gym** sepsis dataset: a GAN-generated (Wasserstein GAN), longitudinal dataset of vital signs, laboratory results, and administered fluids/vasopressors for 2,164 ICU sepsis patients, built from MIMIC-III using previously validated cohort-selection criteria, with an estimated identity-disclosure risk of just 0.045% [6,7]. It is hosted on PhysioNet and was explicitly designed for machine-learning prototyping and education, which makes it well suited to a time-limited research/competition setting. The related **PhysioNet/CinC 2019 Challenge** dataset (40,336 ICU patients, Sepsis-3-labelled) is a useful secondary benchmark for validating pipeline design choices against a much larger, published leaderboard [14,15].

For NSW-specific context — rather than model training — the **Bureau of Health Information (BHI)** publishes quarterly ED activity and performance data (attendance volumes, triage categories, time-to-treatment) through its Data Portal and *Healthcare Quarterly* reports [8], and **HealthStats NSW** provides population-level health-status and infection indicators [9]. These will not be used to train the classifier (they are not case-level clinical records) but will be used to benchmark plausible caseloads, discuss deployment feasibility, and frame the "why NSW, why now" narrative — for example, Westmead ED alone recorded more than 80,000 attendances in the last reporting year [1].

### 2.4 Equity, fairness, and governance considerations

A "high-end" review of this space cannot stop at discrimination metrics. Sepsis prediction models are well documented to risk encoding pre-existing disparities: a 2025 ethics-focused analysis found that social determinants of health and inconsistent labelling practices can distort model training data and disadvantage marginalised patients unless explicitly audited [15]. Related work on "fairness drift" shows that even well-validated sepsis alerts can degrade unevenly across subgroups over time — one US sepsis model saw a 43% jump in false alerts when the COVID-19 pandemic shifted the underlying patient population, and had to be temporarily disabled [16]. Methodologically, recent work argues that clinical fairness should be evaluated through *subgroup net benefit* rather than naïve parity metrics, since forcing equal performance across groups can "level down" a model's utility for everyone [17].

This is especially relevant in the NSW/Australian context: Aboriginal and Torres Strait Islander peoples experience substantially higher sepsis incidence than the non-Indigenous population (age-adjusted hospital admission rates of 40.8 vs 11.8 per 1,000 in one Northern Territory cohort) and disproportionately higher ICU mortality from invasive infection in children [18]. Any model proposed for NSW deployment should therefore explicitly report subgroup performance and flag where synthetic or historical training data may under-represent these populations, in line with the ACSQHC standard's own equity and cultural-safety provisions [18].

Finally, any AI tool intended for real NSW Health use — even a research prototype positioned as a stepping stone toward one — sits inside a defined governance structure. The **NSW AI Assessment Framework (AIAF)**, mandatory under Circular DCS-2024-04, requires agencies to document risk, mitigations, and (for high-risk systems) escalate to the NSW AI Review Committee [19]. In 2026, NSW Health additionally launched its own **AI Framework**, covering seven priority areas (consumers, workforce, privacy/security, governance, safety/ethics/quality, research and development, and industry) and an AI Advisory Service that reviewed the SAFE-WAIT project itself [2,20]. Framing this research explicitly against that governance structure — rather than as a standalone technical exercise — is itself a differentiator for a NSW-focused submission.

## 3. Research Gap and Aims

SAFE-WAIT demonstrates that ED-waiting-room sepsis prediction works in NSW practice, but as a live clinical program its underlying training data and code are not publicly reproducible, and its published evaluation (to date) emphasises clinical process outcomes (time-to-antibiotics) over model-development detail (feature engineering choices, calibration, subgroup fairness) [2]. This leaves room for a complementary, openly reproducible project that:

1. Builds a transparent, end-to-end ML pipeline (baseline → ensemble) on **open** synthetic ICU sepsis data, since real ED data is not accessible in this setting;
2. Benchmarks that pipeline against both the published ML literature (Section 2.2) and SAFE-WAIT's real-world results (Section 2.1), being explicit about where an ICU-trained model would and would not transfer to an ED waiting-room context;
3. Adds an explicit fairness and interpretability layer, addressing a gap in most benchmark studies (Section 2.4); and
4. Grounds the whole exercise in real NSW system data (BHI, HealthStats NSW) so the resulting poster/abstract speaks to *NSW capacity and demand*, not just model accuracy in the abstract.

## 4. Proposed End-to-End Research Design

**4.1 Data sources**
- *Primary modelling data:* Health Gym synthetic sepsis dataset (2,164 ICU patients, time-series vitals/labs/interventions) [6,7], obtained via free PhysioNet registration.
- *Secondary benchmark data:* PhysioNet/CinC 2019 Challenge dataset, used to sanity-check feature engineering and model behaviour against a much larger published cohort [14,15].
- *Contextual data (not for training):* BHI Healthcare Quarterly / Data Portal (ED volumes, triage-category mix, time-to-treatment) [8]; HealthStats NSW (population health and infection indicators) [9]; published SAFE-WAIT results as an external, real-world performance anchor [2].

**4.2 Feature engineering**
Derive triage-relevant, single-time-point and short-window features from the time-series data: latest and worst-in-window vital signs (heart rate, respiratory rate, systolic BP, temperature, SpO₂), key labs (WBC, lactate where available, creatinine), simple comorbidity flags, and demographics. This mirrors the "single time-point, non-invasive vitals" design shown to work well in recent multi-centre XGBoost sepsis studies [32 in web literature / see Section 2.2], and keeps the feature set close to what a real triage nurse could enter in under a minute — a design constraint SAFE-WAIT itself follows [1,2].

**4.3 Model development**
Train and compare, in increasing complexity:
1. **Logistic regression** (interpretable baseline, with L2 regularisation);
2. **Random forest**;
3. **XGBoost** (matching SAFE-WAIT's own architecture, enabling a like-for-like comparison) [2,11,12,13].
Class imbalance (sepsis being a minority outcome) will be handled via class weighting and/or SMOTE applied only within training folds, and hyperparameters tuned via stratified k-fold cross-validation.

**4.4 Evaluation strategy**
- Discrimination: AUROC and AUPRC (AUPRC is prioritised given class imbalance);
- Calibration: reliability curves / Brier score;
- Clinical utility: decision-curve analysis at low probability thresholds, following the ED-triage precedent in [21];
- Benchmarking: compare against SIRS/qSOFA/NEWS2-style rule-based scores computed on the same data, and against published AUROC/accuracy ranges from the meta-analyses in Section 2.2 [11,12,13].

**4.5 Interpretability and fairness**
- Global and per-patient SHAP values to identify top risk drivers, framed for a triage-nurse audience rather than a data-science audience [21,22];
- Subgroup performance breakdown (at minimum by age band and sex, and by any available proxy for disadvantage) using a subgroup net-benefit lens rather than strict parity [17], with findings reported transparently rather than optimised away.

**4.6 NSW contextualisation**
Overlay model outputs with BHI/HealthStats NSW figures to estimate what a given sensitivity/specificity trade-off would mean in practice — e.g. expected daily "high-risk" flags per ED of a given size, using Westmead's >80,000 annual attendances as a reference point [1,8] — and discuss how the pipeline's outputs and documentation would need to be extended (real ED-waiting-room data, prospective validation, NSW AIAF risk assessment) before it could be considered for a real pilot [19,20].

## 5. Feasibility, Scope, and Timeline

This design is intentionally scoped to be achievable in a short research/competition window: the modelling work is feature engineering and comparison of three well-understood, fast-training algorithms (logistic regression, random forest, XGBoost) on a dataset of a few thousand patients, rather than building new sensors or novel architectures. A realistic breakdown:
- Days 1–2: data access/registration, exploratory analysis, feature engineering;
- Days 3–4: baseline and ensemble model training, cross-validation, benchmarking against clinical scores;
- Day 5: SHAP interpretability, subgroup fairness check, NSW contextualisation figures;
- Final day(s): poster/abstract assembly and narrative framing against SAFE-WAIT.

## 6. Limitations

- **Domain mismatch:** Health Gym is an *ICU* cohort, not an *ED waiting-room* cohort; physiological patterns and prevalence will differ from SAFE-WAIT's triage population, so absolute performance figures are not directly comparable to SAFE-WAIT's — this should be stated explicitly in any poster, not glossed over.
- **Synthetic data:** GAN-generated data preserves distributional and correlational structure but is not real patient data; it is appropriate for prototyping and educational benchmarking, not for claims of clinical validity [6,7].
- **Retrospective, not prospective:** as with most of the reviewed literature, this design evaluates predictive discrimination on held-out data rather than real-time clinical impact (time-to-antibiotics, mortality), which is what actually matters and is what SAFE-WAIT's own evaluation measured [2].
- **Contextual, not linked, NSW data:** BHI/HealthStats NSW figures provide system-level context but cannot be linked at the patient level to the synthetic cohort, so they inform narrative and feasibility discussion only, not model training.

## 7. Ethical and Governance Considerations

All data used are open, de-identified/synthetic, and require no new ethics approval for this prototyping exercise; however, the write-up should still explicitly (a) name the domain-mismatch and synthetic-data limitations above, (b) report subgroup performance rather than only aggregate metrics, and (c) frame any "real-world deployment" language against the actual NSW AI Assessment Framework and NSW Health AI Framework processes that a genuine pilot would have to pass through [19,20], mirroring how SAFE-WAIT itself was governed under WSLHD ethics approval and the NSW AI Assurance Approach [2].

## 8. Expected Contribution and Significance

Sepsis recognition is a stated NSW Health research and funding priority right now, evidenced by SAFE-WAIT's translational grant and staged five-hospital rollout [3,4,5]. A project that (a) reproduces the core modelling logic behind that program on transparent, open data, (b) benchmarks it rigorously against both the ML literature and SAFE-WAIT's own published results, and (c) foregrounds fairness and NSW governance rather than treating them as an afterthought, offers a timely, life-saving-adjacent, and distinctly NSW-grounded contribution — while remaining feasible to deliver as analysis-plus-poster in a short timeframe.

---

## References

1. NSW Government (WSLHD Pulse). *Using AI to recognise sepsis in the ED*, 2026. https://www.nsw.gov.au/departments-and-agencies/wslhd/pulse/all-news/ed-sepsis-ai
2. Kabil G, Shetty A, et al. *Evaluation of a real time machine learning sepsis risk algorithm for Emergency Department waiting rooms (SAFE-WAIT)*. medRxiv, 2025. https://www.medrxiv.org/content/10.1101/2025.06.23.25330154v1
3. Hollingworth D. *Medtech: NSW hospitals to trial AI model to detect sepsis faster*. Cyber Daily, Sep 2026. https://www.cyberdaily.au/digital-transformation/14180-medtech-nsw-hospitals-to-trial-ai-model-to-detect-sepsis-faster
4. NSW Government. *World Sepsis Day highlights new AI tool identifying sepsis quicker than before*, 2026. https://www.nsw.gov.au/ministerial-releases/new-ai-tool-identifying-sepsis
5. The Indian Sun. *AI tool to flag sepsis risk in Western Sydney emergency rooms*, 2026. https://www.theindiansun.com.au/2026/09/15/ai-tool-to-flag-sepsis-risk-in-western-sydney-emergency-rooms/
6. Kuo N, Finfer S, Jorm L, et al. *Synthetic Acute Hypotension and Sepsis Datasets Based on MIMIC-III and Published as Part of the Health Gym Project*, v1.0.0. PhysioNet. https://physionet.org/content/synthetic-mimic-iii-health-gym/1.0.0/
7. Kuo N, et al. *The Health Gym: synthetic health-related datasets for the development of reinforcement learning algorithms*. Scientific Data, 2022. https://www.nature.com/articles/s41597-022-01784-7
8. Bureau of Health Information NSW. *Healthcare in Focus / Data Portal*. https://www.bhi.nsw.gov.au/
9. NSW Ministry of Health. *HealthStats NSW / Emergency Department data sources*. https://www.health.nsw.gov.au/data/sources/Pages/Emergency-Department.aspx
10. Clinical Excellence Commission NSW. *SEPSIS KILLS program*. https://cec.health.nsw.gov.au/keep-patients-safe/sepsis/program
11. *Machine Learning and Deep Learning Models for Early Sepsis Prediction: A Scoping Review*. PMC, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12186070/
12. *Machine learning for the prediction of sepsis-related death: a systematic review and meta-analysis*. PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10712076/
13. *Predicting sepsis onset in ICU using machine learning models: a systematic review and meta-analysis*. BMC Infectious Diseases, 2023. https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-023-08614-0
14. Reyna MA, et al. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*. Critical Care Medicine 48(2):210-217, 2019. https://physionet.org/content/challenge-2019/1.0.0/
15. *Mitigating Bias in Machine Learning Models with Ethics-Based Initiatives: The Case of Sepsis*. American Journal of Bioethics / PMC, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12353398/
16. *Emerging algorithmic bias: fairness drift as the next dimension of model maintenance and sustainability*. JAMIA, 2025. https://academic.oup.com/jamia/article/32/5/845/8074959
17. *Understanding algorithmic fairness for clinical prediction in terms of subgroup net benefit and health equity*. arXiv, 2024. https://arxiv.org/pdf/2412.07879
18. Australian Commission on Safety and Quality in Health Care. *Sepsis Clinical Care Standard*, 2022. https://www.safetyandquality.gov.au/standards/clinical-care-standards/sepsis-clinical-care-standard
19. Digital NSW. *NSW AI Assessment Framework*. https://www.digital.nsw.gov.au/policy/artificial-intelligence/ai-governance-assurance-and-frameworks/nsw-ai-assessment-framework
20. Agency for Clinical Innovation NSW. *AI Framework*, 2026. https://aci.health.nsw.gov.au/ai-framework
21. *Interpretable machine learning for predicting sepsis risk in emergency triage patients*. Scientific Reports, 2025. https://www.nature.com/articles/s41598-025-85121-z
22. *A Global and Local SHAP-Driven Interpretable Framework for Early Hospital Admission Prediction With Machine Learning at Emergency Department Triage*. PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC13380983/

---

*Prepared as a research/competition proposal. All clinical and system-level figures above are drawn from the cited public sources; no patient-level real-world data was accessed or used in preparing this document.*
