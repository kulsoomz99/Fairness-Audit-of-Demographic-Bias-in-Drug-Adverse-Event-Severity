# Demographic Bias in Drug Adverse Event Severity Prediction
### A Fairness Audit of a Pharmacovigilance Machine Learning Model Using Real FDA Reporting Data

**Student:** Kulsoom Zaidi  
**Email ID:** kulsoom.zaidi@studio.unibo.it  
**Course:** Ethics in Artificial Intelligence,  University of Bologna, Master in Artificial Intelligence  
**Data:** FDA FAERS Q1 2026, [FDA FAERS Portal](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html)

---

## Overview

The FDA Adverse Event Reporting System (FAERS) is one of the world's largest post-market drug safety databases. Machine learning models trained on this data risk encoding and amplifying historical demographic biases women and elderly patients were systematically underrepresented in clinical trials for decades, producing skewed adverse event records.

This project implements a full end-to-end fairness audit pipeline for an XGBoost classifier trained to predict serious vs non-serious drug adverse events. The audit covers three fairness metrics (Demographic Parity, Equalised Odds, Predictive Parity), an ablation study to distinguish data-level from model-level bias, intersectional analysis, SHAP-based explainability, and two bias mitigation strategies.

**Key Finding:** Female patients experience a statistically significant False Negative Rate of 48.3% versus 19.4% for male patients (gap = 28.9 pp, p < 0.0001). Young Adult Female patients face a catastrophic FNR of 78.6% nearly four in five serious adverse events missed. Both findings are confirmed as data-level bias through ablation analysis.

---

## Project Structure

```
Ethics_Project/
│
├── ascii/                                    # Raw FAERS Q1 2026 files (excluded from git)
│   ├── DEMO26Q1.txt                          # Patient demographics
│   ├── OUTC26Q1.txt                          # Outcome codes
│   ├── REAC26Q1.txt                          # Adverse reaction terms (MedDRA)
│   └── DRUG26Q1.txt                          # Drug names and roles
│
├── notebooks/
│   ├── 01_data_exploration.ipynb             # Load, clean, merge, EDA, base rate analysis
│   ├── 02_Train_test_split.ipynb             # 60/20/20 split, encoding, model training
│   ├── 03_Fairness_audit.ipynb               # DP, EO, PP metrics + Fisher exact tests
│   ├── 04_Shap_analysis.ipynb                # SHAP values, feature importance, waterfall
│   └── 05_Mitigation_strategies.ipynb        # Reweighing + per-group threshold
│
│
├── plots/                                    # All generated visualisations
│   ├── 01_dataset_overview.png
│   ├── 02_serious_by_sex.png
│   ├── 03_serious_by_age.png
│   ├── 04_intersectional_heatmap.png
│   └── 05_top_drugs_reactions.png
│
├── report/
│   ├── FinalReport.pdf                       
│
├── README.md
└── requirements.txt
```

---

## The Five Notebooks

| Notebook | Purpose | Key Output |
|---|---|---|
| `01_data_exploration.ipynb` | Load FAERS files, clean, merge, EDA, base rate plots | `df_clean.csv` |
| `02_train_test_split.ipynb` | 60/20/20 stratified split, encoding, XGBoost training, ablation study, K-Fold, hyperparameter tuning | `model_xgb_tuned.pkl`, `test_df_with_preds.csv` |
| `03_fairness_audit.ipynb` | Demographic Parity, Equalised Odds, Predictive Parity, Fisher tests, impossibility theorem, intersectional analysis | `fairness_summary.csv` |
| `04_shap_analysis.ipynb` | SHAP TreeExplainer, global importance, sex/age group analysis, waterfall plot for false negative | `shap_values.npy` |
| `05_mitigation_strategies.ipynb` | Reweighing (Kamiran & Calders), per-group threshold adjustment, final comparison | `mitigation_comparison.csv` |

---

## Key Results

### Model Performance

| Model | AUC | Recall | FNR | Precision |
|---|---|---|---|---|
| LR + reaction | 0.582 | 0.541 | 0.458 | 0.608 |
| LR - reaction | 0.567 | 0.541 | 0.458 | 0.597 |
| XGB + reaction | 0.708 | 0.553 | 0.447 | 0.714 |
| XGB - reaction | 0.644 | 0.653 | 0.346 | 0.637 |
| **XGB tuned - reaction (PRIMARY)** | **0.651** | **0.651** | **0.348** | **0.632** |

### Fairness Audit (Primary Model: by Sex)

| Metric | Female | Male | Gap | Violated |
|---|---|---|---|---|
| Pred rate (DP) | 0.421 | 0.738 | 0.317 | Yes |
| FNR (EO) | 0.483 | 0.194 | 0.289 | Yes |
| FPR (EO) | 0.316 | 0.641 | 0.324 | Yes |
| Precision (PP) | 0.641 | 0.643 | 0.002 | No |

Fisher's exact test: FNR gap p < 0.0001, FPR gap p < 0.0001

### Intersectional Finding

| Subgroup | FNR |
|---|---|
| Female Young Adult | **0.786** - most harmed |
| Male Elderly | **0.115** - least harmed |
| Intersectional gap | **0.671** |

### Mitigation Results

| Model | AUC | FNR F | FNR M | FNR Gap |
|---|---|---|---|---|
| Baseline | 0.651 | 0.483 | 0.194 | 0.289 |
| Reweighing | 0.646 | 0.308 | 0.338 | 0.031 |
| Per-group threshold | 0.651 | 0.327 | 0.335 | 0.008 |

Both mitigations bring all gaps below the 0.05 practical significance threshold. No mitigation achieves exact equality - consistent with Chouldechova's (2017) impossibility theorem.

---

## Reproducibility

### Prerequisites

```
Python 3.10+
Google Colab (recommended) or Jupyter Notebook
```

### Dataset Setup

Download the four FAERS Q1 2026 ASCII files from the FDA:

```
https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
```

Place the files in your Google Drive:

```
MyDrive/Ethics project/ascii/
├── DEMO26Q1.txt
├── OUTC26Q1.txt
├── REAC26Q1.txt
└── DRUG26Q1.txt
```

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `scipy`

### Execution Order

Notebooks must be run in order: each notebook loads artefacts saved by the previous one via Google Drive:

```
01_data_exploration.ipynb
        ↓  saves: df_clean.csv
02_train_test_split.ipynb
        ↓  saves: model_xgb_tuned.pkl, test_df_with_preds.csv, X_test_B.npy ...
03_fairness_audit.ipynb
        ↓  saves: fairness_summary.csv, intersectional_results.csv
04_shap_analysis.ipynb
        ↓  saves: shap_values.npy, shap_summary.csv
05_mitigation_strategies.ipynb
           saves: mitigation_comparison.csv, test_df_final.csv
```

All random seeds are fixed at `RANDOM_SEED = 42`.


## Methodology Summary

### Target Variable

Adverse events are labelled `serious = 1` if any of the following ICH E2A outcome codes appear in the OUTC file:

| Code | Meaning |
|---|---|
| DE | Death |
| LT | Life-threatening condition |
| HO | Hospitalisation |
| DS | Disability (permanent) |
| CA | Congenital anomaly |
| RI | Required medical intervention |

All other outcomes (`OT = Other`) are labelled `serious = 0`. This follows the ICH E2A guideline definition adopted by the FDA and EMA, under which SAEs require expedited reporting within 15 days.

### Ablation Study

Two model variants were trained:
- **Model A** : with reaction feature: `[age_years, sex_binary, age_group_enc, reaction_enc, drug_enc]`
- **Model B** : without reaction feature: `[age_years, sex_binary, age_group_enc, drug_enc]`

Reaction terms such as "cardiac arrest" and "death" are causally downstream of the target label, risking near-tautological predictions. The ablation confirms that the demographic FNR gap persists without reaction, proving the bias is data-level.

### Fairness Metrics

- **Demographic Parity (Independence):** Equal prediction rates across groups
- **Equalised Odds (Separation):** Equal TPR and FPR across groups
- **Predictive Parity (Sufficiency):** Equal precision across groups
- **Practical significance threshold:** 0.05 (Feldman et al., 2015)

### Mitigations

- **Reweighing** (pre-processing): Sample weights computed as Expected/Observed frequency per sex × outcome group (Kamiran & Calders, 2012)
- **Per-group threshold** (post-processing): Separate decision thresholds for female and male patients selected on the validation set to equalise FNR

---

## Ethical Context

Under the **EU AI Act (2024)**, drug safety monitoring AI is classified as high-risk, requiring bias testing, documentation, and human oversight before deployment. The baseline model's 78.6% FNR for Young Adult Females would likely fail non-discrimination requirements under Article 9. This project demonstrates the necessity of intersectional fairness auditing as a baseline compliance requirement for pharmacovigilance AI systems.

---

## References

- Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data*, 5(2), 153-163.
- Cirillo, D. et al. (2020). Sex and gender differences in AI for biomedicine. *NPJ Digital Medicine*, 3(81).
- Kamiran, F. & Calders, T. (2012). Data preprocessing for classification without discrimination. *KAIS*, 33(1), 1-33.
- Harpaz, R. et al. (2012). Novel data-mining for adverse drug event discovery. *CPT*, 91(6).
- European Parliament. (2024). Regulation (EU) 2024/1689 - The AI Act.
- ICH E2A. (1994). Clinical Safety Data Management: Expedited Reporting Guidelines.
