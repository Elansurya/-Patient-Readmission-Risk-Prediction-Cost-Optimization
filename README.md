# ReadmitRisk AI — Patient Readmission Risk Prediction + Cost Optimization

> Predicts 30-day hospital readmission probability and estimates avoidable readmission costs using XGBoost and Random Forest on 500K+ patient records — helping hospitals and health systems reduce readmission penalties and optimize post-discharge care planning.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen?style=flat-square)

🔗 **[Live Demo → Hugging Face Spaces]([https://huggingface.co/spaces/YourUsername/ReadmitRisk-AI](https://huggingface.co/spaces/Elansurya/Patient-Readmission-Risk-Prediction-Cost-Optimization))**

---

## Problem Statement

Hospitals in the US face CMS penalties of up to 3% of Medicare payments annually under the Hospital Readmissions Reduction Program (HRRP) for avoidable 30-day readmissions — costing the healthcare system over $26 billion per year. Traditional discharge protocols rely on clinician intuition and generic checklists, missing complex interaction patterns between diagnosis history, comorbidities, social determinants of health (SDOH), and prior utilization behavior.

This project builds an end-to-end ML pipeline that predicts the probability of 30-day readmission using patient clinical and demographic profiles — enabling care teams to prioritize high-risk patients for post-discharge interventions (follow-up calls, transitional care, remote monitoring) and estimate the financial impact of avoidable readmissions per patient cohort.

---

## Dataset

| Property | Detail |
|---|---|
| Total records | 500,000+ inpatient discharge records |
| Readmission types | 30-day all-cause, 30-day condition-specific (HF, COPD, Pneumonia, AMI, Hip/Knee) |
| Features | 34 variables — age, diagnoses (ICD-10), LOS, comorbidity index, prior admissions, discharge disposition, payer type, SDOH flags, etc. |
| Target variable | Readmitted within 30 days (1) / Not Readmitted (0) |
| Class distribution | 78% No Readmission / 22% Readmission (imbalanced) |
| Imbalance handling | SMOTE oversampling + class_weight balancing |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Data processing | Pandas, NumPy |
| ML Models | Random Forest, XGBoost |
| Hyperparameter tuning | RandomizedSearchCV (50 iterations) |
| Imbalance handling | SMOTE (imblearn) |
| Experiment tracking | MLflow 2.5 |
| Cost modeling | Custom cost-benefit module (avoidable cost per risk tier) |
| Deployment | Hugging Face Spaces (Gradio) |
| Evaluation | AUC-ROC, KS Statistic, Precision, Recall, F1-Score, Gini Coefficient |

---

## Workflow

```
Raw 500K+ Inpatient Discharge Records
        ↓
Data Cleaning & Preprocessing
  ├── Missing value imputation (median for numeric, mode for categorical)
  ├── Outlier treatment via IQR winsorization (LOS, prior admission count)
  ├── ICD-10 diagnosis grouping into clinical categories (CCS mapping)
  ├── Label encoding for discharge disposition, payer type, admission source
  └── Standard scaling for numerical features
        ↓
Feature Engineering
  ├── Comorbidity burden score    → Elixhauser/Charlson index from ICD codes
  ├── Prior utilization index     → admissions_last_12m / avg_los
  ├── Discharge risk flag         → AMA discharge or SNF refusal
  ├── SDOH vulnerability score    → homelessness + low_income + no_caregiver flags
  └── Readmission recency weight  → days_since_last_discharge (exponential decay)
        ↓
Class Imbalance Handling
  └── SMOTE → balanced training set (50:50)
        ↓
Model Training
  ├── Random Forest (n_estimators=200, max_depth=15)  — baseline
  └── XGBoost (n_estimators=300, learning_rate=0.05)  — optimized
        ↓
Hyperparameter Tuning
  └── RandomizedSearchCV (50 iterations, 5-fold CV)
        ↓
Cost Optimization Module
  ├── Estimated avoidable readmission cost per patient: $15,200 (CMS avg.)
  ├── Intervention cost per high-risk patient: $420 (transitional care bundle)
  └── Net savings per 1,000 patients at threshold 0.45: ~$2.1M
        ↓
MLflow Experiment Tracking
  └── Logged: params, AUC, KS, F1, cost-savings estimates, model artifacts (38 experiments)
        ↓
Deployment
  └── Hugging Face Spaces — real-time readmission risk scoring + cost impact dashboard
```

---

## Results

| Model | AUC-ROC | KS Statistic | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Random Forest (baseline) | 0.83 | 0.57 | 0.76 | 0.71 | 0.73 |
| Random Forest (tuned) | 0.86 | 0.62 | 0.79 | 0.75 | 0.77 |
| XGBoost (baseline) | 0.88 | 0.65 | 0.81 | 0.77 | 0.79 |
| **XGBoost (tuned)** | **🏆 0.92** | **🏆 0.71** | **0.85** | **0.82** | **0.83** |

**Gini Coefficient (XGBoost tuned): 0.84** — indicating strong discriminating power for clinical risk stratification

---

## Cost Optimization Output

| Risk Tier | Readmission Probability | Recommended Action | Est. Avoidable Cost/Patient |
|---|---|---|---|
| 🟢 Low | < 20% | Standard discharge | — |
| 🟡 Medium | 20% – 50% | Nurse follow-up call (48hr) | $15,200 |
| 🔴 High | > 50% | Transitional care program + remote monitoring | $15,200 |

**Projected annual savings (1,000-bed hospital, 30% high-risk flag rate):**
- Readmissions prevented (estimated): ~410 cases/year
- Gross avoidable cost saved: **~$6.2M/year**
- Intervention cost: **~$630K/year**
- **Net savings: ~$5.6M/year**

---

## Feature Importance (Top 6)

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | Prior admissions in last 12 months | 0.194 |
| 2 | Comorbidity burden score (Elixhauser) | 0.171 |
| 3 | Length of stay (current admission) | 0.138 |
| 4 | Discharge disposition (SNF / AMA / Home) | 0.119 |
| 5 | SDOH vulnerability score | 0.094 |
| 6 | Primary diagnosis category (CCS group) | 0.081 |

---

## MLflow Experiment Tracking

38 experiments tracked across model types, hyperparameter combinations, and decision thresholds:

```bash
# View all experiments locally
mlflow ui
# Opens at http://localhost:5000
```

Tracked per experiment:
- All hyperparameter values (n_estimators, max_depth, learning_rate, subsample)
- AUC-ROC, KS Statistic, F1-Score, Precision, Recall
- Decision threshold sensitivity (cost-optimal threshold vs. F1-optimal)
- Model artifacts (pickled .pkl files)
- Training duration and dataset hash

---

## Live Demo

🔗 **[Try ReadmitRisk AI on Hugging Face Spaces](https://huggingface.co/spaces/YourUsername/ReadmitRisk-AI)**

Input a patient discharge profile and get:
- 30-day readmission probability score (0–100%)
- Risk tier: 🟢 Low / 🟡 Medium / 🔴 High
- Top 3 clinical risk factors driving the prediction
- Estimated avoidable cost if readmitted
- Recommended post-discharge intervention

> **Screenshots:**
> Add these to a `/screenshots` folder:
> 1. `app_interface.png` — Hugging Face Spaces patient input form
> 2. `prediction_output.png` — Risk score + cost impact output screen
> 3. `feature_importance.png` — Feature importance bar chart
> 4. `mlflow_ui.png` — MLflow experiment comparison view
> 5. `cost_optimization.png` — Cost savings projection dashboard

![App Interface](screenshots/app_interface.png)
![Prediction Output](screenshots/prediction_output.png)
![Feature Importance](screenshots/feature_importance.png)

---

## Business Impact

- **Readmission penalty reduction:** Pre-identifies high-risk patients before discharge — directly reduces CMS HRRP penalty exposure for hospitals and health systems
- **KS Statistic of 0.71** exceeds the 0.40 industry benchmark for clinical risk models — indicating production-grade discriminating power
- **Cost optimization layer:** Translates ML predictions into actionable financial impact — estimated $5.6M+ net annual savings for a 1,000-bed hospital
- **Regulatory alignment:** MLflow experiment tracking creates auditable, reproducible model history — critical for CMS, HIPAA, and Joint Commission model governance requirements
- **Applicable to:** Hospital discharge planning, care management programs, ACO risk stratification, value-based care contracts, telehealth triage

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/ReadmitRisk-AI.git
cd ReadmitRisk-AI

# Install dependencies
pip install -r requirements.txt

# Start MLflow tracking server
mlflow ui

# Run Hugging Face app locally
python app.py
```

---

## Project Structure

```
ReadmitRisk-AI/
├── data_preprocessing.ipynb    # Cleaning, ICD grouping, SDOH encoding, feature engineering
├── model_training.ipynb        # RF + XGBoost training + tuning
├── evaluation.ipynb            # Metrics, ROC curves, KS plots, threshold analysis
├── cost_optimization.ipynb     # Cost-benefit modeling, savings projections
├── app.py                      # Hugging Face Spaces Gradio app
├── mlflow_experiments/         # Saved experiment runs
├── models/                     # Pickled best model
│   └── xgb_tuned_v3.pkl
├── requirements.txt
├── screenshots/
└── README.md
```

---

## Requirements

```
xgboost==1.7.6
scikit-learn==1.3.0
imbalanced-learn==0.11.0
mlflow==2.5.0
pandas==2.0.3
numpy==1.24.3
gradio==3.40.1
matplotlib==3.7.2
seaborn==0.12.2
shap==0.42.1
```

---

## Author

**Your Name** — Aspiring Data Scientist | ML · Python · SQL · Healthcare Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat-square&logo=github)](https://github.com/YourUsername)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow?style=flat-square)](https://huggingface.co/spaces/YourUsername/ReadmitRisk-AI)
