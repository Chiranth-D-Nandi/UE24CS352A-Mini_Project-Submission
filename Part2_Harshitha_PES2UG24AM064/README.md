# HitPredict — Part 2: Machine Learning Modeling, Calibration, Evaluation & Streamlit Demo Application

**Course:** UE24CS352A - Machine Learning Mini-Project
**Team Member 1:** Chiranth D Nandi (PES2UG24AM048) — *Data Engineering & Ingestion Lead*
**Team Member 2:** Harshitha (PES2UG24AM064) — *ML Modeling & Application Lead*

---

## 📌 Executive Summary & Part 2 Scope

This package represents **Part 2 (50% codebase share)** of the HitPredict mini-project. It consumes the clean, leakage-controlled dataset produced by Part 1 and implements the complete modeling, hyperparameter optimization, probability calibration, metric evaluation, sensitivity analysis, and interactive Streamlit reviewer demo.

### Key Components Built by Harshitha (PES2UG24AM064):
1. **Candidate Model Architectures (`src/hitpredict/models.py`):**
   - **Baseline:** Dummy prevalence classifier.
   - **Linear:** Calibrated Logistic Regression with StandardScaler.
   - **Decision Tree:** Depth-controlled diagnostic tree.
   - **Random Forest:** Bagged ensemble (`n_estimators=250`, `max_depth=14`) capturing non-linear audio interactions.
   - **Histogram Gradient Boosting:** LightGBM-style tabular tree booster.
   - **Multi-Layer Perceptron (MLP):** 2-layer neural network with early stopping.
2. **Training & Calibration Pipeline (`src/hitpredict/training.py`):**
   - Class imbalance mitigation via 10:1 negative undersampling on 1990–2014 training partition.
   - Model selection on 2015–2016 validation period using **PR-AUC (Average Precision)**.
   - **Sigmoid Probability Calibration** fitted on the 2017 partition to align predicted probabilities with natural catalog prevalence (~0.28%).
   - Final evaluation on the held-out 2018–2020 test period.
3. **Comprehensive Metric Suite & Analysis (`src/hitpredict/metrics.py`, `src/hitpredict/reporting.py`):**
   - Precision-Recall curves, ROC curves, Brier score, balanced accuracy, F1 score.
   - Threshold optimization on validation set.
   - Bootstrapped 95% confidence intervals.
   - Top false-positive and false-negative error analysis.
4. **Interactive Streamlit Reviewer Demo (`app/app.py`):**
   - Web application for live inference on user-specified tracks.
   - Interactive audio feature sliders and one-at-a-time sensitivity explainability plots.
5. **Unit Test Suite:**
   - Tests covering evaluation metrics, balanced subsets, candidate builders, and inference contracts (`tests/test_metrics.py`, `tests/test_validation_and_model.py`).

---

## 🛠️ Package Structure

```text
app/                     Streamlit live reviewer demo application
artifacts/               Trained and calibrated Random Forest pipeline (final_model.joblib)
configs/                 Model configurations and feature subset definitions
docs/                    Model selection report, decision log, results slides, writeup draft
reports/
  ├── figures/           PR curves, ROC curves, calibration curves, confusion matrix
  └── metrics/           Final test evaluation, model comparison, feature importance, top FP/FN
src/hitpredict/
  ├── __init__.py        Package initialization
  ├── cli.py             Command-line interface for Part 2
  ├── config.py          Configuration parsing
  ├── constants.py       Feature definitions and schemas
  ├── inference.py       Production predictor and sensitivity explainability
  ├── metrics.py         Classification metrics, thresholding, bootstrap CIs
  ├── models.py          Model candidates (LR, DT, RF, HistGB, MLP)
  ├── reporting.py       Figure generation and error analysis
  ├── training.py        Training, validation selection, and calibration engine
  ├── utils.py           Serialization helpers
  └── data/
      └── validation.py  Input schema verification for inference
tests/                   Unit test suite for metrics and models
```

---

## 🚀 Getting Started

### 1. Installation
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[app,dev]"
```

### 2. Run Model Training and Evaluation
```bash
hitpredict train --config configs/default.yaml --profile quick
```

### 3. Launch Streamlit Live Review Demo
```bash
streamlit run app/app.py
```

### 4. Run Unit Tests
```bash
pytest -q
```

---

## 📊 Key Results on 2018–2020 Test Cohort
- **Winning Model:** Random Forest Classifier (PR-AUC: **0.101**, compared to prevalence baseline **0.005**).
- **Operating F1 Score:** 0.171 (Recall: 30.6%, Precision: 11.9%).
- **Calibrated Brier Score:** 0.0048.
- **Top Predictive Audio Features:** Danceability, Valence, Energy, Loudness (along with Prior Artist Hits).

---

## 🤝 Project Division
- **Part 1 (Chiranth D Nandi - PES2UG24AM048):** Data Engineering, Acquisition, Normalization, Linkage, Audit & Tests.
- **Part 2 (Harshitha - PES2UG24AM064):** ML Modeling, Training, Probability Calibration, Metric Evaluation & Streamlit Demo UI.
