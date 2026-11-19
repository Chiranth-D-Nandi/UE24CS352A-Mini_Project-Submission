# Team Contribution Record

**Course:** UE24CS352A - Machine Learning Mini-Project  
**Project:** HitPredict — Leakage-Aware Prediction of Billboard Hot 100 Hits using Spotify Audio Features  
**Team Composition:**
- **Member 1:** Chiranth D Nandi (PES2UG24AM048) — *Data Engineering & Ingestion Lead*
- **Member 2:** Harshitha (PES2UG24AM064) — *ML Modeling & Application Lead*

---

## 50 / 50 Equal Codebase & Work Breakdown

| Component | Member 1: Chiranth D Nandi (PES2UG24AM048) [50%] | Member 2: Harshitha (PES2UG24AM064) [50%] |
|---|---|---|
| **Primary Domain** | **Data Engineering, Linkage & Leakage Audit** | **ML Modeling, Calibration & Streamlit UI** |
| **Data Ingestion** | Billboard Hot 100 automated download (`acquire.py`), SHA-256 integrity validation, Kaggle Spotify ingestion | Data schema validation and feature vector preparation |
| **Text Normalization** | Comprehensive title and artist canonicalization engine (`normalization.py`) handling aliases, featuring artists, and remasters | Feature scaling and categorical feature alignment |
| **Linkage Pipeline** | Exact string matching (`build_dataset.py`, `labels.py`) preventing false-positive labels | Linkage audit review and target distribution verification |
| **Leakage Controls** | Strict prior-year artist hit accumulation (`prior_artist_hits`), reissue deduplication, and chronological partitioning (1990–2014 / 2015–2016 / 2017 / 2018–2020) | Pre-model feature availability confirmation and inference schema enforcement |
| **Model Architecture** | Baseline prevalence dummy model specification | Logistic Regression, Decision Tree, Random Forest, Histogram Gradient Boosting, and MLP architectures (`models.py`) |
| **Training Pipeline** | Class imbalance negative sampling design (10:1) | Hyperparameter tuning, cross-validation, and model fitting (`training.py`) |
| **Calibration & Eval** | Pre-modeling data audit (`audit.py`, `validation.py`), range verification | Sigmoid probability calibration on 2017 partition, PR-AUC / ROC-AUC / F1 evaluation, single test reveal (`metrics.py`) |
| **Explainability & Demo** | Match audit reporting, unmatched Billboard analysis | Sensitivity perturbation analysis (`inference.py`), error analysis (FP/FN), and Streamlit Web Application (`app/app.py`) |
| **Testing & Quality** | Tests for normalization, labels, and dataset builder (`test_normalization.py`, `test_labels.py`, `test_build_dataset.py`) | Tests for metrics, model pipeline, and inference contract (`test_metrics.py`, `test_validation_and_model.py`) |
| **Documentation & Slides** | Dataset Card, Leakage Audit, Architecture, Part 1 Technical Report | Model Selection Report, Decision Log, Slide Deck content, Final Two-Page Summary |

---

## Member 1 (Chiranth D Nandi) Completed Work Evidence
- `src/hitpredict/data/acquire.py`: SHA-256 verified download of UT Data Billboard archive.
- `src/hitpredict/data/normalization.py`: Full regex and unicode cleaning for titles/artists.
- `src/hitpredict/data/labels.py`: Weekly Hot 100 aggregation to song entity.
- `src/hitpredict/data/build_dataset.py`: Exact linkage with strictly earlier artist hit counts.
- `src/hitpredict/data/audit.py`: Pre-model leakage detection and physical range auditing.
- `tests/test_normalization.py`, `tests/test_labels.py`, `tests/test_build_dataset.py`: 100% passing test coverage.
