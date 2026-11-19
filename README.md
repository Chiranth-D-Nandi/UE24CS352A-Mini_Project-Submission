# HitPredict — Part 1: Data Engineering, Acquisition, Exact Linkage & Leakage Audit Pipeline

**Course:** UE24CS352A - Machine Learning Mini-Project  
**Team Member 1:** Chiranth D Nandi (PES2UG24AM048) — *Data Engineering & Ingestion Lead*  
**Team Member 2:** Harshitha (PES2UG24AM064) — *ML Modeling & Application Lead*  
**Repository:** [https://github.com/Chiranth-D-Nandi/UE24CS352A-Mini_Project-Submission/](https://github.com/Chiranth-D-Nandi/UE24CS352A-Mini_Project-Submission/)

---

## 📌 Executive Summary & Part 1 Scope

This repository represents **Part 1 (50% codebase share)** of the HitPredict machine learning project. It implements the complete, leak-free data engineering and pre-modeling verification pipeline.

Predicting whether a song will ever enter the Billboard Hot 100 using acoustic descriptors requires addressing subtle forms of data leakage. Part 1 constructs a sound, reproducible data pipeline ensuring zero temporal leakage and strict entity resolution.

### Key Components Built by Chiranth D Nandi (PES2UG24AM048):
1. **Automated Source Acquisition (`src/hitpredict/data/acquire.py`):**
   - Automated ingestion of MIT-licensed UT Data Billboard Hot 100 historical archives (`hot100_archive_1958_2021.csv` and `hot-100-current.csv`).
   - Pinned commit verification (`7037d25`) and SHA-256 hash auditing to guarantee provenance.
   - Generates `reports/audits/data_manifest.json`.
2. **Text Normalization Engine (`src/hitpredict/data/normalization.py`):**
   - Deterministic normalization of artist names and song titles: case folding, punctuation stripping, unicode decomposition.
   - Robust regex parsing to extract primary artists from collaborations (`feat.`, `featuring`, `with`, `vs.`, `&`).
   - Stripping reissue, remaster, live, and version artifacts to ensure sound entity resolution.
3. **Billboard Label Construction (`src/hitpredict/data/labels.py`):**
   - Collapsing over 340,000 weekly chart entries into unique song entities.
   - Exact extraction of debut year, peak rank, and chart history.
4. **Exact Linkage & Leakage Guard (`src/hitpredict/data/build_dataset.py`):**
   - High-precision exact string linkage between 1.2M Spotify audio tracks and Billboard records.
   - **Zero future leakage:** strictly calculating artist historical hit counts using only songs from strictly prior years (`prior_artist_hits`).
   - Chronological year-blocked dataset partitioning:
     - **Train:** 1990–2014 (10:1 negative undersampling)
     - **Validation:** 2015–2016 (full catalog)
     - **Calibration:** 2017 (full catalog)
     - **Test:** 2018–2020 (untouched future period)
5. **Data Audit & Quality Verification (`src/hitpredict/data/audit.py`, `src/hitpredict/data/validation.py`):**
   - Schema validation, physical range checks for all 14 Spotify audio descriptors.
   - Missingness audits and deduplication verification.
6. **Unit Test Suite:**
   - Passing test coverage for normalization, label extraction, and dataset builder (`tests/test_normalization.py`, `tests/test_labels.py`, `tests/test_build_dataset.py`).

---

## 🛠️ Repository Structure

```text
configs/                 Pipeline configuration and split definitions
docs/                    Architecture, dataset card, leakage audit, team contributions
src/hitpredict/
  ├── __init__.py        Package initialization
  ├── cli.py             Command-line interface for Part 1
  ├── config.py          Configuration parsing and directory setup
  ├── constants.py       Feature definitions, bounds, and schema
  ├── utils.py           SHA-256 hashing and I/O helpers
  └── data/
      ├── acquire.py         Source acquisition and integrity checks
      ├── audit.py           Pre-model data audit and leakage validation
      ├── build_dataset.py   Exact linkage and temporal split engine
      ├── labels.py          Billboard chart collapse and label creation
      ├── normalization.py   Text normalization engine
      └── validation.py      Schema and range verification
tests/                   Unit test suite for Part 1 components
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
pip install -e ".[dev]"
```

### 2. Download and Verify Billboard Archive
```bash
hitpredict acquire --config configs/default.yaml
```

### 3. Build Modeling Dataset
```bash
hitpredict prepare --config configs/default.yaml
```

### 4. Run Leakage and Data Audit
```bash
hitpredict audit --config configs/default.yaml
```

### 5. Run Unit Tests
```bash
pytest -q
```

---

## 🤝 Project Division
- **Part 1 (This Repository):** Chiranth D Nandi (PES2UG24AM048) — Data Engineering, Acquisition, Normalization, Linkage, Audit & Tests.
- **Part 2 (Delivered via Package):** Harshitha (PES2UG24AM064) — ML Modeling, Training, Probability Calibration, Metric Evaluation & Streamlit Demo UI.
