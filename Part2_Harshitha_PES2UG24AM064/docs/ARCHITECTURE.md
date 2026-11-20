# End-to-End Architecture

## Decision

Use a single Python/scikit-learn pipeline with CSV acquisition, exact record linkage, explicit year-blocked evaluation, a serialized sklearn estimator, and a thin Streamlit UI. Do not add a database, API service, feature store, GPU training, MLflow, or orchestration platform unless the measured project needs change.

## Layers

```text
DATA
Spotify CSV ── schema validation ── identity/audio cleaning ─┐
                                                            ├─ exact linkage ─ dedupe ─ temporal partitions
Billboard CSVs ─ hashes ─ weekly-to-song aggregation ───────┘

ML
partitions ─ preprocessing Pipeline ─ baselines/candidates ─ validation selection
           ─ selected refit ─ sigmoid calibration ─ threshold selection ─ artifact

EVALUATION
artifact + untouched test ─ PR-AUC/ROC-AUC/F1/calibration/confusion matrix
                          ─ bootstrap intervals ─ errors ─ year/artist segments
                          ─ post-selection permutation importance

APPLICATION
manual descriptor form ─ validation ─ saved Pipeline.predict_proba
                       ─ thresholded decision ─ local sensitivity ─ limitations

DEPLOYMENT
private GitHub repository ─ reproducible local demo (primary)
                          └ optional Streamlit Community Cloud (if visibility permits)
```

## Interface contracts

### Prepared data

One row per `song_group`; required metadata: `id`, `name`, `artists`, `year`, `split`, target; model columns come only from named feature sets in YAML. Duplicate `song_group` values fail validation.

### Artifact

`artifacts/final_model.joblib` is a dictionary containing the calibrated sklearn model, exact ordered feature names, threshold, reference values, model metadata, test metrics, and limitations. The UI never reimplements scaling, encoding, or imputation.

### Generated evidence

Every candidate run writes an experiment-ledger row. Final evaluation writes both natural-prevalence and 1:1 comparison-only metrics. The test is read for final evaluation once; later error/importance analyses are post-selection and cannot alter the chosen design.

## Folder responsibilities

| Path | Belongs here | Does not belong here |
|---|---|---|
| `configs/` | declared split, features, metric, sampling policy | results or secrets |
| `src/hitpredict/data/` | source validation, linkage, cleaning, audit | model tuning |
| `src/hitpredict/` | reusable ML/evaluation/inference code | notebook-only logic |
| `app/` | presentation and calls to inference contract | duplicated preprocessing/training |
| `tests/` | deterministic behavior and interface contracts | fabricated benchmark values |
| `data/` | README and local ignored files | committed Kaggle/raw chart data |
| `artifacts/` | generated model locally/deployment | source code |
| `reports/` | generated metrics/plots/audits | hand-edited “results” |
| `docs/` | methodology and decisions | secrets or large raw extracts |

## Deployment debate

**Selected for demo:** local Streamlit, with Community Cloud as optional publication.

- **Why:** pure Python, same process as inference, minimal integration failure, small classical artifact, clear live controls.
- **Why not a separate FastAPI backend:** no second client currently consumes it; it doubles failure surfaces.
- **Why not AWS/Azure/GCP:** billing/IAM/container setup adds no grading value for a short demo.
- **Why not GPU hosting:** inference is a small tabular model on CPU.
- **Constraint:** assignment calls for a private GitHub repository; free public hosting can conflict. Keep a tested local demo and screen recording fallback.

## Security controls

- `.env`, Streamlit secrets, raw data, artifacts, and generated reports are ignored.
- No API key is needed at training or inference time.
- Joblib is loaded only from the repository's own trusted artifact path; never accept user-uploaded pickle/joblib files.
- User input is numeric and range-validated; no shell/database execution exists.
- Pin dependencies and scan before submission (`pip-audit` may be added as a reviewer check, not a runtime dependency).
