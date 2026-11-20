# Final Project Audit

Mark only after checking the real artifacts.

## Data

- [x] source URLs/hashes recorded
- [x] Spotify unknown-licence risk acknowledged; raw data not committed
- [x] Billboard MIT attribution retained
- [x] exact matches and unmatched sample manually reviewed
- [x] missing/invalid/outlier report reviewed
- [x] duplicate entities removed before split
- [x] temporal split counts and prevalence documented

## ML

- [x] dummy and Logistic Regression baselines exist
- [x] serious nonlinear candidates evaluated
- [x] test untouched until final design fixed
- [x] preprocessing fitted inside Pipeline
- [x] popularity/chart outcomes absent from model inputs
- [x] strict prior artist history verified
- [x] calibration and threshold use 2017 only
- [x] full-prevalence PR-AUC and confusion matrix reported
- [x] CIs, errors, temporal/artist segments reviewed
- [x] failed experiments retained

## Engineering

- [x] tests pass from clean environment
- [x] README commands reproduced
- [x] final artifact loads and predicts
- [x] no secrets/raw data/cache committed
- [x] dependency versions pinned
- [x] meaningful, truthful Git history

## Demo/deployment

- [x] valid track input produces probability/explanation
- [x] invalid/missing artifact fails clearly
- [x] model metadata and limitations visible
- [x] local demo tested immediately before review
- [x] hosted URL tested if used; local fallback ready

## Academic

- [x] official requirement mapping complete
- [x] PDF obeys confirmed page limit
- [x] all reported values trace to generated files
- [x] prior-work comparisons use matching targets/prevalence or state why not
- [x] contribution record reflects actual work
- [x] slides, live demo, Q&A rehearsal complete
