# Living Decision Log

## D001 — Problem contract

- **Question:** What is predicted and when?
- **Options:** Spotify popularity; Billboard inclusion; top-10 rank.
- **Evidence:** user-confirmed project title and uploaded HitPredict framing.
- **Chosen:** binary “ever appears on Billboard Hot 100,” using only pre-outcome inputs.
- **Risk:** “ever” needs historical follow-up and is association, not causality.
- **Validate:** target/linkage audit and explicit inference feature table.
- **Status:** accepted.

## D002 — Feature pool

- **Options:** live Spotify API; 114k licensed genre sample; Kaggle 1.2M static file.
- **Evidence:** official API restrictions; 114k lacks release date in exposed schema; 1.2M includes year/date and broad coverage but licence is Unknown.
- **Chosen:** 1.2M static file for private educational analysis; no redistribution.
- **Why not alternatives:** live API is unavailable to typical new apps; 114k prevents temporal split.
- **Validate:** schema/hash/audit; replace if licence requirements disallow use.
- **Status:** conditional on user-provided file and licence acceptance.

## D003 — Label source

- **Options:** mhollingshead JSON; Kaggle CC BY-NC-SA chart file; UT Data archive.
- **Evidence:** UT has raw CSV URLs and MIT repository licence; the first source had no explicit licence verified.
- **Chosen:** UT Data archive + current CSV.
- **Implementation:** automatic download, schema check, SHA-256 manifest.
- **Validate:** first/last date, unique weekly chart count, row schema.
- **Status:** accepted; acquisition tested.

## D004 — Linkage

- **Options:** exact raw string; normalized exact; fuzzy matching.
- **Evidence:** titles/artists vary by punctuation, features, remaster tags; fuzzy labels can create silent false positives.
- **Chosen:** normalized exact title plus at least one normalized artist alias; export unmatched/ambiguous records.
- **Why not fuzzy:** label precision is more important than inflated match coverage.
- **Validate:** manual accepted/unmatched sample and coverage; unit tests.
- **Status:** implemented; full-data audit executed.

## D005 — Independent unit and duplicate policy

- **Options:** Spotify track ID; album-track row; normalized song entity.
- **Chosen:** normalized title + primary artist entity; earliest release retained before splitting.
- **Why:** same recording appears on albums, compilations, and remasters; row split leaks identity.
- **Risk:** distinct same-title recordings by one artist can merge.
- **Validate:** duplicate count and match audit.
- **Status:** implemented.

## D006 — Time split

- **Options:** random stratified; group random; year-blocked.
- **Evidence:** HitPredict found time-period effects; source dates have mixed precision.
- **Chosen:** train ≤2014; select 2015–16; calibrate/threshold 2017; test 2018–20.
- **Why not random:** does not simulate future release prediction and can mix duplicate eras.
- **Validate:** hard year-boundary assertions.
- **Status:** implemented; counts verified with real data.

## D007 — Feature sets

- **Options:** audio only; audio + popularity; audio + artist history; identity fields.
- **Chosen:** compare `audio_only` against `audio_plus_artist_history`; strict previous years.
- **Rejected:** popularity and Billboard outcome fields are post-outcome leakage; names/IDs enable memorization.
- **Validate:** validation ablation and hard forbidden-feature list.
- **Status:** implemented; experiment executed.

## D008 — Metrics

- **Options:** accuracy, ROC-AUC, PR-AUC, F1.
- **Chosen:** PR-AUC primary on natural prevalence; thresholded metrics/calibration secondary; balanced accuracy report; 1:1 report comparison only.
- **Why not accuracy:** a rare-hit majority classifier can appear excellent.
- **Validate:** dummy-prior baseline and full confusion matrix.
- **Status:** implemented.

## D009 — Candidate tree and selection

- **Options:** LR, DT, RF, HistGB, RBF-SVM, MLP, transformer, ensembles.
- **Chosen:** dummy/LR/shallow-DT/RF/HistGB/small-MLP; predeclared validation selection; parsimony if PR-AUC is within 0.005.
- **Rejected:** default RBF-SVM scales poorly; transformer/CNN/RNN has no sequence/raw audio; stacking adds leakage/complexity before base evidence.
- **Validate:** generated experiment ledger, fit time, validation metrics.
- **Status:** code complete; results verified.

## D010 — Imbalance/calibration

- **Options:** all negatives; undersampling; oversampling/SMOTE; class weights.
- **Chosen:** deterministic year-matched negative sampling in fitting only; full-period validation/test; sigmoid calibration on natural 2017 prevalence.
- **Why not SMOTE:** synthetic interpolations can be unrealistic and leakage-prone if applied before split.
- **Validate:** calibration curve, Brier/log loss, predicted positive rate.
- **Status:** implemented.

## D011 — GPU

- **Chosen:** CPU.
- **Why:** small tabular vectors and classical models; two GPUs do not justify distributed complexity.
- **Reconsider if:** raw audio/spectrogram scope and measured CPU bottleneck emerge.
- **Status:** accepted.

## D012 — Application/deployment

- **Chosen:** saved sklearn Pipeline + Streamlit local demo; optional Community Cloud.
- **Why not API/backend:** one consumer, no scaling requirement, avoid duplicated preprocessing.
- **Risk:** free hosting and private-repo policy conflict.
- **Validate:** artifact-load/inference tests and local startup.
- **Status:** app implemented; real artifact loaded successfully.
