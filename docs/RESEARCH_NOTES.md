# Research-to-Design Traceability

Research date: 2026-10-06. Firecrawl CLI was used for the web sources listed below; uploaded PDFs were extracted and read page by page.

## Uploaded official material

### Assignment (2 pages)

Mandatory artifacts are a private GitHub repository with setup README, PDF write-up, slide deck, and live code demonstration. Evaluation covers deliverable quality, functionality, repository maintenance, writing clarity, presentation/demo/Q&A, and individual contribution. The heading says “One-Page Write-up” while its bullet says “concise, two-page summary”; do not invent a resolution.

### HitPredict paper (5 pages) and poster (1 page)

- **Dataset:** roughly 4,000 balanced songs. Random Million Song Dataset tracks released 1990–2018 supplied non-hits; unique Billboard Hot 100 songs supplied hits; about two-thirds of collected hits were removed to balance. The paper says labels mark chart appearance “between 1991–2010,” despite describing Billboard collection through 2018—an internal scope ambiguity.
- **Inputs:** danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence, loudness, tempo, plus binary artist score (“previous Billboard hit,” look-back to 1986).
- **Split/evaluation:** random 75/25 train/validation for supervised models; paper explicitly says an additional test set was not needed. Accuracy, precision, recall; the validation set was also used to inspect epochs/depth/bagging.
- **LR/GDA:** similar boundaries; 75.9% and 73.7% validation accuracy; train/validation similarity was interpreted as little overfit.
- **SVM:** linear/RBF/polynomial tested; validation deterioration indicated overfit. Bagging was used for linear/polynomial variants.
- **Decision Tree:** could reach 100% train accuracy and reported 51.5% validation in prose; depth restriction and bagging reduced overfit. Poster shows 51.5%, while a paper figure/table appears internally inconsistent.
- **Random forests/bagging:** used as an overfit correction; did not materially improve LR precision/recall according to prose. Some displayed bagging values are difficult to reconcile, so they are not copied as benchmarks.
- **Neural network:** one hidden layer, six sigmoid units, L2 regularization; poster reports 76.5% validation accuracy. Validation peaked near 19,000 epochs. NN precision was described as higher than LR.
- **Feature/error analysis:** greedy ablation ranked artist score first for LR and danceability first for NN; later features sometimes reduced accuracy. Holiday songs performed differently; five-year subsets often improved results, suggesting temporal drift.
- **Conclusion/future:** LR and NN strongest; SVM/DT overfit; collect more data/longer chart history and consider duration.

**Finding → implication → design**

- Trend drift → random split can overstate prospective performance → future year blocks and temporal segment metrics.
- Artist score matters but definition is coarse → compute count strictly from chart years before release and run as an ablation.
- SVM/DT overfit → retain a shallow DT diagnostic; omit default RBF-SVM at 1.2M scale.
- No independent test → reserve 2018–2020 and record one reveal.
- Balanced data/accuracy → report natural-prevalence PR-AUC first; 1:1 results only as comparison.

## Firecrawl dataset findings

| Candidate | Verified facts | Fit | Decision |
|---|---|---|---|
| Kaggle Spotify 1.2M+ | 345.74 MB, 24 columns, >1.2M rows, audio features; data card licence “Unknown”; user-provided schema includes `year`/`release_date` through 2020 | Broad feature pool and temporal split possible; no labels | **Selected with non-redistribution warning** |
| Kaggle/HF Spotify Tracks | 114,000 rows, 21 columns, 125 genres; Kaggle page says ODbL; exposed schema lacks release year/date | Clearer licence, but cannot support the required temporal evaluation and is genre-sampled | Reject for primary data |
| HF 45M Spotify tracks | 45M rows mentioned by dataset search | Excessive for a mini-project; licensing/quality not verified | Reject without deeper need |
| Spotify Web API live collection | Spotify official 2024-11-27 notice restricts Audio Features for new/development-mode apps without prior extended access | Not reproducible for a new student app | Reject; static file only |
| UT Data Billboard archive | Weekly Hot 100 archive/current CSVs; repository MIT licence; raw URLs pinned to commit `7037d25289a6d705b631f1dbc1c313cf8d47de5a` | Exact chart label source with provenance | **Selected** |
| Kaggle Billboard Historical Charts | Hot 100 CSV plus other charts; CC BY-NC-SA 4.0 | Valid alternative but adds manual Kaggle dependency | Reject because licensed UT raw URLs are simpler |
| mhollingshead Billboard JSON | Convenient 1958+ JSON but no explicit licence verified during review | Schema is good; redistribution uncertainty | Replaced by UT archive |

Pinned-source validation on 2026-10-06 produced 355,700 weekly rows across 3,557 chart weeks, 32,696 collapsed song entities, and dates 1958-08-04 through 2026-10-03. The archive/current SHA-256 values are recorded automatically by `hitpredict acquire`.

The selected Spotify source's unknown licence remains an explicit project risk. It is acceptable only as private educational analysis unless permission/terms establish more.

## Firecrawl literature findings

### Middlebrook & Sheik (2019), arXiv:1908.08609

They joined roughly 1.8M Spotify records to ~16k chart songs, balanced to ~24k, split into 20k/2k/2k, and reported RF test accuracy 87.7%, LR 81.5%, NN 82.1%, SVM 83.9%. However, their listed predictors include `weeks`, `rank`, custom chart `score`, and Spotify `popularity`. The first three are direct chart outcomes; popularity is post-release. Their random split also does not test future years.

**Implication:** do not treat 88% as a prospective benchmark. Exclude all outcome/post-release fields and expect lower but more defensible results.

### SpotHitPy (2023), arXiv:2301.07978

It collected ~18k tracks, 861 hits, used RandomOverSampler and feature scaling/PCA, then reported RF validation accuracy 0.86 and optimized SVM 0.83. It recognizes trend change but does not establish a future-period test. Popularity is included, and the described balance/split ordering creates a possible duplicate-resampling leakage risk.

**Implication:** RF/SVM are candidate evidence, not benchmark truth. Split before any sampling; exclude popularity; use time blocks.

### Zhao et al. (2023), arXiv:2301.13507

The target is top-10 status among Billboard Hot 100 songs, using lyrics/audio/metadata; it is a different conditional task from chart inclusion.

**Implication:** its headline numbers are not comparable; richer metadata is future work only if available before prediction.

## Deployment finding

Firecrawl retrieved Streamlit's official site: Community Cloud is presented as free and GitHub-based, while Streamlit supports pure-Python/scikit-learn apps. Because the assignment requires a private repository and hosted visibility/resource terms can change, local Streamlit is the guaranteed demo; hosted deployment is optional after checking current account/repository rules.

## Sources

- Uploaded: `Guidelines and Instructions_Mini Project Assignment.pdf`, `16 (1).pdf`, `16.pdf`.
- <https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs>
- <https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset>
- <https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset>
- <https://github.com/utdata/rwd-billboard-data>
- <https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api>
- <https://arxiv.org/abs/1908.08609>
- <https://arxiv.org/abs/2301.07978>
- <https://arxiv.org/abs/2301.13507>
- <https://streamlit.io/cloud>
