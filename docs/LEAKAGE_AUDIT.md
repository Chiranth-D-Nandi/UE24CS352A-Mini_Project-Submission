# Leakage Audit

The final test must not be evaluated until all items marked **gate** pass.

| Leakage type | Risk here | Control | Gate/evidence |
|---|---|---|---|
| Target leakage | Billboard rank/weeks/score directly encode chart membership | hard feature allow-list; forbidden-field assertion | `validate_feature_sets` |
| Post-outcome leakage | Spotify `popularity` is based on plays and recency | never read as a model feature | feature availability CSV |
| Temporal leakage | random rows expose future taste | year-blocked 1990–2014 / 2015–16 / 2017 / 2018–20 | split validation |
| Artist-history leakage | counting future or same-year hits | `bisect_left(first_chart_year, release_year)`; strict earlier years only | unit test |
| Duplicate leakage | album, remaster, and compilation copies | normalized title+primary artist `song_group`; keep one earliest entity | uniqueness gate |
| Entity leakage | same song across partitions | split only after entity deduplication | uniqueness gate |
| Preprocessing leakage | imputer/scaler fit on all rows | transformations inside sklearn Pipeline fit on training only | code contract |
| Feature-selection leakage | selecting on test importance | feature sets predeclared; permutation importance is post-selection | ledger/report label |
| Resampling leakage | duplicate positives copied across splits | no oversampling; negative sampling occurs after split | training code |
| Calibration leakage | sampled train prior gives false probabilities | calibrate on full 2017 catalog only | final flow |
| Threshold leakage | choosing threshold on test | choose F1 threshold on 2017 only | final flow |
| Hyperparameter leakage | repeated test checking | select candidates on 2015–16; test reveal count recorded as 1 | final report |
| Label-linkage leakage | chart metadata accidentally enters model | model receives only ordered named features | artifact schema |

## Features excluded

- `popularity`: post-release plays/recency.
- `peak_position`, `weeks_on_chart`, rank, chart score: target outcomes.
- IDs, title, album, artist ID/name: identity/memorization risk in the primary tabular model.
- `year`: used for splitting/segment analysis; excluded to prevent a temporal shortcut.
- `release_date`: inconsistent precision and split metadata only.

## Residual risks (cannot be claimed away)

1. Exact linkage misses aliases and formatting variants, creating some false negatives.
2. Title/artist collisions can create false positives; matches charting before the recorded release by more than one year are excluded for audit.
3. Billboard first-chart year is not the actual song release date.
4. The static Spotify pool has selection and coverage bias.
5. Artist-history matching remains name based; aliases/group membership are imperfect.

## Required final checks

```bash
hitpredict audit --config configs/default.yaml
pytest -q
```

Review `preparation_summary.json`, `match_audit.csv.gz`, and a manual sample of accepted/unmatched links before training. If match coverage or precision is poor, fix linkage and regenerate all downstream evidence; never patch labels after seeing test errors.
