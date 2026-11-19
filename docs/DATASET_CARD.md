# Dataset Card (Pre-Run)

## Composition

The modelling dataset is built, not downloaded as one labelled table:

1. Spotify static track metadata/audio descriptors (`tracks_features.csv`, user supplied).
2. MIT-licensed weekly Billboard Hot 100 rows from UT Data.
3. Exact normalized title + artist alias linkage.

Actual shape: 1,030,990 rows (2,881 positive hits, 0.28% prevalence). Match coverage: 23.6%. Missingness, removals, and split counts are generated after `hitpredict prepare` and can be inspected in `reports/audits/preparation_summary.json` and `reports/audits/data_audit.json`.

## Label semantics

`billboard_hit = 1` means the linked song entity appears at least once in available Billboard Hot 100 history. `0` means no exact link was found. It does **not** prove that a track was commercially unsuccessful, nor that it never appeared under an unmatched alias.

## Inclusion/exclusion

- Release years 1990–2020.
- Remove unavailable audio-analysis rows (all-zero signal or no tempo/meter).
- Exclude exact links where chart appearance predates the recorded release by more than one year; export for audit.
- Collapse normalized title + primary artist duplicates; retain earliest release.
- Keep unusual valid values; impute missing values inside the training Pipeline.

## Known biases

- Spotify catalog/collection bias and duplicated releases.
- Billboard measures US chart inclusion, not global or artistic success.
- Exact linkage lowers false-positive labels but raises false-negative labels.
- Recent songs have shorter follow-up than older songs, though the test ends in 2020 and charts extend through 2026.
- Artist-name aliases and group membership are incomplete.
- Music taste and production trends drift over time.

## Licence/provenance

- Spotify Kaggle data card: **Unknown** licence. Do not redistribute; review Kaggle/Spotify terms.
- UT Data Billboard repository: MIT repository licence. Preserve attribution/licence when redistributing permitted source/code.
- Source URLs and hashes: generated manifest.

## Suitable use

Coursework on leakage-aware classification, temporal evaluation, and model comparison. Not suitable for automated investment, artist evaluation, causal claims, or use outside source terms.
