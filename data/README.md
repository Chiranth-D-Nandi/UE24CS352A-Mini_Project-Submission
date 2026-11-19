# Local data contract

Raw and row-level derived data are intentionally ignored by Git.

## Required manual file

Download `tracks_features.csv` from:

<https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs>

Place it at:

```text
data/raw/spotify/tracks_features.csv
```

Expected columns used by the pipeline:

```text
id, name, artists, explicit, danceability, energy, key, loudness, mode,
speechiness, acousticness, instrumentalness, liveness, valence, tempo,
duration_ms, time_signature, year, release_date
```

Kaggle reports the dataset licence as **Unknown**. Do not redistribute the raw file or commit derived row-level records. The project records a SHA-256 hash locally for reproducibility.

## Automatically acquired files

`hitpredict acquire` downloads two MIT-licensed files from UT Data's Billboard archive into `data/raw/billboard/` and records source URLs/hashes.

## Generated files

- `data/interim/billboard_hits.csv.gz`
- `data/processed/modeling_dataset.csv.gz`

These are rebuildable and ignored. Reports contain aggregate evidence plus limited audit records; review privacy/licence constraints before sharing them outside the private assignment repository.
