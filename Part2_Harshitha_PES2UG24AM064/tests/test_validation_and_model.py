import pandas as pd

from hitpredict.constants import BASE_AUDIO_FEATURES
from hitpredict.data.validation import validate_modeling_data
from hitpredict.inference import predict_record
from hitpredict.models import build_candidate, candidate_specs


def make_frame():
    rows = []
    split_years = {"train": 2014, "validation": 2016, "calibration": 2017, "test": 2019}
    for split, year in split_years.items():
        for index in range(12):
            label = index % 2
            rows.append(
                {
                    "id": f"{split}-{index}",
                    "name": f"song-{split}-{index}",
                    "artists": "['artist']",
                    "primary_artist": "artist",
                    "year": year,
                    "release_date": f"{year}-01-01",
                    "song_group": f"{split}-{index}",
                    "split": split,
                    "explicit": label,
                    "danceability": 0.3 + 0.4 * label,
                    "energy": 0.4 + 0.3 * label,
                    "key": index % 12,
                    "loudness": -12 + 5 * label,
                    "mode": label,
                    "speechiness": 0.05,
                    "acousticness": 0.7 - 0.5 * label,
                    "instrumentalness": 0.2 - 0.15 * label,
                    "liveness": 0.1,
                    "valence": 0.4 + 0.2 * label,
                    "tempo": 100 + 20 * label,
                    "duration_ms": 180000 + 20000 * label,
                    "time_signature": 4,
                    "prior_artist_hits": label,
                    "billboard_hit": label,
                }
            )
    return pd.DataFrame(rows)


def test_validation_and_serializable_inference_contract():
    frame = make_frame()
    config = {
        "data": {
            "min_release_year": 1990,
            "train_end_year": 2014,
            "selection_end_year": 2016,
            "calibration_year": 2017,
            "test_end_year": 2020,
        },
        "features": {"sets": {"audio_only": BASE_AUDIO_FEATURES}},
    }
    result = validate_modeling_data(frame, config)
    assert result["unique_song_groups"] == len(frame)

    spec = next(item for item in candidate_specs("quick") if item.family == "logistic")
    model = build_candidate(spec, BASE_AUDIO_FEATURES, seed=42)
    train = frame[frame["split"].eq("train")]
    model.fit(train[BASE_AUDIO_FEATURES], train["billboard_hit"])
    reference = {feature: float(train[feature].median()) for feature in BASE_AUDIO_FEATURES}
    artifact = {
        "schema_version": 1,
        "model": model,
        "threshold": 0.5,
        "feature_names": BASE_AUDIO_FEATURES,
        "reference_values": reference,
    }
    record = frame.iloc[0][BASE_AUDIO_FEATURES].to_dict()
    prediction = predict_record(artifact, record)
    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["prediction"] in {0, 1}
