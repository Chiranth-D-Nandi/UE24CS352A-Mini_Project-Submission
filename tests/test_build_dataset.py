import pandas as pd

from hitpredict.constants import REQUIRED_SPOTIFY_COLUMNS
from hitpredict.data.build_dataset import build_modeling_frame
from hitpredict.data.labels import build_billboard_hits


def spotify_row(track_id, name, artist, year, **overrides):
    row = {
        "id": track_id,
        "name": name,
        "artists": str([artist]),
        "explicit": False,
        "danceability": 0.5,
        "energy": 0.6,
        "key": 4,
        "loudness": -7.0,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.5,
        "tempo": 120.0,
        "duration_ms": 210000,
        "time_signature": 4,
        "year": year,
        "release_date": f"{year}-01-01",
    }
    row.update(overrides)
    return row


def test_build_labels_deduplicates_and_computes_strict_prior_history():
    charts = [
        {
            "date": "2010-06-01",
            "data": [
                {"song": "First Hit", "artist": "Artist A", "this_week": 1, "last_week": None, "peak_position": 1, "weeks_on_chart": 1}
            ],
        },
        {
            "date": "2018-06-01",
            "data": [
                {"song": "Future Hit", "artist": "Artist B", "this_week": 20, "last_week": None, "peak_position": 20, "weeks_on_chart": 1}
            ],
        },
    ]
    hits = build_billboard_hits(charts)
    spotify = pd.DataFrame(
        [
            spotify_row("1", "First Hit", "Artist A", 2010),
            spotify_row("2", "First Hit (Remastered 2011)", "Artist A", 2011),
            spotify_row("3", "Follow Up", "Artist A", 2012),
            spotify_row("4", "Future Hit", "Artist B", 2018),
            spotify_row("5", "Ordinary", "Artist C", 2019),
            spotify_row("6", "Broken", "Artist Z", 2014, tempo=0, time_signature=0, danceability=0, energy=0, speechiness=0, acousticness=0, instrumentalness=0, liveness=0, valence=0),
        ],
        columns=REQUIRED_SPOTIFY_COLUMNS,
    )
    config = {
        "min_release_year": 1990,
        "max_release_year": 2020,
        "train_end_year": 2014,
        "selection_end_year": 2016,
        "calibration_year": 2017,
        "test_end_year": 2020,
        "max_chart_before_release_years": 1,
    }
    modeling, summary, audit, unmatched = build_modeling_frame(spotify, hits, config)
    first_hit = modeling[modeling["name"].eq("First Hit")].iloc[0]
    follow_up = modeling[modeling["name"].eq("Follow Up")].iloc[0]
    assert first_hit["billboard_hit"] == 1
    assert follow_up["billboard_hit"] == 0
    assert follow_up["prior_artist_hits"] == 1
    assert "Broken" not in set(modeling["name"])
    assert summary["invalid_audio_rows_removed"] == 1
    assert modeling["song_group"].is_unique
