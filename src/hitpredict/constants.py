"""Project-wide schema constants.

Only columns listed in an approved feature set may enter a model. Keeping the
allow-list here makes accidental target/post-outcome leakage fail loudly.
"""

BASE_AUDIO_FEATURES = [
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
]

CONTEXT_FEATURES = ["prior_artist_hits"]

CONTINUOUS_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "prior_artist_hits",
]

CATEGORICAL_FEATURES = ["explicit", "key", "mode", "time_signature"]

REQUIRED_SPOTIFY_COLUMNS = [
    "id",
    "name",
    "artists",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
    "year",
    "release_date",
]

METADATA_COLUMNS = [
    "id",
    "name",
    "artists",
    "primary_artist",
    "year",
    "release_date",
    "song_group",
    "split",
]

TARGET_COLUMN = "billboard_hit"

# These fields are either direct outcomes, post-release measurements, unique
# identifiers, or source metadata. They must never be model inputs.
FORBIDDEN_MODEL_FEATURES = {
    "billboard_hit",
    "billboard_rank",
    "peak_position",
    "weeks_on_chart",
    "chart_score",
    "first_chart_date",
    "last_chart_date",
    "popularity",
    "id",
    "album_id",
    "artist_ids",
    "track_number",
    "disc_number",
    "name",
    "album",
    "artists",
    "release_date",
    "song_group",
    "split",
}

BOUNDED_ZERO_ONE = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

VALID_RANGES = {
    "explicit": (0, 1),
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "key": (0, 11),
    "loudness": (-60.0, 10.0),
    "mode": (0, 1),
    "speechiness": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "tempo": (0.0, 300.0),
    "duration_ms": (1_000.0, 7_200_000.0),
    "time_signature": (0, 7),
    "prior_artist_hits": (0, 100_000),
}
