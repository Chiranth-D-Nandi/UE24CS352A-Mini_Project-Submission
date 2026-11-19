"""Construct the leakage-aware modelling table from Spotify and Billboard."""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hitpredict.constants import (
    BASE_AUDIO_FEATURES,
    BOUNDED_ZERO_ONE,
    METADATA_COLUMNS,
    REQUIRED_SPOTIFY_COLUMNS,
    TARGET_COLUMN,
)
from hitpredict.data.acquire import acquire_sources, spotify_missing_message
from hitpredict.data.labels import (
    build_billboard_hits_from_rows,
    load_billboard_rows,
    make_artist_history,
    make_hit_lookup,
    save_billboard_hits,
)
from hitpredict.data.normalization import (
    artist_aliases,
    make_match_keys,
    make_song_group,
    parse_spotify_artists,
    primary_artist,
)
from hitpredict.utils import utc_now_iso, write_json


def _read_spotify(path: str | Path) -> pd.DataFrame:
    """Read only columns that are needed for identity, audit, and modelling."""
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(REQUIRED_SPOTIFY_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"Spotify CSV is missing required columns: {missing}")

    frame = pd.read_csv(
        path,
        usecols=REQUIRED_SPOTIFY_COLUMNS,
        low_memory=False,
        dtype={
            "id": "string",
            "name": "string",
            "artists": "string",
            "release_date": "string",
        },
    )
    return frame


def _coerce_spotify_schema(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    numeric = [feature for feature in BASE_AUDIO_FEATURES if feature != "explicit"] + ["year"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    explicit_map = {
        True: 1,
        False: 0,
        "true": 1,
        "false": 0,
        "True": 1,
        "False": 0,
        "1": 1,
        "0": 0,
    }
    frame["explicit"] = frame["explicit"].map(explicit_map)
    frame["year"] = frame["year"].astype("Int64")
    return frame


def _invalid_audio_mask(frame: pd.DataFrame) -> pd.Series:
    """Flag unavailable audio-analysis rows, not merely unusual music."""
    signal_columns = BOUNDED_ZERO_ONE + ["tempo"]
    all_zero = frame[signal_columns].fillna(0).abs().sum(axis=1).eq(0)
    no_meter_or_tempo = frame["time_signature"].fillna(0).eq(0) & frame["tempo"].fillna(0).eq(0)
    empty_identity = frame["name"].fillna("").str.strip().eq("") | frame["artists"].fillna("").str.strip().eq("")
    return all_zero | no_meter_or_tempo | empty_identity


def _select_match(keys: list[str], hit_lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    matches = {hit_lookup[key]["hit_id"]: hit_lookup[key] for key in keys if key in hit_lookup}
    if not matches:
        return None
    return min(matches.values(), key=lambda value: value["first_chart_date"])


def _prior_hit_count(aliases: list[str], year: int, history: dict[str, list[int]]) -> int:
    # Use the most established listed artist rather than summing collaborators.
    # This prevents a many-artist track from receiving an inflated context score.
    return max((bisect_left(history.get(alias, []), year) for alias in aliases), default=0)


def assign_temporal_split(years: pd.Series, data_config: dict[str, Any]) -> pd.Series:
    train_end = int(data_config["train_end_year"])
    selection_end = int(data_config["selection_end_year"])
    calibration_year = int(data_config["calibration_year"])
    test_end = int(data_config["test_end_year"])
    values = np.select(
        [
            years <= train_end,
            (years > train_end) & (years <= selection_end),
            years == calibration_year,
            (years > calibration_year) & (years <= test_end),
        ],
        ["train", "validation", "calibration", "test"],
        default="excluded",
    )
    return pd.Series(values, index=years.index, dtype="string")


def build_modeling_frame(
    spotify: pd.DataFrame,
    hits: pd.DataFrame,
    data_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Pure transformation used by both the CLI and tests."""
    spotify = _coerce_spotify_schema(spotify)
    source_rows = len(spotify)
    min_year = int(data_config["min_release_year"])
    max_year = int(data_config["max_release_year"])

    spotify = spotify[spotify["year"].between(min_year, max_year, inclusive="both")].copy()
    after_year_filter = len(spotify)
    invalid_mask = _invalid_audio_mask(spotify)
    invalid_rows = spotify.loc[invalid_mask].copy()
    spotify = spotify.loc[~invalid_mask].copy()

    spotify["primary_artist"] = spotify["artists"].map(primary_artist)
    spotify["artist_list"] = spotify["artists"].map(parse_spotify_artists)
    spotify["artist_aliases"] = spotify["artist_list"].map(artist_aliases)
    spotify["match_keys"] = [
        make_match_keys(title, artists)
        for title, artists in zip(spotify["name"], spotify["artist_list"], strict=True)
    ]
    spotify["song_group"] = [
        make_song_group(title, artists)
        for title, artists in zip(spotify["name"], spotify["artists"], strict=True)
    ]

    hit_lookup = make_hit_lookup(hits)
    history = make_artist_history(hits)
    matches = [_select_match(keys, hit_lookup) for keys in spotify["match_keys"]]
    spotify["matched_hit_id"] = [match["hit_id"] if match else pd.NA for match in matches]
    spotify["first_chart_date"] = [match["first_chart_date"] if match else pd.NaT for match in matches]
    spotify["first_chart_year"] = pd.array(
        [int(match["first_chart_year"]) if match else pd.NA for match in matches], dtype="Int64"
    )
    spotify["peak_position"] = pd.array(
        [int(match["peak_position"]) if match else pd.NA for match in matches], dtype="Int64"
    )
    spotify["chart_release_gap_years"] = spotify["first_chart_year"] - spotify["year"]
    spotify["exact_chart_match"] = spotify["matched_hit_id"].notna()

    max_before = int(data_config.get("max_chart_before_release_years", 1))
    ambiguous_match = spotify["exact_chart_match"] & (
        spotify["first_chart_year"] < spotify["year"] - max_before
    )
    ambiguous_rows = spotify.loc[ambiguous_match].copy()
    spotify = spotify.loc[~ambiguous_match].copy()
    spotify[TARGET_COLUMN] = spotify["exact_chart_match"].astype("int8")

    spotify["prior_artist_hits"] = [
        _prior_hit_count(aliases, int(year), history)
        for aliases, year in zip(spotify["artist_aliases"], spotify["year"], strict=True)
    ]

    # Collapse album/reissue duplicates before assigning time splits. Propagate an
    # exact hit label across duplicate recordings, then retain the earliest release.
    spotify["group_hit"] = spotify.groupby("song_group")[TARGET_COLUMN].transform("max")
    hit_metadata = (
        spotify.loc[spotify[TARGET_COLUMN].eq(1)]
        .groupby("song_group", as_index=True)
        .agg(
            matched_hit_id_group=("matched_hit_id", "first"),
            first_chart_date_group=("first_chart_date", "min"),
            first_chart_year_group=("first_chart_year", "min"),
            peak_position_group=("peak_position", "min"),
        )
    )
    spotify = spotify.join(hit_metadata, on="song_group")
    spotify = spotify.sort_values(["year", "release_date", "id"], kind="stable")
    duplicate_rows = int(spotify.duplicated("song_group", keep="first").sum())
    spotify = spotify.drop_duplicates("song_group", keep="first").copy()
    spotify[TARGET_COLUMN] = spotify["group_hit"].astype("int8")
    positive = spotify[TARGET_COLUMN].eq(1)
    for target, grouped in [
        ("matched_hit_id", "matched_hit_id_group"),
        ("first_chart_date", "first_chart_date_group"),
        ("first_chart_year", "first_chart_year_group"),
        ("peak_position", "peak_position_group"),
    ]:
        spotify.loc[positive, target] = spotify.loc[positive, grouped]
    spotify["chart_release_gap_years"] = spotify["first_chart_year"] - spotify["year"]

    spotify["split"] = assign_temporal_split(spotify["year"].astype(int), data_config)
    spotify = spotify.loc[spotify["split"].ne("excluded")].copy()

    keep_columns = list(
        dict.fromkeys(
            METADATA_COLUMNS
            + BASE_AUDIO_FEATURES
            + [
                "prior_artist_hits",
                TARGET_COLUMN,
                "matched_hit_id",
                "first_chart_date",
                "first_chart_year",
                "peak_position",
                "chart_release_gap_years",
            ]
        )
    )
    modeling = spotify[keep_columns].sort_values(["year", "song_group"]).reset_index(drop=True)

    matched_ids_all = set(spotify.loc[spotify[TARGET_COLUMN].eq(1), "matched_hit_id"].dropna())
    eligible_hits = hits[hits["first_chart_year"].between(min_year, max_year)].copy()
    eligible_ids = set(eligible_hits["hit_id"])
    matched_ids = matched_ids_all & eligible_ids
    unmatched_hits = eligible_hits.loc[~eligible_hits["hit_id"].isin(matched_ids)].copy()

    audit_columns = [
        "id",
        "name",
        "artists",
        "year",
        "release_date",
        "song_group",
        TARGET_COLUMN,
        "matched_hit_id",
        "first_chart_date",
        "first_chart_year",
        "chart_release_gap_years",
    ]
    positive_audit = spotify.loc[spotify[TARGET_COLUMN].eq(1), audit_columns].copy()
    if not ambiguous_rows.empty:
        ambiguous_audit = ambiguous_rows.reindex(columns=audit_columns).copy()
        ambiguous_audit["audit_status"] = "excluded_chart_before_recorded_release"
    else:
        ambiguous_audit = pd.DataFrame(columns=audit_columns + ["audit_status"])
    positive_audit["audit_status"] = "accepted_exact_match"
    audit_frames = [positive_audit]
    if not ambiguous_audit.empty:
        audit_frames.append(ambiguous_audit)
    match_audit = pd.concat(audit_frames, ignore_index=True)

    split_counts = (
        modeling.groupby(["split", TARGET_COLUMN], observed=True).size().unstack(fill_value=0).to_dict(orient="index")
    )
    summary = {
        "created_at_utc": utc_now_iso(),
        "source_spotify_rows": source_rows,
        "rows_after_year_filter": after_year_filter,
        "invalid_audio_rows_removed": int(len(invalid_rows)),
        "ambiguous_matches_removed": int(len(ambiguous_rows)),
        "duplicate_song_entities_removed": duplicate_rows,
        "final_rows": int(len(modeling)),
        "final_positive_rows": int(modeling[TARGET_COLUMN].sum()),
        "final_positive_prevalence": float(modeling[TARGET_COLUMN].mean()),
        "billboard_hits_first_charting_in_release_window": int(len(eligible_hits)),
        "billboard_hit_entities_matched": int(len(matched_ids)),
        "billboard_entity_match_coverage": (
            float(len(matched_ids) / len(eligible_hits)) if len(eligible_hits) else None
        ),
        "split_class_counts": split_counts,
        "label_definition": "1 iff an exact normalized title+artist alias appears on Billboard Hot 100; 0 otherwise",
    }
    return modeling, summary, match_audit, unmatched_hits


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    acquire_sources(config, force=False)
    data = config["data"]
    spotify_path = Path(data["spotify_csv"])
    if not spotify_path.exists():
        raise FileNotFoundError(spotify_missing_message(config))

    chart_rows = load_billboard_rows(
        data["billboard_archive_csv"], data["billboard_current_csv"]
    )
    hits = build_billboard_hits_from_rows(chart_rows)
    save_billboard_hits(hits, data["billboard_hits"])

    spotify = _read_spotify(spotify_path)
    modeling, summary, match_audit, unmatched_hits = build_modeling_frame(spotify, hits, data)

    Path(data["modeling_dataset"]).parent.mkdir(parents=True, exist_ok=True)
    modeling.to_csv(data["modeling_dataset"], index=False, compression="infer")
    match_audit.to_csv(data["match_audit"], index=False, compression="infer")
    serializable_unmatched = unmatched_hits.copy()
    if "artist_aliases" in serializable_unmatched:
        serializable_unmatched["artist_aliases"] = serializable_unmatched["artist_aliases"].map(str)
    serializable_unmatched.to_csv(data["unmatched_billboard"], index=False, compression="infer")

    summary_path = Path(config["_meta"]["project_root"]) / "reports/audits/preparation_summary.json"
    write_json(summary_path, summary)
    return summary
