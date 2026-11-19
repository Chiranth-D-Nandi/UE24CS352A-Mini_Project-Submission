"""Build auditable Billboard hit records and exact-match lookup tables."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from hitpredict.data.normalization import artist_aliases, normalize_artist, normalize_title


def load_charts(path: str | Path) -> list[dict[str, Any]]:
    """Legacy helper retained for fixture tests and compatible chart JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        charts = json.load(handle)
    if not isinstance(charts, list):
        raise ValueError("Expected Billboard JSON to contain a list")
    return charts


def _as_int(value: Any, fallback: int) -> int:
    return fallback if value is None or pd.isna(value) else int(value)


def _collapse_entries(entries: Iterable[dict[str, Any]]) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    for entry in entries:
        chart_date = str(entry["chart_date"])
        title = str(entry.get("song", "")).strip()
        artist = str(entry.get("artist", "")).strip()
        title_norm = normalize_title(title)
        artist_norm = normalize_artist(artist)
        if not title_norm or not artist_norm:
            continue
        hit_id = f"{title_norm}||{artist_norm}"
        peak = _as_int(entry.get("peak_position"), _as_int(entry.get("this_week"), 100))
        weeks = _as_int(entry.get("weeks_on_chart"), 0)
        if hit_id not in records:
            records[hit_id] = {
                "hit_id": hit_id,
                "song": title,
                "artist": artist,
                "title_norm": title_norm,
                "artist_norm": artist_norm,
                "artist_aliases": artist_aliases(artist),
                "first_chart_date": chart_date,
                "last_chart_date": chart_date,
                "peak_position": peak,
                "max_weeks_on_chart": weeks,
                "chart_observations": 1,
            }
        else:
            record = records[hit_id]
            record["first_chart_date"] = min(record["first_chart_date"], chart_date)
            record["last_chart_date"] = max(record["last_chart_date"], chart_date)
            record["peak_position"] = min(record["peak_position"], peak)
            record["max_weeks_on_chart"] = max(record["max_weeks_on_chart"], weeks)
            record["chart_observations"] += 1

    hits = pd.DataFrame(records.values())
    if hits.empty:
        raise ValueError("No Billboard hit records were produced")
    hits["first_chart_date"] = pd.to_datetime(hits["first_chart_date"], errors="raise")
    hits["last_chart_date"] = pd.to_datetime(hits["last_chart_date"], errors="raise")
    hits["first_chart_year"] = hits["first_chart_date"].dt.year.astype("int16")
    hits["last_chart_year"] = hits["last_chart_date"].dt.year.astype("int16")
    return hits.sort_values(["first_chart_date", "hit_id"]).reset_index(drop=True)


def build_billboard_hits(charts: list[dict[str, Any]]) -> pd.DataFrame:
    """Collapse weekly chart objects into unique title/artist hit entities."""

    def entries() -> Iterable[dict[str, Any]]:
        for chart in charts:
            for entry in chart.get("data", []):
                yield {
                    "chart_date": chart["date"],
                    "song": entry.get("song"),
                    "artist": entry.get("artist"),
                    "this_week": entry.get("this_week"),
                    "peak_position": entry.get("peak_position"),
                    "weeks_on_chart": entry.get("weeks_on_chart"),
                }

    return _collapse_entries(entries())


def load_billboard_rows(archive_path: str | Path, current_path: str | Path) -> pd.DataFrame:
    """Load UT Data's pre-2022 archive and 2022+ current file into one schema."""
    archive = pd.read_csv(archive_path, low_memory=False).rename(
        columns={
            "chart_date": "chart_date",
            "current_position": "this_week",
            "title": "song",
            "performer": "artist",
            "peak_position": "peak_position",
            "weeks_on_chart": "weeks_on_chart",
        }
    )
    current = pd.read_csv(current_path, low_memory=False).rename(
        columns={
            "chart_week": "chart_date",
            "current_week": "this_week",
            "title": "song",
            "performer": "artist",
            "peak_pos": "peak_position",
            "wks_on_chart": "weeks_on_chart",
        }
    )
    columns = ["chart_date", "song", "artist", "this_week", "peak_position", "weeks_on_chart"]
    rows = pd.concat([archive[columns], current[columns]], ignore_index=True)
    rows["chart_date"] = pd.to_datetime(rows["chart_date"], errors="raise")
    rows = rows.drop_duplicates(["chart_date", "this_week", "song", "artist"], keep="last")
    return rows.sort_values(["chart_date", "this_week"]).reset_index(drop=True)


def build_billboard_hits_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"chart_date", "song", "artist", "this_week", "peak_position", "weeks_on_chart"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Standardized Billboard rows missing columns: {sorted(missing)}")
    return _collapse_entries(rows.to_dict(orient="records"))


def save_billboard_hits(hits: pd.DataFrame, path: str | Path) -> None:
    serializable = hits.copy()
    serializable["artist_aliases"] = serializable["artist_aliases"].map(json.dumps)
    serializable.to_csv(path, index=False, compression="infer")


def load_billboard_hits(path: str | Path) -> pd.DataFrame:
    hits = pd.read_csv(path, parse_dates=["first_chart_date", "last_chart_date"])
    hits["artist_aliases"] = hits["artist_aliases"].map(json.loads)
    return hits


def make_hit_lookup(hits: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Map normalized title/artist aliases to the earliest matching hit record."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in hits.to_dict(orient="records"):
        for alias in row["artist_aliases"]:
            key = f"{row['title_norm']}||{alias}"
            current = lookup.get(key)
            if current is None or row["first_chart_date"] < current["first_chart_date"]:
                lookup[key] = row
    return lookup


def make_artist_history(hits: pd.DataFrame) -> dict[str, list[int]]:
    """Map artist aliases to sorted years of unique prior Billboard hits."""
    history: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in hits.to_dict(orient="records"):
        for alias in row["artist_aliases"]:
            history[alias].append((row["hit_id"], int(row["first_chart_year"])))

    result: dict[str, list[int]] = {}
    for alias, observations in history.items():
        unique = {hit_id: year for hit_id, year in observations}
        result[alias] = sorted(unique.values())
    return result
