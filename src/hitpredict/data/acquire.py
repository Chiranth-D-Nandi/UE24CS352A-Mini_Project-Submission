"""Reproducible source acquisition.

The two Billboard CSVs come from UT Data's MIT-licensed chart archive. The
large Spotify/Kaggle file is deliberately not redistributed or silently
downloaded: Kaggle marks its licence as "Unknown". The user must obtain it from
the source and place it at the configured path.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from hitpredict.config import ensure_output_directories
from hitpredict.utils import sha256_file, utc_now_iso, write_json


def download_file(url: str, destination: str | Path, timeout: int = 180) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "HitPredict/0.1 academic-use"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def validate_billboard_csvs(archive_path: str | Path, current_path: str | Path) -> dict[str, Any]:
    archive = pd.read_csv(archive_path, nrows=5)
    current = pd.read_csv(current_path, nrows=5)
    archive_required = {
        "chart_date",
        "current_position",
        "title",
        "performer",
        "peak_position",
        "weeks_on_chart",
    }
    current_required = {
        "chart_week",
        "current_week",
        "title",
        "performer",
        "peak_pos",
        "wks_on_chart",
    }
    if not archive_required.issubset(archive.columns):
        raise ValueError(f"Billboard archive schema changed: {archive_required - set(archive.columns)}")
    if not current_required.issubset(current.columns):
        raise ValueError(f"Billboard current schema changed: {current_required - set(current.columns)}")

    archive_dates = pd.read_csv(archive_path, usecols=["chart_date"])["chart_date"]
    current_dates = pd.read_csv(current_path, usecols=["chart_week"])["chart_week"]
    combined = pd.to_datetime(pd.concat([archive_dates, current_dates], ignore_index=True), errors="raise")
    return {
        "first_chart_date": combined.min().date().isoformat(),
        "last_chart_date": combined.max().date().isoformat(),
        "weekly_chart_count": int(combined.nunique()),
    }


def acquire_sources(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    ensure_output_directories(config)
    data = config["data"]
    archive_path = Path(data["billboard_archive_csv"])
    current_path = Path(data["billboard_current_csv"])
    spotify_path = Path(data["spotify_csv"])

    for path, url in [
        (archive_path, data["billboard_archive_url"]),
        (current_path, data["billboard_current_url"]),
    ]:
        if force or not path.exists():
            download_file(url, path)
    billboard_summary = validate_billboard_csvs(archive_path, current_path)

    manifest = {
        "created_at_utc": utc_now_iso(),
        "billboard": {
            "repository": "https://github.com/utdata/rwd-billboard-data",
            "archive_url": data["billboard_archive_url"],
            "archive_path": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "current_url": data["billboard_current_url"],
            "current_path": str(current_path),
            "current_sha256": sha256_file(current_path),
            **billboard_summary,
            "license": "MIT (repository LICENSE, copyright UT Data 2019)",
        },
        "spotify": {
            "source": "https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs",
            "path": str(spotify_path),
            "present": spotify_path.exists(),
            "sha256": sha256_file(spotify_path) if spotify_path.exists() else None,
            "license_status": "Kaggle data card says Unknown; do not commit or redistribute raw data.",
        },
    }
    write_json(data["manifest"], manifest)
    return manifest


def spotify_missing_message(config: dict[str, Any]) -> str:
    path = config["data"]["spotify_csv"]
    return (
        "Spotify source file is missing. Download tracks_features.csv from "
        "https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs and place it at:\n"
        f"  {path}\n"
        "The source is not auto-downloaded because Kaggle reports an Unknown licence."
    )
