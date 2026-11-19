"""Conservative title/artist normalization for exact record linkage.

Fuzzy matching is intentionally not used to assign labels. It can raise match
coverage, but it can also silently turn unrelated same-title tracks into hits.
Unmatched records are exported for manual audit instead.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Iterable

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_FEATURE_SUFFIX = re.compile(
    r"\s*[\(\[]\s*(?:feat(?:uring)?|ft)\.?\s+[^\)\]]+[\)\]]\s*$",
    flags=re.IGNORECASE,
)
_VERSION_TERMS = (
    r"remaster(?:ed)?|mono|stereo|radio edit|edit|live|version|mix|remix|"
    r"instrumental|karaoke|demo|acoustic|sped up|slowed"
)
_NOISY_VERSION = re.compile(
    rf"\s*[\(\[][^\)\]]*\b(?:{_VERSION_TERMS})\b[^\)\]]*[\)\]]\s*$",
    flags=re.IGNORECASE,
)
_NOISY_DASH_VERSION = re.compile(
    rf"\s*[-–—]\s*[^-–—]*\b(?:{_VERSION_TERMS})\b[^-–—]*$",
    flags=re.IGNORECASE,
)
_ARTIST_SEPARATORS = re.compile(
    r"\b(?:feat(?:uring)?|ft|with|vs)\.?\b|\s+[xX]\s+|[;/+]",
    flags=re.IGNORECASE,
)


def ascii_fold(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(character for character in text if not unicodedata.combining(character))


def normalize_text(value: object) -> str:
    text = ascii_fold(value).lower().replace("&", " and ")
    text = text.replace("’", "'")
    text = _NON_ALNUM.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_title(value: object) -> str:
    """Normalize without deleting meaningful parenthetical title text."""
    text = ascii_fold(value)
    previous = None
    while previous != text:
        previous = text
        text = _FEATURE_SUFFIX.sub("", text)
        text = _NOISY_VERSION.sub("", text)
        text = _NOISY_DASH_VERSION.sub("", text)
    return normalize_text(text)


def normalize_artist(value: object) -> str:
    return normalize_text(value)


def parse_spotify_artists(value: object) -> list[str]:
    """Parse Kaggle's stringified Python list safely."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (SyntaxError, ValueError):
            pass
    return [text]


def artist_aliases(value: object | Iterable[object]) -> list[str]:
    """Return full and collaborator aliases, preserving deterministic order."""
    raw_values: list[str]
    if isinstance(value, str) or value is None:
        raw_values = parse_spotify_artists(value)
    elif isinstance(value, Iterable):
        raw_values = [str(item) for item in value if str(item).strip()]
    else:
        raw_values = parse_spotify_artists(value)

    aliases: list[str] = []
    # Spotify represents collaborations as a list while Billboard often stores a
    # single "A & B" string. A combined alias links those without splitting real
    # band names such as "Earth, Wind & Fire" into unsafe generic tokens.
    if len(raw_values) > 1:
        combined = normalize_artist(" and ".join(raw_values))
        if combined:
            aliases.append(combined)
    for raw in raw_values:
        full = normalize_artist(raw)
        if full:
            aliases.append(full)
        for part in _ARTIST_SEPARATORS.split(raw):
            normalized = normalize_artist(part)
            if normalized:
                aliases.append(normalized)

    return list(dict.fromkeys(alias for alias in aliases if alias))


def primary_artist(value: object) -> str:
    artists = parse_spotify_artists(value)
    return artists[0] if artists else ""


def make_match_keys(title: object, artists: object | Iterable[object]) -> list[str]:
    normalized_title = normalize_title(title)
    if not normalized_title:
        return []
    return [f"{normalized_title}||{artist}" for artist in artist_aliases(artists)]


def make_song_group(title: object, artists: object) -> str:
    """Group duplicate releases so one song entity cannot cross time splits."""
    normalized_title = normalize_title(title)
    parsed = parse_spotify_artists(artists)
    main_artist = normalize_artist(parsed[0]) if parsed else ""
    return f"{normalized_title}||{main_artist}"
