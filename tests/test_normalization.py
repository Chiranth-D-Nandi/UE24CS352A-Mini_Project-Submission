from hitpredict.data.normalization import (
    artist_aliases,
    make_match_keys,
    normalize_title,
    parse_spotify_artists,
)


def test_title_normalization_removes_version_noise_but_keeps_meaningful_parenthetical():
    assert normalize_title("Song Name - Remastered 2011") == "song name"
    assert normalize_title("Song Name (Remastered 2011)") == "song name"
    assert normalize_title("Escape (The Piña Colada Song)") == "escape the pina colada song"


def test_artist_parsing_and_aliases_are_safe_and_deterministic():
    parsed = parse_spotify_artists("['Ariana Grande', 'The Weeknd']")
    assert parsed == ["Ariana Grande", "The Weeknd"]
    aliases = artist_aliases("Ariana Grande Featuring The Weeknd")
    assert "ariana grande" in aliases
    assert "the weeknd" in aliases
    assert "artist a and artist b" in artist_aliases(["Artist A", "Artist B"])
    # Full band aliases are kept; ampersands are not split into unsafe tokens.
    assert artist_aliases("Earth, Wind & Fire") == ["earth wind and fire"]
    assert len(aliases) == len(set(aliases))


def test_match_keys_require_both_title_and_artist():
    keys = make_match_keys("Example (feat. Guest)", ["Main Artist", "Guest"])
    assert "example||main artist" in keys
    assert "example||guest" in keys
