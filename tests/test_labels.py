from hitpredict.data.labels import build_billboard_hits, make_artist_history, make_hit_lookup


def test_weekly_charts_collapse_to_unique_hit_and_history():
    charts = [
        {
            "date": "2010-01-02",
            "data": [
                {
                    "song": "Example",
                    "artist": "Artist A Featuring Guest",
                    "this_week": 10,
                    "last_week": None,
                    "peak_position": 10,
                    "weeks_on_chart": 1,
                }
            ],
        },
        {
            "date": "2010-01-09",
            "data": [
                {
                    "song": "Example",
                    "artist": "Artist A Featuring Guest",
                    "this_week": 5,
                    "last_week": 10,
                    "peak_position": 5,
                    "weeks_on_chart": 2,
                }
            ],
        },
    ]
    hits = build_billboard_hits(charts)
    assert len(hits) == 1
    assert hits.iloc[0]["peak_position"] == 5
    assert hits.iloc[0]["max_weeks_on_chart"] == 2
    lookup = make_hit_lookup(hits)
    assert "example||artist a" in lookup
    history = make_artist_history(hits)
    assert history["artist a"] == [2010]
