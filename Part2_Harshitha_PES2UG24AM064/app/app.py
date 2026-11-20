"""Reviewer-facing Streamlit demo: input → saved pipeline → output → explanation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hitpredict.inference import InputValidationError, load_artifact, local_sensitivity, predict_record  # noqa: E402

st.set_page_config(page_title="HitPredict", page_icon="🎵", layout="wide")

MODEL_PATH = Path(os.environ.get("HITPREDICT_MODEL_PATH", ROOT / "artifacts/final_model.joblib"))


@st.cache_resource
def get_artifact(path: str):
    return load_artifact(path)


st.title("HitPredict — Billboard Hot 100 likelihood")
st.caption("A historical, association-based classifier using pre-outcome audio descriptors. Not a guarantee of success.")

if not MODEL_PATH.exists():
    st.error(
        f"Model artifact not found at `{MODEL_PATH}`. Run `make pipeline` after placing the Spotify CSV."
    )
    st.stop()

try:
    artifact = get_artifact(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

KEYS = {
    "C": 0,
    "C♯/D♭": 1,
    "D": 2,
    "D♯/E♭": 3,
    "E": 4,
    "F": 5,
    "F♯/G♭": 6,
    "G": 7,
    "G♯/A♭": 8,
    "A": 9,
    "A♯/B♭": 10,
    "B": 11,
}

with st.form("prediction_form"):
    st.subheader("Track descriptors")
    first, second, third = st.columns(3)
    with first:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.55, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.65, 0.01)
        valence = st.slider("Valence", 0.0, 1.0, 0.50, 0.01)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.20, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.01, 0.01)
    with second:
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.06, 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01)
        loudness = st.slider("Loudness (dB)", -60.0, 7.0, -7.0, 0.1)
        tempo = st.number_input("Tempo (BPM)", min_value=1.0, max_value=300.0, value=120.0)
        duration_seconds = st.number_input(
            "Duration (seconds)", min_value=1.0, max_value=7200.0, value=210.0
        )
    with third:
        key_label = st.selectbox("Key", list(KEYS))
        mode_label = st.selectbox("Mode", ["Major", "Minor"])
        time_signature = st.selectbox("Time signature", [3, 4, 5, 1, 2, 6, 7], index=1)
        explicit = st.toggle("Explicit lyrics", value=False)
        prior_artist_hits = st.number_input(
            "Prior Billboard hits by the most established listed artist",
            min_value=0,
            max_value=1000,
            value=0,
            disabled="prior_artist_hits" not in artifact["feature_names"],
            help="Counts only hits from years strictly before this track's release year.",
        )
    submitted = st.form_submit_button("Estimate likelihood", type="primary")

record = {
    "explicit": int(explicit),
    "danceability": danceability,
    "energy": energy,
    "key": KEYS[key_label],
    "loudness": loudness,
    "mode": 1 if mode_label == "Major" else 0,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "valence": valence,
    "tempo": tempo,
    "duration_ms": duration_seconds * 1000,
    "time_signature": time_signature,
    "prior_artist_hits": prior_artist_hits,
}

if submitted:
    try:
        result = predict_record(artifact, record)
        explanation = local_sensitivity(artifact, record).head(5)
    except InputValidationError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Prediction failed safely: {exc}")
        st.stop()

    left, right = st.columns([1, 1.4])
    with left:
        st.metric("Calibrated historical hit likelihood", f"{result['probability']:.1%}")
        if result["prediction"]:
            st.success(f"Above the validation-selected decision threshold ({result['threshold']:.1%}).")
        else:
            st.info(f"Below the validation-selected decision threshold ({result['threshold']:.1%}).")
        st.caption(
            "The probability was calibrated on the 2017 catalog. Changing market conditions can make it miscalibrated."
        )
    with right:
        st.subheader("Largest one-at-a-time sensitivities")
        display = explanation[["feature", "probability_change_from_reference"]].copy()
        display["probability_change_from_reference"] = display[
            "probability_change_from_reference"
        ].map(lambda value: f"{value:+.2%}")
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption("Association only: each feature is replaced by its training reference while other inputs stay fixed.")

    with st.expander("Model card and limitations"):
        st.write(
            {
                "model_family": artifact["model_family"],
                "candidate": artifact["candidate"],
                "feature_set": artifact["feature_set"],
                "test_period": artifact["data_periods"]["untouched_test"],
            }
        )
        for limitation in artifact["limitations"]:
            st.markdown(f"- {limitation}")
