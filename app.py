"""Step 3: Streamlit demo.  Run:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

from data_loader import FEATURES, load_data
from recommender import SongRecommender

st.set_page_config(page_title="Song Genre Predictor & Recommender", page_icon="🎵")
st.title("🎵 Song Genre Predictor & Recommender")


@st.cache_resource
def get_models():
    df = load_data()
    return df, SongRecommender(df), joblib.load("models/genre_classifier.joblib")


df, rec, clf = get_models()
tab1, tab2 = st.tabs(["Recommend songs", "Predict genre"])

with tab1:
    name = st.text_input("Type a song name", value=df.song_name.iloc[0])
    k = st.slider("How many recommendations?", 3, 20, 10)
    if st.button("Recommend"):
        try:
            st.dataframe(rec.recommend(name, k)[["song_name", "genre", "similarity"]],
                         use_container_width=True)
        except ValueError as e:
            st.error(str(e))

with tab2:
    st.write("Set the audio features of a song and the model predicts its genre.")
    c1, c2 = st.columns(2)
    vals = {
        "danceability": c1.slider("danceability", 0.0, 1.0, 0.7),
        "energy": c1.slider("energy", 0.0, 1.0, 0.7),
        "speechiness": c1.slider("speechiness", 0.0, 1.0, 0.1),
        "acousticness": c1.slider("acousticness", 0.0, 1.0, 0.1),
        "instrumentalness": c1.slider("instrumentalness", 0.0, 1.0, 0.0),
        "liveness": c2.slider("liveness", 0.0, 1.0, 0.15),
        "valence": c2.slider("valence", 0.0, 1.0, 0.5),
        "loudness": c2.slider("loudness (dB)", -30.0, 0.0, -6.0),
        "tempo": c2.slider("tempo (BPM)", 60, 220, 120),
    }
    row = {f: float(df[f].median()) for f in FEATURES}   # defaults for the other features
    row.update(vals)
    X = pd.DataFrame([row])[FEATURES]
    if st.button("Predict genre"):
        proba = pd.Series(clf.predict_proba(X)[0], index=clf.classes_).sort_values(ascending=False)
        st.success(f"Predicted genre: **{proba.index[0]}**  ({proba.iloc[0]:.0%} confidence)")
        st.bar_chart(proba.head(5))
