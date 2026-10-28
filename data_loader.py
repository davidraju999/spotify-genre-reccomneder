"""Loads the song dataset.

If data/genres_v2.csv exists (Kaggle "Spotify Genre Classification" file), it is used.
Otherwise a SYNTHETIC dataset is generated so the project still runs out of the box.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/genres_v2.csv")

FEATURES = ["danceability", "energy", "key", "loudness", "mode", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo",
            "duration_ms", "time_signature"]

# rough audio "personality" of each genre (used only for synthetic data)
_PROFILES = {
    "Trap":     dict(danceability=.78, energy=.65, loudness=-6.5, speechiness=.25, acousticness=.10, instrumentalness=.02, liveness=.17, valence=.40, tempo=140),
    "Hiphop":   dict(danceability=.74, energy=.62, loudness=-7.0, speechiness=.28, acousticness=.15, instrumentalness=.01, liveness=.18, valence=.55, tempo=100),
    "RnB":      dict(danceability=.68, energy=.55, loudness=-7.5, speechiness=.10, acousticness=.30, instrumentalness=.02, liveness=.15, valence=.50, tempo=105),
    "Pop":      dict(danceability=.65, energy=.65, loudness=-6.0, speechiness=.06, acousticness=.20, instrumentalness=.00, liveness=.16, valence=.55, tempo=118),
    "Techno":   dict(danceability=.78, energy=.80, loudness=-8.0, speechiness=.06, acousticness=.02, instrumentalness=.70, liveness=.13, valence=.20, tempo=128),
    "Trance":   dict(danceability=.60, energy=.85, loudness=-7.0, speechiness=.05, acousticness=.02, instrumentalness=.60, liveness=.14, valence=.25, tempo=138),
    "DnB":      dict(danceability=.55, energy=.90, loudness=-5.5, speechiness=.07, acousticness=.03, instrumentalness=.45, liveness=.17, valence=.30, tempo=174),
    "Hardstyle": dict(danceability=.50, energy=.95, loudness=-4.5, speechiness=.08, acousticness=.01, instrumentalness=.40, liveness=.20, valence=.25, tempo=150),
}


def _synthetic(n_per_genre: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for genre, p in _PROFILES.items():
        n = n_per_genre
        d = pd.DataFrame({
            "danceability": np.clip(rng.normal(p["danceability"], .12, n), 0, 1),
            "energy": np.clip(rng.normal(p["energy"], .12, n), 0, 1),
            "key": rng.integers(0, 12, n),
            "loudness": np.clip(rng.normal(p["loudness"], 2.5, n), -30, 0),
            "mode": rng.choice([0, 1], n, p=[.4, .6]),
            "speechiness": np.clip(rng.normal(p["speechiness"], .08, n), 0, 1),
            "acousticness": np.clip(rng.normal(p["acousticness"], .12, n), 0, 1),
            "instrumentalness": np.clip(rng.normal(p["instrumentalness"], .25, n), 0, 1),
            "liveness": np.clip(rng.normal(p["liveness"], .08, n), 0, 1),
            "valence": np.clip(rng.normal(p["valence"], .18, n), 0, 1),
            "tempo": np.clip(rng.normal(p["tempo"], 14, n), 60, 220),
            "duration_ms": np.clip(rng.normal(210_000, 55_000, n), 60_000, 600_000).astype(int),
            "time_signature": rng.choice([3, 4, 5], n, p=[.05, .92, .03]),
        })
        d["genre"] = genre
        d["song_name"] = [f"{genre} Track {i + 1}" for i in range(n)]
        frames.append(d)
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_data() -> pd.DataFrame:
    """Returns a DataFrame with FEATURES + song_name + genre."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, low_memory=False)
        # song_name is often empty in this file, so fall back to the 'title' column
        name = df["song_name"] if "song_name" in df else pd.Series(index=df.index, dtype=object)
        if "title" in df:
            name = name.fillna(df["title"])
        df["song_name"] = name
        df = df.dropna(subset=["song_name", "genre"]).drop_duplicates(subset=["song_name"])
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
        print(f"[data] Loaded REAL dataset: {len(df):,} songs, {df.genre.nunique()} genres")
    else:
        df = _synthetic()
        print(f"[data] data/genres_v2.csv not found -> using SYNTHETIC data: {len(df):,} songs, "
              f"{df.genre.nunique()} genres")
    return df[FEATURES + ["song_name", "genre"]].reset_index(drop=True)


if __name__ == "__main__":
    print(load_data().head())
