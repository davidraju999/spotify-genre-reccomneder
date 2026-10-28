"""Step 2: Content-based song recommender + evaluation.

Improvements over a basic "euclidean distance inside one genre" approach:
  * features are standardised so tempo / loudness don't dominate
  * cosine similarity with sklearn NearestNeighbors (fast, no Python loops)
  * fuzzy song-name search (difflib)
  * measured with precision@k so the improvement is provable
"""
import difflib

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from data_loader import FEATURES, load_data


class SongRecommender:
    def __init__(self, df: pd.DataFrame, metric: str = "cosine", scale: bool = True):
        self.df = df.reset_index(drop=True)
        self.scaler = StandardScaler() if scale else None
        X = self.df[FEATURES].values
        self.X = self.scaler.fit_transform(X) if scale else X
        self.nn = NearestNeighbors(metric=metric, algorithm="brute").fit(self.X)
        self._names_lower = self.df.song_name.str.lower().tolist()

    def find_song(self, query: str):
        """Fuzzy match a typed name to a song in the dataset; returns row index or None."""
        q = query.lower().strip()
        if q in self._names_lower:
            return self._names_lower.index(q)
        match = difflib.get_close_matches(q, self._names_lower, n=1, cutoff=0.5)
        return self._names_lower.index(match[0]) if match else None

    def recommend(self, song_name: str, k: int = 10) -> pd.DataFrame:
        idx = self.find_song(song_name)
        if idx is None:
            raise ValueError(f"No song similar to '{song_name}' found.")
        dist, ind = self.nn.kneighbors(self.X[idx:idx + 1], n_neighbors=k + 1)
        out = self.df.iloc[ind[0][1:]][["song_name", "genre"]].copy()
        out["similarity"] = (1 - dist[0][1:]) if self.nn.metric == "cosine" else -dist[0][1:]
        out.insert(0, "query_song", self.df.song_name.iloc[idx])
        return out.reset_index(drop=True)

    def precision_at_k(self, k: int = 10, n_queries: int = 1500, seed: int = 42) -> float:
        """Share of the top-k recommendations that have the SAME genre as the query song."""
        rng = np.random.default_rng(seed)
        q = rng.choice(len(self.df), size=min(n_queries, len(self.df)), replace=False)
        _, ind = self.nn.kneighbors(self.X[q], n_neighbors=k + 1)
        genres = self.df.genre.values
        return float(np.mean([(genres[ind[i][1:]] == genres[q[i]]).mean() for i in range(len(q))]))


if __name__ == "__main__":
    df = load_data()
    k = 10
    baseline = SongRecommender(df, metric="euclidean", scale=False)   # "naive" version
    improved = SongRecommender(df, metric="cosine", scale=True)        # this project
    p_base, p_new = baseline.precision_at_k(k), improved.precision_at_k(k)
    random_guess = float((df.genre.value_counts(normalize=True) ** 2).sum())
    print(f"Precision@{k} (genre match)")
    print(f"  Random guess               : {random_guess:.3f}")
    print(f"  Euclidean, unscaled (naive): {p_base:.3f}")
    print(f"  Cosine, scaled (improved)  : {p_new:.3f}")

    demo = df.song_name.iloc[0]
    print(f"\nTop {k} songs similar to '{demo}':")
    print(improved.recommend(demo, k)[["song_name", "genre", "similarity"]].to_string(index=False))
