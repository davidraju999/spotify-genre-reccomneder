# Song Genre Predictor & Recommender

Two ML systems built on Spotify audio features (danceability, energy, tempo, loudness, etc.):
1. **Genre classifier** - predicts a song's genre from its audio features.
2. **Content-based recommender** - finds the most similar songs, evaluated with precision@k.

A Streamlit app lets you try both.

## Setup (VS Code terminal)
```bash
python -m venv venv
venv\Scripts\activate        # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Use REAL data (recommended for your resume)
1. Download `genres_v2.csv` from Kaggle (search "Spotify Genre Classification" / "Dataset of songs in Spotify").
2. Put it in the `data/` folder.
3. Run everything below. The code detects the file automatically.

Without that file, the project runs on a **synthetic** dataset (8 genres, 12,000 songs) so you can test it.

## Run
```bash
python train_genre_classifier.py   # EDA charts + model comparison + saves model   -> outputs/, models/
python recommender.py              # precision@10: naive vs improved recommender
streamlit run app.py               # interactive demo in your browser
```

## What's different from a basic Spotify recommender
| Basic version | This version |
|---|---|
| Euclidean distance on raw features | Standardised features + cosine similarity |
| Python loops over every row | sklearn NearestNeighbors |
| Recommends only inside one genre | Searches all songs |
| Exact name match | Fuzzy name search |
| No metrics | Precision@k, CV accuracy, macro F1 |
| No genre prediction | 3-model classifier comparison |
| Terminal input() | Streamlit app |

## Resume bullets (fill the XX after running on the REAL dataset)
**Song Genre Prediction & Recommendation System** | Python, scikit-learn, pandas, Streamlit | GitHub link
- Built a genre classifier on XX,XXX Spotify tracks across XX genres, comparing Logistic Regression, Random Forest and Gradient Boosting with 5-fold CV (best: XX% accuracy, XX macro F1).
- Developed a content-based recommender using standardised audio features and cosine k-NN; raised precision@10 from XX to XX versus an unscaled Euclidean baseline.
- Used permutation importance to find the audio features that drive genre (e.g. tempo, speechiness) and visualised results with matplotlib/seaborn.
- Deployed an interactive Streamlit app for song recommendations and live genre prediction.

> Do NOT paste numbers from the synthetic run onto your resume. Run on the real data and use those.

## Interview talking points
- Why scale features? Tempo (~120) and loudness (~-6) would swamp 0-1 features in distance calculations.
- Why cosine? It compares the "shape" of the audio profile, and works well on standardised features.
- Why precision@k? There are no user ratings, so genre match is a proxy for relevance. Its limit: two songs can be similar without sharing a genre label.
- Limitations / next steps: add user listening data (collaborative filtering), hybrid model, time-based evaluation.
