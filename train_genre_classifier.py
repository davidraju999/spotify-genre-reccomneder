"""Step 1: EDA + genre classification (predict a song's genre from its audio features)."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import FEATURES, load_data

SEED = 42
sns.set_theme(style="whitegrid")
df = load_data()

# ---------------- EDA ----------------
df.genre.value_counts().plot(kind="barh", figsize=(7, 4), color="#1db954")
plt.title("Songs per genre"); plt.tight_layout(); plt.savefig("outputs/01_genre_counts.png", dpi=150); plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df[FEATURES].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("Audio feature correlations"); plt.tight_layout()
plt.savefig("outputs/02_correlation.png", dpi=150); plt.close()

df.groupby("genre")[["danceability", "energy", "valence", "speechiness", "acousticness"]].mean().plot(
    kind="bar", figsize=(10, 5))
plt.title("Average audio profile by genre"); plt.tight_layout()
plt.savefig("outputs/03_genre_profiles.png", dpi=150); plt.close()

# ---------------- Modeling ----------------
X, y = df[FEATURES], df.genre
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
    cv_acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1).mean()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    rows.append({"Model": name, "CV accuracy": cv_acc, "Test accuracy": accuracy_score(y_te, pred),
                 "Macro F1": f1_score(y_te, pred, average="macro")})
    fitted[name] = pipe
res = pd.DataFrame(rows).set_index("Model").round(4)
print(res)
res.to_csv("outputs/model_comparison.csv")

best_name = res["Macro F1"].idxmax()
best = fitted[best_name]
print("\nBest model:", best_name)
print(classification_report(y_te, best.predict(X_te)))
joblib.dump(best, "models/genre_classifier.joblib")

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_estimator(best, X_te, y_te, xticks_rotation=45, cmap="Greens",
                                      normalize="true", values_format=".2f", ax=ax)
ax.set_title(f"Confusion matrix (normalized) - {best_name}")
plt.tight_layout(); plt.savefig("outputs/04_confusion_matrix.png", dpi=150); plt.close()

imp = permutation_importance(best, X_te, y_te, scoring="f1_macro", n_repeats=5,
                             random_state=SEED, n_jobs=-1)
imp_s = pd.Series(imp.importances_mean, index=FEATURES).sort_values().tail(10)
plt.figure(figsize=(8, 5)); imp_s.plot(kind="barh", color="#1db954")
plt.title("Top features for genre prediction"); plt.xlabel("Drop in macro F1 when shuffled")
plt.tight_layout(); plt.savefig("outputs/05_feature_importance.png", dpi=150); plt.close()
print("\nTop 5 features:\n", imp_s.tail(5)[::-1].round(4))

json.dump({"best_model": best_name, **res.loc[best_name].to_dict(),
           "top_features": list(imp_s.tail(5)[::-1].index)},
          open("outputs/classifier_metrics.json", "w"), indent=2)
