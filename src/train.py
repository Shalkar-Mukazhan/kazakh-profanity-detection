"""Train char n-gram TF-IDF + logistic regression.   python -m src.train

The vectoriser (vocabulary, IDF) is fitted on the TRAIN split only. n-gram range, C and the decision threshold
are selected on the VALIDATION split only. The test split is not touched here.
"""
import json, os, sys, itertools
import numpy as np, pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.textnorm import normalize

cfg = json.load(open("config.json"))
df = pd.read_csv("data/processed/dataset.csv", keep_default_na=False)
tr, va = df[df.split == "train"], df[df.split == "val"]
print("train", len(tr), "val", len(va))


def make_pipeline(ngram, C):
    t = cfg["tfidf"]
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, analyzer=t["analyzer"], ngram_range=tuple(ngram), min_df=t["min_df"],
                                  max_features=t["max_features"], sublinear_tf=t["sublinear_tf"])),
        ("clf", LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=cfg["seed"])),
    ])


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.argmax(f1[:-1]))
    return float(thr[i]), float(f1[i])


if __name__ == "__main__":
    log = []
    for ngram, C in itertools.product(cfg["ngram_grid"], cfg["C_grid"]):
        pipe = make_pipeline(ngram, C).fit(tr.text, tr.label)
        p = pipe.predict_proba(va.text)[:, 1]
        thr, f1 = best_threshold(va.label, p)
        log.append({"ngram": ngram, "C": C, "val_f1_at_0.5": round(f1_score(va.label, p >= 0.5), 4), "threshold": round(thr, 4), "val_f1_at_best_thr": round(f1, 4)})
        print(log[-1])
    # Many settings tie on the validation set (differences of 0.0005 are noise). Among all settings within 0.001 of the best
    # validation F1 we take the simplest one: strongest regularisation (smallest C), then the smaller n-gram range.
    top = max(r["val_f1_at_best_thr"] for r in log)
    best = min((r for r in log if r["val_f1_at_best_thr"] >= top - 0.001), key=lambda r: (r["C"], r["ngram"][1]))
    print("selected:", best)
    final = make_pipeline(best["ngram"], best["C"]).fit(tr.text, tr.label)
    os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
    joblib.dump({"pipeline": final, "threshold": best["threshold"], "C": best["C"], "ngram": best["ngram"], "config": cfg}, cfg["model_path"])
    os.makedirs("results", exist_ok=True)
    json.dump({"selected": best, "grid": log, "n_vocab": len(final.named_steps["tfidf"].vocabulary_)}, open("results/validation.json", "w"), indent=1)
    print("saved", cfg["model_path"], "| vocabulary size", len(final.named_steps["tfidf"].vocabulary_))
