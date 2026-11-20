"""Predict with the saved model (no retraining).

  python -m src.predict "Тауар сапасы нашар"        # one or more comments as arguments
  python -m src.predict                               # interactive: type comments, empty line to stop
"""
import os, sys
import joblib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODEL = os.path.join(ROOT, "models", "profanity_tfidf_logreg.joblib")
_bundle = None


def load(path=MODEL):
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def predict(texts):
    """texts: str or list of str -> list of dicts {text, label (0/1), prob_profanity, threshold}"""
    b = load()
    if isinstance(texts, str):
        texts = [texts]
    probs = b["pipeline"].predict_proba(list(texts))[:, 1]
    return [{"text": t, "label": int(p >= b["threshold"]), "prob_profanity": round(float(p), 4), "threshold": round(b["threshold"], 4)}
            for t, p in zip(texts, probs)]


def explain(text, top=8):
    """Trace one comment: raw text -> normalised text -> n-gram features -> probability. Returns a dict."""
    import numpy as np
    from src.textnorm import normalize
    b = load(); vec, clf = b["pipeline"].named_steps["tfidf"], b["pipeline"].named_steps["clf"]
    x = vec.transform([text])
    names = vec.get_feature_names_out()
    contrib = x.multiply(clf.coef_[0]).tocoo()           # tf-idf value * logistic-regression weight, per n-gram
    parts = sorted(zip(contrib.data, names[contrib.col]), reverse=True)
    p = float(b["pipeline"].predict_proba([text])[0, 1])
    return {"raw": text, "normalized": normalize(text), "n_features": int(x.nnz), "bias": float(clf.intercept_[0]),
            "top_n_grams_for_profanity": [(g, round(float(v), 3)) for v, g in parts[:top] if v > 0],
            "top_n_grams_for_clean": [(g, round(float(v), 3)) for v, g in parts[::-1][:top] if v < 0],
            "prob_profanity": round(p, 4), "threshold": round(b["threshold"], 4), "label": int(p >= b["threshold"])}


def _show(r):
    print(f'{r["label"]} ({"PROFANITY" if r["label"] else "clean"})  p={r["prob_profanity"]:.3f}  thr={r["threshold"]:.3f}  | {r["text"]}')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for r in predict(sys.argv[1:]): _show(r)
    else:
        print("Type a Kazakh comment (empty line to quit):")
        for line in sys.stdin:
            if not line.strip(): break
            _show(predict(line.strip())[0])
