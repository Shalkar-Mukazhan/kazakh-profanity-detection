"""Evaluate the saved model ONCE on the held-out test split (+ synthetic product reviews).   python -m src.evaluate"""
import json, os, sys
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, average_precision_score, precision_recall_curve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.textnorm import normalize

b = joblib.load("config.json" and json.load(open("config.json"))["model_path"])
pipe, thr = b["pipeline"], b["threshold"]
os.makedirs("results", exist_ok=True)


def metrics(y, p, name):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"set": name, "n": int(len(y)), "n_label_1": int(np.sum(y)), "n_label_0": int(len(y) - np.sum(y)), "threshold": round(thr, 4),
            "precision_1": round(precision_score(y, pred, zero_division=0), 4), "recall_1": round(recall_score(y, pred), 4),
            "f1_1": round(f1_score(y, pred), 4), "average_precision": round(average_precision_score(y, p), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


df = pd.read_csv("data/processed/dataset.csv", keep_default_na=False)
te = df[df.split == "test"].copy()
te["prob"] = pipe.predict_proba(te.text)[:, 1]; te["pred"] = (te.prob >= thr).astype(int)
out = {"test": metrics(te.label.values, te.prob.values, "held-out test")}
out["test"]["by_domain"] = {d: {"n": int(len(g)), "n_label_1": int(g.label.sum()), "fp": int(((g.pred == 1) & (g.label == 0)).sum()),
                                "fn": int(((g.pred == 0) & (g.label == 1)).sum())} for d, g in te.groupby("domain")}
out["test"]["by_label_method"] = {m: {"n": int(len(g)), "accuracy": round(float((g.pred == g.label).mean()), 4)} for m, g in te.groupby("label_method")}

# product-review test: REAL clean market reviews from the test split + SYNTHETIC reviews (35 profane / 30 clean)
syn = pd.read_csv("data/product_test_synthetic.csv")
syn["prob"] = pipe.predict_proba(syn.text)[:, 1]; syn["pred"] = (syn.prob >= thr).astype(int)
mk = te[te.domain == "market"]
y = np.r_[syn.label.values, mk.label.values]; p = np.r_[syn.prob.values, mk.prob.values]
out["product_test"] = metrics(y, p, f"product reviews: {len(syn)} synthetic + {len(mk)} real clean market reviews")
out["synthetic_only"] = metrics(syn.label.values, syn.prob.values, "synthetic only")
out["synthetic_by_type"] = {t: {"n": int(len(g)), "accuracy": round(float((g.pred == g.label).mean()), 3)} for t, g in syn.groupby("type")}
json.dump(out, open("results/test_metrics.json", "w"), indent=1, ensure_ascii=False)

# test texts are not republished (they may contain personal names); only ids, labels, probabilities. The short error list keeps
# the text, truncated, with person names masked by hand (see NAMES).
NAMES = ["Madi"]
te[["custom_id", "label", "pred", "prob", "label_method", "domain"]].to_csv("results/test_predictions.csv", index=False)
err = te[te.pred != te.label].sort_values("prob", ascending=False)[["custom_id", "label", "pred", "prob", "label_method", "text"]].copy()
err["text"] = err.text.str.replace("|".join(NAMES), "<name>", regex=True).str.slice(0, 200)
err.to_csv("results/test_errors.csv", index=False)
syn.to_csv("results/synthetic_predictions.csv", index=False)

cm = np.array([[out["test"]["confusion_matrix"]["tn"], out["test"]["confusion_matrix"]["fp"]], [out["test"]["confusion_matrix"]["fn"], out["test"]["confusion_matrix"]["tp"]]])
fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"]); ax.set_title(f"Test confusion matrix (n={len(te)})")
plt.tight_layout(); plt.savefig("results/confusion_matrix_test.png", dpi=150); plt.close()
pr, rc, _ = precision_recall_curve(te.label, te.prob)
fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.plot(rc, pr); ax.scatter([out["test"]["recall_1"]], [out["test"]["precision_1"]], c="r", zorder=3, label=f"threshold {thr:.2f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Test precision-recall (class 1)"); ax.legend(); plt.tight_layout(); plt.savefig("results/pr_curve_test.png", dpi=150); plt.close()

vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
names = np.array(vec.get_feature_names_out()); w = clf.coef_[0]; o = np.argsort(w)
pd.DataFrame({"ngram": list(names[o[-25:]][::-1]) + list(names[o[:15]]), "weight": list(w[o[-25:]][::-1].round(3)) + list(w[o[:15]].round(3)),
              "direction": ["profanity"] * 25 + ["clean"] * 15}).to_csv("results/top_ngrams.csv", index=False)
print(json.dumps({k: v for k, v in out.items() if k != "synthetic_by_type"}, indent=1, ensure_ascii=False)); print(out["synthetic_by_type"])
