"""Label, clean, de-duplicate and split the data.   python -m src.prepare_data

1. pool the KazSAnDRA files; mask personal data (URLs, e-mails, phones, @mentions)
2. label with the lexicon (strong -> 1; ambiguous -> hand labels in data/labels/manual_review_ambiguous.csv)
3. remove exact duplicates (an agreeing or conflicting duplicate keeps label 1 if any copy is profane)
4. group near-duplicates (letters only, repeats collapsed) and keep every group inside ONE split
5. sample clean reviews, add the swear-word rows to TRAIN only, split 70/15/15 with the seed from config.json
"""
import hashlib, json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.textnorm import normalize, skeleton, mask_pii, dup_key
from src import lexicon as L

cfg = json.load(open("config.json"))
rng = np.random.default_rng(cfg["seed"])


def h01(key):
    """Deterministic number in [0, 1) from the seed and a key. Used for sampling and for the split, so that correcting a few
    labels does not reshuffle which reviews are in which split."""
    return int(hashlib.md5(f"{cfg['seed']}:{key}".encode()).hexdigest(), 16) / 16**32
os.makedirs("data/processed", exist_ok=True); os.makedirs("data/splits", exist_ok=True)

# 1. pool -------------------------------------------------------------------
parts = [pd.read_csv(os.path.join(cfg["raw_dir"], f), keep_default_na=False) for f in cfg["raw_files"]]
df = pd.concat(parts, ignore_index=True)[["custom_id", "text", "label", "domain"]].rename(columns={"label": "sentiment"})
df["text"] = df.text.astype(str).map(mask_pii).str.replace(r"\s+", " ", regex=True).str.strip()
df = df[df.text.str.len() >= 2].drop_duplicates("custom_id").reset_index(drop=True)
print("pooled reviews:", len(df))

# 2. label ------------------------------------------------------------------
norm = df.text.map(normalize); skel = df.text.map(skeleton)
strong = [L.strong_hits(a, b, raw) for a, b, raw in zip(skel, norm, df.text)]
ambig = [L.ambiguous_hits(a, b) if not s else [] for a, b, s in zip(skel, norm, strong)]
manual = pd.read_csv("data/labels/manual_review_ambiguous.csv").set_index("custom_id").label.to_dict()
label, method = [], []
for cid, s, a in zip(df.custom_id, strong, ambig):
    if cid in manual:                       # a human decision always wins over the lexicon
        label.append(manual[cid]); method.append("manual")
    elif s:
        label.append(1); method.append("lexicon_strong")
    elif a:
        raise SystemExit(f"ambiguous review without a hand label: {cid}")
    else:
        label.append(0); method.append("no_lexicon_hit")
df["label"], df["label_method"] = label, method
# second-pass corrections: items the lexicon missed, found while checking TRAIN/VALIDATION disagreements (never test)
corr = pd.read_csv("data/labels/label_corrections.csv").set_index("custom_id").label.to_dict()
m = df.custom_id.isin(corr)
df.loc[m, "label"] = df.loc[m, "custom_id"].map(corr); df.loc[m, "label_method"] = "manual_correction"
print(df.label_method.value_counts().to_dict(), "| profane:", int(df.label.sum()))

# 3. exact duplicates -------------------------------------------------------
df["exact"] = df.text.str.lower().str.replace(r"\s+", " ", regex=True)
df = df.sort_values("label", ascending=False).drop_duplicates("exact").reset_index(drop=True)
print("after exact-duplicate removal:", len(df))

# 4. near-duplicate groups --------------------------------------------------
df["group"] = df.text.map(dup_key)
df.loc[df.group == "", "group"] = df.exact
df["source"] = "kazsandra"

# 5. sampling -----------------------------------------------------------------
prof = df[df.label == 1]
manual0 = df[(df.label == 0) & (df.label_method == "manual")]       # hard negatives: look suspicious, are clean
clean = df[(df.label == 0) & (df.label_method == "no_lexicon_hit")]
def take(pool, n):
    # market (product) reviews first: they are the target domain; rows are ranked by a hash of their id (deterministic)
    pool = pool.assign(_r=pool.custom_id.map(h01)).sort_values("_r")
    mk = pool[pool.domain == "market"]; ot = pool[pool.domain != "market"]
    n_mk = min(len(mk), n // 2)
    return pd.concat([mk.head(n_mk), ot.head(n - n_mk)]).drop(columns="_r")
neg = take(clean[clean.sentiment == 0], cfg["n_clean_negative_sentiment"])    # sentiment 0 = negative review
pos = take(clean[clean.sentiment == 1], cfg["n_clean_positive_sentiment"])
data = pd.concat([prof, manual0, neg, pos], ignore_index=True)

# group-level split by hash: every group is assigned to ONE split, 70 / 15 / 15 -------------------------------
def split_of_group(g):
    u = h01("group:" + g)
    return "train" if u < cfg["split"]["train"] else "val" if u < cfg["split"]["train"] + cfg["split"]["val"] else "test"
split_of = {g: split_of_group(g) for g in data.group.unique()}
data["split"] = data.group.map(split_of)

# swear-word dataset: TRAIN only (hand-written words/phrases, not real reviews) ----------
sw = pd.concat([pd.read_json(f"data/raw/swear/{f}", lines=True) for f in ("data.jsonl", "sentences.jsonl")], ignore_index=True)
sw_lab = [int(c in ("vulgar", "offensive") or bool(L.strong_hits(skeleton(t), normalize(t)))) for t, c in zip(sw.text, sw.category)]
sw_rows = pd.DataFrame({"custom_id": [f"swear{i:03d}" for i in range(len(sw))], "text": sw.text, "label": sw_lab, "domain": "swear_lexicon",
                        "sentiment": -1, "label_method": "swear_dataset_" + sw.category, "group": [dup_key(t) for t in sw.text],
                        "source": "swear_lexicon", "split": "train"})
sw_rows = sw_rows[~sw_rows.group.isin(data.group)]
data = pd.concat([data, sw_rows], ignore_index=True)

cols = ["custom_id", "text", "label", "label_method", "domain", "sentiment", "source", "group", "split"]
data[cols].to_csv("data/processed/dataset.csv", index=False)
for s in ("train", "val", "test"):
    data[data.split == s][["custom_id", "label", "label_method", "domain", "source", "group", "split"]].to_csv(f"data/splits/{s}_ids.csv", index=False)

# leak check + statistics ---------------------------------------------------------
gs = data.groupby("group").split.nunique()
stats = {"seed": cfg["seed"], "total": len(data), "groups_in_more_than_one_split": int((gs > 1).sum()),
         "by_split": {s: {"n": int((data.split == s).sum()), "label_0": int(((data.split == s) & (data.label == 0)).sum()),
                          "label_1": int(((data.split == s) & (data.label == 1)).sum())} for s in ("train", "val", "test")},
         "by_label_method": data.label_method.value_counts().to_dict(), "by_domain_label1": data[data.label == 1].domain.value_counts().to_dict(),
         "by_domain_total": data.domain.value_counts().to_dict()}
json.dump(stats, open("data/splits/stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(stats, indent=1, ensure_ascii=False))
