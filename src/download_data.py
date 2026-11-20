"""Download the public data into data/raw/ (no account or token needed).

KazSAnDRA (Yeshpanov & Varol, 2024, CC BY 4.0) is published on GitHub: https://github.com/IS2AI/KazSAnDRA
We use the polarity-classification files 01 (train), 04 (valid) and 05 (test) as ONE pool of reviews and
re-split it ourselves (see prepare_data.py). The sentiment labels are not used as profanity labels.
Kazakh Swear Words (MIT): https://huggingface.co/datasets/Rustem-Kaimolla/kazakh-swear-words
"""
import io, os, urllib.request, zipfile

GH = "https://github.com/IS2AI/KazSAnDRA/raw/main/dataset/"
SWEAR = "https://huggingface.co/datasets/Rustem-Kaimolla/kazakh-swear-words/resolve/main/"
os.makedirs("data/raw/kazsandra", exist_ok=True)
os.makedirs("data/raw/swear", exist_ok=True)

for z in ("01_pc_train_ib", "04_pc_valid", "05_pc_test"):
    print("downloading", z)
    blob = urllib.request.urlopen(GH + z + ".zip", timeout=120).read()
    zipfile.ZipFile(io.BytesIO(blob)).extractall("data/raw/kazsandra")
for f in ("data.jsonl", "sentences.jsonl"):
    print("downloading", f)
    open("data/raw/swear/" + f, "wb").write(urllib.request.urlopen(SWEAR + f, timeout=120).read())
print("done -> data/raw/")
