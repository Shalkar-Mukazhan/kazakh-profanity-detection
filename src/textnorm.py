"""Text normalisation used both for labelling support and inside the trained model.

Two levels:
  normalize(text)  - what the classifier sees (keeps Kazakh letters, undoes obfuscation)
  skeleton(text)   - aggressive form used only to match the lexicon (merges look-alike letters)
"""
import re

URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\-\s()]{8,}\d)(?!\w)")
MENTION = re.compile(r"(?<!\w)@\w+")

# Latin letters that look like Cyrillic ones (used inside mixed-script words, e.g. "кoтaк")
LOOKALIKE = str.maketrans("aeopcxykhmtbu", "аеорсхукнмтьи")
# Latin -> Cyrillic for words typed fully in Latin ("qotaq" -> "қотақ")
TRANSLIT_MULTI = [("sh", "ш"), ("ch", "ч"), ("ya", "я"), ("yu", "ю"), ("yo", "ё"), ("zh", "ж"), ("kh", "х"), ("gh", "ғ"), ("ng", "ң")]
TRANSLIT = {"a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "e": "е", "z": "з", "i": "и", "j": "ж", "k": "к",
            "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т", "u": "у", "f": "ф",
            "h": "х", "c": "с", "q": "қ", "w": "в", "x": "кс", "y": "й",
            "ä": "ә", "ǵ": "ғ", "ń": "ң", "ó": "ө", "ú": "ү", "ı": "і", "ý": "у", "á": "а", "ş": "ш", "ç": "ч"}

LATIN = re.compile(r"[a-zäǵńóúıýáşç]", re.I)
CYR = re.compile(r"[а-яёәғқңөұүһі]", re.I)
SPACED = re.compile(r"(?<!\w)(?:\w[\s.\-_]){2,}\w(?!\w)")   # "к о т а к", "қ.о.т.а.қ"
REPEAT3 = re.compile(r"(.)\1{2,}")


def _translit_word(w):
    for a, b in TRANSLIT_MULTI:
        w = w.replace(a, b)
    return "".join(TRANSLIT.get(ch, ch) for ch in w)


def _fix_word(w):
    has_lat, has_cyr = bool(LATIN.search(w)), bool(CYR.search(w))
    if has_lat and not has_cyr:
        return _translit_word(w)
    if has_lat and has_cyr:                      # mixed script: fold look-alike Latin letters
        return w.translate(LOOKALIKE)
    return w


def mask_pii(text):
    """Replace URLs, e-mails, phone numbers and @mentions by placeholders (first step of normalize)."""
    t = str(text)
    t = URL.sub(" <url> ", t)
    t = EMAIL.sub(" <email> ", t)
    t = PHONE.sub(" <phone> ", t)
    return MENTION.sub(" <user> ", t)


def normalize(text):
    t = mask_pii(text).lower().replace("\u0259", "\u04d9")   # Latin schwa -> Kazakh ә
    t = SPACED.sub(lambda m: re.sub(r"[\s.\-_]", "", m.group(0)), t)       # join spaced letters
    t = re.sub(r"(?<=\w)[*#@$%]+(?=\w)", "", t)                              # b**t -> bt, q*taq -> qtaq
    t = re.sub(r"<(?:url|email|phone|user)>|[\w']+", lambda m: m.group(0) if m.group(0).startswith("<") else _fix_word(m.group(0)), t)
    t = REPEAT3.sub(r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


# --- aggressive skeleton for lexicon matching only ---
SKEL = str.maketrans({"қ": "к", "ә": "а", "ө": "о", "ү": "у", "ұ": "у", "і": "и", "ң": "н", "ғ": "г", "һ": "х", "ё": "е", "й": "и", "ы": "и"})


def skeleton(text):
    t = normalize(text).translate(SKEL)
    return re.sub(r"(.)\1+", r"\1", t)           # collapse every repeated letter


def dup_key(text):
    """Key for exact/near duplicates: letters only, repeats collapsed."""
    t = re.sub(r"[^а-яёәғқңөұүһіa-z0-9]+", "", str(text).lower())
    return re.sub(r"(.)\1+", r"\1", t)
