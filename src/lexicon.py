"""Lexicon of obscene word roots (Kazakh, plus Russian roots used inside Kazakh text).

It is used ONLY to find candidate profane reviews when labelling the data. The trained model never sees it.

Two matching levels (see textnorm.py):
  skeleton(text)   - aggressive: қ->к, ә->а, ы/і/й->и, repeats collapsed. Good for spelling variants.
  normalize(text)  - keeps ы/і/қ. Needed where the skeleton would merge a profane word with an innocent one
                     (сіктір vs сықтыр, сіг-ейін vs сыйға "to fit", амын vs әмин "Amen").
STRONG    = obscene in practically every use -> auto label 1 (100 random hits are audited by hand).
AMBIGUOUS = also matches innocent words/uses -> every hit is read and labelled by hand.
"""
import re

# ---- STRONG, matched on skeleton ----
STRONG_SKEL = {
    "kotak": r"\bкотак|\bкотаг(?:и|м|ы|а)",           # қотақ, қотақбас, қотағым(ды)
    "tasak": r"\bтасак",                              # тасақ
    "kanshik": r"\bканшик",                           # қаншық
    "zhalep": r"\bжалеп|\bжалаптар",                  # жәлеп (pimp, slur), жалаптар (plural slur)
    "kotok": r"\bкотокт[ао]р",                        # котоктар / котоктор
    "mother_am": r"\b(?:шешен|шешес|шеше|щещен|щеще|шишин|шишен|шен|енен|ене|анан|анаң)\w*?\s?(?:ам(?:и|ин|ик|ына|ыс)?\b|сги\w*|сигеин\w*|сигин\w*)",   # шешеңнің амы, шешенамы, шешең сгин
    "eba_slang": r"\bеба(?:и|а|\b)",                  # ебаит / ебайт / еба(а): Russian 'ебать' as exclamation
    "ru_hui": r"\bх(?:у|ую)(?:и|е|я|ю)|\bнах(?:уи|уя|у)\b|\bнихуя",   # хуй, хуе-, хуя, наху, нихуя
    "ru_pizd": r"\bпизд",
    "ru_blya": r"\bбля(?:д|т|х|к|\b)|\bблия",
    "ru_eb": r"\b(?:за|на|от|вы|съ|у|до)?е(?:б(?:ал|ан|ат|ну|ис|ут|ли|ач|ак)|иб(?:ал|ан|ат))|\bзаеб|\bзаи ?б(?:ал|ис)|\bзаиб",
    "ru_dalbaeb": r"\b(?:далба|долба|далбо|долбо|балба)(?:еб|ев|еп|об|иоп|иов|иоб|оп)",
    "ru_tvoyu_mat": r"тво[юи] ?мать",
    "ru_mudak": r"\bмудак|\bмудил",
    "ru_pidor": r"\bпидор|\bпидар|\bпедик(?!ю)",
    "ru_gandon": r"\bганд?он",
    "ru_zalup": r"\bзалуп",
    "ru_shlyuh": r"\bшлюх",
    "ru_suka": r"\bсук(?:а|и|ам|ами|ой|у)\b|\bсучк|\bсучар",
}
# ---- STRONG, matched on normalize(text) ----
STRONG_NORM = {
    "shshs": r"\b(?:шшс|щщс|шщс|щшс)\b",
    "kot_rude": r"\bкөтт?\b",                                                   # standalone көт (with ө): the vulgar word, usually evaluative ("көт ойын")
    "blya_kz": r"\bбл[ә]+\b",                                                   # блә / бләә: Kazakh spelling of бля                                       # abbreviation of a Kazakh obscene phrase
    "sik": r"\bс[іи]кк[іи]р|\bс[іи]к(?:т(?!ивкар)|ей[іи]н|ем\b|пе(?:[іи]|\b))",             # сікті, сіктір, сікейін, сікпеймін
    "sig_verb": r"\bс[іи]г(?:ей[іи]н|ем\b|[іи]л(?:[іи]п|\b|[іи]н|ген)|[іи]п\b)",   # сігейін, сігіл, сігіліп, сігілген
}
# ---- AMBIGUOUS (skeleton) ----
AMBIG_SKEL = {
    "kot": r"\bкот(?!ак)",                  # қот/көт (arse) vs көтеру, SMS code "кот", Kotlin
    "am": r"\bам(?:и|ин|ик|иш|бас|дар|ги|из)?\b",   # ам / амы / амын (vulgar) vs "I am", AM, Instagram...
    "zhalap": r"\bжалап",                   # жалап (to lick) vs slur
}
# ---- AMBIGUOUS (normalize) ----
AMBIG_NORM = {
    "sig_other": r"\bс[іи]г(?!нал|ма\b|ар)",   # other сіг-; сыйға / сығады are spelled with ы and do not match
}
# Words that contain an ambiguous root but are clean. A review whose ONLY ambiguous hits are these gets label 0.
INNOCENT = re.compile(r"\b(?:кот[еи]р\w*|котел\w*|котле\w*|котедж\w*|котор\w*|котлин\w*|сигнал\w*|сигма|сигар\w*)\b")
# 'Amen' (әмин/амин) after a prayer is not the vulgar ам-
AMEN = re.compile(r"\b[аә]ми+н\b")
AMEN_CONTEXT = re.compile(r"алла|аллах|аллаһ|раз[ыі]|береке|иншал|иншаа|бата|құдай|кудай|тілей|тилей|дұға|дуга|нәсіп|насип")

_c = lambda d: {k: re.compile(v) for k, v in d.items()}
STRONG_SKEL_RE, STRONG_NORM_RE, AMBIG_SKEL_RE, AMBIG_NORM_RE = map(_c, (STRONG_SKEL, STRONG_NORM, AMBIG_SKEL, AMBIG_NORM))


def strong_hits(skel, norm, raw=None):
    h = [k for k, r in STRONG_SKEL_RE.items() if r.search(skel)] + [k for k, r in STRONG_NORM_RE.items() if r.search(norm)]
    if raw is not None:
        h += masked_hits(raw)
    return h


def ambiguous_hits(skel, norm):
    """Ambiguous matches that are not explained by an innocent word or an 'Amen'."""
    out = []
    for k, r in AMBIG_SKEL_RE.items():
        for m in r.finditer(skel):
            word = re.findall(r"\w+", skel[m.start():])[0]
            if INNOCENT.fullmatch(word):
                continue
            if k == "am" and AMEN.search(norm) and AMEN_CONTEXT.search(norm) and re.fullmatch(r"ам(?:и|ин)", word):
                continue
            out.append(k); break
    for k, r in AMBIG_NORM_RE.items():
        if r.search(norm):
            out.append(k)
    return out


# ---- masked profanity: letters kept, the rest replaced by * # @ $ % ! (қ*тақ, sig*****p, за#!%#) ----
MASK_CH = "*#@$%!"
MASK_WORDS = ["котак", "котакбас", "сикти", "сиктир", "сигеин", "сигил", "сигилип", "сикпеи", "заибал", "заебал", "пиздец", "пиздес",
              "блят", "блять", "блядь", "бляд", "хуи", "хуиня", "нахуи", "ебат", "ебать", "ебаит", "ебан", "сука", "далбаеб", "каншик", "пидар", "мудак", "жалеп"]
_TOK = re.compile(r"[\w*#@$%!]+")


def masked_hits(raw):
    """Match tokens that contain mask characters against MASK_WORDS: letters must match as a prefix/suffix pattern and each
    mask run of k characters may stand for k-2 .. k+2 letters. Needs at least 2 letters and a mask between or after them."""
    from src.textnorm import SKEL, normalize
    out = []
    for tok in _TOK.findall(str(raw).lower()):
        if not any(c in MASK_CH for c in tok) or sum(c.isalpha() for c in tok) < 2 or tok[0] in MASK_CH:
            continue
        tok = tok.rstrip("!")                     # "!" at the very end is punctuation, not a mask ("бля!")
        if not any(c in MASK_CH for c in tok):
            continue
        parts = re.split("([" + re.escape(MASK_CH) + "]+)", tok)
        rx = ""
        for part in parts:
            if not part:
                continue
            if part[0] in MASK_CH:
                rx += "[а-яәғқңөұүһі]{%d,%d}" % (max(0, len(part) - 2), len(part) + 2)
            else:
                rx += re.escape(re.sub(r"(.)\1+", r"\1", normalize(part).translate(SKEL)))
        r = re.compile("^" + rx + "$")
        for w in MASK_WORDS:
            if r.match(w):
                out.append("masked:" + w); break
    return out
