"""Text normalisation shared by the index (codes) and the queries (notes).

Both sides must go through the same functions, otherwise "anaemia" in a title
and "anemia" in a note would never meet.
"""
from __future__ import annotations

import re
import unicodedata

from .lexicon import ABBREVIATIONS, FOREIGN_PHRASES

_TOKEN = re.compile(r"[a-z0-9]+")
# Words replaced (not kept) by their expansion: in ICD-11 titles "cancer"
# appears almost only in "Fear of cancer", so keeping it misleads retrieval.
_REPLACE_ONLY = {"cancer"}

# Very common words that carry no diagnostic signal. Kept short on purpose:
# words such as "acute", "chronic", "without" matter for ICD-11 specificity.
STOP = set("""
a an and are as at be been but by can did do does for from had has have he her his i if in into is it its
me my no not of on or our she so than that the their them then there these they this to was we were what
when which who will with you your patient pt c o year years old yo day days week weeks today also please
""".split())

_FOREIGN = sorted(FOREIGN_PHRASES.items(), key=lambda kv: -len(kv[0]))


def _phrase_rx(src: str) -> re.Pattern:
    # Latin-script phrases need word boundaries ("homa" must not hit "lymphoma");
    # Chinese has no spaces, and Devanagari/Arabic marks confuse \b, so those match as substrings.
    if all(ord(ch) < 0x0250 for ch in src):
        return re.compile(r"(?<![\w])" + re.escape(src) + r"(?![\w])")
    return re.compile(re.escape(src))


_FOREIGN_RX = [(_phrase_rx(src), dst) for src, dst in _FOREIGN]


def translate_phrases(text: str) -> str:
    """Replace known non-English diagnosis phrases with English, in place.

    In-place (not appended) so that the surrounding words still apply: in
    "sin fiebre ni dolor lumbar" the gloss "low back pain" stays inside the
    negated clause. Longest phrases first. Runs on NFC lower-case text, before
    accents are stripped, because Devanagari vowel signs are combining marks.
    """
    t = unicodedata.normalize("NFC", text).lower()
    changed = False
    for rx, dst in _FOREIGN_RX:
        t2 = rx.sub(" " + dst + " ", t)
        if t2 != t:
            t, changed = t2, True
    return t if changed else text


def fold(text: str) -> str:
    """Lower-case and strip Latin accents ("fièvre" -> "fievre")."""
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def _spell(tok: str) -> str:
    """Collapse British/American spellings: anaemia/anemia, oedema/edema."""
    if len(tok) > 4:
        tok = tok.replace("ae", "e").replace("oe", "e")
    return tok


def stem(tok: str) -> str:
    """Tiny suffix stripper, enough to join plurals and adjectival forms."""
    if len(tok) <= 4 or tok.isdigit():
        return tok
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ic", ""), ("al", ""), ("es", ""), ("s", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            if suf == "s" and tok.endswith(("ss", "us", "is")):
                continue
            return tok[: len(tok) - len(suf)] + rep
    return tok


def tokens(text: str, expand_abbrev: bool = True, keep_stop: bool = False) -> list[str]:
    """Normalised tokens. Abbreviations are expanded in place (original kept)."""
    out: list[str] = []
    for raw in _TOKEN.findall(fold(text)):
        if expand_abbrev and raw in ABBREVIATIONS:
            if raw not in _REPLACE_ONLY:
                out.append(raw)
            out.extend(_TOKEN.findall(ABBREVIATIONS[raw]))
            continue
        out.append(raw)
    res = []
    for t in out:
        if not keep_stop and t in STOP:
            continue
        res.append(stem(_spell(t)))
    return res


def char_text(text: str) -> str:
    """Text for the character n-gram model: folded, abbreviations expanded."""
    toks = []
    for raw in _TOKEN.findall(fold(text)):
        toks.append(ABBREVIATIONS.get(raw, raw))
    return " ".join(toks)
