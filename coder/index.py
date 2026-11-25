"""Retrieval index over ICD-11 MMS categories.

Each code becomes a small document:  title (x2) + synonyms (x2) + parent title.
Two scorers run over those documents and are blended:

* BM25 over normalised word tokens (exact vocabulary, handles abbreviations
  via ``coder.text``),
* cosine similarity over character 3-5-grams (robust to misspellings,
  morphology and spelling variants, e.g. "candidiasis" vs "candidosis").

The index is built once (``python -m coder.index``, run by the Dockerfile) and
pickled to ``artifacts/index.pkl``. If the pickle is missing at run time it is
rebuilt in-process (a few seconds), so a stale image can never fail outright.
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import text as T
from .lexicon import SYNONYMS
from .mms import Mms

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "artifacts" / "index.pkl"
WHO_TERMS = ROOT / "data" / "who_terms.jsonl.gz"
INDEX_VERSION = 3


def _identity(x):
    return x


# Title words that describe the coding convention rather than the patient.
_GENERIC = set(T.tokens(
    "unspecified other specified not elsewhere classified site type due to without with "
    "mention certain disease disorder disorders condition conditions or and of in the a "
    "infection infections acute chronic"))
_WITHOUT = __import__("re").compile(r"\bwithout\b.*$", __import__("re").I)


def coverage_tokens(title: str) -> frozenset:
    """Content words of a title, ignoring 'without ...' tails and coding boilerplate."""
    t = _WITHOUT.sub("", title)
    return frozenset(x for x in T.tokens(t) if x not in _GENERIC and not x.isdigit())


def load_who_terms(path: Path = WHO_TERMS) -> dict[str, list[str]]:
    """Optional WHO index terms / inclusions, produced by scripts/fetch_who.py."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.setdefault(r["code"], []).extend(t for t in r.get("terms", []) if t)
    return out


class Index:
    def __init__(self, mms: Mms):
        self.mms = mms
        self.codes: list[str] = []
        self.pos: dict[str, int] = {}

    # -- build --------------------------------------------------------------
    def build(self, verbose: bool = False) -> "Index":
        t0 = time.time()
        mms = self.mms
        who = load_who_terms()
        syn: dict[str, list[str]] = {}
        bad = []
        for code, terms in SYNONYMS.items():
            if code in mms.cats:
                syn.setdefault(code, []).extend(terms)
            else:
                bad.append(code)
        if bad:
            print(f"[index] lexicon codes not in MMS release, skipped: {bad}", file=sys.stderr)
        for code, terms in who.items():
            if code in mms.cats:
                syn.setdefault(code, []).extend(terms)
        self.synonyms = syn

        word_docs, char_docs = [], []
        for code, cat in mms.cats.items():
            parent = mms.title(cat.parent) if cat.parent else ""
            extra = syn.get(code, [])
            word = T.tokens(cat.title) * 2 + T.tokens(" ".join(extra)) * 2 + T.tokens(parent)
            word_docs.append(word)
            char_docs.append(T.char_text(" | ".join([cat.title] + extra)))
            self.codes.append(code)
        self.pos = {c: i for i, c in enumerate(self.codes)}
        self.title_toks = [coverage_tokens(mms.cats[c].title) for c in self.codes]

        # BM25 weights, precomputed per (doc, term).
        cv = CountVectorizer(analyzer=_identity, lowercase=False)
        tf = cv.fit_transform(word_docs).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean()
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        k1, b = 1.2, 0.75
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)
        w = idf[tf.col] * tf.data * (k1 + 1) / denom
        self.bm25 = sparse.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.vocab = cv.vocabulary_

        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                     min_df=1, dtype=np.float32)
        self.char_mat = self.tfidf.fit_transform(char_docs).tocsr()

        # Exact-phrase table for the lexicon (normalised token tuples -> codes).
        self.phrases: dict[tuple, set[str]] = {}
        for code, terms in syn.items():
            for t in terms:
                key = tuple(T.tokens(t, expand_abbrev=False))
                if key:
                    self.phrases.setdefault(key, set()).add(code)
        self.max_phrase = max((len(k) for k in self.phrases), default=1)

        # Static priors.
        prior = np.ones(len(self.codes), dtype=np.float32)
        for i, c in enumerate(self.codes):
            cat = mms.cats[c]
            if cat.chapter == "26":
                prior[i] = 0.25           # traditional medicine patterns
            elif cat.chapter == "23":
                prior[i] = 0.4            # external causes: a secondary code, rarely the visit diagnosis
            if cat.residual and c.endswith("Y"):
                prior[i] *= 0.85          # "other specified" needs explicit detail
        self.prior = prior
        if verbose:
            print(f"[index] {len(self.codes)} codes, {len(self.vocab)} terms, "
                  f"{len(syn)} codes with synonyms, {sum(map(len, who.values()))} WHO terms, "
                  f"{time.time() - t0:.1f}s", file=sys.stderr)
        return self

    # -- query --------------------------------------------------------------
    def bm25_scores(self, query: str) -> np.ndarray:
        cols = [self.vocab[t] for t in set(T.tokens(query)) if t in self.vocab]
        if not cols:
            return np.zeros(len(self.codes), dtype=np.float32)
        return np.asarray(self.bm25[:, cols].sum(axis=1)).ravel()

    def char_scores(self, query: str) -> np.ndarray:
        q = self.tfidf.transform([T.char_text(query)])
        return np.asarray((self.char_mat @ q.T).todense()).ravel()

    def scores(self, query: str, w_bm25: float = 0.55, w_char: float = 0.45) -> np.ndarray:
        """Blended, prior-weighted relevance of every code to ``query`` (0..~1)."""
        b = self.bm25_scores(query)
        if b.max() > 0:
            b = b / b.max()
        c = self.char_scores(query)
        if c.max() > 0:
            c = c / c.max()
        return (w_bm25 * b + w_char * c) * self.prior

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        s = self.scores(query)
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return [(self.codes[i], float(s[i])) for i in top if s[i] > 0]

    def phrase_hits(self, query: str) -> dict[str, int]:
        """Lexicon phrases found verbatim in the query -> phrase length."""
        toks = T.tokens(query, expand_abbrev=False)
        hits: dict[str, int] = {}
        for n in range(1, self.max_phrase + 1):
            for i in range(len(toks) - n + 1):
                for code in self.phrases.get(tuple(toks[i:i + n]), ()):
                    hits[code] = max(hits.get(code, 0), n)
        return hits


def build_and_save(path: Path = ARTIFACT) -> Index:
    idx = Index(Mms()).build(verbose=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"version": INDEX_VERSION, "index": idx}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[index] saved {path} ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return idx


def load(path: Path = ARTIFACT) -> Index:
    """Load the prebuilt index, or build it if missing/outdated."""
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("version") == INDEX_VERSION:
            return blob["index"]
        print("[index] outdated artifact, rebuilding", file=sys.stderr)
    except FileNotFoundError:
        print("[index] no artifact, building in-process", file=sys.stderr)
    except Exception as e:  # corrupt pickle etc.
        print(f"[index] could not load artifact ({e}), rebuilding", file=sys.stderr)
    try:
        return build_and_save(path)
    except OSError:
        return Index(Mms()).build(verbose=True)


if __name__ == "__main__":
    # Import through the package so the pickle records ``coder.index.Index``
    # rather than ``__main__.Index``.
    from coder.index import build_and_save as _build
    _build()
