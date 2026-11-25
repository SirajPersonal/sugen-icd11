"""No-key coder: retrieval + rules. Also used to seed the LLM's candidate list.

For a note:
1. pull out the diagnosis statement and split it into non-negated clauses;
2. score every code against each clause (earlier clauses weigh more), against
   the whole statement, and weakly against the whole note;
3. add a bonus for lexicon phrases that appear verbatim;
4. walk each ranked code down to its most specific default
   (``Mms.specify``: parent -> ``.Z`` child / "without complication" child);
5. return the top 3 distinct codes.
"""
from __future__ import annotations

import numpy as np

from .extract import assessment_text, clauses
from .index import Index
from .text import tokens, translate_phrases

CLAUSE_WEIGHTS = (1.0, 0.6, 0.45, 0.35, 0.3)


def score_note(idx: Index, note: str) -> tuple[np.ndarray, str]:
    note_x = translate_phrases(note)
    dx = assessment_text(note_x)
    total = np.zeros(len(idx.codes), dtype=np.float32)
    parts = clauses(dx) if dx else []
    for i, c in enumerate(parts[:8]):
        w = CLAUSE_WEIGHTS[min(i, len(CLAUSE_WEIGHTS) - 1)]
        total += w * idx.scores(c)
        for code, n in idx.phrase_hits(c).items():
            total[idx.pos[code]] += w * (0.35 + 0.1 * n)
    if dx:
        total += 0.5 * idx.scores(dx)
    total += (0.25 if parts else 1.0) * idx.scores(note_x)
    if not parts:
        for code, n in idx.phrase_hits(note_x).items():
            total[idx.pos[code]] += 0.2 + 0.05 * n
    total = apply_coverage(idx, total, note_x)
    return total, dx


def apply_coverage(idx: Index, total: np.ndarray, note: str, top: int = 300) -> np.ndarray:
    """Penalise codes whose title adds content words the note never mentions.

    "HIV disease clinical stage 1 associated with tuberculosis" should lose to
    "... without mention of tuberculosis or malaria" when the note has no TB.
    Only the current top candidates are adjusted (cheap, and enough).
    """
    note_toks = set(tokens(note))
    out = total.copy()
    for i in np.argsort(-total)[:top]:
        tt = idx.title_toks[i]
        if not tt or total[i] <= 0:
            continue
        cov = len(tt & note_toks) / len(tt)
        out[i] = total[i] * (0.55 + 0.45 * cov)
    return out


def rank(idx: Index, note: str, k: int = 30) -> list[tuple[str, float]]:
    """Top-k specified codes with scores (deduplicated after specification)."""
    s, _ = score_note(idx, note)
    order = np.argsort(-s)[: k * 3]
    seen, out = set(), []
    for i in order:
        if s[i] <= 0:
            break
        code = specify_scored(idx, idx.codes[i], s)
        if code in seen:
            continue
        seen.add(code)
        out.append((code, float(s[i])))
        if len(out) >= k:
            break
    return out


def specify_scored(idx: Index, code: str, s: np.ndarray, ratio: float = 0.75) -> str:
    """Descend from a non-leaf code, preferring a child the note itself supports.

    If a non-residual child scores at least ``ratio`` of its parent, take it
    (e.g. EK00 -> EK00.6 for nickel); otherwise take the default child (.Z).
    """
    mms = idx.mms
    for _ in range(3):
        cat = mms.cats[code]
        if cat.leaf or not cat.children:
            break
        base = s[idx.pos[code]]
        best, best_s = None, -1.0
        for k in cat.children:
            if mms.cats[k].residual:
                continue
            ks = s[idx.pos[k]]
            if ks > best_s:
                best, best_s = k, ks
        if best is not None and base > 0 and best_s >= ratio * base:
            code = best
        else:
            nxt = mms.default_child(code)
            if not nxt:
                break
            code = nxt
    return code


def predict(idx: Index, note: str) -> list[str]:
    codes = [c for c, _ in rank(idx, note, k=3)]
    return codes or ["MG26"]  # never return nothing; a generic symptom code beats a miss
