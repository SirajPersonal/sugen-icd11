"""Find the part of a note that states the diagnosis, and drop negated clauses.

Typed notes usually carry an assessment line ("A:", "Dx", "Imp:",
"Assessment:", or the equivalent in other languages). Ambient transcripts bury
it in the doctor's speech ("This is chickenpox", "So he has some dehydration").
Everything here is heuristic and only used by the no-key path and to seed
candidate retrieval; the LLM reads the whole note.
"""
from __future__ import annotations

import re

_LABELS = (
    r"A|Ax|Dx|DDx|Imp|Impression|Assessment|Assess|Diagnos[ie]s|Diagnosis|Working dx|Final dx|Clinical dx|"
    r"Diagnostic|Diagnóstico|Diagnostico|Impressão|Avaliação|Évaluation|Utambuzi|Tathmini|"
    r"निदान|निष्कर्ष|التشخيص|تشخيص"
)
# A label at line start (or after a sentence end), followed by ":" / "-" / space.
_LABEL_RX = re.compile(
    rf"(?:^|\n|(?<=[.;]\s))\s*(?:{_LABELS})\b\s*(?:[:\-–=/>]|\s)\s*(?P<dx>.+?)"
    r"(?=\n|\.\s+(?=[A-Z][a-z]|[A-Z]{2,}\b|P\b|Rx\b)|\bP\s*:|\bPlan\b|\bRx\b|\bMx\b|$)",
    re.I | re.S,
)
_SPEAKER = re.compile(r"^\s*(doctor|dr|clinician|physician|médecin|medico|médico|daktari|डॉक्टर|الطبيب)\s*[:\-]", re.I)
_DX_CUES = re.compile(
    r"\b(this is|this looks like|it'?s|it is|you have|you've got|he has|she has|they have|so (he|she|you) (has|have)|"
    r"diagnosis|consistent with|in keeping with|suggest(s|ive of)|confirms?|shows?|we'?re dealing with|"
    r"looks like|that means|which means|is in the .* range|c'est|il s'agit|tiene|tienes|es una?|una|ni|una)\b",
    re.I,
)
_NEG = re.compile(
    r"\b(no|not|nil|neg|negative|excluded|ruled out|rule out|r/o|unlikely|denies|without evidence|"
    r"sans|pas de|sin|descartad[oa]|hakuna|नहीं)\b",
    re.I,
)
_SPLIT = re.compile(r"[;,()\[\]]|\s-\s|\s–\s|\.\s|\n|\+")


def assessment_text(note: str) -> str:
    """Best guess at the diagnosis statement(s) of a note ("" if none found)."""
    hits = [m.group("dx").strip() for m in _LABEL_RX.finditer(note)]
    hits = [h for h in hits if len(h) > 1]
    if hits:
        return " . ".join(hits)
    doctor = [ln for ln in note.splitlines() if _SPEAKER.match(ln)]
    if doctor:
        sents = []
        for ln in doctor:
            ln = _SPEAKER.sub("", ln)
            sents.extend(s for s in re.split(r"(?<=[.!?])\s+", ln) if _DX_CUES.search(s))
        # Diagnoses are usually stated late in the consultation.
        return " . ".join(sents[-4:])
    return ""


def clauses(dx: str) -> list[str]:
    """Split a diagnosis statement into clauses, dropping negated ones."""
    out = []
    for c in _SPLIT.split(dx):
        c = c.strip(" .:-")
        if len(c) < 2:
            continue
        if _NEG.search(c):
            continue
        out.append(c)
    return out
