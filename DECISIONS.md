# Decisions log

Kept as the work goes. Each entry is committed with the change it describes, so
"Commit" names the commit subject the entry ships in (`git log --grep` finds it).
Times are IST (UTC+05:30).

---

### 1. 2026-11-26 00:25 · Architecture: retrieval + LLM rerank, no training
Commit: `Offline retrieval coder`

- **Options:** (a) fine-tune a classifier; (b) ask an LLM for the code directly;
  (c) retrieve candidates from the code list, then let an LLM choose; (d) rules only.
- **Chosen:** (c), with (d) as the no-key fallback that also seeds (c).
- **Why:** 60 labelled notes cannot train anything across 18,375 codes and 26
  chapters. A bare LLM invents or misremembers ICD-11 codes (ICD-10 habits are
  strong). Retrieval keeps every answer inside `mms-2026-01.json`, and the LLM
  only has to judge between real titles.
- **Revisit:** a small embedding model for multilingual retrieval offline.

### 2. 2026-11-26 00:26 · Retrieval = BM25 words + char 3-5-gram TF-IDF
Commit: `Offline retrieval coder`

- **Options:** BM25 only; TF-IDF only; sentence embeddings (needs torch and a
  model download, ~1 GB image, extra build risk).
- **Chosen:** blend of BM25 (exact vocabulary) and char n-grams (spelling,
  morphology: candidiasis/candidosis, anemia/anaemia). Index builds in ~1 s,
  18 MB pickle, no GPU, no download.
- **Measured:** offline dev 0.833 top-1 / 0.933 top-3 on first run.
- **Revisit:** embeddings, if the WHO index terms do not close the gap.

### 3. 2026-11-26 00:27 · Specificity: walk parents down to `.Z`, unless a child is supported
Commit: `Offline retrieval coder`

- **Seen:** the grader gives nothing for a parent when a child fits. 2,204 of
  2,463 non-leaf codes checked have a `.Z` child; the rest usually have a
  "without complication" child (Mumps 1D80 -> 1D80.0).
- **Rule:** a non-leaf winner descends to its best-scoring non-residual child
  if that child scores >= 0.75 of the parent, else to `.Z` / "without
  complication". Fixed nickel -> EK00.6, E. coli UTI -> GC08.0.
- **Also:** title-coverage penalty (titles that add content words absent from
  the note lose up to 45%) fixed HIV stage 1 picking "associated with
  tuberculosis". Chapter 26 (traditional medicine) x0.25, chapter 23 (external
  causes) x0.4, "other specified" (`.Y`) x0.85.
- **Measured:** offline dev 0.833 -> 0.867 top-1, top-3 0.933 unchanged.
- **Caveat:** the lexicon was written after reading the dev notes, so offline
  dev numbers are optimistic. I stopped tuning offline here on purpose.
