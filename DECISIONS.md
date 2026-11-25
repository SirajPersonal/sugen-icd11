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

### 4. 2026-11-26 00:40 · LLM pipeline: two calls per note, offline-first writing
Commit: `LLM pipeline`

- **Options:** one call (note + offline candidates -> codes); two calls
  (extract + translate -> retrieve -> choose); agentic multi-round search.
- **Chosen:** two calls. Call 1 reads the note in any language and returns an
  English diagnosis, ICD-style search terms and code guesses. Those drive a
  second retrieval; call 2 picks 3 codes from ~35-45 real candidates (with
  parents, siblings and children so every specificity level is visible).
  Multi-round search was dropped: latency and cost for little expected gain.
- **Measured:** candidate-list recall on dev is 59/60 from offline seeds
  alone, before the LLM's own search terms are added. Estimated cost with
  gpt-4.1-mini about $0.0012/note, so ~$0.35 for 300 notes. With
  claude-haiku-4-5 two calls would be ~$0.004/note (> budget), so the code
  switches to one choose-only call per note when two calls would exceed 80% of
  the budget.
- **Safety:** every note gets an offline answer written to disk before any API
  call. LLM upgrades rewrite the file atomically (`os.replace`) as they land.
  The budget guard reserves worst-case cost before each call (cap $0.90).
  The run stops starting LLM work at 50 min. A circuit breaker turns the LLM
  off after 12 failures in a row (bad key, proxy not reachable). A model that
  is not found falls through to the next one in the list.
- **Not measured yet:** real LLM accuracy (no API key in my build
  environment). The plumbing is tested with a fake provider.
- **Revisit:** self-consistency (2 samples, vote) if budget allows; prompt
  caching to cut call-2 cost.

### 5. 2026-11-26 00:55 · Docker: python:3.13-slim, index baked at build, pinned deps
Commit: `Docker image and compose`

- **Options:** build the index at run time (simpler image) or at build time.
- **Chosen:** build time (`RUN python -m coder.index`), so the run needs no
  network and starts in ~1.5 s. If the pickle is missing or stale, the code
  rebuilds it in-process anyway.
- **Pinned** numpy/scipy/scikit-learn/openai/anthropic to the versions the dev
  score was produced with, so a pickle/library mismatch cannot happen.
- **Tested:** Docker Hub is blocked from my build box, so I built a stand-in
  `python:3.13-slim` from the local Python install and ran the exact grading
  command: no key (66.7/75 dev, 4 s); fake key with an unreachable proxy
  (circuit breaker trips, offline answers kept, $0 spent, 13 s).
- **Caught:** the compose file passed `SSL_CERT_FILE` through. On my box it
  points at a host-only path, so the OpenAI client crashed at start-up and the
  run quietly went offline. Removed the passthrough, and the code now ignores CA
  variables that point at missing files.
- **Revisit:** confirm the real `python:3.13-slim` build on a machine with Docker Hub.

### 6. 2026-11-26 01:00 · SDK compatibility checked against a mock API
Commit: `Docker image and compose`

- Ran the pipeline against a local mock of the OpenAI and Anthropic HTTP APIs
  (`OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL`).
- **Caught:** the pinned `anthropic` 2.x SDK rejects `temperature=` in
  `messages.create`. Every Anthropic call would have failed, and the failure
  was being counted as spend. Removed the argument; local errors
  (TypeError/ValueError) now cost $0.
- **Verified:** a 404 on the first model falls through to the next; 300 notes
  on Haiku pricing switch to single-call mode automatically.
- **Measured prompt sizes:** extract ≈640 tokens, choose ≈1,430 tokens. Cost
  per 300 notes: gpt-4.1-mini two-stage $0.41, gpt-4o-mini $0.15,
  claude-haiku-4-5 single-call $0.52.
