# Design notes

Companion to the one-page README. It covers the layout, the data flow, every
setting, and how to extend or debug the system.

## Repository layout

```
coder/
  __main__.py   CLI entry: python -m coder IN.jsonl OUT.jsonl
  pipeline.py   orchestration: read input, offline pass, LLM pass, atomic writer, deadline, signals
  llm.py        OpenAI / Anthropic providers, prices, budget guard, model fallback, circuit breaker
  prompts.py    the two prompts (extract, choose)
  offline.py    no-key coder: clause scoring, phrase bonus, coverage penalty, specificity walk
  index.py      retrieval index: BM25 (words) + TF-IDF (char 3-5-grams), priors; build/load/pickle
  extract.py    finds the assessment line / diagnostic sentences; negation filter
  text.py       normalisation shared by index and queries (tokens, stemming, spelling, translation)
  lexicon.py    abbreviations, foreign phrases, code synonyms (hand-curated)
  mms.py        MMS release: hierarchy, .Z / default child, specify()
scripts/
  evaluate.py      full-system score on dev (slice A) + extra set (slice B) via score.py
  eval_offline.py  quick offline-only score with the misses listed
  fetch_who.py     WHO index terms / inclusions / translations -> data/who_terms.jsonl.gz
eval/
  make_extra.py    writes eval/extra_notes.jsonl (45 out-of-distribution notes, gold checked)
tests/test_coder.py   pytest, no key or network needed
Dockerfile, docker-compose.yml, requirements.txt (pinned)
```

## Data flow for one note

```
note ──► translate_phrases ──► assessment_text ──► clauses (negated ones dropped)
                                     │
             BM25 + char-gram score of all 18,375 codes per clause (weights 1.0, .6, .45 …)
             + whole statement x0.5 + whole note x0.25 + lexicon phrase bonus
             x priors (ch.26 x0.25, ch.23 x0.4, ".Y" x0.85) x title-coverage penalty
                                     │
                         specify_scored: parent → supported child, else .Z / "without complication"
                                     │
                      offline top-30 (seeds) ──► top-3 written to disk immediately
                                     │ (if key)
       LLM call 1 (extract): English primary dx, details, search terms, ICD-11 guesses
                                     │
       candidates = seeds ∪ search(terms) ∪ guesses ∪ family of the top 10 (≤45, sorted by code)
                                     │
       LLM call 2 (choose): 3 codes under coding rules ──► finalize (validate, specify, pad) ──► write
```

## Why each rule exists

| Rule | Reason |
|---|---|
| Parent → `.Z` | Grader: a parent scores 0 when a child fits; `.Z` is right when detail is missing. |
| Child if it scores ≥ 75% of parent | Note supports the detail (nickel → EK00.6, E. coli → GC08.0). |
| "without complication" child | Parents with no `.Z` (Mumps 1D80 → 1D80.0). |
| Coverage penalty | Titles adding unstated features lose ("HIV stage 1 *with tuberculosis*"). |
| Chapter 26 / 23 down-weighted | Traditional-medicine and external-cause codes are valid but not visit diagnoses. |
| Negated clauses dropped | "malaria excluded", "mRDT NEG", "sin dolor lumbar". |
| In-place translation | Keeps foreign glosses inside their (possibly negated) clause. |

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | – | Enable the LLM pass; OpenAI is preferred when both are set (cheaper). |
| `CODER_PROVIDER` | `auto` | `openai`, `anthropic` or `none` (force offline). |
| `CODER_OPENAI_MODELS` | `gpt-4.1-mini,gpt-4o-mini,gpt-5-mini` | Tried in order; a 404/unknown model falls through. |
| `CODER_ANTHROPIC_MODELS` | `claude-haiku-4-5,claude-haiku-5-5,claude-3-5-haiku-latest` | Same. |
| `CODER_BUDGET_USD` | `0.90` | Hard cap enforced before each call (worst-case reservation). |
| `CODER_DEADLINE_S` | `3000` | No new LLM work after this many seconds (limit is 3600). |
| `CODER_WORKERS` | `8` | Parallel notes. |
| `CODER_MODE` | auto | `two_stage` or `single`; auto picks single if two calls would exceed 80% of budget. |
| `CODER_MAX_CANDIDATES` | `45` | Candidate list size for call 2. |

Prices for unknown models default to $3/$15 per M tokens, so the guard stops
early rather than late. Add new models to `PRICES` in `coder/llm.py`.

## Failure behaviour

| Situation | What happens |
|---|---|
| No key | Offline answers for all notes. |
| Bad key / proxy unreachable | 12 consecutive failures disable the LLM; offline answers stand; connection errors cost $0. |
| Model name not available | Next model in the list. |
| Budget would be crossed | Call not made; that note keeps its offline answer. |
| 50 min reached | No new LLM work; file already complete. |
| SIGTERM / Ctrl-C | Current results flushed, exit 0. |
| Broken input line | Id salvaged by regex and the note answered; lines with no id skipped. |
| Missing/stale index pickle | Rebuilt in-process (~1 s). |
| CA env var pointing at a missing file | Ignored (it would crash the SDK clients). |

## Common tasks

- **Add WHO terms:** see `scripts/fetch_who.py` docstring, then `python -m coder.index` and commit `data/who_terms.jsonl.gz`.
- **Add a synonym:** edit `SYNONYMS` in `coder/lexicon.py` (parents are fine), rebuild the index, run `pytest` (it fails on any code not in the release).
- **See why a note was coded X:** `python -m scripts.eval_offline FILE` lists misses; `offline.score_note` returns the full score vector and the extracted diagnosis text.
- **Check cost before a run:** prompt sizes are about 640 tokens (extract) and 1,430 (choose); multiply by `PRICES`.
