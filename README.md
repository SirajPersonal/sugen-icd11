# ICD-11 coder for outpatient notes

Reads a consult note (typed, abbreviated or an ambient transcript, in any language) and proposes up to 3 ICD-11 MMS 2026-01 codes, best first.

```
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl     # ./in and ./out are mounted
```
`OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` and `HTTPS_PROXY` are passed through from the host. With no key it still answers every note (offline path). Tests: `python -m pytest -q`. Full evaluation: `python scripts/evaluate.py`. Design details: [`docs/DESIGN.md`](docs/DESIGN.md). Decision log: [`DECISIONS.md`](DECISIONS.md).

## How it works
1. **Offline pass (every note, ~20 ms each).** Pull the assessment / "Dx" line (or the doctor's diagnostic sentences in a transcript), drop negated clauses ("malaria excluded"), and score all 18,375 codes with BM25 plus character n-grams over titles, a clinical lexicon and WHO index terms. Then walk a parent code down to its supported child or its `.Z` (unspecified) child. The full output file is written before any API call.
2. **LLM pass (when a key is set).** Call 1 reads the note in any language and returns an English diagnosis, ICD-style search terms and code guesses. These drive a second retrieval. Call 2 picks 3 codes from about 40 real candidates, shown with their parents, siblings and children, under explicit ICD-11 coding rules. The output is validated against the release, parents are walked down, and the list is padded from the offline ranking.
3. **Guards.** Each call reserves its worst-case cost before it is sent, with a hard cap of $0.90 of the $1. If two calls per note would cost too much, as with Claude Haiku, it drops to one call. Results are written atomically as they land. No new LLM work starts after 50 minutes. A model that isn't found falls through to the next one. After 12 failures in a row the LLM is switched off and the offline answers stand.

Cost measured from real prompt sizes: gpt-4.1-mini about $0.41 per 300 notes (two calls each); gpt-4o-mini $0.15; claude-haiku-4-5 $0.52 (one call each).

## Scores
| Set | Path | top-1 | top-3 | points |
|---|---|---|---|---|
| dev (60, slice A) | offline, no key | 0.867 | 0.933 | 66.7 / 75 |
| extra (45, other chapters + 7 languages) | offline, no key | 0.800 | 0.956 | 21.3 / 25 |
| dev + extra | LLM (gpt-4.1-mini) | _fill in from `scripts/evaluate.py`_ | | |

**Hidden-score estimate: about 60-75 / 100.** The offline dev number is optimistic, because I wrote the lexicon after reading the dev notes and wrote the extra notes myself. I'd expect the hidden core slice to come in a few points below dev, and generalisation to rest mostly on the LLM pass, since real notes in other chapters and languages are harder than mine.

## Data added
- `coder/lexicon.py`: hand-written clinical abbreviations (URTI, HTN, PET, SAM…), synonyms for about 210 common outpatient codes across chapters, and diagnosis phrases in French, Spanish, Portuguese, Swahili, Hindi, Arabic and Chinese. Every code is checked against the release when the index is built.
- `scripts/fetch_who.py`: pulls WHO ICD-11 index terms, inclusions and translated titles from the WHO API (or WHO's offline `whoicd/icd-api` container) into `data/who_terms.jsonl.gz`, which the index merges automatically. **Status: written and tested against a mock; the real file still needs fetching** (my build machine could not reach WHO).

## Evaluation beyond the 60 dev notes
`eval/extra_notes.jsonl` has 45 notes covering neoplasms, mental health, nervous system, eye, ear, injuries, neonatal and COVID, plus French, Spanish, Portuguese, Swahili, Hindi, Arabic and Chinese. `scripts/evaluate.py` scores it as slice B with `score.py`. The tests cover a fake LLM, the budget guard, model fallback, the circuit breaker, broken input lines and an end-to-end no-key run. A mock OpenAI/Anthropic server checked the real SDK calls, and the exact `docker compose run` command was tested with no key and with an unreachable proxy.

## Where the AI tooling was wrong and I caught it
_To write in your own words. Candidates are in DECISIONS.md: the Anthropic SDK `temperature` argument (#6), the `SSL_CERT_FILE` leak (#5), acute otitis media mapped to the wrong code (#7)._

## Time spent
_N hours._
