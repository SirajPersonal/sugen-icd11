"""End-to-end run: read notes, code them, write predictions safely as we go.

Order of work, chosen so that a crash, a timeout or an exhausted budget can
never leave a note without an answer:

1. Load the index and code every note offline (fast: ~20 ms/note). Write the
   complete output file immediately.
2. If an API key is set, upgrade notes with the LLM in parallel threads.
   Each finished note is merged into the results and the file is rewritten
   atomically (temp file + ``os.replace``) every few notes, so the file on
   disk is always complete and valid.
3. Stop starting new LLM work at the deadline (``CODER_DEADLINE_S``, default
   50 min of the 60 allowed) or when the budget guard refuses a call.
"""
from __future__ import annotations

import json
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import index as index_mod
from . import offline
from .llm import Budget, BudgetExceeded, ModelUnavailable, log, make_provider, parse_json, retry_call
from .mms import norm_code
from .prompts import CHOOSE_SYSTEM, EXTRACT_SYSTEM, choose_user, extract_user

START = time.time()
DEADLINE_S = float(os.environ.get("CODER_DEADLINE_S", 50 * 60))
WORKERS = int(os.environ.get("CODER_WORKERS", 8))
BUDGET_USD = float(os.environ.get("CODER_BUDGET_USD", 0.90))
MAX_CANDIDATES = int(os.environ.get("CODER_MAX_CANDIDATES", 45))
_ID_RX = re.compile(r'"id"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+)')


# --------------------------------------------------------------------------- IO
def read_notes(path: str) -> list[dict]:
    """Read input JSONL. Tolerates blank lines, BOMs and some broken lines."""
    notes, seen = [], set()
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                nid, text = r.get("id"), r.get("text") or ""
            except Exception:
                m = _ID_RX.search(line)  # salvage the id so the note still gets an answer
                if not m:
                    log(f"line {n}: unparseable and no id, skipped")
                    continue
                nid, text = json.loads(m.group(1)), line
            if nid is None or json.dumps(nid) in seen:
                continue
            seen.add(json.dumps(nid))
            notes.append({"id": nid, "text": str(text)})
    return notes


class Writer:
    """Keeps the latest codes per note and rewrites the output atomically."""

    def __init__(self, path: str, notes: list[dict]):
        self.path = path
        self.order = [n["id"] for n in notes]
        self.codes: dict = {}
        self._lock = threading.RLock()  # re-entrant: the SIGTERM handler may flush mid-flush
        self._n = 0
        self._dirty = 0
        self._last = 0.0
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)

    def set(self, nid, codes: list[str]) -> None:
        with self._lock:
            self.codes[json.dumps(nid)] = codes[:3]
            self._dirty += 1

    def flush(self, force: bool = False) -> None:
        with self._lock:
            if not force and self._dirty < 5 and time.time() - self._last < 15:
                return
            self._n += 1
            tmp = f"{self.path}.tmp{os.getpid()}.{self._n}"
            with open(tmp, "w", encoding="utf-8") as f:
                for nid in self.order:
                    f.write(json.dumps({"id": nid, "codes": self.codes.get(json.dumps(nid), ["MG26"])},
                                       ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self._dirty = 0
            self._last = time.time()


# ------------------------------------------------------------------ candidates
def _line(mms, code: str) -> str:
    cat = mms.cats[code]
    parent = mms.title(cat.parent) if cat.parent else ""
    tag = " [has subcodes]" if not cat.leaf else ""
    return f"{code} | {cat.title}{tag} | {parent}"


def build_candidates(idx, note: str, seeds: list[tuple[str, float]], summary: dict | None) -> list[str]:
    """Union of offline ranking, LLM search terms and LLM code guesses, with
    the family (parent, siblings, children) of the strongest ones, so the model
    can see every specificity level it may need to choose between."""
    mms = idx.mms
    score: dict[str, float] = {}

    def add(code, s):
        code = norm_code(code)
        if code and code in mms.cats and not mms.is_traditional_medicine(code):
            score[code] = max(score.get(code, 0.0), s)

    for i, (c, s) in enumerate(seeds[:15]):
        add(c, 1.0 - i * 0.03)
    if summary:
        terms = [summary.get("primary") or ""] + list(summary.get("search_terms") or [])[:6]
        for t in terms:
            if isinstance(t, str) and t.strip():
                for j, (c, s) in enumerate(idx.search(t, k=8)):
                    add(mms.specify(c), 0.95 - j * 0.05)
                    add(c, 0.9 - j * 0.05)
        for g in list(summary.get("icd11_guesses") or [])[:4]:
            add(g, 1.0)
            if isinstance(g, str) and norm_code(g) in mms.cats:
                add(mms.specify(norm_code(g)), 0.98)
        for t in list(summary.get("secondary") or [])[:2]:
            if isinstance(t, str) and t.strip():
                for j, (c, s) in enumerate(idx.search(t, k=3)):
                    add(mms.specify(c), 0.6 - j * 0.05)

    # Family expansion around the top candidates.
    top = sorted(score, key=lambda c: -score[c])[:10]
    for c in top:
        cat = mms.cats[c]
        fam = list(cat.children)
        if cat.parent:
            fam.append(cat.parent)
            fam.extend(mms.cats[cat.parent].children)
        for k in fam[:14]:
            add(k, score[c] * 0.7)
    ordered = sorted(score, key=lambda c: -score[c])[:MAX_CANDIDATES]
    # Present families together: sort by code so parents sit above children.
    return [_line(mms, c) for c in sorted(ordered)]


# ------------------------------------------------------------------ coding one note
def finalize(idx, codes: list, fallback: list[str]) -> list[str]:
    """Validate, specify parents, dedupe and pad to 3 with the offline ranking."""
    mms = idx.mms
    out: list[str] = []
    for c in list(codes or []) + list(fallback):
        c = norm_code(c)
        if not c or c not in mms.cats or mms.is_traditional_medicine(c):
            continue
        c = mms.specify(c)
        if c not in out:
            out.append(c)
        if len(out) == 3:
            break
    return out or ["MG26"]


class Coder:
    def __init__(self, idx, provider, budget: Budget, mode: str = "two_stage"):
        self.idx, self.provider, self.budget, self.mode = idx, provider, budget, mode
        self.stats = {"llm_ok": 0, "llm_fail": 0, "budget_refused": 0, "skipped": 0}
        self._lock = threading.Lock()

    def _bump(self, k):
        with self._lock:
            self.stats[k] += 1

    def code_llm(self, note: str, seeds: list[tuple[str, float]], fallback: list[str]) -> list[str] | None:
        if time.time() - START > DEADLINE_S or self.provider is None or self.provider.dead:
            self._bump("skipped")
            return None
        try:
            summary = None
            if self.mode == "two_stage":
                raw = retry_call(lambda: self.provider.complete(EXTRACT_SYSTEM, extract_user(note), 320))
                summary = parse_json(raw)
            cands = build_candidates(self.idx, note, seeds, summary)
            raw = retry_call(lambda: self.provider.complete(CHOOSE_SYSTEM, choose_user(note, summary, cands), 90))
            pick = parse_json(raw).get("codes") or []
            if not isinstance(pick, list):
                pick = [pick]
            if not pick:
                raise ValueError(f"no codes in reply: {raw[:120]!r}")
            self._bump("llm_ok")
            return finalize(self.idx, pick, fallback)
        except BudgetExceeded:
            self._bump("budget_refused")
        except ModelUnavailable:
            self._bump("llm_fail")
        except Exception as e:
            self._bump("llm_fail")
            log(f"LLM error: {type(e).__name__}: {str(e)[:160]}")
        return None


def choose_mode(provider, budget: Budget, n_notes: int) -> str:
    """Two calls per note unless the model is too pricey for the budget, in
    which case a single (choose-only) call per note."""
    from .llm import price
    if provider is None:
        return "offline"
    pin, pout = price(provider.model)
    two = (2600 * pin + 260 * pout) / 1e6      # measured-ish token budget per note, two calls
    one = (1900 * pin + 90 * pout) / 1e6
    forced = os.environ.get("CODER_MODE")
    if forced in ("two_stage", "single"):
        return forced
    if two * n_notes <= 0.8 * budget.limit:
        return "two_stage"
    log(f"{provider.model}: est ${two * n_notes:.2f} for two-stage > 80% budget; using single call "
        f"(est ${one * n_notes:.2f})")
    return "single"


# ------------------------------------------------------------------ main
def run(in_path: str, out_path: str) -> int:
    notes = read_notes(in_path)
    log(f"{len(notes)} notes from {in_path}")
    writer = Writer(out_path, notes)

    def _on_term(signum, frame):  # make sure what we have is on disk
        log(f"signal {signum}: flushing and exiting")
        writer.flush(force=True)
        os._exit(0)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_term)
        except Exception:
            pass

    # Placeholder answers first, so the file is complete from the first second.
    for n in notes:
        writer.set(n["id"], ["MG26"])
    writer.flush(force=True)

    idx = index_mod.load()
    seeds, fallback = {}, {}
    for n in notes:
        key = json.dumps(n["id"])
        try:
            seeds[key] = offline.rank(idx, n["text"], k=30)
            fallback[key] = [c for c, _ in seeds[key][:3]] or ["MG26"]
        except Exception as e:  # never let one odd note stop the run
            log(f"offline failed on {n['id']}: {e}")
            seeds[key], fallback[key] = [], ["MG26"]
        writer.set(n["id"], fallback[key])
    writer.flush(force=True)
    log(f"offline pass done in {time.time() - START:.1f}s")

    budget = Budget(BUDGET_USD)
    provider = make_provider(budget)
    mode = choose_mode(provider, budget, len(notes))
    if provider is None:
        log("no API key set: offline answers only")
        return 0
    log(f"LLM: {provider.name}/{provider.model}, mode={mode}, workers={WORKERS}, budget=${BUDGET_USD}")
    coder = Coder(idx, provider, budget, mode)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(coder.code_llm, n["text"], seeds[json.dumps(n["id"])],
                          fallback[json.dumps(n["id"])]): n["id"] for n in notes}
        for fut in as_completed(futs):
            nid = futs[fut]
            try:
                codes = fut.result()
            except Exception as e:
                log(f"worker error on {nid}: {e}")
                codes = None
            if codes:
                writer.set(nid, codes)
                writer.flush()
    writer.flush(force=True)
    log(f"done in {time.time() - START:.1f}s; {coder.stats}; spent ~${budget.spent:.4f} "
        f"over {budget.calls} calls (model {provider.model})")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m coder IN.jsonl OUT.jsonl", file=sys.stderr)
        return 2
    return run(argv[0], argv[1])
