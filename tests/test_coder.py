"""Tests that need no API key. Run: ``python -m pytest -q``.

The LLM path is exercised with a fake provider, so the plumbing (budget guard,
model fallback, circuit breaker, candidate building, output writing) is
covered without network access.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coder import index as index_mod  # noqa: E402
from coder import llm, offline, pipeline  # noqa: E402
from coder.mms import Mms, norm_code  # noqa: E402


@pytest.fixture(scope="module")
def idx():
    return index_mod.load()


# ---------------------------------------------------------------- code list
def test_hierarchy_and_specify():
    m = Mms()
    assert m.cats["1F40.Z"].parent == "1F40"
    assert "CA23.30" in m.cats["CA23.3"].children
    assert m.specify("1F40") == "1F40.Z"          # .Z child
    assert m.specify("1D80") == "1D80.0"          # "without complication" child
    assert m.specify("5A11") == "5A11"            # leaf stays
    assert m.specify("FA25") .startswith("FA25")
    assert norm_code(" nc72.2&XK8G ") == "NC72.2"


def test_lexicon_codes_are_valid(idx):
    from coder.lexicon import SYNONYMS
    bad = [c for c in SYNONYMS if c not in idx.mms.cats]
    assert not bad, f"lexicon codes missing from MMS: {bad}"


# ---------------------------------------------------------------- offline
@pytest.mark.parametrize("note,expect", [
    ("Dx: Uncomplicated falciparum malaria. Coartem.", "1F40.Z"),
    ("A: T2DM, new dx. Start metformin.", "5A11"),
    ("Imp: acute exacerbation of COPD", "CA22.0"),
    ("Diagnostic : paludisme simple à P. falciparum", "1F40.Z"),
    ("Dx: ankle sprain after a fall playing football", "ND14.7Z"),
])
def test_offline_examples(idx, note, expect):
    assert expect in offline.predict(idx, note)


def test_offline_negation(idx):
    p = offline.predict(idx, "Dx viral URTI, malaria excluded. Panadol.")
    assert not p[0].startswith("1F4")


def test_offline_never_empty(idx):
    for note in ["", "   ", "asdf qwer", "Patient seen."]:
        p = offline.predict(idx, note)
        assert p and all(c in idx.mms.cats for c in p)


# ---------------------------------------------------------------- budget
def test_budget_refuses_before_crossing():
    b = llm.Budget(0.01)
    b.reserve(0.006)
    with pytest.raises(llm.BudgetExceeded):
        b.reserve(0.006)
    b.settle(0.006, 0.001)
    b.reserve(0.006)  # fits again after settling to real cost


class FakeProvider(llm.Provider):
    """Answers extraction with the note's own words, and picks the first
    candidate whose title shares a word with the note."""
    name = "fake"

    def __init__(self, models, budget, fail_models=(), fail_all=False):
        super().__init__(models, budget)
        self.fail_models, self.fail_all, self.seen = set(fail_models), fail_all, []

    def _call(self, model, system, user, max_tokens):
        self.seen.append(model)
        if self.fail_all:
            raise ConnectionError("proxy refused")  # name contains "connection"
        if model in self.fail_models:
            raise llm.ModelUnavailable("model_not_found")
        if "CANDIDATE CODES" in user:
            lines = user.split("CANDIDATE CODES (code | title | parent):\n", 1)[1].splitlines()
            return json.dumps({"codes": [ln.split(" | ")[0] for ln in lines[:3]]}), 1500, 30
        return json.dumps({"primary": "malaria", "search_terms": ["Malaria due to Plasmodium falciparum"],
                           "icd11_guesses": ["1F40", "J45.9"]}), 600, 80


def test_model_fallback():
    b = llm.Budget(1.0)
    p = FakeProvider(["bad-model", "gpt-4o-mini"], b, fail_models={"bad-model"})
    out = p.complete("sys", "user", 50)
    assert json.loads(out) and p.model == "gpt-4o-mini"
    assert b.spent > 0 and b.reserved == pytest.approx(0, abs=1e-9)


def test_circuit_breaker():
    b = llm.Budget(1.0)
    p = FakeProvider(["gpt-4o-mini"], b, fail_all=True)
    for _ in range(p.max_consecutive_failures):
        with pytest.raises(ConnectionError):
            p.complete("s", "u", 10)
    assert p.dead and b.spent == 0  # connection failures cost nothing


def test_llm_path_with_fake(idx):
    b = llm.Budget(1.0)
    coder = pipeline.Coder(idx, FakeProvider(["gpt-4o-mini"], b), b, "two_stage")
    note = "A: Uncomplicated falciparum malaria. P: Coartem."
    seeds = offline.rank(idx, note, k=30)
    out = coder.code_llm(note, seeds, [c for c, _ in seeds[:3]])
    assert out and all(c in idx.mms.cats for c in out) and len(out) == 3
    assert idx.mms.cats[out[0]].leaf          # a parent pick is walked down


def test_finalize_filters_and_pads(idx):
    out = pipeline.finalize(idx, ["J45.9", "SA00", "1F40", "1F40.Z"], ["CA23.30", "MG26"])
    assert out == ["1F40.Z", "CA23.30", "MG26"]


# ---------------------------------------------------------------- end to end (no key)
def test_cli_end_to_end(tmp_path):
    notes = tmp_path / "notes.jsonl"
    with open(notes, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a", "text": "Dx: UTI"}) + "\n")
        f.write("\n")
        f.write('{"id": "b", "text": "broken json \n')       # salvaged by id
        f.write(json.dumps({"id": 7, "text": "Utambuzi: kisukari aina ya pili"}) + "\n")
    out = tmp_path / "out" / "pred.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    r = subprocess.run([sys.executable, "-m", "coder", str(notes), str(out)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["a", "b", 7]
    assert all(1 <= len(r["codes"]) <= 3 for r in rows)
