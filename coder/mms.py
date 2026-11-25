"""ICD-11 MMS code list: loading, hierarchy and specificity rules.

The release file (``data/mms-2026-01.json``) is a flat list of categories with
``code``, ``title``, ``chapter``, ``leaf``, ``residual`` and ``depth``. Only these
codes score, so every code this system emits passes through :meth:`Mms.valid`.

The hierarchy is not given explicitly. It is rebuilt from the codes: a child of
``1F40`` is any code starting with ``1F40`` whose depth is one more
(``1F40.0`` ... ``1F40.Z``); a child of ``CA23.3`` is ``CA23.30``/``CA23.31``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MMS = Path(__file__).resolve().parent.parent / "data" / "mms-2026-01.json"

_STEM = re.compile(r"[&/]")

# Phrases that mark the "default" child of a parent that has no ``.Z`` child,
# e.g. Mumps (1D80) -> 1D80.0 "Mumps without complication".
_DEFAULT_CHILD = re.compile(r"\b(without (mention of )?complications?|uncomplicated)\b", re.I)


def norm_code(code) -> str | None:
    """Normalise a code the way ``score.py`` does (stem before ``&``/``/``)."""
    if not isinstance(code, str):
        return None
    return _STEM.split(code.strip())[0].strip().upper() or None


@dataclass
class Category:
    code: str
    title: str
    chapter: str
    leaf: bool
    residual: bool
    depth: int
    parent: str | None = None
    children: list[str] = field(default_factory=list)


class Mms:
    """In-memory view of the MMS release with parent/child links."""

    def __init__(self, path: str | Path = DEFAULT_MMS):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        self.cats: dict[str, Category] = {}
        for r in rows:
            c = norm_code(r["code"])
            self.cats[c] = Category(c, r["title"], r["chapter"], bool(r["leaf"]),
                                    bool(r["residual"]), int(r["depth"]))
        self._link()

    # -- construction -------------------------------------------------------
    def _link(self) -> None:
        # Index codes by (depth) and find the longest proper prefix one level up.
        by_depth: dict[int, set[str]] = {}
        for c, cat in self.cats.items():
            by_depth.setdefault(cat.depth, set()).add(c)
        for c, cat in self.cats.items():
            if cat.depth <= 1:
                continue
            ups = by_depth.get(cat.depth - 1, set())
            for n in range(len(c) - 1, 0, -1):
                p = c[:n]
                if p in ups:
                    cat.parent = p
                    self.cats[p].children.append(c)
                    break
        for cat in self.cats.values():
            cat.children.sort()

    # -- queries ------------------------------------------------------------
    def valid(self, code) -> bool:
        c = norm_code(code)
        return c is not None and c in self.cats

    def title(self, code: str) -> str:
        cat = self.cats.get(code)
        return cat.title if cat else ""

    def path_titles(self, code: str) -> list[str]:
        """Titles from the top ancestor down to ``code`` (for context)."""
        out = []
        cur = self.cats.get(code)
        while cur is not None:
            out.append(cur.title)
            cur = self.cats.get(cur.parent) if cur.parent else None
        return list(reversed(out))

    def unspecified_child(self, code: str) -> str | None:
        """The ``.Z``-style child of a non-leaf code, if there is one."""
        cat = self.cats.get(code)
        if cat is None:
            return None
        for k in cat.children:
            if k.endswith("Z") and self.cats[k].residual:
                return k
        return None

    def default_child(self, code: str) -> str | None:
        """Child to use when the note gives no extra detail.

        ``.Z`` (unspecified) if it exists, else a "without complication" child.
        """
        z = self.unspecified_child(code)
        if z:
            return z
        cat = self.cats.get(code)
        if cat is None:
            return None
        for k in cat.children:
            if _DEFAULT_CHILD.search(self.cats[k].title):
                return k
        return None

    def specify(self, code: str, max_steps: int = 3) -> str:
        """Walk a non-leaf code down to its default leaf.

        The grader does not credit a parent when a child fits, and an
        unspecified child is correct when the note lacks the detail, so a
        parent is almost never the right answer.
        """
        cur = code
        for _ in range(max_steps):
            cat = self.cats.get(cur)
            if cat is None or cat.leaf:
                break
            nxt = self.default_child(cur)
            if not nxt:
                break
            cur = nxt
        return cur

    def is_traditional_medicine(self, code: str) -> bool:
        """Chapter 26 (TM1/TM2) codes: valid but practically never the gold."""
        cat = self.cats.get(code)
        return bool(cat and cat.chapter == "26")
