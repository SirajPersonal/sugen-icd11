"""Score the full system (as the grader would) on dev + the extra out-of-distribution set.

    python scripts/evaluate.py                  # uses keys from the environment if set
    CODER_PROVIDER=none python scripts/evaluate.py   # force the offline path

Dev notes are slice A, eval/extra_notes.jsonl is slice B, so score.py prints both
the core and the generalisation share. Writes eval/out/{notes,gold,pred,per_note}.jsonl.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "out"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for path, slice_ in ((ROOT / "data" / "dev_notes.jsonl", "A"), (ROOT / "eval" / "extra_notes.jsonl", "B")):
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            r["slice"] = r.get("slice", slice_)
            rows.append(r)
with open(OUT / "notes.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({"id": r["id"], "text": r["text"]}, ensure_ascii=False) + "\n")
with open(OUT / "gold.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({"id": r["id"], "gold": r["gold"], "slice": r["slice"]}, ensure_ascii=False) + "\n")

subprocess.run([sys.executable, "-m", "coder", str(OUT / "notes.jsonl"), str(OUT / "pred.jsonl")], cwd=ROOT, check=True)
subprocess.run([sys.executable, str(ROOT / "score.py"), "--gold", str(OUT / "gold.jsonl"), "--pred", str(OUT / "pred.jsonl"),
                "--mms", str(ROOT / "data" / "mms-2026-01.json"), "--per-note", str(OUT / "per_note.jsonl")], check=True)

titles = {x["code"]: x["title"] for x in json.load(open(ROOT / "data" / "mms-2026-01.json", encoding="utf-8"))}
print("\nmisses (top-1):")
for line in open(OUT / "per_note.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if not r["top1"]:
        p = r["pred"][0] if r["pred"] else None
        print(f"  {r['id']:10} {r['slice']} gold={r['gold']} pred={r['pred']}  "
              f"[{titles.get(r['gold'][0], '')[:40]} | {titles.get(p, '')[:40]}]")
