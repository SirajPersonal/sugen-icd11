"""Run the offline coder over a gold file and print misses. Usage: python -m scripts.eval_offline data/dev_notes.jsonl"""
import json, sys
from coder import index as I, offline
idx = I.load()
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
t1 = t3 = 0
for r in rows:
    p = offline.predict(idx, r["text"]); g = set(r["gold"])
    a, b = p[0] in g, bool(g & set(p)); t1 += a; t3 += b
    if not a and "-q" not in sys.argv:
        print(f"{r['id']:10} gold={r['gold']} pred={p}  [{idx.mms.title(r['gold'][0])[:45]} | {idx.mms.title(p[0])[:45]}]")
n = len(rows); print(f"n={n} top1={t1/n:.3f} top3={t3/n:.3f} score/75={75*(2/3*t1/n+1/3*t3/n):.1f}")
