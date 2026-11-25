"""Fetch WHO ICD-11 index terms and inclusions for every code in the MMS file.

Output: data/who_terms.jsonl.gz, one row per (code, language):
    {"code": "1F23.10", "lang": "en", "terms": ["Vaginal thrush", "Candidal vulvovaginitis", ...]}
The index builder (coder/index.py) adds these to each code's document, which is
the single biggest lever on retrieval recall. Run it once, commit the file, and
the Docker image ships it. (No network is needed at run time.)

Two ways to reach the data (stdlib only, no extra packages):

1. WHO's public API (free registration at https://icd.who.int/icdapi):
       export ICD_CLIENT_ID=...  ICD_CLIENT_SECRET=...
       python scripts/fetch_who.py
2. WHO's offline API container, no credentials needed:
       docker run -p 8382:80 -e acceptLicense=true -e include=2026-01_en-fr-es whoicd/icd-api
       python scripts/fetch_who.py --base http://localhost:8382 --no-auth

Options: --release 2026-01 (default), --langs en,fr,es,pt, --workers 16.
The script resumes: codes already in the output are skipped, so it can be
interrupted and re-run.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"


class Auth:
    def __init__(self, client_id: str | None, secret: str | None, enabled: bool):
        self.cid, self.secret, self.enabled = client_id, secret, enabled
        self.token, self.expires = None, 0.0
        self.lock = threading.Lock()

    def header(self) -> dict:
        if not self.enabled:
            return {}
        with self.lock:
            if self.token is None or time.time() > self.expires - 120:
                data = urllib.parse.urlencode({"client_id": self.cid, "client_secret": self.secret,
                                               "scope": "icdapi_access", "grant_type": "client_credentials"}).encode()
                with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data), timeout=30) as r:
                    j = json.load(r)
                self.token = j["access_token"]
                self.expires = time.time() + float(j.get("expires_in", 3600))
            return {"Authorization": f"Bearer {self.token}"}


def _val(x) -> str:
    if isinstance(x, dict):
        return (x.get("@value") or "").strip()
    return (x or "").strip() if isinstance(x, str) else ""


def terms_from_entity(j: dict) -> list[str]:
    """Title, inclusions and index terms of one MMS entity (deduplicated)."""
    out = [_val(j.get("title"))]
    for key in ("inclusion", "indexTerm"):
        for it in j.get(key) or []:
            out.append(_val(it.get("label")))
    seen, res = set(), []
    for t in out:
        k = t.lower()
        if t and k not in seen:
            seen.add(k)
            res.append(t)
    return res


def fetch(base: str, release: str, tail: str, lang: str, auth: Auth, tries: int = 4) -> dict | None:
    url = f"{base.rstrip('/')}/icd/release/11/{release}/mms/{tail}"
    for i in range(tries):
        try:
            hdr = {"Accept": "application/json", "Accept-Language": lang, "API-Version": "v2", **auth.header()}
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 401:
                auth.token = None
            time.sleep(1.5 * (i + 1))
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="https://id.who.int")
    ap.add_argument("--release", default="2026-01")
    ap.add_argument("--langs", default="en,fr,es,pt")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-auth", action="store_true")
    ap.add_argument("--mms", default=str(ROOT / "data" / "mms-2026-01.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "who_terms.jsonl.gz"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N codes (for a quick test)")
    a = ap.parse_args()

    auth = Auth(os.environ.get("ICD_CLIENT_ID"), os.environ.get("ICD_CLIENT_SECRET"), not a.no_auth)
    if auth.enabled and not (auth.cid and auth.secret):
        sys.exit("set ICD_CLIENT_ID / ICD_CLIENT_SECRET, or use --no-auth with a local whoicd/icd-api container")

    rows = json.load(open(a.mms, encoding="utf-8"))
    if a.limit:
        rows = rows[: a.limit]
    done: set[tuple[str, str]] = set()
    kept = []
    if os.path.exists(a.out):
        with gzip.open(a.out, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["code"], r["lang"]))
                kept.append(line)
    jobs = []
    for r in rows:
        tail = r["uri"].split("/mms/", 1)[1]
        for lang in a.langs.split(","):
            if (r["code"], lang) not in done:
                jobs.append((r["code"], tail, lang))
    print(f"{len(jobs)} requests to make ({len(done)} already done)", file=sys.stderr)

    tmp = a.out + ".part"
    lock = threading.Lock()
    n_ok = n_miss = 0
    out = gzip.open(tmp, "wt", encoding="utf-8")
    try:
        for line in kept:
            out.write(line)
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(fetch, a.base, a.release, t, lang, auth): (c, lang) for c, t, lang in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                code, lang = futs[fut]
                j = fut.result()
                with lock:
                    if j is None:
                        n_miss += 1
                    else:
                        n_ok += 1
                        out.write(json.dumps({"code": code, "lang": lang, "terms": terms_from_entity(j)},
                                             ensure_ascii=False) + "\n")
                if i % 500 == 0:
                    print(f"  {i}/{len(jobs)} ok={n_ok} missing={n_miss}", file=sys.stderr)
    finally:  # keep partial progress on Ctrl-C; the next run resumes from it
        out.close()
        os.replace(tmp, a.out)
    print(f"wrote {a.out}: ok={n_ok} missing={n_miss}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
