"""Polite, resumable MediaWiki crawler for the frozen GameWiki snapshot.

For each game in configs/wikis.yaml it fetches every non-redirect main-namespace page of the
primary wiki (50 pages per request, <= 1 request/s per host) plus the Wikipedia article, and
writes DATA_ROOT/wiki/<game>/pages.jsonl with:
    {source, pageid, title, categories, wikitext, text, infobox, fetched_at}

Resumable: page ids already in pages.jsonl are skipped. snapshot.json records the crawl date.

    python -m streamduo.wiki.crawl --games elden_ring lol      # or --all
"""
import argparse
import datetime
import json
import threading
import time

import mwparserfromhell
import requests
import yaml

from streamduo.paths import DATA_ROOT, REPO_ROOT

UA = "RtC-GameWiki-snapshot/0.1 (academic research crawler; respects 1 req/s)"
BATCH = 50


class Api:
    def __init__(self, url, min_interval=1.0):
        self.url = url
        self.min_interval = min_interval
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
        self._last = 0.0

    def get(self, **params):
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        for attempt in range(8):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.s.get(self.url, params=params, timeout=60)
                if r.status_code in (429, 503) or "maxlag" in r.text[:200]:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError):
                time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f"giving up on {self.url} {params}")


def all_page_ids(api):
    ids, cont = [], {}
    while True:
        d = api.get(action="query", list="allpages", apnamespace=0, apfilterredir="nonredirects",
                    aplimit="max", **cont)
        ids += [p["pageid"] for p in d["query"]["allpages"]]
        if "continue" not in d:
            return ids
        cont = d["continue"]


def parse_wikitext(wt):
    code = mwparserfromhell.parse(wt or "")
    infobox = {}
    for tpl in code.filter_templates(recursive=False):
        name = str(tpl.name).strip().lower()
        if "infobox" in name or name.startswith(("character", "boss", "item", "weapon", "champion", "card")):
            infobox = {"_template": str(tpl.name).strip()}
            for p in tpl.params:
                val = p.value.strip_code().strip()
                if val:
                    infobox[str(p.name).strip()] = val[:500]
            break
    text = code.strip_code(normalize=True, collapse=True)
    return text.strip(), infobox


def fetch_batch(api, pageids, source):
    d = api.get(action="query", pageids="|".join(map(str, pageids)), prop="revisions|categories",
                rvprop="content", rvslots="main", cllimit="max", clshow="!hidden")
    out = []
    for p in d["query"]["pages"]:
        if p.get("missing") or "revisions" not in p:
            continue
        rev = p["revisions"][0]
        # MediaWiki >= 1.32 nests content under slots; older installs (e.g. Yugipedia) do not.
        wt = rev["slots"]["main"].get("content", "") if "slots" in rev else rev.get("content", "")
        text, infobox = parse_wikitext(wt)
        out.append({
            "source": source, "pageid": p["pageid"], "title": p["title"],
            "categories": [c["title"].split(":", 1)[-1] for c in p.get("categories", [])],
            "wikitext": wt, "text": text, "infobox": infobox,
            "fetched_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })
    return out


def crawl_game(game, spec, max_pages=None):
    out_dir = DATA_ROOT / "wiki" / game
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pages.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            done.add((r["source"], r["pageid"]))

    log = open(out_dir / "crawl.log", "a")
    with open(path, "a") as f:
        # Wikipedia article (single page).
        wp = Api("https://en.wikipedia.org/w/api.php")
        d = wp.get(action="query", titles=spec["wikipedia"], redirects=1)
        page = d["query"]["pages"][0]
        if not page.get("missing") and ("wikipedia", page["pageid"]) not in done:
            for r in fetch_batch(wp, [page["pageid"]], "wikipedia"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        api = Api(spec["primary"])
        ids = [i for i in all_page_ids(api) if ("primary", i) not in done]
        if max_pages:
            ids = ids[:max_pages]
        print(f"[{game}] {len(ids)} pages to fetch", file=log, flush=True)
        for k in range(0, len(ids), BATCH):
            for r in fetch_batch(api, ids[k:k + BATCH], "primary"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if k // BATCH % 20 == 0:
                print(f"[{game}] {k + BATCH}/{len(ids)}", file=log, flush=True)

    snap = {"game": game, "primary": spec["primary"], "wikipedia": spec["wikipedia"],
            "completed_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "n_pages": sum(1 for _ in open(path))}
    (out_dir / "snapshot.json").write_text(json.dumps(snap, indent=2))
    print(f"[{game}] done {snap['n_pages']} pages", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-pages", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "wikis.yaml"))["games"]
    games = list(cfg) if args.all else args.games
    # One thread per game: different hosts crawl in parallel, each host stays at <= 1 req/s.
    # (Wikipedia is hit once per game, so its rate stays negligible.)
    threads = [threading.Thread(target=crawl_game, args=(g, cfg[g], args.max_pages)) for g in games]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
