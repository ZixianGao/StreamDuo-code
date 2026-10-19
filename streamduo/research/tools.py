"""Tools available to the research agent.

  wiki_search(query)  BM25 over the frozen GameWiki snapshot of the target game (always on)
  read_page(title)    text of one snapshot page (truncated)
  popular_pages(k)    most-linked in-scope pages of the wiki (in-link count from the wiki's own
                      link graph) with their lead sentence -- a salience prior the agent can use
                      to decide which entities matter
  web_search(query)   live web search; enabled only if TAVILY_API_KEY or SERPER_API_KEY is set
  fetch(url)          plain text of a web page (only with web search enabled)

Leakage guard: web results whose URL or title contains a blocked string (test video IDs,
test match names) are dropped before the agent sees them.
"""
import json
import os
import re

import bm25s
import requests

from streamduo.paths import DATA_ROOT
from streamduo.wiki.ontology import is_disambiguation
from streamduo.wiki.scope import in_scope

PAGE_CHARS = 3000
SNIPPET_CHARS = 300


LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def inlink_counts(game):
    """In-link counts over the game's primary wiki (cached). Counts each linking page once."""
    cache = DATA_ROOT / "wiki" / game / "inlinks.json"
    if cache.exists():
        return json.loads(cache.read_text())
    counts = {}
    with open(DATA_ROOT / "wiki" / game / "pages.jsonl") as f:
        for line in f:
            p = json.loads(line)
            if p["source"] != "primary":
                continue
            targets = {m.group(1).strip().replace("_", " ") for m in LINK_RE.finditer(p.get("wikitext") or "")}
            for t in targets:
                if ":" in t:  # File:, Category:, interwiki
                    continue
                k = t[:1].upper() + t[1:]
                counts[k] = counts.get(k, 0) + 1
    cache.write_text(json.dumps(counts))
    return counts


class WikiIndex:
    def __init__(self, game):
        self.pages = []
        with open(DATA_ROOT / "wiki" / game / "pages.jsonl") as f:
            for line in f:
                p = json.loads(line)
                if p["text"].strip() and not is_disambiguation(p) and in_scope(game, p):
                    self.pages.append({"title": p["title"], "text": p["text"], "source": p["source"]})
        self.by_title = {p["title"].lower(): p for p in self.pages}
        self.game = game
        corpus = [f"{p['title']}\n{p['title']}\n{p['text'][:4000]}" for p in self.pages]
        self.retriever = bm25s.BM25()
        self.retriever.index(bm25s.tokenize(corpus, stopwords="en", show_progress=False),
                             show_progress=False)

    def search(self, query, k=5, pool=50):
        """BM25 top-`pool`, re-ranked with a bonus for query terms that appear in the title."""
        docs, scores = self.retriever.retrieve(bm25s.tokenize([query], stopwords="en", show_progress=False),
                                               k=min(pool, len(self.pages)), show_progress=False)
        q_terms = set(re.findall(r"\w+", query.lower()))
        ranked = []
        for i, s in zip(docs[0], scores[0]):
            p = self.pages[int(i)]
            t_terms = set(re.findall(r"\w+", p["title"].lower()))
            overlap = len(q_terms & t_terms) / max(len(t_terms), 1)
            ranked.append((float(s) * (1 + overlap), p))
        ranked.sort(key=lambda x: -x[0])
        out = []
        for s, p in ranked[:k]:
            snippet = re.sub(r"\s+", " ", p["text"][:SNIPPET_CHARS])
            out.append({"title": p["title"], "snippet": snippet, "score": round(s, 2)})
        return out

    def popular(self, k=60, skip=()):
        """Most-linked in-scope primary pages with a one-sentence lead."""
        counts = inlink_counts(self.game)
        skip = {s.lower() for s in skip}
        ranked = sorted((p for p in self.pages if p["source"] == "primary" and p["title"].lower() not in skip),
                        key=lambda p: -counts.get(p["title"][:1].upper() + p["title"][1:], 0))
        out = []
        for p in ranked[:k]:
            lead = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", p["text"][:600]).strip())[0][:200]
            out.append({"title": p["title"], "inlinks": counts.get(p["title"][:1].upper() + p["title"][1:], 0),
                        "lead": lead})
        return out

    def read(self, title):
        p = self.by_title.get(title.lower())
        if p is None:
            hits = self.search(title, k=1)
            if not hits:
                return None
            p = self.by_title[hits[0]["title"].lower()]
        return {"title": p["title"], "text": p["text"][:PAGE_CHARS]}


class Web:
    def __init__(self, blocklist=()):
        self.blocklist = [b.lower() for b in blocklist if b]
        self.tavily = os.environ.get("TAVILY_API_KEY")
        self.serper = os.environ.get("SERPER_API_KEY")

    @property
    def enabled(self):
        return bool(self.tavily or self.serper)

    def _blocked(self, *fields):
        blob = " ".join(f or "" for f in fields).lower()
        return any(b in blob for b in self.blocklist)

    def search(self, query, k=5):
        if self.tavily:
            r = requests.post("https://api.tavily.com/search", timeout=60,
                              json={"api_key": self.tavily, "query": query, "max_results": k + 5})
            items = [{"title": x["title"], "url": x["url"], "snippet": x.get("content", "")[:SNIPPET_CHARS]}
                     for x in r.json().get("results", [])]
        elif self.serper:
            r = requests.post("https://google.serper.dev/search", timeout=60,
                              headers={"X-API-KEY": self.serper}, json={"q": query, "num": k + 5})
            items = [{"title": x["title"], "url": x["link"], "snippet": x.get("snippet", "")}
                     for x in r.json().get("organic", [])]
        else:
            return []
        return [x for x in items if not self._blocked(x["url"], x["title"])][:k]

    def fetch(self, url):
        if self._blocked(url):
            return {"url": url, "text": "[blocked by leakage guard]"}
        r = requests.get(url, timeout=60, headers={"User-Agent": "RtC-research-agent/0.1"})
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", r.text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return {"url": url, "text": re.sub(r"\s+", " ", text)[:PAGE_CHARS]}
