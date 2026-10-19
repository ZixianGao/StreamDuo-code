"""Gold ontology from the GameWiki snapshot, and an n-gram entity matcher.

Ontology entry: {name, type, pageid, generic}
  * type is inferred from the infobox template name and page categories (keyword rules).
  * generic=True if the same (normalized) title is a page in >= GENERIC_MIN_GAMES game wikis
    (e.g. "Sword", "Health"), or if it is a single common English word (Zipf frequency >=
    COMMON_WORD_ZIPF, e.g. "Will", "Time", "Flash") -- such names are not reliable evidence of
    game-specific knowledge when string-matched in commentary.
  * Disambiguation pages are skipped.

The ontology is used ONLY for evaluation (Entity-P/R, hallucinated-entity rate, Clean-setting
residual checks), never shown to the research agent.

    python -m streamduo.wiki.ontology            # builds DATA_ROOT/wiki/<game>/ontology.jsonl for all crawled games
"""
import collections
import json
import re


from streamduo.paths import DATA_ROOT
from streamduo.wiki.scope import in_scope

GENERIC_MIN_GAMES = 3
COMMON_WORD_ZIPF = 4.0
MAX_NGRAM = 6

TYPE_RULES = [  # first match wins; tried on the infobox template name first, then on categories
    ("PLAYER", r"\b(players?|casters?|commentators?|personalit(y|ies)|streamers?|coaches)\b"),
    ("CHARACTER", r"\b(characters?|npcs?|bosses|boss|enemies|enemy|champions?|heroes|hero|units?|"
                  r"creatures?|monsters?|fighters?|companions?|mobs?|yaoguais?|races?)\b"),
    ("ABILITY", r"\b(abilit(y|ies)|skills?|spells?|moves?|techniques?|spirit ?skills?|cantrips?|"
                r"special moves?|super arts?|transformations?|incantations?|sorceries|sorcery)\b"),
    ("ITEM", r"\b(items?|weapons?|armou?r|cards?|equipment|consumables?|materials?|ingredients?|"
             r"talismans?|gear|blocks?|tools?|cyberware|quickhacks?|runes?|curios?|vessels?)\b"),
    ("LOCATION", r"\b(locations?|areas?|maps?|regions?|dungeons?|places?|districts?|zones?|"
                 r"biomes?|chapters?|levels?|sites? of grace|stages?)\b"),
    ("TEAM", r"\b(teams?|organizations?|clans?)\b"),
    ("EVENT", r"\b(tournaments?|leagues?|championships?|events?|seasons?|majors?)\b"),
    ("MECHANIC", r"\b(mechanics?|gameplay|status effects?|attributes?|stats?|conditions?|systems?)\b"),
]
_TYPE_RES = [(t, re.compile(p)) for t, p in TYPE_RULES]


def norm(s):
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)  # drop disambiguation suffix "Foo (boss)"
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^\w'+:&-]+", " ", s)
    return " ".join(s.split())


def is_disambiguation(page):
    return "(disambiguation)" in page["title"].lower() or any(
        "disambiguation" in c.lower() for c in page.get("categories", []))


def is_common_word(n):
    from wordfreq import zipf_frequency  # lazy: only needed when building ontologies
    return " " not in n and zipf_frequency(n, "en") >= COMMON_WORD_ZIPF


def infer_type(page):
    template = page.get("infobox", {}).get("_template", "").lower()
    cats = " | ".join(c.lower() for c in page.get("categories", []))
    for blob in (template, cats):
        for t, rx in _TYPE_RES:
            if blob and rx.search(blob):
                return t
    return "OTHER"


def load_pages(game):
    path = DATA_ROOT / "wiki" / game / "pages.jsonl"
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def build_all():
    games = sorted(p.name for p in (DATA_ROOT / "wiki").iterdir() if (p / "pages.jsonl").exists())
    per_game = {}
    for g in games:
        ents = {}
        for p in load_pages(g):
            if p["source"] != "primary" or is_disambiguation(p) or not in_scope(g, p):
                continue
            n = norm(p["title"])
            if len(n) < 3 or n in ents:
                continue
            ents[n] = {"name": p["title"], "norm": n, "type": infer_type(p), "pageid": p["pageid"]}
        per_game[g] = ents

    df = collections.Counter(n for ents in per_game.values() for n in ents)
    for g, ents in per_game.items():
        out = DATA_ROOT / "wiki" / g / "ontology.jsonl"
        types = collections.Counter()
        with open(out, "w") as f:
            for e in ents.values():
                e["generic"] = df[e["norm"]] >= GENERIC_MIN_GAMES or is_common_word(e["norm"])
                types[e["type"]] += 1
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        n_gen = sum(e["generic"] for e in ents.values())
        print(f"{g:24s} entities={len(ents):7d} generic={n_gen:5d} types={dict(types.most_common(6))}")


class EntityMatcher:
    """Longest-match-first n-gram lookup of ontology names in free text."""

    def __init__(self, entries, include_generic=False):
        self.lex = {e["norm"]: e for e in entries if include_generic or not e["generic"]}

    @classmethod
    def for_game(cls, game, **kw):
        path = DATA_ROOT / "wiki" / game / "ontology.jsonl"
        return cls([json.loads(l) for l in open(path)], **kw)

    def find(self, text):
        toks = norm(text).split()
        hits, i = [], 0
        while i < len(toks):
            for n in range(min(MAX_NGRAM, len(toks) - i), 0, -1):
                cand = " ".join(toks[i:i + n])
                if cand in self.lex:
                    hits.append(self.lex[cand])
                    i += n
                    break
            else:
                i += 1
        return hits


if __name__ == "__main__":
    build_all()
