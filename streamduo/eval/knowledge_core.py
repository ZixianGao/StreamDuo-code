"""Dependency-light entity helpers shared by knowledge metrics and RL rewards (no LLM imports)."""
import json

from streamduo.paths import DATA_ROOT
from streamduo.wiki.ontology import EntityMatcher

_M = {}
_WIKI_TEXT = {}


def wiki_text(game):
    """Lower-cased concatenated text of the game's wiki snapshot (in-scope pages), cached."""
    if game not in _WIKI_TEXT:
        from streamduo.wiki.scope import in_scope
        parts = []
        with open(DATA_ROOT / "wiki" / game / "pages.jsonl") as f:
            for line in f:
                p = json.loads(line)
                if in_scope(game, p):
                    parts.append(p["title"].lower())
                    parts.append(p["text"].lower())
        _WIKI_TEXT[game] = " \n ".join(parts)
    return _WIKI_TEXT[game]


_SEEN = {}
RARE_ZIPF = 2.5


def distinctive(norm):
    from wordfreq import zipf_frequency
    return any(zipf_frequency(w, "en") < RARE_ZIPF for w in norm.replace("'", " ").split() if w.isalpha())


def in_own_wiki(game, norm):
    key = (game, norm)
    if key not in _SEEN:
        _SEEN[key] = norm in wiki_text(game)
    return _SEEN[key]


def matcher(game):
    if game not in _M:
        try:
            _M[game] = EntityMatcher.for_game(game)
        except FileNotFoundError:
            _M[game] = None
    return _M[game]


def other_games(game):
    games = [p.name for p in (DATA_ROOT / "wiki").iterdir() if (p / "ontology.jsonl").exists()]
    return [g for g in games if g != game]


