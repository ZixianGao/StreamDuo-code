"""Per-game page scope for series-wide wikis (configs/wikis.yaml: scope_include / scope_exclude)."""
import functools
import re

import yaml

from streamduo.paths import REPO_ROOT

LEAD_CHARS = 1500


@functools.lru_cache(maxsize=None)
def _rules(game):
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "wikis.yaml"))["games"].get(game, {})
    inc = cfg.get("scope_include")
    exc = cfg.get("scope_exclude")
    return (re.compile(inc, re.I) if inc else None, re.compile(exc, re.I) if exc else None)


def in_scope(game, page):
    """Wikipedia pages are always in scope; primary-wiki pages are filtered by the game's rules."""
    if page.get("source") == "wikipedia":
        return True
    inc, exc = _rules(game)
    head = page["title"] + " | " + " | ".join(page.get("categories", []))
    if exc is not None and exc.search(head):
        return False
    if inc is not None and not inc.search(head + " | " + page.get("text", "")[:LEAD_CHARS]):
        return False
    return True
