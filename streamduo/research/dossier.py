"""Game Dossier schema, rendering, and token-budgeted compression.

The dossier is stored as JSON and rendered to plain text for the commentator's system prompt.
Rendering is priority-ordered so that truncation to a token budget drops the least useful
material first.
"""
import functools

from streamduo.paths import DATA_ROOT

FIELDS = {
    "game": "Game title",
    "genre": "Genre and perspective in one line",
    "mode": "esports | walkthrough | guide",
    "core_rules": "Win/lose conditions, core loop, resources (list of short strings)",
    "ui_glossary": "On-screen elements -> meaning (list of {element, meaning})",
    "entities": "Characters/bosses/items/abilities/locations a commentator will likely need "
                "(list of {name, type, visual_cue, facts})",
    "terminology": "Community jargon with one-line definitions (list of {term, meaning})",
    "people": "Esports only: notable players/teams/events (list of {name, role, facts})",
    "commentary_phrases": "Typical caster phrases/hype lines for this game (list of strings)",
}

# Render order = truncation priority (earlier survives longer).
ORDER = ["game", "genre", "core_rules", "ui_glossary", "entities", "terminology", "people",
         "commentary_phrases"]


def empty_dossier(game, mode=""):
    return {"game": game, "genre": "", "mode": mode, "core_rules": [], "ui_glossary": [],
            "entities": [], "terminology": [], "people": [], "commentary_phrases": []}


def _render_item(field, it):
    if isinstance(it, str):
        return f"- {it}"
    if field == "entities":
        cue = f" [looks like: {it['visual_cue']}]" if it.get("visual_cue") else ""
        return f"- {it.get('name', '?')} ({it.get('type', '')}){cue}: {it.get('facts', '')}"
    if field == "ui_glossary":
        return f"- {it.get('element', '?')}: {it.get('meaning', '')}"
    if field == "terminology":
        return f"- {it.get('term', '?')}: {it.get('meaning', '')}"
    if field == "people":
        return f"- {it.get('name', '?')} ({it.get('role', '')}): {it.get('facts', '')}"
    return "- " + "; ".join(f"{k}: {v}" for k, v in it.items())


def render_lines(d):
    """Yield (field, line) in priority order; headers are emitted with their first item."""
    titles = {"core_rules": "Rules", "ui_glossary": "Screen/UI", "entities": "Key entities",
              "terminology": "Terminology", "people": "People", "commentary_phrases": "Caster phrases"}
    yield "game", f"=== Game Dossier: {d.get('game', '')} ==="
    if d.get("genre"):
        yield "genre", f"Genre: {d['genre']}"
    for field in ORDER[2:]:
        items = d.get(field) or []
        for i, it in enumerate(items):
            line = _render_item(field, it)
            yield field, (f"{titles[field]}:\n{line}" if i == 0 else line)


@functools.lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(DATA_ROOT / "hf" / "chenjoya__LiveCC-7B-Base")


def n_tokens(text):
    return len(_tokenizer()(text, add_special_tokens=False)["input_ids"])


def render(d, max_tokens=None):
    """Render to text; with max_tokens, fill fields round-robin by priority until the budget is hit.

    Round-robin (one item per field per pass, fields in ORDER) keeps every section represented
    at small budgets instead of spending everything on the first long list.
    """
    lines = list(render_lines(d))
    if max_tokens is None:
        return "\n".join(l for _, l in lines)
    by_field = {}
    for f, l in lines:
        by_field.setdefault(f, []).append(l)
    out, used = [], 0
    queues = [list(by_field[f]) for f in ORDER if f in by_field]
    while any(queues):
        progressed = False
        for q in queues:
            if not q:
                continue
            cost = n_tokens(q[0]) + 1
            if used + cost > max_tokens:
                q.clear()  # this field is full; others may still fit shorter items
                continue
            out.append((len(out), q.pop(0)))
            used += cost
            progressed = True
        if not progressed:
            break
    # Restore document order: lines were generated in render order per field, so sort by
    # the original index of each line.
    index = {l: i for i, (_, l) in enumerate(lines)}
    return "\n".join(sorted((l for _, l in out), key=index.__getitem__))


LIST_FIELDS = [f for f in FIELDS if f not in ("game", "genre", "mode")]


def ground(dossier, n_facts):
    """Keep only items citing at least one valid fact id; flatten {"text"} items to strings."""
    dropped = 0
    for field in LIST_FIELDS:
        kept = []
        for it in dossier.get(field) or []:
            if not isinstance(it, dict):
                dropped += 1
                continue
            src = [s for s in it.get("src", []) if isinstance(s, int) and 1 <= s <= n_facts]
            if not src:
                dropped += 1
                continue
            it = {k: v for k, v in it.items() if k != "src"}
            kept.append(it["text"] if set(it) == {"text"} else it)
        dossier[field] = kept
    return dropped


def ground_by_sources(dossier, seen_sources):
    """Keep items citing at least one source actually retrieved in this episode."""
    seen = {s.lower() for s in seen_sources}
    dropped = 0
    for field in LIST_FIELDS:
        kept = []
        for it in dossier.get(field) or []:
            if not isinstance(it, dict):
                dropped += 1
                continue
            src = [str(s).lower() for s in (it.get("src") or [])]
            if not any(s in seen for s in src):
                dropped += 1
                continue
            it = {k: v for k, v in it.items() if k != "src"}
            kept.append(it["text"] if set(it) == {"text"} else it)
        dossier[field] = kept
    return dropped
