"""Research agent: builds a source-grounded Game Dossier for one game under a tool-call budget.

Plan-and-execute (more robust than free-form ReAct for mid-size open models):
  1. plan:     questions a live commentator needs answered for this game
  2. execute:  per question -> write search queries -> search -> pick pages -> read ->
               extract facts, each tagged with its source page
  3. synth:    facts (numbered) -> dossier JSON; every item must cite fact ids ("src")
  4. ground:   drop items without valid citations, so the dossier contains only retrieved
               knowledge, never the LLM's parametric memory (critical for the unseen-game claim)
  5. render:   dossier -> text within max_tokens

Budget = number of tool calls (searches + page reads), spread evenly across questions.

Optional salience sweep (sweep=True, "v2"): one popular_pages call (the wiki's most-linked pages with
lead sentences, charged as SWEEP_COST tool calls) -> the LLM keeps the entities a commentator would
actually name on screen -> their leads become citable facts and synthesis prioritises them.
Motivation: the v1 budget curve showed coverage is limited by WHICH entities synthesis picks,
not by how much is retrieved.

    python -m streamduo.research.agent --game elden_ring --max-queries 24 --max-tokens 2048 --no-web
Outputs DATA_ROOT/dossiers/<run_name>/<game>.{json,txt,trace.json}
"""
import argparse
import json
import time

from streamduo.data.games import GAMES
from streamduo.llm import LLM
from streamduo.paths import DATA_ROOT
from streamduo.research.dossier import FIELDS, empty_dossier, ground, n_tokens, render
from streamduo.research.tools import Web, WikiIndex

LIST_FIELDS = [f for f in FIELDS if f not in ("game", "genre", "mode")]
SWEEP_COST = 4
SWEEP_K = 80

SWEEP = """These are the most-referenced pages of the "{title}" wiki (title, in-link count, lead):
{listing}
Select up to {n} titles that a live commentator would likely NAME while watching {mode} footage
(characters, bosses, enemies, units, champions, items/weapons/cards, abilities/moves, places/maps,
key mechanics{people}). Skip patch notes, cosmetics/skins, merchandise, real-world companies,
lore-only or meta pages. Return JSON: {{"titles": ["..."]}} ordered by how often they would be named."""

SYSTEM = """You are a research assistant preparing a professional live commentator who has never \
seen the game "{title}" ({mode} footage). The commentator watches gameplay video and must recognise \
what is on screen and talk about it like an expert.{people} Work only from the sources you are given; \
never add facts from your own memory."""

PLAN = """List {n} questions to research so a commentator can follow "{title}" footage. Cover: \
rules/win conditions, how to read the screen/UI, the entities seen most often (and what they look \
like), community jargon{people_q}. Return JSON: {{"questions": ["..."]}}"""

QUERIES = """Question: {question}
Write {n} short keyword search queries (2-5 words each) for the game's own wiki that would answer it. \
Do not include the game's title in the queries (every page is about this game). \
Return JSON: {{"queries": ["..."]}}"""

PICK = """Question: {question}
Search results:
{results}
Pick up to {n} page titles most likely to answer the question (exact titles from the list). \
Return JSON: {{"titles": ["..."]}}"""

EXTRACT = """Question: {question}
Source page "{title}":
<<<
{text}
>>>
Extract the facts from THIS PAGE that a live commentator would use for the question (names, what \
things look like on screen, what they do, jargon). At most {n} facts, each one short sentence. \
If the page is irrelevant return an empty list. Return JSON: {{"facts": ["..."]}}"""

SYNTH = """Facts (numbered, each from a retrieved source):
{facts}

Fill a dossier for "{title}" using ONLY these facts. Field guide:
{fields}

Rules:
- Every list item MUST include "src": [fact numbers it is based on]. Items you cannot support with \
the facts must be omitted. Never use your own knowledge.
- List items are objects (strings are not allowed): core_rules/commentary_phrases items are \
{{"text": ..., "src": [...]}}; ui_glossary {{"element", "meaning", "src"}}; entities {{"name", "type", \
"visual_cue", "facts", "src"}} (visual_cue "" if the facts do not describe the look); terminology \
{{"term", "meaning", "src"}}; people {{"name", "role", "facts", "src"}}.
- core_rules: 3-8 items on objectives, how a match/run is won or lost, and the core gameplay \
loop, whenever any fact covers them (do not leave it empty in that case).
- commentary_phrases: only phrases quoted in the facts; otherwise leave it empty.
- people: only real players, casters, or teams (not games, expansions, or events).
- At most {max_entities} entities, most commentary-relevant first.
Return one JSON object with keys: {keys}."""


_ground = ground  # kept for backward compatibility


def run(game, llm, max_queries=24, max_tokens=2048, max_entities=40, n_questions=8,
        blocklist=(), use_web=False, sweep=False):
    g = GAMES.get(game)
    title = g.title if g else game.replace("fresh_", "").replace("_", " ").title()
    mode = {"multi": "esports tournament", "solo": "solo", "guidance": "guide"}.get(
        getattr(g, "mode", ""), "gameplay")
    is_esports = getattr(g, "source", "") == "tournament"
    wiki = WikiIndex(game)
    web = Web(blocklist) if use_web else None
    sys_msg = {"role": "system", "content": SYSTEM.format(
        title=title, mode=mode,
        people=" Notable players and teams matter too." if is_esports else "")}

    def ask(content, **kw):
        return llm.chat_json([sys_msg, {"role": "user", "content": content}], **kw)

    t0 = time.time()
    trace = {"game": game, "title": title, "max_queries": max_queries, "web": bool(web), "steps": []}
    questions = ask(PLAN.format(n=n_questions, title=title,
                                people_q=", notable players/teams" if is_esports else ""))
    questions = [q for q in questions.get("questions", []) if isinstance(q, str)][:n_questions]
    trace["questions"] = questions

    facts, seen_pages, calls = [], set(), 0
    salient = []
    if sweep and max_queries > SWEEP_COST:
        pop = wiki.popular(SWEEP_K)
        calls += SWEEP_COST
        listing = "\n".join(f"- {p['title']} ({p['inlinks']}): {p['lead']}" for p in pop)
        picked = ask(SWEEP.format(title=title, listing=listing, n=40, mode=mode,
                                  people=", players, teams" if is_esports else "")).get("titles", [])
        by_t = {p["title"]: p for p in pop}
        salient = [t for t in picked if t in by_t]
        facts += [{"fact": by_t[t]["lead"], "source": t} for t in salient if by_t[t]["lead"]]
        trace["steps"].append({"action": "popular_pages", "kept": salient})
    per_q = max(2, (max_queries - calls) // max(len(questions), 1))
    for q in questions:
        budget = min(per_q, max_queries - calls)
        if budget < 2:
            break
        n_search = max(1, budget // 3)
        queries = ask(QUERIES.format(question=q, n=n_search)).get("queries", [])[:n_search]
        results = {}
        for qq in queries:
            for h in wiki.search(str(qq), k=6):
                results.setdefault(h["title"], h)
            if web is not None:
                for h in web.search(str(qq), k=3):
                    results.setdefault(h["url"], {"title": h["url"], "snippet": h["snippet"], "web": True})
            calls += 1
            trace["steps"].append({"q": q, "action": "search", "query": qq})
        cands = [t for t in results if t not in seen_pages]
        if not cands:
            continue
        listing = "\n".join(f"- {t}: {results[t]['snippet'][:150]}" for t in cands[:20])
        picks = ask(PICK.format(question=q, results=listing, n=budget - len(queries))).get("titles", [])
        for t in [t for t in picks if t in results][: budget - len(queries)]:
            seen_pages.add(t)
            page = web.fetch(t) if results[t].get("web") else wiki.read(t)
            calls += 1
            trace["steps"].append({"q": q, "action": "read", "title": t})
            if not page:
                continue
            got = ask(EXTRACT.format(question=q, title=t, text=page["text"], n=8)).get("facts", [])
            facts += [{"fact": f, "source": t} for f in got if isinstance(f, str) and f.strip()]

    dossier = empty_dossier(title, getattr(g, "mode", ""))
    dropped = 0
    if facts:
        numbered = "\n".join(f"[{i + 1}] ({f['source']}) {f['fact']}" for i, f in enumerate(facts))
        prompt = SYNTH.format(facts=numbered, title=title, max_entities=max_entities,
                              fields="\n".join(f"- {k}: {v}" for k, v in FIELDS.items()), keys=list(FIELDS))
        if salient:
            prompt += ("\n\nSalient entities (most likely to be named on screen, in priority order); include "
                       "as many as the facts support, with SHORT facts (<= 15 words each):\n"
                       + ", ".join(salient))
        synth = ask(prompt, max_tokens=8000)
        dossier.update({k: v for k, v in synth.items() if k in FIELDS})
        dossier["game"], dossier["mode"] = title, getattr(g, "mode", "")
        dropped = _ground(dossier, len(facts))
    text = render(dossier, max_tokens=max_tokens)
    trace.update({"n_facts": len(facts), "n_tool_calls": calls, "n_pages_read": len(seen_pages),
                  "n_items_dropped_ungrounded": dropped, "seconds": round(time.time() - t0, 1),
                  "dossier_tokens": n_tokens(text), "facts": facts})
    return dossier, text, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", nargs="+", required=True)
    ap.add_argument("--max-queries", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--sweep", action="store_true", help="v2: wiki salience sweep")
    ap.add_argument("--max-entities", type=int, default=40)
    args = ap.parse_args()
    llm = LLM()
    use_web = not args.no_web and Web().enabled
    run_name = args.run_name or f"q{args.max_queries}_k{args.max_tokens}" + ("_web" if use_web else "")
    out = DATA_ROOT / "dossiers" / run_name
    out.mkdir(parents=True, exist_ok=True)
    for game in args.game:
        dossier, text, trace = run(game, llm, args.max_queries, args.max_tokens,
                                   max_entities=args.max_entities, use_web=use_web, sweep=args.sweep)
        (out / f"{game}.json").write_text(json.dumps(dossier, indent=2, ensure_ascii=False))
        (out / f"{game}.txt").write_text(text)
        (out / f"{game}.trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False))
        print(f"[{game}] calls={trace['n_tool_calls']} pages={trace['n_pages_read']} "
              f"facts={trace['n_facts']} dropped={trace['n_items_dropped_ungrounded']} "
              f"tokens={trace['dossier_tokens']} {trace['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
