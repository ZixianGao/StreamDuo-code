"""Training data for USING research-agent answers in-stream ("Ask What You See", step 3).

For every source-game training clip and every entity e the human names (first mention at second t),
an answer is assumed to arrive at t - 1 (the query was sent earlier) and is injected into the history
block, formatted like the oracle dossier:
    Entities likely on screen now:
    - <name> (<type>): <wiki lead>
The answer lists e plus up to `n_distract` distractor entities of the same game, shuffled; with
probability `miss_p` it contains only distractors (a retrieval miss), so the commentator learns to
check answers against the frames instead of reading out whatever arrives.

    python -m streamduo.interact.train_answers --fold S_bg3
-> manifests/<fold>/train_answers.jsonl  {key, lookups: {t: answer}}  (for StreamClipDataset.lookups)
"""
import argparse
import json
import random
import re

from streamduo.paths import MANIFEST_ROOT


def entry(wiki, e):
    page = wiki.read(e["name"])
    lead = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", page["text"][:500]))[0][:180] if page else ""
    name = re.sub(r" *\([^)]*\)$", "", e["name"])
    return f"- {name} ({e['type'].lower()}): {lead}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="S_bg3")
    ap.add_argument("--n_distract", type=int, default=2)
    ap.add_argument("--miss_p", type=float, default=0.3)
    ap.add_argument("--split", default="train", help="test: oracle answers on test clips (upper bound)")
    args = ap.parse_args()
    from streamduo.research.tools import WikiIndex
    from streamduo.wiki.ontology import EntityMatcher

    rng = random.Random(0)
    recipe = json.loads((MANIFEST_ROOT / args.fold / "train_recipe.json").read_text())
    if args.split == "train":
        out = MANIFEST_ROOT / args.fold / ("train_answers.jsonl" if args.miss_p > 0 else "train_answers_full.jsonl")
        sources = {g: [json.loads(l) for l in open(MANIFEST_ROOT / args.fold / f"train_{g}.jsonl")]
                   for g in recipe["train_games"]}
    else:
        out = MANIFEST_ROOT / args.fold / "test_answers_oracle.jsonl"
        sources = {}
        for l in open(MANIFEST_ROOT / args.fold / "test_clean.jsonl"):
            r = json.loads(l)
            sources.setdefault(r["metadata"]["dataset_name"], []).append(r)
    n_clips = n_ans = n_miss = 0
    with open(out, "w") as f:
        for g, rows in sources.items():
            try:
                m, wiki = EntityMatcher.for_game(g), WikiIndex(g)
            except FileNotFoundError:
                continue
            pool = [e for e in m.lex.values() if not e.get("generic")]
            for r in rows:
                seen, lookups = set(), {}
                for a in sorted((a for a in r["annotations"] if a["role"] == "assistant" and a.get("text")),
                                key=lambda a: a["start"]):
                    t = int(a["start"])
                    for e in m.find(a["text"]):
                        if e.get("generic") or e["norm"] in seen or t - 1 < r["video_begin"]:
                            continue
                        seen.add(e["norm"])
                        miss = rng.random() < args.miss_p
                        ents = [] if miss else [e]
                        ents += rng.sample([x for x in pool if x["norm"] != e["norm"]], args.n_distract)
                        rng.shuffle(ents)
                        text = "Entities likely on screen now:\n" + "\n".join(entry(wiki, x) for x in ents)
                        lookups[str(t - 1)] = (lookups.get(str(t - 1), "") + "\n" + text).strip()
                        n_ans += 1
                        n_miss += miss
                if lookups:
                    n_clips += 1
                    f.write(json.dumps({"key": f"{r['video_path']}@{r['video_begin']}", "lookups": lookups},
                                       ensure_ascii=False) + "\n")
    print(f"wrote {out}: {n_clips} clips with answers, {n_ans} answers ({n_miss} misses)")


if __name__ == "__main__":
    main()
