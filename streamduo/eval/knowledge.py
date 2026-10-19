"""Knowledge metrics -- rule-based, no LLM judge.

Per clip, entities = non-generic gold-ontology entities of the clip's game found in the text.
  Entity-P   = |pred ∩ gt| / |pred|           (distinct entities, micro over clips)
  Entity-R   = |pred ∩ gt| / |gt|
  Ent/100w   = predicted entity mentions per 100 predicted words (how "knowledgeable" it sounds)
  Halluc     = share of predicted entity mentions that are entities of ANOTHER game and never occur
               anywhere in this game's own wiki text (e.g. "Malenia" in a Silksong clip). The
               "never in this game's wiki" condition removes shared names (LoL's champion
               "Wukong", generic words); only DISTINCTIVE foreign names count (at least one word
               with Zipf frequency < 2.5, so "Game Over"/"Perfect Timing"/"Campfire" never do).
               Sanity check: human GT commentary scores near 0 (see released_knowledge.json).
  K-strict R = Entity-R restricted to GT utterances labeled K that contain an entity

    python -m streamduo.eval.knowledge --preds preds.jsonl --manifest test_clean.jsonl [--format ours|released]
"""
import argparse
import collections
import json

from streamduo.eval.judge import load_preds, responses
from streamduo.paths import DATA_ROOT, MANIFEST_ROOT
from streamduo.wiki.ontology import EntityMatcher

from streamduo.eval.knowledge_core import distinctive, in_own_wiki, matcher, other_games  # noqa: E402,F401


def clip_knowledge(game, gt_text, pred_text, others):
    m = matcher(game)
    if m is None:
        return None
    gt = {e["norm"] for e in m.find(gt_text)}
    pred_hits = m.find(pred_text)
    pred = {e["norm"] for e in pred_hits}
    # hallucination: entity strings from other games' ontologies that are not entities of this game
    own_norms = set(m.lex)
    foreign = 0
    for og in others:
        om = matcher(og)
        if om is None:
            continue
        foreign += sum(1 for e in om.find(pred_text)
                       if e["norm"] not in own_norms and distinctive(e["norm"])
                       and not in_own_wiki(game, e["norm"]))
    words = max(len(pred_text.split()), 1)
    return {"tp": len(gt & pred), "n_pred": len(pred), "n_gt": len(gt), "mentions": len(pred_hits),
            "foreign": foreign, "words": words}


def aggregate(cs):
    cs = [c for c in cs if c]
    tp = sum(c["tp"] for c in cs)
    npred = sum(c["n_pred"] for c in cs)
    ngt = sum(c["n_gt"] for c in cs)
    ment = sum(c["mentions"] for c in cs)
    words = sum(c["words"] for c in cs)
    foreign = sum(c["foreign"] for c in cs)
    p = tp / npred if npred else 0.0
    r = tp / ngt if ngt else 0.0
    return {"Entity-P": round(p, 3), "Entity-R": round(r, 3),
            "Entity-F1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "Ent/100w": round(100 * ment / words, 2) if words else 0.0,
            "Halluc": round(foreign / (ment + foreign), 3) if ment + foreign else 0.0, "n": len(cs)}


def evaluate(rows, preds_by_key, fmt):
    by_game = collections.defaultdict(list)
    for r in rows:
        game = r.get("dataset_name") or r["metadata"]["dataset_name"]
        key = (game, r["video_path"].split("/")[-1], r["video_begin"], r.get("idx") if fmt == "released" else None)
        if key not in preds_by_key:
            continue
        gt_text = " ".join(a["text"] for a in r["annotations"] if a["role"] == "assistant" and a.get("text"))
        pred_text = " ".join(t for _, t in responses(preds_by_key[key]))
        by_game[game].append(clip_knowledge(game, gt_text, pred_text, other_games(game)))
    return {g: aggregate(cs) for g, cs in sorted(by_game.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST_ROOT / "all" / "test_orig.jsonl"))
    ap.add_argument("--format", default="ours", choices=["ours", "released"])
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest)]
    res = evaluate(rows, load_preds(args.preds, args.format), args.format)
    for g, v in res.items():
        print(f"{g:22s} {v}")


if __name__ == "__main__":
    main()
