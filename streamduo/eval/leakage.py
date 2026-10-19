"""Source-game leakage: how often a commentator trained on a source game says that game's names in an
unseen game.

    Leak = (# mentions of DISTINCTIVE source-game entities that never occur in the target game's own wiki)
           / (# entity mentions of the target game + those leaked mentions)

Same filters as the cross-game hallucination metric (streamduo.eval.knowledge): shared names and common words
are excluded. Undefined for models that were never trained on a source game (zero-shot) -> reported as
N/A. The broader cross-game version (any other game's names) stays in streamduo.eval.report as "Halluc".

    python -m streamduo.eval.leakage --preds preds.jsonl --manifest $STREAMDUO_DATA/manifests/S_bg3/test_clean.jsonl
"""
import argparse
import json

from streamduo.eval.judge import load_preds, responses
from streamduo.eval.knowledge_core import distinctive, in_own_wiki, matcher



def leakage(rows, preds, source):
    src = matcher(source)
    leak = ment = 0
    per_game = {}
    for r in rows:
        g = r["metadata"]["dataset_name"]
        key = (g, r["video_path"].split("/")[-1], r["video_begin"], None)
        m = matcher(g)
        if key not in preds or m is None or g == source:
            continue
        text = " ".join(t for _, t in responses(preds[key]))
        own = set(m.lex)
        n_own = len(m.find(text))
        n_leak = sum(1 for e in src.find(text)
                     if e["norm"] not in own and distinctive(e["norm"]) and not in_own_wiki(g, e["norm"]))
        leak += n_leak
        ment += n_own
        a, b = per_game.get(g, (0, 0))
        per_game[g] = (a + n_leak, b + n_own)
    out = {g: round(a / max(a + b, 1), 3) for g, (a, b) in sorted(per_game.items())}
    out["_overall"] = round(leak / max(leak + ment, 1), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", default="baldurs_gate_3", help="the training game")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest)]
    res = leakage(rows, load_preds(args.preds, "ours"), args.source)
    print(f"Leak={res['_overall']:.3f}  " + " ".join(f"{g}={v}" for g, v in res.items() if g != "_overall"))


if __name__ == "__main__":
    main()
