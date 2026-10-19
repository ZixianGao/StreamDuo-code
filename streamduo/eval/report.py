"""Rule-based metrics for a predictions file, per game and overall: timing (Proact-VL-aligned)
and knowledge, plus K-strict entity recall: entities of GT utterances labeled K (needs game
knowledge, streamduo.data.kvs) that the prediction names anywhere in the clip. LLM-judged metrics are
computed separately (streamduo.eval.judge).

    python -m streamduo.eval.report --preds preds.jsonl --manifest test_clean.jsonl --out metrics.json
"""
import argparse
import collections
import json

from streamduo.eval.judge import load_preds, responses
from streamduo.eval.knowledge import aggregate as k_aggregate, clip_knowledge, other_games
from streamduo.eval.knowledge_core import matcher
from streamduo.eval.timing import aggregate as t_aggregate, clip_timing
from streamduo.paths import MANIFEST_ROOT


def load_k_texts():
    """key -> GT texts of utterances labeled K."""
    out = collections.defaultdict(list)
    path = MANIFEST_ROOT / "all" / "kvs_labels.jsonl"
    if path.exists():
        for u in map(json.loads, open(path)):
            if u.get("label") == "K":
                out[u["key"]].append(u["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="ours", choices=["ours", "released"],
                    help="released = oaaoaa/proactvl_results prediction files (Orig-setting baselines)")
    ap.add_argument("--word_budget", type=int, default=0,
                    help="knowledge metrics on the first N predicted words of each clip (time order); 0 = all. "
                         "Main table: 126 = 2x the human mean, so verbose models cannot win recall by volume")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest)]
    preds = load_preds(args.preds, args.format)
    if args.format == "released":  # same key space as ours: (game, video basename, int begin, None)
        preds = {(g, v, int(b), None): p for (g, v, b, _), p in preds.items()}
    t_by, k_by = collections.defaultdict(list), collections.defaultdict(list)
    ks_by = collections.defaultdict(lambda: [0, 0])   # K-strict: [hit, total] entity mentions
    k_texts = load_k_texts()
    for r in rows:
        g = r["metadata"]["dataset_name"]
        key = (g, r["video_path"].split("/")[-1], r["video_begin"], None)
        if key not in preds:
            continue
        p = preds[key]
        t_by[g].append(clip_timing(r, list(p)))
        gt = " ".join(a["text"] for a in r["annotations"] if a["role"] == "assistant" and a.get("text"))
        pred_text = " ".join(t for _, t in responses(p))
        if args.word_budget:
            pred_text = " ".join(pred_text.split()[:args.word_budget])
        k_by[g].append(clip_knowledge(g, gt, pred_text, other_games(g)))
        m = matcher(g)
        if m is not None:
            said = {e["norm"] for e in m.find(pred_text)}
            for t in k_texts.get(f"{r['video_path']}@{r['video_begin']}", []):
                for e in {e["norm"] for e in m.find(t)}:
                    ks_by[g][1] += 1
                    ks_by[g][0] += e in said
    out = {}
    for g in sorted(t_by):
        h, n = ks_by[g]
        out[g] = {**{k: v for k, v in t_aggregate(t_by[g]).items() if k in ("TimeDiff", "F1", "n_clips")},
                  **k_aggregate(k_by[g]), "Kstrict-R": round(h / n, 3) if n else None}
    allt = [c for cs in t_by.values() for c in cs]
    allk = [c for cs in k_by.values() for c in cs]
    H, N = sum(v[0] for v in ks_by.values()), sum(v[1] for v in ks_by.values())
    out["_overall"] = {**{k: v for k, v in t_aggregate(allt).items() if k in ("TimeDiff", "F1", "n_clips")},
                       **k_aggregate(allk), "Kstrict-R": round(H / N, 3) if N else None}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    for g, v in out.items():
        print(f"{g:22s} " + " ".join(f"{k}={v[k]:.3f}" if isinstance(v[k], float) else f"{k}={v[k]}" for k in v))


if __name__ == "__main__":
    main()
