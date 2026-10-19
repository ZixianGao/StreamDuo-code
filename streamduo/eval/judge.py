"""LLM-judged text metrics (Proact-VL appendix C): PAUC, CC win rate, LiveU, FinalQ.

* PAUC  -- Proact-VL's judge prompt (verbatim, streamduo.eval.pauc); responses = runs of consecutive
           spoken seconds; the judge sees all responses so far (cumulative), as in their release.
* CC    -- LiveCC's A/B prompt (verbatim), judged in both orders against Gemini-2.5-Pro captions
           released by Proact-VL; win rate = model wins / (2 x clips).
* LiveU -- mean of Time, Rate, TextU (1-10); FinalQ -- mean of Fidelity, Continuity, Substance.
           Proact-VL released only the scores, not these prompts: the prompts below are ours,
           written from the dimension definitions in their appendix C.2.

The judge model is whatever streamduo.llm.LLM points to (dev: local Qwen3-30B; final: GPT-5.1).

    python -m streamduo.eval.judge --preds preds.jsonl --manifest test_clean.jsonl --out judged.jsonl \\
        [--metrics pauc,cc,liveu,finalq] [--format ours|released]
"""
import argparse
import json
import re
import statistics as st

from streamduo.data.games import GAMES
from streamduo.eval.pauc import JUDGE_SYSTEM, judge_user_message, pauc_turn, raw_to_score
from streamduo.eval.timing import gt_seconds
from streamduo.llm import LLM
from streamduo.paths import DATA_ROOT

GEMINI_CAPTIONS = DATA_ROOT / "hf" / "oaaoaa__proactvl_results" / "results" / "baseline" / "captions" / "gemini-2.5-pro.jsonl"

CC_PROMPT = (
    'You are an expert in video commentary. '
    'Your task is to review two commentaries (Commentary A and Commentary B), and select the one that better '
    'aligns with the human commentary. '
    'You should consider the criteria:\n'
    '1. Semantic Alignment: The commentary should convey the same meaning, details, and key points as the human commentary.\n'
    'If the above criteria is not enough to judge, then consider:\n'
    '2. Stylistic Consistency: The commentary should maintain a tone, word choice, and structure similar to the human commentary.\n'
    '\n---Commentary A---\n{a}\n----------\n'
    '\n---Commentary B---\n{b}\n----------\n'
    '\n---Human Commentary---\n{gt}\n----------\n'
    '\nYour response should be "Commentary A is better aligned with the human commentary" or '
    '"Commentary B is better aligned with the human commentary".\n'
)

LIVEU_PROMPT = """You are evaluating a LIVE {mode} system for the game "{game}" on a {dur}-second clip.
Each line is "[second] text" for a second in which the system spoke; unlisted seconds were silent.
The human commentator's timeline is given as a reference for when speaking was appropriate.

Human reference timeline:
{gt}

System timeline:
{pred}

Score the SYSTEM from 1 (very poor) to 10 (excellent) on:
- Time: does it speak at the right moments (covers the salient moments, stays quiet otherwise{overlap})?
- Rate: is the pacing listenable in real time (brief bursts, no dumping long monologues{guide})?
- TextU: is each spoken unit immediately clear and natural as live speech?
Return JSON: {{"Time": int, "Rate": int, "TextU": int}}"""

FINALQ_PROMPT = """You are evaluating the full script of a {mode} system for the game "{game}" \
(its spoken lines concatenated in order; silences removed).
Context given to the system before the clip: {context}

Human reference script:
{gt}

System script:
{pred}

Score the SYSTEM script from 1 (very poor) to 10 (excellent) on:
- Fidelity: avoids contradicting the events (use the human script as evidence) and avoids misleading concrete claims.
- Continuity: coherent with the context and with itself as one thread.
- Substance: useful, readable, non-redundant commentary.
Return JSON: {{"Fidelity": int, "Continuity": int, "Substance": int}}"""

MODE = {"Solo commentators": "solo commentary", "Multiple commentators": "co-commentary",
        "Guidance": "game guidance"}


def clean_text(t):
    return re.sub(r"\s*\.\.\.\s*$", "", t.strip())


def responses(pred_by_sec):
    """Merge runs of consecutive spoken seconds into responses -> [(start_sec, text)]."""
    out = []
    for t in sorted(pred_by_sec):
        txt = clean_text(pred_by_sec[t])
        if not txt:
            continue
        if out and t == out[-1][2] + 1:
            out[-1] = (out[-1][0], out[-1][1] + " " + txt, t)
        else:
            out.append((t, txt, t))
    return [(s, txt) for s, txt, _ in out]


def gt_text(row):
    return " ".join(a["text"] for a in sorted(row["annotations"], key=lambda a: a["start"])
                    if a["role"] == "assistant" and a.get("text"))


def timeline(by_sec):
    return "\n".join(f"[{t}] {clean_text(x)}" for t, x in sorted(by_sec.items()) if clean_text(x)) or "(silent)"


def judge_pauc(llm, row, preds):
    resp = responses(preds)
    gold = gt_text(row)
    judged, cum = [], []
    for t, txt in resp:
        cum.append(txt)
        raw = llm.chat([{"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": judge_user_message(gold, " ".join(cum))}], max_tokens=4)
        judged.append((float(t), raw_to_score(raw)))
    return pauc_turn(row["video_begin"], row["video_end"], judged, omega=0.5, start=0.0)


def judge_cc(llm, row, preds, baseline):
    if baseline is None:
        return None
    ours = " ".join(txt for _, txt in responses(preds))
    gt = gt_text(row)
    wins = 0.0
    for a, b, ours_is in ((ours, baseline, "A"), (baseline, ours, "B")):
        out = llm.chat([{"role": "user", "content": CC_PROMPT.format(a=a, b=b, gt=gt)}], max_tokens=30)
        if f"Commentary {ours_is}" in out:
            wins += 1
    return wins / 2


def _chat_json_safe(llm, messages):
    """The 60-token budget can cut a verbose judge reply mid-JSON (seen on near-silent models): retry once with a
    larger budget, else count the clip as unjudged. Temperature 0, so replies that parsed before are unchanged."""
    for max_tokens in (60, 300):
        try:
            return llm.chat_json(messages, max_tokens=max_tokens)
        except ValueError:   # json.JSONDecodeError
            continue
    return {}


def _scores(out, keys):
    vals = [out.get(k) for k in keys]
    if not all(isinstance(v, (int, float)) for v in vals):
        return None
    return {k: float(v) for k, v in zip(keys, vals)}


def judge_liveu(llm, row, preds):
    tag = row["metadata"].get("tag", "")
    gt_by_sec = {a["start"]: a["text"] for a in row["annotations"] if a["role"] == "assistant" and a.get("text")}
    p = LIVEU_PROMPT.format(
        mode=MODE.get(tag, "commentary"), game=GAMES[row["metadata"]["dataset_name"]].title,
        dur=row["video_end"] - row["video_begin"], gt=timeline(gt_by_sec), pred=timeline(preds),
        overlap=", and does not talk over the co-commentator" if tag == "Multiple commentators" else "",
        guide="; step-by-step in guidance" if tag == "Guidance" else "")
    return _scores(_chat_json_safe(llm, [{"role": "user", "content": p}]), ["Time", "Rate", "TextU"])


def judge_finalq(llm, row, preds):
    tag = row["metadata"].get("tag", "")
    p = FINALQ_PROMPT.format(
        mode=MODE.get(tag, "commentary"), game=GAMES[row["metadata"]["dataset_name"]].title,
        context=(row.get("history") or "(none)")[-800:], gt=gt_text(row) or "(silent)",
        pred=" ".join(txt for _, txt in responses(preds)) or "(silent)")
    return _scores(_chat_json_safe(llm, [{"role": "user", "content": p}]), ["Fidelity", "Continuity", "Substance"])


def load_preds(path, fmt):
    """-> {(dataset_name, video_path_basename, begin): {sec: text}}"""
    out = {}
    for d in map(json.loads, open(path)):
        if fmt == "released":
            out[(d["dataset_name"], d["video_id"], d["begin"], d.get("idx"))] = {int(t): x for t, x in d["pred"].items()}
        else:
            vp, b = d["key"].rsplit("@", 1)
            out[(d["game"], vp.split("/")[-1], int(b), None)] = {p["t"]: p["text"] for p in d["preds"] if p["text"]}
    return out


def load_gemini():
    return {(d["video_id"], d["begin"]): d["pred"] for d in map(json.loads, open(GEMINI_CAPTIONS))}


def run(rows, preds, metrics, workers=48, llm=None):
    llm = llm or LLM()
    gem = load_gemini() if "cc" in metrics else {}

    def one(llm_, item):
        row, pr = item
        res = {}
        if "pauc" in metrics:
            res["pauc"] = judge_pauc(llm_, row, pr)
        if "cc" in metrics:
            res["cc"] = judge_cc(llm_, row, pr, gem.get((row["video_path"].split("/")[-1], row["video_begin"])))
        if "liveu" in metrics:
            res["liveu"] = judge_liveu(llm_, row, pr)
        if "finalq" in metrics:
            res["finalq"] = judge_finalq(llm_, row, pr)
        return res

    return llm.map(one, list(zip(rows, preds)), workers, "judge")


def summarize(results):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(st.mean(xs), 4) if xs else None
    s = {"n": len(results)}
    if any("pauc" in r for r in results):
        s["PAUC"] = mean(r.get("pauc") for r in results)
    if any("cc" in r for r in results):
        s["CC"] = mean(r.get("cc") for r in results)
    if any("liveu" in r for r in results):
        s["LiveU"] = mean(st.mean(r["liveu"].values()) if r.get("liveu") else None for r in results)
    if any("finalq" in r for r in results):
        s["FinalQ"] = mean(st.mean(r["finalq"].values()) if r.get("finalq") else None for r in results)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="ours", choices=["ours", "released"])
    ap.add_argument("--metrics", default="pauc,cc,liveu,finalq")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    metrics = set(args.metrics.split(","))
    rows = [json.loads(l) for l in open(args.manifest)]
    preds = load_preds(args.preds, args.format)
    pairs = []
    for r in rows:
        key = (r.get("dataset_name") or r["metadata"]["dataset_name"], r["video_path"].split("/")[-1],
               r["video_begin"], r.get("idx") if args.format == "released" else None)
        if key in preds:
            pairs.append((r, preds[key]))
    if args.limit:
        pairs = pairs[:args.limit]
    results = run([r for r, _ in pairs], [p for _, p in pairs], metrics)
    with open(args.out, "w") as f:
        for (r, _), res in zip(pairs, results):
            f.write(json.dumps({"key": f"{r['video_path']}@{r['video_begin']}", "game": r["metadata"]["dataset_name"],
                                "tag": r["metadata"].get("tag"), **res}) + "\n")
    print(json.dumps(summarize(results)))


if __name__ == "__main__":
    main()
