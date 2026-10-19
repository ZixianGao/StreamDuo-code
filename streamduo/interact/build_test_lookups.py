"""Cache of the frozen visual research agent's answers for EVERY second of the test clips.

The commentator decides itself when to ask (its learned ask head). Because the research agent is frozen
and decodes greedily, its answer to the frames of second t is deterministic, so it is computed once for
every second and looked up when the commentator asks at t (answer arrives at t + delta in
streamduo.model.stream_infer). Identical to calling the deployed agent online; online deployment is only
used to measure latency.

    python -m streamduo.interact.build_test_lookups candidates           # wiki entries per game (once)
    python -m streamduo.interact.build_test_lookups frames --every 1     # train env (decord)
    python -m streamduo.interact.build_test_lookups agent --vlm VLM      # vllm env, one call per shard
    python -m streamduo.interact.build_test_lookups merge
-> manifests/<fold>/test_agent_cache_e{every}.jsonl   {key, answers: {request second t: answer}}
"""
import argparse
import json
import re
from multiprocessing import Pool

from streamduo.paths import DATA_ROOT, MANIFEST_ROOT

WORK = DATA_ROOT / "interact" / "test_lookups"
PROPOSE = ("These frames are from a live {game} video. A commentator does not recognise what is shown. "
           "List up to 5 names or search keywords that would identify the main characters, enemies, items, "
           "places or abilities on screen; include any readable on-screen text (names, titles, labels). "
           "One per line, no explanations.")
PICK = ("These frames are from a live {game} video. Which of these wiki entries are actually shown or "
        "referred to on screen?\n{cands}\nReply with up to 3 candidate numbers, most likely first, "
        "separated by commas, or 0 if none. Numbers only.")


def _dump(job):
    from PIL import Image

    from streamduo.model.stream_data import read_clip
    row, every, out_dir, idx = job
    begin, end = row["video_begin"], row["video_end"]
    times = list(range(begin, end - 1, every))
    if not times:
        return []
    try:
        clip = read_clip(row["video_abspath"], begin, end)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for t in times:
        paths = []
        for k in range(2):
            fr = clip[(t - begin) * 2 + k]
            im = Image.fromarray(fr.permute(1, 2, 0).clamp(0, 255).byte().numpy())
            im.thumbnail((640, 640))
            p = out_dir / f"c{idx:05d}_t{t}_{k}.jpg"
            im.save(p, quality=90)
            paths.append(str(p))
        out.append({"key": f"{row['video_path']}@{begin}", "game": row["metadata"]["dataset_name"], "t": t,
                    "frames": paths})
    return out


def candidates(args):
    """Every non-generic wiki entity of every game as "name (type): lead", the research agent's search index."""
    from streamduo.data.games import GAMES
    from streamduo.research.tools import WikiIndex
    from streamduo.wiki.ontology import EntityMatcher
    norm = lambda n: re.sub(r"[^a-z0-9]+", " ", re.sub(r"\s*\([^)]*\)$", "", n.lower())).strip()  # noqa: E731
    cands = {}
    for g in sorted(GAMES):
        try:
            m, w = EntityMatcher.for_game(g), WikiIndex(g)
        except FileNotFoundError:
            continue
        rows = []
        for e in m.lex.values():
            if e.get("generic"):
                continue
            page = w.by_title.get(e["name"].lower())
            lead = re.sub(r"\s+", " ", page["text"][:300]).strip() if page else ""
            rows.append({"name": e["name"], "norm": norm(e["name"]), "text": f"{e['name']} ({e['type'].lower()}): {lead}"})
        cands[g] = rows
        print(g, len(rows), "candidates")
    out = DATA_ROOT / "interact" / "candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cands, ensure_ascii=False))


def frames(args):
    rows = [json.loads(l) for l in open(MANIFEST_ROOT / args.fold / f"test_{args.setting}.jsonl")]
    seen, uniq = set(), []
    for r in rows:  # co-commentary clips appear once per target speaker: same frames
        k = f"{r['video_path']}@{r['video_begin']}"
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    fdir = WORK / f"frames_e{args.every}"
    fdir.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool, open(WORK / f"moments_e{args.every}.jsonl", "w") as f:
        n = 0
        for res in pool.imap_unordered(_dump, [(r, args.every, fdir, i) for i, r in enumerate(uniq)], chunksize=2):
            for m in res:
                f.write(json.dumps(m) + "\n")
                n += 1
    print(f"{len(uniq)} clips, {n} lookup moments")


def agent(args):
    """Sharded, chunked, resumable: each chunk of moments goes through propose -> wiki search -> pick and is
    appended to this shard's pick file at once (images are only decoded chunk by chunk)."""
    from PIL import Image
    from vllm import LLM, SamplingParams

    from streamduo.interact.bm25 import BM25Okapi
    src = WORK / f"moments_e{args.every}.jsonl"
    moms = [json.loads(l) for l in open(src)][args.shard::args.num_shards]
    from pathlib import Path
    out_path = WORK / f"picks_e{args.every}.shard{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        done = {(d["key"], d["t"]) for d in map(json.loads, open(out_path))}
    moms = [m for m in moms if (m["key"], m["t"]) not in done]
    cands = json.loads((DATA_ROOT / "interact" / "candidates.json").read_text())
    bm = {g: BM25Okapi([re.findall(r"[a-z0-9]+", c["text"].lower()) for c in rows]) for g, rows in cands.items()}
    llm = LLM(model=args.vlm, max_model_len=8192, gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 2},
              mm_processor_cache_gb=0, tensor_parallel_size=args.tp)
    sp = SamplingParams(temperature=0.0, max_tokens=60)
    title = lambda g: g.replace("_", " ")  # noqa: E731
    search_cache = {}

    def search(g, kw):
        k = (g, kw.lower())
        if k not in search_cache:
            search_cache[k] = [j for j, _ in bm[g].top_k(re.findall(r"[a-z0-9]+", kw.lower()), 3)]
        return search_cache[k]

    def batch(prompts):
        msgs = [[{"role": "user", "content": [*({"type": "image_pil", "image_pil": Image.open(p).convert("RGB")}
                                               for p in m["frames"]), {"type": "text", "text": text}]}]
                for m, text in prompts]
        return [o.outputs[0].text for o in llm.chat(msgs, sp, use_tqdm=False)]

    with open(out_path, "a") as f:
        for c0 in range(0, len(moms), args.chunk):
            chunk = moms[c0:c0 + args.chunk]
            out1 = batch([(m, PROPOSE.format(game=title(m["game"]))) for m in chunk])
            for m, o in zip(chunk, out1):
                kws = [k.strip("-•* 0123456789.").strip() for k in o.splitlines()]
                m["keywords"] = [k for k in kws if k][:5]
                pool = []
                for k in m["keywords"]:
                    for j in search(m["game"], k):
                        if j not in pool:
                            pool.append(j)
                m["pool"] = pool[:15]
            todo = [m for m in chunk if m["pool"]]
            out2 = batch([(m, PICK.format(game=title(m["game"]), cands="\n".join(
                f"{n + 1}. {cands[m['game']][j]['text'][:160]}" for n, j in enumerate(m["pool"])))) for m in todo])
            for m, o in zip(todo, out2):
                nums = [int(x) - 1 for x in re.findall(r"\d+", o)]
                m["picked"] = [m["pool"][n] for n in nums if 0 <= n < len(m["pool"])][:3]
            for m in chunk:
                f.write(json.dumps({k: m.get(k) for k in ("key", "t", "game", "keywords", "pool", "picked")}) + "\n")
            f.flush()
            print(f"[agent] shard {args.shard}: {c0 + len(chunk)}/{len(moms)}", flush=True)


def merge(args):
    """All shards' picks -> the answer cache keyed by REQUEST second."""
    cands = json.loads((DATA_ROOT / "interact" / "candidates.json").read_text())
    by_key, n, n_ans = {}, 0, 0
    for p in sorted(WORK.glob(f"picks_e{args.every}.shard*.jsonl")):
        for m in map(json.loads, open(p)):
            n += 1
            if not m.get("picked"):
                continue
            rows = cands[m["game"]]
            lines = [f"- {rows[j]['text'][:220]}" for j in m["picked"]]  # "name (type): lead", as in training
            by_key.setdefault(m["key"], {})[str(m["t"])] = "Entities likely on screen now:\n" + "\n".join(lines)
            n_ans += 1
    out = MANIFEST_ROOT / args.fold / f"test_agent_cache_e{args.every}.jsonl"
    with open(out, "w") as f:
        for k, ans in by_key.items():
            f.write(json.dumps({"key": k, "answers": ans}, ensure_ascii=False) + "\n")
    print(f"{n} moments, {n_ans} answers, {len(by_key)} clips -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["candidates", "frames", "agent", "merge"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size of the research agent VLM")
    ap.add_argument("--fold", default="S_bg3")
    ap.add_argument("--setting", default="clean")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--delta", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--vlm", default="")
    ap.add_argument("--work", default="", help="working directory (default: interact/test_lookups)")
    args = ap.parse_args()
    if args.work:   # keep dev / test frames and picks apart
        global WORK
        from pathlib import Path
        WORK = Path(args.work)
        WORK.mkdir(parents=True, exist_ok=True)
    {"candidates": candidates, "frames": frames, "agent": agent, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
