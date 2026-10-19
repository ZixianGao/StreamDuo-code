"""Model-aware ask labels: ask only where the commentator does NOT know and an answer WOULD help.

For each training sample (built exactly as streamduo.model.train builds it: same dossier dropout, same
Knowledge-Swap draw, same task template, keyed by sample index), a frozen commentator (the SFT init)
scores the human commentary twice, teacher-forced:
    with   the ideal answer (always contains the entity) injected at its arrival second a
    without any answer
For the turn at a + 1 (where the human names the entity), the value of asking is
    gain(a) = sum_tokens [log p(turn | with) - log p(turn | without)]
Label: ask at a - ASK_DELTA iff gain(a) > --min_gain (the model did not already know, and help helps).
Under Knowledge-Swap the names are pseudo names the model cannot know -> high gain -> ask; real BG3
names it has memorised -> low gain -> no ask, which is what transfers to unseen games.

    torchrun --nproc_per_node 4 -m streamduo.interact.ask_labels --init $STREAMDUO_DATA/ckpt/stage0 --max_rows 3400
-> manifests/<fold>/ask_labels.jsonl  {i, key, gains: {a: gain}}   (thresholded at training time, matched by key)
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

from streamduo.paths import DATA_ROOT, MANIFEST_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--fold", default="S_bg3")
    ap.add_argument("--max_rows", type=int, default=3400)
    ap.add_argument("--dossier_run", default="v2_sweep_q48_k2048")
    ap.add_argument("--dossier_dropout", type=float, default=0.3)
    ap.add_argument("--knowledge_swap_p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoProcessor

    from streamduo.model.proactvl import ProactCommentator
    from streamduo.model.stream_data import StreamClipDataset, setup_processor
    from streamduo.model.train import load_dossiers, make_swap

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    recipe = json.loads((MANIFEST_ROOT / args.fold / "train_recipe.json").read_text())
    files = [f for f in recipe["files"] if "livecc" not in f and "ego4d" not in f]
    processor = AutoProcessor.from_pretrained(args.init, padding_side="right")
    setup_processor(processor)
    dossiers, dossier_jsons = load_dossiers(args.dossier_run)
    ds = StreamClipDataset(files, processor, recipe["video_root"], dossiers=dossiers,
                           dossier_dropout=args.dossier_dropout, max_rows=args.max_rows, seed=args.seed,
                           transform=make_swap(args.knowledge_swap_p, dossier_jsons))
    full = {d["key"]: {int(t): a for t, a in d["lookups"].items()}
            for d in map(json.loads, open(MANIFEST_ROOT / args.fold / "train_answers_full.jsonl"))}
    # liger's fused CE (training mode + labels) never materialises the (seq x vocab) logits; Qwen2-VL has no
    # dropout, and everything below runs under no_grad, so train() changes nothing else.
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
    apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=True, rms_norm=True, swiglu=True, rope=False)
    model = ProactCommentator.from_pretrained(args.init, n_vocab=len(processor.tokenizer), torch_dtype=torch.bfloat16,
                                              attn_implementation="flash_attention_2").to("cuda").train()
    tok = processor.tokenizer
    im_start, im_end = tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
    head = tok("assistant\n", add_special_tokens=False).input_ids

    def turn_spans(ids):
        ids = ids.tolist()
        out = []
        for i, x in enumerate(ids):
            if x == im_start and ids[i + 1:i + 1 + len(head)] == head:
                s = i + 1 + len(head)
                out.append((s, ids.index(im_end, s)))
        return out

    @torch.no_grad()
    def turn_logps(enc):
        enc = {k: v.to("cuda") for k, v in enc.items() if k in ("input_ids", "attention_mask", "pixel_values_videos",
                                                               "video_grid_thw")}
        model.base(**enc, use_cache=False, labels=enc["input_ids"])  # labels -> fused path, no full logits
        spans = turn_spans(enc["input_ids"][0])
        pos = torch.cat([torch.arange(s, e, device="cuda") for s, e in spans])
        lp = model.token_logprobs(enc["input_ids"], pos).cpu()
        out, k = [], 0
        for s, e in spans:
            out.append(float(lp[k:k + e - s].sum()))
            k += e - s
        return out  # one log-prob per second (assistant turn)

    out_path = MANIFEST_ROOT / args.fold / f"ask_labels.rank{rank}.jsonl"
    import random
    with open(out_path, "w") as f:
        for i in range(rank, len(ds), world):
            row = ds.rows[i]
            key = f"{row['video_path']}@{row['video_begin']}"
            if key not in full:
                continue
            try:
                res = {}
                for cond in ("with", "without"):
                    ds.lookups = {key: full[key]} if cond == "with" else {}
                    rng = random.Random(ds.seed * 1_000_003 + i)   # same draw as StreamClipDataset.__getitem__
                    text, videos, flags, _ = ds.build(row, rng)
                    res[cond] = turn_logps(ds.encode(text, videos, flags))
                begin = row["video_begin"]
                gains = {}
                for a in full[key]:
                    j = a + 1 - begin   # the turn where the human names the entity
                    if 0 <= j < len(res["with"]):
                        gains[str(a)] = round(res["with"][j] - res["without"][j], 4)
                f.write(json.dumps({"i": i, "key": key, "gains": gains}) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001
                print(f"[ask_labels] {i} failed: {e!r}", flush=True)
    dist.barrier()
    if rank == 0:
        rows = []
        for r in range(world):
            rows += [json.loads(l) for l in open(MANIFEST_ROOT / args.fold / f"ask_labels.rank{r}.jsonl")]
        with open(MANIFEST_ROOT / args.fold / "ask_labels.jsonl", "w") as f:
            for x in sorted(rows, key=lambda x: x["i"]):
                f.write(json.dumps(x) + "\n")
        g = [v for x in rows for v in x["gains"].values()]
        g.sort()
        q = lambda p: g[int(p * (len(g) - 1))] if g else None  # noqa: E731
        print(f"[ask_labels] {len(rows)} samples, {len(g)} candidate moments; gain quantiles "
              f"10% {q(.1)} 25% {q(.25)} 50% {q(.5)} 75% {q(.75)} 90% {q(.9)}; >1 nat: {sum(v > 1 for v in g)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
