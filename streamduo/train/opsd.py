"""Stage 2: on-policy self-distillation (OPSD) from a privileged copy of the commentator.

For one second t right after a research-agent answer arrives, the student (Stage 1 + LoRA) samples its own
lines while seeing the NOISY answer it would get at test time (distractors, sometimes no right entry). The
teacher is the same model with the adapter disabled (= Stage 1) that sees the IDEAL answer (always contains
the entity the human names next), with the effect of that answer amplified by context-aware decoding:
    log p_T(.) = log_softmax((1 + alpha) * l(. | ideal answer) - alpha * l(. | no answer))
Loss = reverse KL(p_S || p_T) over the student's own tokens. Random entity renaming (p = 0.5) is applied
consistently to the clip, the dossier and both answers.

    torchrun --nproc_per_node 4 -m streamduo.train.opsd --init $STREAMDUO_DATA/ckpt/stage1 --out $STREAMDUO_DATA/ckpt/stage2
    python -m streamduo.train.common --init $STREAMDUO_DATA/ckpt/stage1 --out $STREAMDUO_DATA/ckpt/stage2   # merge
"""
import argparse
import json
import os
import random
import re
import time

import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from streamduo.data.games import GAMES
from streamduo.model.stream_data import setup_processor
from streamduo.model.train import load_dossiers
from streamduo.paths import MANIFEST_ROOT
from streamduo.train.common import build_prompt, decode, repeat_inputs


def cont_logits(model, enc, seqs, n_prompt, eos):
    """Logits [K, L, V] predicting each continuation token, and the mask of real (non-padding) tokens."""
    k = seqs.shape[0]
    inp = repeat_inputs(enc, k)
    cont = seqs[:, n_prompt:]
    mask = torch.ones_like(cont, dtype=torch.bool)
    for i in range(k):
        stop = (cont[i] == eos).nonzero()
        if len(stop):
            mask[i, stop[0, 0] + 1:] = False
    attn = torch.cat([inp["attention_mask"], mask.long()], dim=1)
    out = model(input_ids=seqs, attention_mask=attn, pixel_values_videos=inp["pixel_values_videos"],
                video_grid_thw=inp["video_grid_thw"], use_cache=False)
    return out.logits[:, n_prompt - 1:-1].float(), mask


def with_prompt(seqs, n_prompt, enc):
    """Same continuations after a different prompt."""
    return torch.cat([enc["input_ids"].repeat(seqs.shape[0], 1), seqs[:, n_prompt:]], dim=1)


def rename_text(text, mapping):
    if not mapping or not text:
        return text
    pat = re.compile("|".join(rf"(?<!\w){re.escape(n)}(?!\w)" for n in sorted(mapping, key=len, reverse=True)), re.I)
    low = {k.lower(): v for k, v in mapping.items()}
    return pat.sub(lambda m: low[m.group(0).lower()], text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", default="S_bg3")
    ap.add_argument("--dossier_run", default="v2_sweep_q48_k2048")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--alpha", type=float, default=1.0, help="CAD amplification of the teacher (0 = plain privileged)")
    ap.add_argument("--swap_p", type=float, default=0.5)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noisy_file", default="train_answers.jsonl", help="student's answers (with misses)")
    ap.add_argument("--ideal_file", default="train_answers_full.jsonl", help="teacher's answers (always right)")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from streamduo.model.knowledge_swap import KnowledgeSwap

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dev = torch.device("cuda")
    rng = random.Random(args.seed * 1000 + rank)

    processor = AutoProcessor.from_pretrained(args.init, padding_side="right")
    setup_processor(processor)
    tok = processor.tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        args.init, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(dev)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        target_modules=r".*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    recipe = json.loads((MANIFEST_ROOT / args.fold / "train_recipe.json").read_text())
    games = [g for g in recipe["train_games"] if g in GAMES]
    rows = [json.loads(l) for g in games for l in open(MANIFEST_ROOT / args.fold / f"train_{g}.jsonl")]
    load = lambda f: {d["key"]: {int(t): a for t, a in d["lookups"].items()}  # noqa: E731
                      for d in map(json.loads, open(MANIFEST_ROOT / args.fold / f))}
    noisy, ideal = load(args.noisy_file), load(args.ideal_file)
    dossiers, dossier_jsons = load_dossiers(args.dossier_run)
    swap = KnowledgeSwap(p=1.0)
    os.makedirs(args.out, exist_ok=True)
    adapter_dir = os.path.join(args.out, "adapter")
    log = open(os.path.join(args.out, f"log_rank{rank}.jsonl"), "a")

    def sample_prompt():
        """A training clip and the second right after one of its ideal answers arrives."""
        while True:
            row = rows[rng.randrange(len(rows))]
            key = f"{row['video_path']}@{row['video_begin']}"
            if key not in ideal:
                continue
            t = rng.choice(sorted(ideal[key])) + 1
            if t < row["video_end"]:
                return row, t, key

    for step in range(args.steps):
        t0 = time.time()
        row, t, key = sample_prompt()
        g = row["metadata"]["dataset_name"]
        d_game = dossiers.get(g)
        lk_s, lk_t = dict(noisy.get(key, {})), dict(ideal.get(key, {}))
        if rng.random() < args.swap_p:
            row, d_game = swap(row, d_game, rng, dossier_json=dossier_jsons.get(g, {}), game_title=GAMES[g].title)
            lk_s = {k: rename_text(v, row.get("_swap")) for k, v in lk_s.items()}
            lk_t = {k: rename_text(v, row.get("_swap")) for k, v in lk_t.items()}
        opt.zero_grad(set_to_none=True)
        stats = {}
        try:
            vr = recipe["video_root"]   # same frames and dossier everywhere; only the in-stream answer differs
            e_s = {k: v.to(dev) for k, v in build_prompt(row, t, d_game, processor, args.window, vr, lk_s).items()}
            e_o = {k: v.to(dev) for k, v in build_prompt(row, t, d_game, processor, args.window, vr, lk_t).items()}
            e_n = {k: v.to(dev) for k, v in build_prompt(row, t, d_game, processor, args.window, vr, {}).items()}
        except Exception as e:  # corrupt video etc.
            print(f"[opd] prompt failed ({e!r})", flush=True)
            e_s = None
        if e_s is not None:
            n_s = e_s["input_ids"].shape[1]
            model.eval()
            with torch.no_grad():
                seqs = model.generate(**repeat_inputs(e_s, args.group), do_sample=True, temperature=1.0, top_p=1.0,
                                      top_k=0, repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                                      eos_token_id=im_end, pad_token_id=im_end)
                with model.disable_adapter():   # teacher = Stage 1 (adapter off), ideal answer, amplified
                    lo, _ = cont_logits(model, e_o, with_prompt(seqs, n_s, e_o), e_o["input_ids"].shape[1], im_end)
                    ln, _ = cont_logits(model, e_n, with_prompt(seqs, n_s, e_n), e_n["input_ids"].shape[1], im_end)
                logp_t = torch.log_softmax((1 + args.alpha) * lo - args.alpha * ln, -1)
                del lo, ln
            model.train()
            ls, mask = cont_logits(model, e_s, seqs, n_s, im_end)
            logp_s = torch.log_softmax(ls, -1)
            rkl = (logp_s.exp() * (logp_s - logp_t)).sum(-1)          # reverse KL per token
            m = mask.float()
            loss = (rkl * m).sum() / m.sum().clamp(min=1)
            loss.backward()
            preds = decode(tok, seqs, n_s)
            stats = {"rkl": float(loss), "len": sum(len(p.split()) for p in preds) / len(preds),
                     "distinct": len(set(preds)) / len(preds), "example": preds[0][:120]}
            del seqs, ls, logp_s, logp_t, rkl, loss
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        rec = {"step": step + 1, "t": t, "swapped": bool(row.get("_swap")), "sec": round(time.time() - t0, 1), **stats}
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        if rank == 0 and (step + 1) % 5 == 0:
            print(json.dumps(rec, ensure_ascii=False), flush=True)
        if rank == 0 and ((step + 1) % args.save_every == 0 or step + 1 == args.steps):
            model.save_pretrained(adapter_dir)
            open(os.path.join(args.out, "rl_state.json"), "w").write(json.dumps({"step": step + 1, "args": vars(args)}))
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
