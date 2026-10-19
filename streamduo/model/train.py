"""Supervised training of the commentator (Stage 0 and Stage 1), HF Trainer + DeepSpeed.

Stage 0 (renaming + dossier):
    torchrun --nproc_per_node 4 -m streamduo.model.train --fold S_bg3 \\
        --dossier_run v2_sweep_q48_k2048 --dossier_dropout 0.3 --knowledge_swap_p 0.5 \\
        --output_dir $STREAMDUO_DATA/ckpt/stage0 --deepspeed configs/ds_zero3.json ...
Stage 1 (when to ask, how to read answers), initialised from Stage 0:
    ... --base_model $STREAMDUO_DATA/ckpt/stage0 \\
        --lookup_file $STREAMDUO_DATA/manifests/S_bg3/train_answers.jsonl \\
        --ask_labels_file $STREAMDUO_DATA/manifests/S_bg3/ask_labels.jsonl --ask_min_gain 1.0
See scripts/2_stage0.sh and scripts/4_stage1.sh for the full argument lists.
"""
import glob
import json
import os
from dataclasses import dataclass

import torch
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from streamduo.model.proactvl import ASK_DELTA, ASK_FILE, HEAD_FILE, ProactCommentator
from streamduo.model.stream_data import StreamClipDataset, setup_processor
from streamduo.paths import DATA_ROOT, MANIFEST_ROOT


@dataclass
class StreamDuoArgs:
    fold: str = "S_bg3"
    base_model: str = str(DATA_ROOT / "hf" / "chenjoya__LiveCC-7B-Base")
    dossier_run: str = ""            # DATA_ROOT/dossiers/<run>/<game>.txt; empty = no dossier
    dossier_dropout: float = 0.0
    knowledge_swap_p: float = 0.0    # probability of consistent random entity renaming per sample
    max_rows: int = 0
    liger: bool = True               # fused linear cross-entropy (needed to fit ~20k-token clips)
    lookup_file: str = ""            # jsonl {key, lookups: {t: answer}}: in-stream research-agent answers
    ask_labels_file: str = ""        # jsonl {i, gains: {a: gain}} from streamduo.interact.ask_labels
    ask_min_gain: float = 1.0        # ask at a - ASK_DELTA iff the ideal answer raises log p(turn) by > gamma


def load_dossiers(run):
    if not run:
        return {}, {}
    d = DATA_ROOT / "dossiers" / run
    texts = {p.stem: p.read_text() for p in d.glob("*.txt")}
    jsons = {g: json.loads((d / f"{g}.json").read_text()) for g in texts if (d / f"{g}.json").exists()}
    return texts, jsons


def make_swap(p, dossier_jsons):
    """Per-sample transform: consistent random renaming of the source game's entities."""
    if p <= 0:
        return None
    from streamduo.data.games import GAMES
    from streamduo.model.knowledge_swap import KnowledgeSwap
    ks = KnowledgeSwap(p=p)

    def transform(row, dossier, rng):
        g = row["metadata"]["dataset_name"]
        return ks(row, dossier, rng, dossier_json=dossier_jsons.get(g, {}),
                  game_title=GAMES[g].title if g in GAMES else "")
    return transform


class StreamDuoTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(**inputs)
        if self.state.global_step % max(self.args.logging_steps, 1) == 0 and out.stats:
            self.log({f"loss_{k}": float(v) for k, v in out.stats.items()})
        return (out.loss, out) if return_outputs else out.loss

    def _save(self, output_dir=None, state_dict=None):
        """Under ZeRO-3 the Trainer passes the gathered full state_dict (keys 'base.*', 'head.*', 'ask_head.*')."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        part = lambda pre: {k[len(pre):]: v.detach().clone().cpu() for k, v in state_dict.items() if k.startswith(pre)}  # noqa: E731
        self.model.base.save_pretrained(output_dir, state_dict={k[5:]: v for k, v in state_dict.items()
                                                                if k.startswith("base.")}, safe_serialization=True)
        torch.save(part("head."), os.path.join(output_dir, HEAD_FILE))
        ask_sd = part("ask_head.")
        if ask_sd:
            torch.save(ask_sd, os.path.join(output_dir, ASK_FILE))
        self.processing_class.save_pretrained(output_dir)


def main():
    args, targs = HfArgumentParser((StreamDuoArgs, TrainingArguments)).parse_args_into_dataclasses()
    targs.remove_unused_columns = False
    recipe = json.loads((MANIFEST_ROOT / args.fold / "train_recipe.json").read_text())
    files = [f for f in recipe["files"] if "livecc" not in f and "ego4d" not in f]

    processor = AutoProcessor.from_pretrained(args.base_model, padding_side="right")
    setup_processor(processor)
    dossiers, dossier_jsons = load_dossiers(args.dossier_run)
    ds = StreamClipDataset(files, processor, recipe["video_root"], dossiers=dossiers,
                           dossier_dropout=args.dossier_dropout, max_rows=args.max_rows or None, seed=targs.seed,
                           transform=make_swap(args.knowledge_swap_p, dossier_jsons))
    if args.lookup_file:
        ds.lookups = {d["key"]: {int(t): a for t, a in d["lookups"].items()} for d in map(json.loads, open(args.lookup_file))}
    if args.ask_labels_file:
        # keyed by clip, not by sample index (the labels are computed on a shuffled subset of the clips)
        ds.ask_override = {d["key"]: {int(a) - ASK_DELTA for a, g in d["gains"].items() if g > args.ask_min_gain}
                           for d in map(json.loads, open(args.ask_labels_file))}
    if args.liger:
        # fused linear + CE: never materialises the (seq ~20k x vocab 151k) logits
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
        apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=True, rms_norm=True, swiglu=True, rope=False)
    model = ProactCommentator.from_pretrained(args.base_model, n_vocab=len(processor.tokenizer),
                                              torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    if targs.gradient_checkpointing:
        model.base.config.use_cache = False
    trainer = StreamDuoTrainer(model=model, args=targs, train_dataset=ds, data_collator=ds.collate,
                               processing_class=processor)
    ckpts = sorted(glob.glob(os.path.join(targs.output_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
    trainer.train(resume_from_checkpoint=ckpts[-1] if ckpts else None)
    trainer.save_model(targs.output_dir)


if __name__ == "__main__":
    main()
