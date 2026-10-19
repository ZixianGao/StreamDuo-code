"""Helpers shared by Stage 2: the streaming prefix of one second, batching, decoding, LoRA merge."""
import argparse
import os
import random
import shutil

import torch
from transformers import Qwen2VLForConditionalGeneration

from streamduo.model.prompts import SILENCE, system_prompt
from streamduo.model.stream_data import FPS, build_messages, read_clip


def build_prompt(row, t, dossier, processor, window, video_root="", lookups=None):
    """Streaming prefix for the last `window` seconds ending at the user turn of second t.
    lookups: {second: research-agent answer} injected into the history block."""
    sub = dict(row)
    sub["video_begin"] = max(row["video_begin"], t - window)
    sub["video_end"] = t + 1
    msgs, _ = build_messages(sub, system_prompt(row, random.Random(0), dossier, task_idx=0), lookups)
    text = processor.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    path = row.get("video_abspath") or os.path.join(video_root, row["video_path"])
    clip = read_clip(path, sub["video_begin"], sub["video_end"])
    n = sub["video_end"] - sub["video_begin"]
    videos = [clip[i * FPS:(i + 1) * FPS] for i in range(n)]
    return processor(text=[text], videos=videos, return_tensors="pt")


def repeat_inputs(enc, k):
    return {"input_ids": enc["input_ids"].repeat(k, 1),
            "attention_mask": enc["attention_mask"].repeat(k, 1),
            "pixel_values_videos": enc["pixel_values_videos"].repeat(k, 1),
            "video_grid_thw": enc["video_grid_thw"].repeat(k, 1)}


def decode(tok, seqs, n_prompt):
    texts = tok.batch_decode(seqs[:, n_prompt:], skip_special_tokens=True)
    return ["" if t.strip() in ("", SILENCE) else t.strip() for t in texts]


def merge(init, out):
    """Merge the Stage 2 LoRA adapter (out/adapter) into the Stage 1 weights and copy the two heads."""
    from peft import PeftModel
    base = Qwen2VLForConditionalGeneration.from_pretrained(init, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, os.path.join(out, "adapter")).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    for f in os.listdir(init):  # response / ask heads + processor files
        if f.endswith((".pt", ".json", ".txt", ".jinja", ".model")) and not f.startswith(("config", "model.safetensors", "generation")):
            shutil.copy(os.path.join(init, f), os.path.join(out, f))
    print("merged ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="python -m streamduo.train.common --init STAGE1 --out STAGE2")
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge(a.init, a.out)
