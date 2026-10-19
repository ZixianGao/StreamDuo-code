"""Clip -> Proact-VL-style per-second conversation sample.

For a clip [begin, end) and every second t:
  user:      <|history_start|> H_t <|history_end|> <video: 2 frames of [t, t+1)>
             <|query_start|> Q_t <|query_end|> <|FLAG|>
  assistant: the target speaker's text for second t, or "..." when silent
where H_t = other speakers' lines from second t-1 (plus the clip's prior `history` at the first step)
and Q_t = player queries that start at t (guidance mode).

Outputs (one sample, batch size 1):
  processor outputs (input_ids, attention_mask, pixel_values_videos, video_grid_thw)
  labels       LM targets on speaking assistant turns only (silent turns fully masked)
  flag_index   positions of <|FLAG|> tokens, one per second
  flag_labels  1 if the target speaks in that second else 0
"""
import json
import os
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from streamduo.model.prompts import SILENCE, SPECIAL_TOKENS, system_prompt

FPS = 2
PARTNER_OFFSET = 1   # partner lines of second t-1 enter the history block at step t
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 540 * 28 * 28
MAX_VIDEO_PIXELS = 36 * 540 * 28 * 28


def setup_processor(processor):
    """Add the 5 Proact-VL special tokens; returns their ids."""
    processor.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return {t: processor.tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}


def read_clip(path, begin, end):
    """Frames at exactly FPS in [begin, end), resized with the Proact-VL pixel budget. -> (T, C, H, W)"""
    import decord  # noqa: imported lazily (after torch) to avoid decord/torch import-order crashes
    from qwen_vl_utils.vision_process import smart_resize

    vr = decord.VideoReader(path, num_threads=2)
    fps = vr.get_avg_fps()
    n = (end - begin) * FPS
    times = [begin + (i + 0.5) / FPS for i in range(n)]
    idx = [min(int(round(t * fps)), len(vr) - 1) for t in times]
    frames = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
    h, w = frames.shape[2:]
    budget = min(MAX_PIXELS, MAX_VIDEO_PIXELS / max(end - begin, 1))
    rh, rw = smart_resize(h, w, factor=28, min_pixels=MIN_PIXELS, max_pixels=budget)
    return TF.resize(frames, [rh, rw], interpolation=TF.InterpolationMode.BICUBIC, antialias=True).float()


def per_second(row):
    """Group annotations by second: target text, partner lines, queries."""
    begin, end = row["video_begin"], row["video_end"]
    target = row["active_speaker"]["name"]
    speak = {t: [] for t in range(begin, end)}
    partner = {t: [] for t in range(begin, end)}
    query = {t: [] for t in range(begin, end)}
    for a in row["annotations"]:
        t = int(a["start"])
        if t not in speak:
            continue
        if a["role"] == "assistant" and a.get("speaker", target) == target and a.get("text"):
            speak[t].append(a["text"])
        elif a.get("query"):
            query[t].append(a["query"])
        elif a.get("text"):
            partner[t].append(a["text"])
    return speak, partner, query


def build_messages(row, sys_text, lookups=None):
    """lookups: {second t: research-agent answer} injected into the history block (same format as
    streamduo.model.stream_infer --lookup_file)."""
    begin, end = row["video_begin"], row["video_end"]
    speak, partner, query = per_second(row)
    msgs = [{"role": "system", "content": [{"type": "text", "text": sys_text}]}]
    labels = []
    lookups = lookups or {}
    for i, t in enumerate(range(begin, end)):
        hist = " ".join(partner.get(t - PARTNER_OFFSET, []))
        if i == 0 and row.get("history"):
            hist = (row["history"][-1500:] + " " + hist).strip()
        if t in lookups:
            hist = (f"[Research agent] {lookups[t]} " + hist).strip()
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": f"<|history_start|>{hist}<|history_end|>"},
            {"type": "video", "second": i},
            {"type": "text", "text": f"<|query_start|>{' '.join(query[t])}<|query_end|><|FLAG|>"},
        ]})
        said = " ".join(speak[t]).strip()
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": said or SILENCE}]})
        labels.append(1 if said else 0)
    return msgs, labels


class StreamClipDataset(Dataset):
    def __init__(self, files, processor, video_root, dossiers=None, dossier_dropout=0.0,
                 seed=0, max_rows=None, transform=None):
        """files: annotation jsonl paths; dossiers: {dataset_name: text} or None;
        transform: optional fn(row, dossier, rng) -> (row, dossier), e.g. Knowledge-Swap."""
        self.rows = []
        for f in files:
            with open(f) as fh:
                self.rows += [json.loads(l) for l in fh]
        if max_rows:  # random subset (clips are stored grouped by video)
            random.Random(seed).shuffle(self.rows)
            self.rows = self.rows[:max_rows]
        self.processor = processor
        self.video_root = video_root
        self.dossiers = dossiers or {}
        self.dossier_dropout = dossier_dropout
        self.transform = transform
        self.seed = seed
        self.lookups = {}         # {key: {t: answer}}: in-stream research-agent answers
        self.ask_override = None  # {clip key: set of ask seconds}: information-gain labels (Stage 1)
        self.tok = processor.tokenizer
        self.im_start, self.im_end = self.tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
        self.assistant_ids = self.tok("assistant\n", add_special_tokens=False).input_ids
        self.flag_id = self.tok.convert_tokens_to_ids("<|FLAG|>")
        self.silence_ids = self.tok(SILENCE, add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.rows)

    def build(self, row, rng, with_video=True):
        dossier = self.dossiers.get(row["metadata"]["dataset_name"])
        if dossier and rng.random() < self.dossier_dropout:
            dossier = None
        if self.transform is not None:
            row, dossier = self.transform(row, dossier, rng)
        lk = self.lookup_for(row)
        msgs, flag_labels = build_messages(row, system_prompt(row, rng, dossier), lk)
        ask_labels = None
        if self.lookups:  # ask at t if an answer is due at t + ASK_DELTA
            from streamduo.model.proactvl import ASK_DELTA
            ask_labels = [1 if (lk and (t + ASK_DELTA) in lk) else 0 for t in range(row["video_begin"], row["video_end"])]
            if self.ask_override is not None:   # ask only where the model did not know and help helps
                asks = self.ask_override.get(f"{row['video_path']}@{row['video_begin']}", set())
                ask_labels = [1 if t in asks else 0 for t in range(row["video_begin"], row["video_end"])]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        videos = None
        if with_video:
            path = row.get("video_abspath") or os.path.join(self.video_root, row["video_path"])
            clip = read_clip(path, row["video_begin"], row["video_end"])
            videos = [clip[i * FPS:(i + 1) * FPS] for i in range(len(flag_labels))]
        return text, videos, flag_labels, ask_labels

    def encode(self, text, videos, flag_labels, ask_labels=None):
        inputs = self.processor(text=[text], videos=videos, return_tensors="pt")
        ids = inputs["input_ids"][0]
        labels = torch.full_like(ids, -100)
        starts = (ids == self.im_start).nonzero().flatten().tolist()
        ends = (ids == self.im_end).nonzero().flatten().tolist()
        n_a = len(self.assistant_ids)
        for s in starts:
            if ids[s + 1:s + 1 + n_a].tolist() != self.assistant_ids:
                continue
            e = next(x for x in ends if x > s)
            body = ids[s + 1 + n_a:e].tolist()
            if body == self.silence_ids:
                continue  # silent turn: no LM loss
            labels[s + 1 + n_a:e + 1] = ids[s + 1 + n_a:e + 1]
        flag_index = (ids == self.flag_id).nonzero().flatten()
        assert len(flag_index) == len(flag_labels), (len(flag_index), len(flag_labels))
        inputs["labels"] = labels[None]
        inputs["flag_index"] = flag_index
        inputs["flag_labels"] = torch.tensor(flag_labels, dtype=torch.float)
        if ask_labels is not None:
            inputs["ask_labels"] = torch.tensor(ask_labels, dtype=torch.float)
        return inputs

    def lookup_for(self, row):
        """This clip's in-stream answers, renamed consistently if Knowledge-Swap was applied."""
        lk = self.lookups.get(f"{row['video_path']}@{row['video_begin']}")
        if not lk:
            return None
        m = row.get("_swap")
        if m:
            import re
            pat = re.compile("|".join(rf"(?<!\w){re.escape(n)}(?!\w)" for n in sorted(m, key=len, reverse=True)), re.I)
            low = {k.lower(): v for k, v in m.items()}
            lk = {t: pat.sub(lambda mt: low[mt.group(0).lower()], a) for t, a in lk.items()}
        return lk

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        for attempt in range(5):
            try:
                return self.encode(*self.build(self.rows[i], rng))
            except Exception as e:  # corrupt video etc.: fall back to another sample
                print(f"[stream_data] sample {i} failed ({e!r}); resampling", flush=True)
                i = rng.randrange(len(self.rows))
        raise RuntimeError("too many failed samples")

    @staticmethod
    def collate(batch):
        assert len(batch) == 1, "batch size must be 1 (use gradient accumulation)"
        return batch[0]
