"""Streaming inference of the commentator with the research agent.

Per clip, the KV cache grows second by second exactly like the training sequence:
  [system (+dossier)] then for each second t:
     user turn up to <|FLAG|>  -> p_t from the response head
     if p_t >= tau: "<|im_end|>\\n<|im_start|>assistant\\n" + sampled tokens (<= max_new_tokens) + "<|im_end|>\\n"
     else:          "<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>\\n"
M-RoPE positions are recomputed with get_rope_index over the FULL accumulated sequence for every
appended chunk, so inference positions match training (LiveCC's demo instead gives later video
chunks 1-D positions).

    python -m streamduo.model.stream_infer --ckpt CKPT --manifest $STREAMDUO_DATA/manifests/S_bg3/test_clean.jsonl \\
        --ask_cache $STREAMDUO_DATA/manifests/S_bg3/test_agent_cache.jsonl --out preds.jsonl [--shard 0 --num_shards 4]
"""
import argparse
import json
import random

import torch
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

from streamduo.model.proactvl import ProactCommentator
from streamduo.model.prompts import SILENCE, system_prompt
from streamduo.model.stream_data import FPS, PARTNER_OFFSET, per_second, read_clip, setup_processor
from streamduo.paths import DATA_ROOT


class StreamRunner:
    def __init__(self, ckpt, device="cuda", tau=0.3, max_new_tokens=12, temperature=0.7, top_p=0.9,
                 repetition_penalty=1.15):
        self.processor = AutoProcessor.from_pretrained(ckpt, padding_side="right")
        setup_processor(self.processor)
        self.tok = self.processor.tokenizer
        self.model = ProactCommentator.from_pretrained(
            ckpt, n_vocab=len(self.tok), torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to(device).eval()
        self.device = device
        self.tau, self.max_new = tau, max_new_tokens
        self.temperature, self.top_p, self.rep = temperature, top_p, repetition_penalty
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    # --- low-level -------------------------------------------------------------------------
    def _reset(self):
        self.cache = DynamicCache()
        self.ids = torch.empty((1, 0), dtype=torch.long, device=self.device)
        self.grids = torch.empty((0, 3), dtype=torch.long, device=self.device)

    def _positions(self, n_new):
        full = self.ids
        attn = torch.ones_like(full)
        pos, _ = self.model.base.get_rope_index(full, None, self.grids if len(self.grids) else None, attn)
        return pos[:, :, -n_new:]

    @torch.inference_mode()
    def _feed(self, input_ids, pixel_values_videos=None, video_grid_thw=None):
        input_ids = input_ids.to(self.device)
        self.ids = torch.cat([self.ids, input_ids], dim=1)
        if video_grid_thw is not None:
            self.grids = torch.cat([self.grids, video_grid_thw.to(self.device)], dim=0)
        out = self.model.base(
            input_ids=input_ids, position_ids=self._positions(input_ids.shape[1]),
            pixel_values_videos=None if pixel_values_videos is None else pixel_values_videos.to(self.device),
            video_grid_thw=None if video_grid_thw is None else video_grid_thw.to(self.device),
            past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        return out.logits[:, -1].float()

    def _text_ids(self, text):
        return self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids

    def _sample(self, logits, generated):
        if self.rep != 1.0 and generated:
            idx = torch.tensor(sorted(set(generated)), device=logits.device)
            sel = logits[0, idx]
            logits[0, idx] = torch.where(sel > 0, sel / self.rep, sel * self.rep)
        if self.temperature <= 0:
            return int(logits.argmax(-1))
        probs = torch.softmax(logits / self.temperature, -1)
        sp, si = probs.sort(descending=True)
        keep = sp.cumsum(-1) - sp < self.top_p
        sp = sp * keep
        return int(si[0, torch.multinomial(sp[0] / sp[0].sum(), 1)])

    @torch.inference_mode()
    def run_clip(self, row, dossier=None, seed=0, ask_cache=None, tau_ask=0.9, delta=2, cooldown=3,
                 cont_after=4, tau_cont=0.6):
        """ask_cache: {second t: the frozen research agent's answer to the frames of t}. When the ask head fires
        at t (q_t >= tau_ask, at least `cooldown` s after the previous question), the answer is placed in the
        history block at t + delta. The agent decodes greedily, so its answer to the frames of t is fixed and
        can be computed once (streamduo.interact.build_test_lookups); this equals calling it online.
        cont_after > 0: after `cont_after` consecutive spoken seconds, continuing needs p_t >= tau_cont."""
        lookups = {}
        last_ask = -10 ** 9
        streak = 0
        rng = random.Random(seed)
        self._reset()
        sys_text = system_prompt(row, rng, dossier, task_idx=0)
        sys_ids = self.processor.apply_chat_template(
            [{"role": "system", "content": [{"type": "text", "text": sys_text}]}], tokenize=False)
        self._feed(self._text_ids(sys_ids))

        begin, end = row["video_begin"], row["video_end"]
        clip = read_clip(row.get("video_abspath"), begin, end)
        _, partner, query = per_second(row)
        outputs = []
        for i, t in enumerate(range(begin, end)):
            hist = " ".join(partner.get(t - PARTNER_OFFSET, []))
            if i == 0 and row.get("history"):
                hist = (row["history"][-1500:] + " " + hist).strip()
            if t in lookups:
                hist = (f"[Research agent] {lookups[t]} " + hist).strip()
            user = [{"role": "user", "content": [
                {"type": "text", "text": f"<|history_start|>{hist}<|history_end|>"},
                {"type": "video"},
                {"type": "text", "text": f"<|query_start|>{' '.join(query[t])}<|query_end|><|FLAG|>"}]}]
            text = self.processor.apply_chat_template(user, tokenize=False)
            text = text[text.index("<|im_start|>user"):]          # drop the default system block
            text = text[:text.rindex("<|FLAG|>") + len("<|FLAG|>")]  # stop right after FLAG
            if i > 0:
                text = "\n" + text
            enc = self.processor(text=[text], videos=[clip[i * FPS:(i + 1) * FPS]], return_tensors="pt")
            self._feed(enc["input_ids"], enc["pixel_values_videos"], enc["video_grid_thw"])
            p = float(torch.sigmoid(self.model.head(self.model._penult[0, -1].to(
                self.model.head.out.weight.dtype)).float()))
            asked, p_ask = False, None
            if ask_cache is not None:
                p_ask = float(torch.sigmoid(self.model.ask_head(self.model._penult[0, -1].to(
                    self.model.ask_head.out.weight.dtype)).float()))
                if p_ask >= tau_ask and t - last_ask >= cooldown:
                    asked, last_ask = True, t
                    ans = ask_cache.get(t)
                    if ans:
                        lookups[t + delta] = (lookups.get(t + delta, "") + " " + ans).strip()

            logits = self._feed(self._text_ids("<|im_end|>\n<|im_start|>assistant\n"))
            said = ""
            speak = p >= self.tau and not (cont_after and streak >= cont_after and p < tau_cont)
            if speak:
                gen = []
                for _ in range(self.max_new):
                    nxt = self._sample(logits.clone(), gen)
                    if nxt == self.im_end:
                        break
                    gen.append(nxt)
                    logits = self._feed(torch.tensor([[nxt]]))
                said = self.tok.decode(gen, skip_special_tokens=True).strip()
                self._feed(self._text_ids("<|im_end|>"))
            else:
                self._feed(self._text_ids(f"{SILENCE}<|im_end|>"))
            streak = streak + 1 if said else 0
            outputs.append({"t": t, "p": round(p, 4), "text": said, **({"ask": True} if asked else {}),
                            **({"q": round(p_ask, 4)} if p_ask is not None else {})})
        return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dossier_run", default="v2_sweep_q48_k2048")
    ap.add_argument("--ask_cache", default="", help="jsonl {key, answers: {t: answer}} (empty: no research agent)")
    ap.add_argument("--tau", type=float, default=0.27, help="speak threshold on p_t")
    ap.add_argument("--tau_ask", type=float, default=0.9, help="ask threshold on q_t")
    ap.add_argument("--ask_delta", type=int, default=2, help="seconds until the answer arrives")
    ap.add_argument("--ask_cooldown", type=int, default=3)
    ap.add_argument("--cont_after", type=int, default=4)
    ap.add_argument("--tau_cont", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=0.7, help="decoding temperature (0 = greedy)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest)][args.shard::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]
    dossiers = {}
    if args.dossier_run:
        dossiers = {p.stem: p.read_text() for p in (DATA_ROOT / "dossiers" / args.dossier_run).glob("*.txt")}
    ask_by_key = None
    if args.ask_cache:
        ask_by_key = {d["key"]: {int(t): a for t, a in d["answers"].items()} for d in map(json.loads, open(args.ask_cache))}
    runner = StreamRunner(args.ckpt, tau=args.tau, temperature=args.temperature)
    done = set()
    try:
        done = {json.loads(l)["key"] for l in open(args.out)}
    except FileNotFoundError:
        pass
    with open(args.out, "a") as f:
        for k, row in enumerate(rows):
            key = f"{row['video_path']}@{row['video_begin']}"
            if key in done:
                continue
            preds = runner.run_clip(row, dossiers.get(row["metadata"]["dataset_name"]), seed=k,
                                    ask_cache=None if ask_by_key is None else ask_by_key.get(key, {}),
                                    tau_ask=args.tau_ask, delta=args.ask_delta, cooldown=args.ask_cooldown,
                                    cont_after=args.cont_after, tau_cont=args.tau_cont)
            f.write(json.dumps({"key": key, "game": row["metadata"]["dataset_name"], "preds": preds}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
