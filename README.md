# StreamDuo

Code for **StreamDuo: A Live Commentator and a Research Agent for Unseen Games**.

A trainable commentator (LiveCC-7B-Base) watches the stream and decides every second whether to speak. When it
sees something it cannot name, it asks a frozen research agent (Qwen3-VL-8B with BM25 search over the game's
wiki) by sending the current frames, and the answer arrives two seconds later while the stream continues. The
commentator is trained on Baldur's Gate 3 only and tested on ten games it has never seen.

This repository contains only what is needed to train and test that system.

## Layout

```
streamduo/
  data/       Proact-VL splits, clean test protocol, K/V/S labels for K-strict
  wiki/       wiki snapshot (MediaWiki API) and entity ontology per game
  research/   the research agent that writes a game dossier before the stream
  interact/   training answers, information-gain ask labels, test-time answer cache of the research agent
  model/      the commentator (response head + ask head), training (Stage 0 / 1), streaming inference
  train/      Stage 2: on-policy self-distillation, LoRA merge
  eval/       TimeDiff, F1, Ent-R, Ent-F1, K-strict, source-game leakage, LLM judge
scripts/      one script per step, in order
configs/      DeepSpeed ZeRO-3, splits, wiki sources
```

## Setup

Two Python 3.11 environments:

```bash
pip install -r requirements-train.txt   # training, inference, metrics (PY_TRAIN)
pip install -r requirements-vllm.txt    # research agent and LLM server (PY_VLLM)
```

Set the paths at the top of `scripts/env.sh` (`STREAMDUO_DATA`, `PY_TRAIN`, `PY_VLLM`, `NGPU`). All data,
checkpoints and results live under `$STREAMDUO_DATA`. Training uses 4 GPUs with 80 GB each.

Data: the Proact-VL game commentary data (`oaaoaa/game_commentary_sft`, `oaaoaa/LiveGamingBenchmark`, gated on
the Hugging Face Hub) and `chenjoya/LiveCC-7B-Base`:

```bash
HF_TOKEN=hf_... python scripts/0_download.py
```

Steps that call a text LLM (cleaning, dossiers, judge) use an OpenAI-compatible server, by default a local vLLM
server with Qwen3-30B-A3B started by `scripts/serve_llm.sh`. Point them to it with
`STREAMDUO_LLM_BASE_URL` and `STREAMDUO_LLM_MODEL`.

## Pipeline

| Step | Script | Output under `$STREAMDUO_DATA` |
|---|---|---|
| Data, wiki, dossiers | `scripts/1_data.sh` | `manifests/S_bg3/`, `wiki/`, `dossiers/v2_sweep_q48_k2048/`, `interact/candidates.json` |
| Stage 0: renaming + dossier | `scripts/2_stage0.sh` | `ckpt/stage0` |
| Training answers, ask labels | `scripts/3_answers_and_ask_labels.sh` | `manifests/S_bg3/{train_answers,train_answers_full,ask_labels}.jsonl` |
| Stage 1: when to ask | `scripts/4_stage1.sh` | `ckpt/stage1` |
| Stage 2: OPSD | `scripts/5_stage2.sh` | `ckpt/stage2` (merged) |
| Research agent answers on the test set | `scripts/6_agent_cache.sh` | `manifests/S_bg3/test_agent_cache_e1.jsonl` |
| Streaming inference | `scripts/7_infer.sh` | `results/streamduo/preds.jsonl` |
| Metrics | `scripts/8_eval.sh` | `results/streamduo/metrics_b126.json`, `judged.jsonl` |

### Training

- **Stage 0** fine-tunes all parameters of LiveCC-7B-Base for one epoch on 10,286 Baldur's Gate 3 clips. The
  game dossier is in the system prompt (dropped with p = 0.3), and with p = 0.5 the entities of a clip are
  renamed consistently and at random, so a name is evidence only within its own clip.
- **Ask labels.** For every moment a human caster names an entity, the Stage 0 model scores that line with and
  without an ideal answer. Asking two seconds earlier is labelled positive when the answer raises the line's
  log-likelihood by more than γ = 1 nat.
- **Stage 1** continues from Stage 0 on the same 3,400 clips, with the ask head and with research-agent answers placed in the history
  (the right entity plus two distractors, missing in 30% of the answers).
- **Stage 2** trains a rank-32 LoRA for 300 steps. The student sees the noisy answer, and its teacher is the
  Stage 1 model with the ideal answer, amplified by context-aware decoding (α = 1). The loss is the reverse KL
  on the student's own samples.

### Inference

At second t the commentator speaks when p_t > 0.27 and asks when q_t > 0.9 (at most every 3 s). The answer is
placed in the history at t + 2. After four consecutive spoken seconds it continues only when p_t > 0.6. Lines
are sampled with temperature 0.7. The research agent decodes greedily, so its answer to the frames of second t
is computed once for every test second (`6_agent_cache.sh`) and looked up when the commentator asks, which
gives the same result as calling it online.

## Citation

```bibtex
@article{streamduo2026,
  title  = {StreamDuo: A Live Commentator and a Research Agent for Unseen Games},
  author = {Gao, Zixian and Luo, Penghao and Cao, Xiangcheng and Li, Zhen and Xiu, Lixin and Cai, Ziqi and Wu, Erwin and Zhang, Kaipeng},
  year   = {2026}
}
```
