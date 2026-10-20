# Scripts

Run the scripts in this folder in order, from the repository root. Each one reads the settings in `env.sh`
(`STREAMDUO_DATA`, `PY_TRAIN`, `PY_VLLM`, `NGPU`) and writes its output under `$STREAMDUO_DATA`.

```bash
bash scripts/1_data.sh
bash scripts/2_stage0.sh
...
bash scripts/8_eval.sh
```

## Steps

| Step | Script | Output under `$STREAMDUO_DATA` |
|---|---|---|
| Data, wiki, dossiers | `1_data.sh` | `manifests/S_bg3/`, `wiki/`, `dossiers/v2_sweep_q48_k2048/`, `interact/candidates.json` |
| Stage 0: renaming + dossier | `2_stage0.sh` | `ckpt/stage0` |
| Training answers, ask labels | `3_answers_and_ask_labels.sh` | `manifests/S_bg3/{train_answers,train_answers_full,ask_labels}.jsonl` |
| Stage 1: when to ask | `4_stage1.sh` | `ckpt/stage1` |
| Stage 2: OPSD | `5_stage2.sh` | `ckpt/stage2` (merged) |
| Research agent answers on the test set | `6_agent_cache.sh` | `manifests/S_bg3/test_agent_cache_e1.jsonl` |
| Streaming inference | `7_infer.sh` | `results/streamduo/preds.jsonl` |
| Metrics | `8_eval.sh` | `results/streamduo/metrics_b126.json`, `judged.jsonl` |

## Training

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

## Inference

At second t the commentator speaks when p_t > 0.27 and asks when q_t > 0.9 (at most every 3 s). The answer is
placed in the history at t + 2. After four consecutive spoken seconds it continues only when p_t > 0.6. Lines
are sampled with temperature 0.7. The research agent decodes greedily, so its answer to the frames of second t
is computed once for every test second (`6_agent_cache.sh`) and looked up when the commentator asks, which
gives the same result as calling it online.
