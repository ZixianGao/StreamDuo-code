#!/bin/bash
# Training answers of the research agent and information-gain ask labels.
set -euo pipefail
source "$(dirname "$0")/env.sh"
# noisy answers (right entry + 2 distractors, 30% misses) and ideal answers (always contain the right entry)
$PY_TRAIN -m streamduo.interact.train_answers --fold $FOLD --miss_p 0.3   # -> train_answers.jsonl
$PY_TRAIN -m streamduo.interact.train_answers --fold $FOLD --miss_p 0     # -> train_answers_full.jsonl
# gain of the ideal answer for the Stage 0 model, on a random subset of 3,400 training clips
$PY_TRAIN -m torch.distributed.run --nproc_per_node $NGPU -m streamduo.interact.ask_labels \
  --init $CKPT/stage0 --fold $FOLD --max_rows 3400                       # -> ask_labels.jsonl
