#!/bin/bash
# Metrics: timing (TimeDiff, F1), knowledge on the first 126 words (Ent-R, Ent-F1, K-strict), source-game
# leakage, and the LLM judge (PAUC, CC, LiveU, FinalQ; needs scripts/serve_llm.sh).
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT=${OUT:-$STREAMDUO_DATA/results/streamduo}
$PY_TRAIN -m streamduo.eval.report --preds $OUT/preds.jsonl --manifest $M/test_clean.jsonl \
  --out $OUT/metrics_b126.json --word_budget 126
$PY_TRAIN -m streamduo.eval.leakage --preds $OUT/preds.jsonl --manifest $M/test_clean.jsonl
$PY_TRAIN -m streamduo.eval.judge --preds $OUT/preds.jsonl --manifest $M/test_clean.jsonl \
  --out $OUT/judged.jsonl --metrics pauc,cc,liveu,finalq
