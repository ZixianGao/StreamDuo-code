#!/bin/bash
# Streaming inference on the 2,470 test clips, one shard per GPU (speak 0.27, ask 0.9, cooldown 3 s,
# answer after 2 s, continuation 0.6 after 4 s, temperature 0.7: the defaults of stream_infer).
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODEL=${MODEL:-$CKPT/stage2}
OUT=${OUT:-$STREAMDUO_DATA/results/streamduo}
mkdir -p $OUT
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i $PY_TRAIN -m streamduo.model.stream_infer --ckpt $MODEL \
    --manifest $M/test_clean.jsonl --ask_cache $M/test_agent_cache_e1.jsonl --dossier_run $DOSSIER \
    --out $OUT/shard$i.jsonl --shard $i --num_shards $NGPU > $OUT/shard$i.log 2>&1 &
done
wait
cat $OUT/shard*.jsonl > $OUT/preds.jsonl
