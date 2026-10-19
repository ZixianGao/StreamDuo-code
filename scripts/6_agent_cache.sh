#!/bin/bash
# The frozen research agent (Qwen3-VL-8B + BM25 over the wiki) answers the frames of every test second once.
# It decodes greedily, so this cache equals calling it online when the commentator asks (streamduo.model.stream_infer).
set -euo pipefail
source "$(dirname "$0")/env.sh"
VLM=${VLM:-Qwen/Qwen3-VL-8B-Instruct}
$PY_TRAIN -m streamduo.interact.build_test_lookups frames --fold $FOLD --every 1 --workers 24
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i $PY_VLLM -m streamduo.interact.build_test_lookups agent --fold $FOLD --every 1 \
    --vlm $VLM --shard $i --num_shards $NGPU &
done
wait
$PY_TRAIN -m streamduo.interact.build_test_lookups merge --fold $FOLD --every 1   # -> test_agent_cache_e1.jsonl
