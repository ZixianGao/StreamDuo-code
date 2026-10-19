#!/bin/bash
# Text LLM used for data cleaning, game dossiers and the LLM judge (Qwen3-30B-A3B, 2 GPUs).
# Run in the background before scripts that need it; they read STREAMDUO_LLM_BASE_URL / STREAMDUO_LLM_MODEL.
source "$(dirname "$0")/env.sh"
$PY_VLLM -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 \
  --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.8 --enable-prefix-caching
# export STREAMDUO_LLM_BASE_URL=http://localhost:8000/v1 STREAMDUO_LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
