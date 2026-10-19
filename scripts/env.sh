# Shared settings, sourced by every script. Edit these three lines for your machine.
export STREAMDUO_DATA=${STREAMDUO_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}
PY_TRAIN=${PY_TRAIN:-python}     # env with torch 2.6, transformers 4.50, deepspeed, liger-kernel, flash-attn (requirements-train.txt)
PY_VLLM=${PY_VLLM:-python}       # env with vllm 0.11 for the research agent and the LLM judge (requirements-vllm.txt)
NGPU=${NGPU:-4}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $REPO
FOLD=S_bg3                        # train on Baldur's Gate 3, test on the ten other games
DOSSIER=v2_sweep_q48_k2048
CKPT=$STREAMDUO_DATA/ckpt
M=$STREAMDUO_DATA/manifests/$FOLD
