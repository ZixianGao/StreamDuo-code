#!/bin/bash
# Stage 2: on-policy self-distillation (rank-32 LoRA, 300 steps, alpha = 1, reverse KL), then merge.
set -euo pipefail
source "$(dirname "$0")/env.sh"
$PY_TRAIN -m torch.distributed.run --nproc_per_node $NGPU -m streamduo.train.opsd \
  --init $CKPT/stage1 --out $CKPT/stage2 --fold $FOLD --dossier_run $DOSSIER --steps 300 --alpha 1.0
$PY_TRAIN -m streamduo.train.common --init $CKPT/stage1 --out $CKPT/stage2   # merged model + heads in ckpt/stage2
