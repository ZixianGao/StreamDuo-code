#!/bin/bash
# Stage 1: learn when to ask (ask at a - 2 where the gain > gamma = 1 nat) and to read in-stream answers.
set -euo pipefail
source "$(dirname "$0")/env.sh"
$PY_TRAIN -m torch.distributed.run --nproc_per_node $NGPU -m streamduo.model.train \
  --fold $FOLD --base_model $CKPT/stage0 \
  --dossier_run $DOSSIER --dossier_dropout 0.3 --knowledge_swap_p 0.5 \
  --lookup_file $M/train_answers.jsonl --ask_labels_file $M/ask_labels.jsonl --ask_min_gain 1.0 \
  --output_dir $CKPT/stage1 --num_train_epochs 1 \
  --deepspeed configs/ds_zero3.json --bf16 --gradient_checkpointing \
  --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.03 --max_grad_norm 1.0 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --save_strategy no --logging_steps 10 --dataloader_num_workers 6 --report_to none --seed 0
