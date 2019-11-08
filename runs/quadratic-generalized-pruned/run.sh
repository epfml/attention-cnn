#!/bin/bash

# OUTPUTDIR is directory containing this run.sh script
OUTPUTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python train.py \
  --num_hidden_layers 6 \
  --num_attention_heads 9 \
  --optimizer_cosine_lr True \
  --optimizer_learning_rate 0.01 \
  --optimizer_warmup_ratio 0.05 \
  --batch_size 100 \
  --num_epochs 100 \
  --hidden_size 400 \
  --attention_isotropic_gaussian False \
  --num_keep_checkpoints 10 \
  --prune_degenerated_heads True \
  --load_checkpoint_file ./quadratic-generalized/final.checkpoint \
  --output_dir $OUTPUTDIR
