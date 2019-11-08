#!/bin/bash

# OUTPUTDIR is directory containing this run.sh script
OUTPUTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python train.py \
  --model "resnet18" \
  --batch_size 100 \
  --num_epochs 300 \
  --output_dir $OUTPUTDIR
