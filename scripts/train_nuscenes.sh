#!/usr/bin/env bash
set -e

# Run from the repository root after activating the environment.
DATA_PATH=${NUSCENES_TRAIN_PATH:-data/nuscenes/training}
PRETRAIN=networks/lite-mono-pretrain.pth

python train.py --dataset nuscenes --split nuscenes --data_path "$DATA_PATH" \
    --model_name model-nuscenes --model lite-mono --mypretrain "$PRETRAIN" \
    --version v1 --num_epochs 30 --batch_size 6 --height 384 --width 640 \
    --weights 1.0 0.5 0.1 0.05 --psedo_depth psedo_depth \
    --psedo_weights 0.01 --normal_ranking_loss "$@"
