#!/usr/bin/env bash
set -e

DATA_PATH=${NUSCENES_TEST_PATH:-data/nuscenes/testing}
WEIGHTS=dimode/weights/dimode-nuscenes

python -m dimode.evaluate_depth --dataset nuscenes --eval_split nuscenes --data_path "$DATA_PATH" \
    --model lite-mono --load_weights_folder "$WEIGHTS" --model_name model-nuscenes \
    --gt_height 900 --gt_width 1600 "$@"
