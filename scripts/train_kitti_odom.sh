#!/usr/bin/env bash
set -e

DATA_PATH=${KITTI_ODOM_PATH:-data/kitti_odom}
PRETRAIN=networks/lite-mono-8m-pretrain.pth
INITIALIZATION=weights/initialization/kitti_odom_pt

python train.py --dataset kitti_odom --split kitti_odom --data_path "$DATA_PATH" \
    --model_name model-kitti_odom --model lite-mono-8m --mypretrain "$PRETRAIN" \
    --version v1 --num_epochs 30 --batch_size 12 --width 640 --height 192 --png \
    --weights 1.0 0.5 0.1 0.05 --load_weights_folder "$INITIALIZATION" --resume_epoch 5 \
    --lr 0.00001 5e-7 31 0.00001 1e-6 31 \
    --eval_frequency 1000 --eval_dataset kitti_odom "$@"
