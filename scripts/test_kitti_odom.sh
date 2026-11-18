#!/usr/bin/env bash
set -e

DATA_PATH=${KITTI_ODOM_PATH:-data/kitti_odom}
WEIGHTS=weights/dimode-kitti-odom
RESULTS=${KITTI_ODOM_RESULTS:-eval/model-kitti_odom}
GT_PATH=${KITTI_ODOM_GT_PATH:-$DATA_PATH/poses}

for i in 9 10
do
    python evaluate_pose.py --data_path "$DATA_PATH" --load_weights_folder "$WEIGHTS" \
        --dataset kitti_odom --eval_split "odom_${i}" --png --height 192 --width 640 \
        --eval_out_dir "$RESULTS" "$@"
done
python utils/kitti_odom_eval/eval_odom.py --result "$RESULTS" --gt "$GT_PATH" --seqs 9 10 --align 7dof
