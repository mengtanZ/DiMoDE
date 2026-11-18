# Checkpoints

Model binaries are excluded from Git. Public download links are pending; use the layout below when placing downloaded checkpoints.

| Bundle | Extract into | Files |
| --- | --- | --- |
| `dimode-nuscenes.zip` | Repository root | `weights/dimode-nuscenes/encoder.pth`, `depth.pth` |
| `dimode-kitti-odom.zip` | Repository root | `weights/dimode-kitti-odom/pose_encoder.pth`, `pose.pth` |
| `dimode-training-init.zip` | Repository root | Lite-Mono ImageNet weights, RAFT weights, KITTI epoch-4 model/optimizer states |

The training initialization bundle contains:

- `networks/lite-mono-pretrain.pth` and `networks/lite-mono-8m-pretrain.pth`.
- `weights/resnet18-f37072fd.pth` (torchvision ImageNet initialization for PoseNet).
- `weights/raft-kitti.pth` (used by the original scripts) and `weights/raft-chairs.pth` (available via `--flow_weights`).
- `weights/initialization/kitti_odom_pt/{encoder,depth,pose_encoder,pose,flow,adam,adam_pose}.pth`. The KITTI training script explicitly sets `--resume_epoch 5`.

The inference bundles do not include optimizer states or the complete three-network training state. Do not use them as a complete training resume checkpoint.

See the main README for the recorded reference metrics and the distinction between those records and the current validation status. Pretrained third-party components retain their upstream terms.

## SHA-256

```text
63e33f9a02b6fe426d6a7890ad2b350d974d947216c489ba8c066cce333513be  dimode-nuscenes.zip
9637b9a751d99dace81889029718503c898c8f9f439b39e111a0c3e35444d856  dimode-kitti-odom.zip
2b6995194cd2b5e66b5825ff46e9cb9ac1fc18f841cd19caa7f33e0ff0d87d59  dimode-training-init.zip
```
