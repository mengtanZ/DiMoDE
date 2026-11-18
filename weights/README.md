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
- `weights/initialization/kitti_odom_pt/weights_4/{encoder,depth,pose_encoder,pose,flow,adam,adam_pose}.pth`.

The inference bundles do not include optimizer states or the complete three-network training state. Do not use them as a complete training resume checkpoint.

See the main README for the recorded reference metrics and the distinction between those records and the current validation status. Pretrained third-party components retain their upstream terms.

## SHA-256

```text
63e33f9a02b6fe426d6a7890ad2b350d974d947216c489ba8c066cce333513be  dimode-nuscenes.zip
9637b9a751d99dace81889029718503c898c8f9f439b39e111a0c3e35444d856  dimode-kitti-odom.zip
7fc9f42b47b69aae237ff70c7f0b2d9092dc58dbba3115ef966eb5d2ed4569f0  dimode-training-init.zip
```
