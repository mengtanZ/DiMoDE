# Checkpoints

Download the checkpoints from [Hugging Face](https://huggingface.co/mengtanZ/DiMoDE/tree/main) and extract each archive into `dimode/weights/`.

| Bundle | Extract into | Files |
| --- | --- | --- |
| [dimode-nuscenes.zip](https://huggingface.co/mengtanZ/DiMoDE/resolve/main/dimode-nuscenes.zip?download=true) | `dimode/weights/` | `dimode/weights/dimode-nuscenes/{encoder,depth}.pth` |
| [dimode-kitti-odom.zip](https://huggingface.co/mengtanZ/DiMoDE/resolve/main/dimode-kitti-odom.zip?download=true) | `dimode/weights/` | `dimode/weights/dimode-kitti-odom/{pose_encoder,pose}.pth` |
| [initialization.zip](https://huggingface.co/mengtanZ/DiMoDE/resolve/main/initialization.zip?download=true) | `dimode/weights/` | KITTI Odometry initialization: `dimode/weights/initialization/kitti_odom_pt/{encoder,depth,pose_encoder,pose,flow,adam,adam_pose}.pth` |

Training also requires the following pretrained weights, prepared separately:

- [Lite-Mono](https://github.com/noahzn/Lite-Mono) ImageNet weights: `dimode/networks/lite-mono-pretrain.pth` and `dimode/networks/lite-mono-8m-pretrain.pth`.
- [ResNet-18](https://download.pytorch.org/models/resnet18-f37072fd.pth) ImageNet weights: `dimode/weights/resnet18-f37072fd.pth`.
- [RAFT](https://github.com/princeton-vl/RAFT) weights: `dimode/weights/raft-kitti.pth` by default; `dimode/weights/raft-chairs.pth` can be selected with `--flow_weights`.
