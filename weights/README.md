# Checkpoints

Download the checkpoints from [Hugging Face](https://huggingface.co/mengtanZ/DiMoDE/tree/main) and extract each archive into `weights/`.

| Bundle | Extract into | Files |
| --- | --- | --- |
| [dimode-nuscenes.zip](https://huggingface.co/mengtanZ/DiMoDE/resolve/main/dimode-nuscenes.zip?download=true) | `weights/` | `weights/dimode-nuscenes/{encoder,depth}.pth` |
| [dimode-kitti-odom.zip](https://huggingface.co/mengtanZ/DiMoDE/resolve/main/dimode-kitti-odom.zip?download=true) | `weights/` | `weights/dimode-kitti-odom/{pose_encoder,pose}.pth` |
| [initialization.zip](https://huggingface.co/mengtanZ/DiMoDE/resolve/main/initialization.zip?download=true) | `weights/` | KITTI Odometry initialization: `weights/initialization/kitti_odom_pt/{encoder,depth,pose_encoder,pose,flow,adam,adam_pose}.pth` |

The KITTI training script sets `--resume_epoch 5` when loading these initialization weights.

Training also requires the following pretrained weights, prepared separately:

- [Lite-Mono](https://github.com/noahzn/Lite-Mono) ImageNet weights: `networks/lite-mono-pretrain.pth` and `networks/lite-mono-8m-pretrain.pth`.
- [ResNet-18](https://download.pytorch.org/models/resnet18-f37072fd.pth) ImageNet weights: `weights/resnet18-f37072fd.pth`.
- [RAFT](https://github.com/princeton-vl/RAFT) weights: `weights/raft-kitti.pth` by default; `weights/raft-chairs.pth` can be selected with `--flow_weights`.

Training and evaluation commands are provided in the main README. Pretrained third-party components retain their upstream terms.

## SHA-256

```text
af30ec24971eb300eb23af8aaa6b5a1bb81c53ae907d792fb42c3d95e23109a1  dimode-nuscenes.zip
46b48cfad17e5e053fe5c7410b02fcff9e8589daa99ed813a9edbd0a0082c89c  dimode-kitti-odom.zip
cca545a1826839957dac4e503b4d52a62a2a622b401b01ebd37db3ccbabb3279  initialization.zip
```
