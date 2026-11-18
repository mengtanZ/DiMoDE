# Third-party notices

The root MIT license applies to the DiMoDE authors' original contributions. It does not replace the terms of inherited components or pretrained checkpoints.

| Component | Upstream source | Terms / local copy |
| --- | --- | --- |
| Depth encoder/decoder and inherited training utilities | [Lite-Mono](https://github.com/noahzn/Lite-Mono) | [MIT](licenses/lite-mono.txt), Copyright (c) 2023 Noah |
| Training geometry, pose components, base data loading and utilities inherited through Lite-Mono/DCPI-Depth | [Monodepth2](https://github.com/nianticlabs/monodepth2) | [Original license](licenses/monodepth2.txt), Copyright Niantic, Inc. 2018. Patent Pending. All rights reserved. |
| ResNet multi-image encoder | [torchvision](https://github.com/pytorch/vision) | [BSD 3-Clause](licenses/torchvision.txt) |
| Optical flow network in `networks/RAFT/` | [RAFT](https://github.com/princeton-vl/RAFT) | [BSD 3-Clause](licenses/RAFT-BSD-3-Clause.txt) |
| KITTI trajectory evaluator in `utils/kitti_odom_eval/` | [kitti-odom-eval](https://github.com/Huangying-Zhan/kitti-odom-eval) | [MIT](licenses/kitti-odom-eval-MIT.txt) |
| Original project implementation and README structure | [DCPI-Depth](https://github.com/mengtanZ/DCPI-Depth) | [MIT](licenses/DCPI-Depth-MIT.txt) |
| Pseudo-depth normal ranking loss | [SC-DepthV3](https://github.com/JiawangBian/sc_depth_pl), with sampling routines originating from [LeReS](https://github.com/aim-uofa/AdelaiDepth) | Inherited through the original working project; upstream terms apply |

The DiMoDE release specializes the supplied implementation to nuScenes and KITTI odometry; removes unused experiments; makes data, checkpoint, device, and output paths explicit; repairs missing dependencies and checkpoint resuming; and removes repeated image reads. The model architectures and active geometric-loss formulas are retained. The scheduler implements the original no-warmup cosine schedule used by the released commands.

The inherited Monodepth2 license includes non-commercial and redistribution conditions. The root MIT notice does not grant different rights to those portions. Dataset and pretrained-checkpoint terms apply separately.
