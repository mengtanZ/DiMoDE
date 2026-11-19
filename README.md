# DiMoDE: Discriminately Treating Motion Components Evolves Joint Depth and Ego-Motion Learning

Official PyTorch implementation for the paper:

> **Discriminately Treating Motion Components Evolves Joint Depth and Ego-Motion Learning**
>
> Mengtan Zhang, Zizhan Guo, Hongbo Zhao, Yi Feng, Zuyi Xiong, Yue Wang, Shaoyi Du, Hanli Wang, and Rui Fan
>
> IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2026
>
> [Paper](https://doi.org/10.1109/TPAMI.2026.3732069) · [Project Page](https://mias.group/DiMoDE/)

DiMoDE exploits the geometric regularities of individual motion components to improve joint depth and ego-motion learning. Optical-axis and imaging-plane alignments introduce geometric constraints on pose and depth. Optical flow is used during training; inference requires only DepthNet or PoseNet.

![Overview of DiMoDE](assets/framework.png)


## Table of Contents

- [DiMoDE: Discriminately Treating Motion Components Evolves Joint Depth and Ego-Motion Learning](#dimode-discriminately-treating-motion-components-evolves-joint-depth-and-ego-motion-learning)
  - [Table of Contents](#table-of-contents)
  - [Install](#install)
  - [Dataset](#dataset)
    - [nuScenes](#nuscenes)
    - [KITTI Odometry](#kitti-odometry)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Pretrained Models](#pretrained-models)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)

## Install

Run the following commands from the repository root:

```bash
conda create -n dimode_env python=3.9
conda activate dimode_env
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Dataset

Download the original datasets from [nuScenes](https://www.nuscenes.org/nuscenes) and [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Data are not included in this repository.

### nuScenes

Preprocess nuScenes front-camera videos into consecutively numbered JPEG frames, following [Dynamo-Depth](https://github.com/YihongSun/Dynamo-Depth). Organize the images and pseudo-depth maps as shown below, with folder names and frame indices matching `splits/nuscenes/`.

| Path | Content |
| --- | --- |
| `data/nuscenes/training/scene-XXXX/000000.jpg` | Consecutive training/validation frames |
| `data/nuscenes/training/scene-XXXX/psedo_depth/000000.png` | Pseudo-depth for the corresponding frame, required for training; see [SC-DepthV3](https://github.com/JiawangBian/sc_depth_pl#generating-pseudo-depth-for-training-v3) for generation instructions |
| `data/nuscenes/testing/color/000000.jpg` | Test images, numbered `000000` through `006018` |
| `data/nuscenes/testing/depth/000000.npy` | Corresponding point-depth array with shape `(N, 3)`: image `x`, image `y`, depth in meters |

Split files are in `splits/nuscenes/`: 79,760 training triplets, 6,019 validation entries, and 6,019 test images. Each line has the format `folder frame_index side`. Training and validation entries refer to scene folders under the **training root**. Test entries refer to `color` under the **testing root**. Point coordinates refer to the original 1600×900 image.

### KITTI Odometry

Download the **color image sequences** and the official ground-truth poses. Retain the original PNG names and both color cameras, since the training split uses left and right images.

| Path | Content |
| --- | --- |
| `data/kitti_odom/sequences/00/image_2/000000.png` | Left color images; sequences `00`–`10` |
| `data/kitti_odom/sequences/00/image_3/000000.png` | Right color images; sequences `00`–`08` used for training |
| `data/kitti_odom/poses/09.txt` | Official ground-truth trajectory for sequence 09 |
| `data/kitti_odom/poses/10.txt` | Official ground-truth trajectory for sequence 10 |

The split files in `splits/kitti_odom/` contain 36,671 training and 4,075 validation entries from sequences 00–08. Evaluation uses 1,590 consecutive image pairs for sequence 09 and 1,200 for sequence 10, producing 1,591 and 1,201 poses, respectively.

To use datasets stored elsewhere, set `NUSCENES_TRAIN_PATH`, `NUSCENES_TEST_PATH`, and `KITTI_ODOM_PATH` as shown below. No data need to be copied into the repository.

## Training

Prepare the initialization checkpoints listed in [weights/README.md](weights/README.md), then run:

```bash
# nuScenes
NUSCENES_TRAIN_PATH=/path/to/nuscenes/training bash scripts/train_nuscenes.sh

# KITTI Odometry
KITTI_ODOM_PATH=/path/to/kitti_odom bash scripts/train_kitti_odom.sh
```

You can adjust the training settings in the scripts to fit your GPU memory.

## Evaluation

Download the [pretrained models](#pretrained-models) and place them as described in [weights/README.md](weights/README.md), then run:

```bash
NUSCENES_TEST_PATH=/path/to/nuscenes/testing bash scripts/test_nuscenes.sh
KITTI_ODOM_PATH=/path/to/kitti_odom bash scripts/test_kitti_odom.sh
```

## Pretrained Models

Pretrained models are available on [Hugging Face](https://huggingface.co/mengtanZ/DiMoDE/tree/main). Extract the downloaded archives into `weights/`; see [weights/README.md](weights/README.md) for the directory layout.

| Model | Task | Input size | Required files |
| --- | --- | --- | --- |
| [DiMoDE (nuScenes)](https://huggingface.co/mengtanZ/DiMoDE/resolve/main/dimode-nuscenes.zip?download=true) | Depth | 640×384 | `weights/dimode-nuscenes/{encoder,depth}.pth` |
| [DiMoDE (KITTI Odometry)](https://huggingface.co/mengtanZ/DiMoDE/resolve/main/dimode-kitti-odom.zip?download=true) | Ego-motion | 640×192 | `weights/dimode-kitti-odom/{pose_encoder,pose}.pth` |

For KITTI Odometry training, download the [initialization weights](https://huggingface.co/mengtanZ/DiMoDE/resolve/main/initialization.zip?download=true) and extract them into `weights/` as well.

Performance of the pretrained models:

| Dataset | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ < 1.25 ↑ | δ < 1.25² ↑ | δ < 1.25³ ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nuScenes | 0.13894 | 1.47151 | 6.25395 | 0.22585 | 0.83556 | 0.93874 | 0.97270 |

| KITTI sequence | Translation (%) ↓ | Rotation (deg/100m) ↓ | ATE (m) ↓ | RPE (m) ↓ | RPE (deg) ↓ |
| --- | --- | --- | --- | --- | --- |
| 09 | 2.861 | 0.740 | 9.835 | 0.044 | 0.045 |
| 10 | 4.203 | 1.220 | 5.816 | 0.047 | 0.055 |

## Acknowledgements

Many thanks to [DCPI-Depth](https://github.com/mengtanZ/DCPI-Depth), [Lite-Mono](https://github.com/noahzn/Lite-Mono), [Monodepth2](https://github.com/nianticlabs/monodepth2), [RAFT](https://github.com/princeton-vl/RAFT), [SC-DepthV3](https://github.com/JiawangBian/sc_depth_pl), [Dynamo-Depth](https://github.com/YihongSun/Dynamo-Depth), and the [KITTI odometry evaluation toolbox](https://github.com/Huangying-Zhan/kitti-odom-eval).

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and upstream terms.

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{zhang2026dimode,
  title={Discriminately Treating Motion Components Evolves Joint Depth and Ego-Motion Learning},
  author={Zhang, Mengtan and Guo, Zizhan and Zhao, Hongbo and Feng, Yi and Xiong, Zuyi and Wang, Yue and Du, Shaoyi and Wang, Hanli and Fan, Rui},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  doi={10.1109/TPAMI.2026.3732069},
  publisher={IEEE}
}
```
