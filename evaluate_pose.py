"""Predict complete trajectories for KITTI odometry sequences 09 and 10."""
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import networks
from datasets import KITTIOdomDataset
from options import Options
from utils.layers import transformation_from_parameters
from utils.runtime import get_device, output_directory
from utils.utils import readlines


def evaluate(opt):
    if opt.dataset != 'kitti_odom' or opt.eval_split not in ['odom_9', 'odom_10']:
        raise ValueError('Pose evaluation in this release supports KITTI odometry 09/10 only.')
    device = get_device(opt)
    sequence = int(opt.eval_split.split('_')[1])
    filenames = readlines(str(Path(opt.split_dir) / 'kitti_odom' / f'test_files_{sequence:02d}.txt'))
    if not filenames:
        raise ValueError('The test split is empty.')
    dataset = KITTIOdomDataset(opt.data_path, filenames, opt.height, opt.width, [0, 1], 1,
                              is_train=False, img_ext='.png' if opt.png else '.jpg')
    # Every listed frame must have its successor; never silently duplicate test frames.
    for line in filenames:
        folder, frame, side = line.split()
        for offset in [0, 1]:
            image = Path(dataset.get_image_path(folder, int(frame) + offset, side))
            if not image.is_file():
                raise FileNotFoundError(f'Required KITTI image is missing: {image}')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers,
                        pin_memory=device.type == 'cuda')
    weights = Path(opt.load_weights_folder)
    encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    decoder = networks.PoseDecoder(encoder.num_ch_enc, 1, 2)
    encoder.load_state_dict(torch.load(weights / 'pose_encoder.pth', map_location='cpu'))
    decoder.load_state_dict(torch.load(weights / 'pose.pth', map_location='cpu'))
    encoder.to(device).eval()
    decoder.to(device).eval()
    global_pose = np.eye(4)
    poses = [global_pose[:3].reshape(12)]
    print(f'Predicting {len(dataset)} image pairs for KITTI {sequence:02d} on {device}')
    start = time.perf_counter()
    with torch.inference_mode():
        for inputs in tqdm(loader):
            color = torch.cat([inputs['color_aug', i, 0] for i in [0, 1]], 1).to(device)
            axisangle, translation = decoder([encoder(color)])
            pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
            global_pose = global_pose @ np.linalg.inv(pose[0].cpu().numpy())
            poses.append(global_pose[:3].reshape(12))
    poses = np.asarray(poses)
    if not np.isfinite(poses).all():
        raise RuntimeError('Pose inference produced non-finite values.')
    output = output_directory(opt) / f'{sequence:02d}.txt'
    np.savetxt(output, poses, delimiter=' ', fmt='%1.8e')
    print(f'Saved {len(poses)} poses to {output}; elapsed {time.perf_counter() - start:.2f}s')
    return poses


if __name__ == '__main__':
    evaluate(Options().parse())
