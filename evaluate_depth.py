"""Evaluate the released nuScenes depth model with the original point-depth protocol."""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import networks
from options import Options
from utils.layers import disp_to_depth
from utils.runtime import get_device, output_directory
from utils.utils import readlines

cv2.setNumThreads(0)


def compute_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    return (np.mean(np.abs(gt - pred) / gt), np.mean((gt - pred) ** 2 / gt),
            np.sqrt(np.mean((gt - pred) ** 2)),
            np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)),
            (thresh < 1.25).mean(), (thresh < 1.25 ** 2).mean(), (thresh < 1.25 ** 3).mean())


def batch_post_process_disparity(left, right):
    _, h, w = left.shape
    mean = 0.5 * (left + right)
    x, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    left_mask = (1 - np.clip(20 * (x - 0.05), 0, 1))[None]
    right_mask = left_mask[:, :, ::-1]
    return right_mask * left + left_mask * right + (1 - left_mask - right_mask) * mean


def evaluate(opt):
    if opt.dataset != 'nuscenes' or opt.eval_split != 'nuscenes':
        raise ValueError('Depth evaluation in this release supports nuScenes only.')
    device = get_device(opt)
    filenames = readlines(str(Path(opt.split_dir) / 'nuscenes/test_files.txt'))
    if not filenames:
        raise ValueError('The test split is empty.')
    gt_path = Path(opt.gt_path or Path(opt.data_path) / 'depth')
    gt_files = [gt_path / f'{int(line.split()[1]):06d}.npy' for line in filenames]
    for line, gt_file in zip(filenames, gt_files):
        folder, frame, _ = line.split()
        image = Path(opt.data_path) / folder / f'{int(frame):06d}.jpg'
        if not image.is_file() or not gt_file.is_file():
            raise FileNotFoundError(f'Required nuScenes test pair is missing: {image}, {gt_file}')
    weights = Path(opt.load_weights_folder)
    encoder_state = torch.load(weights / 'encoder.pth', map_location='cpu')
    decoder_state = torch.load(weights / 'depth.pth', map_location='cpu')
    height, width = int(encoder_state['height']), int(encoder_state['width'])
    encoder = networks.LiteMono(model=opt.model, height=height, width=width)
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=opt.scales)
    encoder.load_state_dict({k: v for k, v in encoder_state.items() if k not in ['height', 'width', 'use_stereo']})
    decoder.load_state_dict(decoder_state)
    encoder.to(device).eval()
    decoder.to(device).eval()
    dataset = datasets.PreprocessedDataset(opt.data_path, filenames, height, width, [0], 1,
                                           is_train=False, img_ext='.jpg')
    loader = DataLoader(dataset, batch_size=opt.eval_batch_size, shuffle=False,
                        num_workers=opt.num_workers, pin_memory=device.type == 'cuda')
    errors, ratios, predictions = [], [], []
    index = 0
    print(f'Evaluating {len(dataset)} nuScenes images at {width}x{height} on {device}')
    with torch.inference_mode():
        for data in tqdm(loader):
            color = data['color', 0, 0].to(device)
            if opt.post_process:
                color = torch.cat([color, torch.flip(color, [3])])
            output = decoder(encoder(color))
            disp, _ = disp_to_depth(output['disp', 0], opt.min_depth, opt.max_depth)
            disp = disp.cpu()[:, 0].numpy()
            if opt.post_process:
                half = len(disp) // 2
                disp = batch_post_process_disparity(disp[:half], disp[half:, :, ::-1])
            if opt.save_pred_disps:
                predictions.append(disp)
            for disparity in disp:
                points = np.load(gt_files[index])
                if points.ndim != 2 or points.shape[1] != 3:
                    raise ValueError(f'{gt_files[index]} must have shape (N, 3): x, y, depth.')
                x, y, gt = points.T
                valid = ((x >= 0) & (x < opt.gt_width) & (y >= 0) & (y < opt.gt_height)
                         & (gt > 1e-3) & (gt < 75))
                if not valid.any():
                    raise ValueError(f'No valid points in {gt_files[index]}')
                prediction = 1 / cv2.resize(disparity, (opt.gt_width, opt.gt_height))
                prediction = prediction[y[valid].astype(int), x[valid].astype(int)]
                prediction *= opt.pred_depth_scale_factor
                gt = gt[valid]
                if not opt.disable_median_scaling:
                    ratio = np.median(gt) / np.median(prediction)
                    ratios.append(ratio)
                    prediction *= ratio
                # Keep the original nuScenes protocol: no extra post-scaling clipping.
                errors.append(compute_errors(gt, prediction))
                index += 1
    mean = np.asarray(errors).mean(0)
    if not np.isfinite(mean).all():
        raise RuntimeError('Depth evaluation produced non-finite metrics.')
    header = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    print(' | '.join(header))
    print(' | '.join(f'{value:.6f}' for value in mean))
    if ratios:
        med = np.median(ratios)
        print(f'Scaling ratios: median={med:.3f}, normalized std={np.std(ratios / med):.3f}')
    output_dir = output_directory(opt)
    with (output_dir / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow(mean.tolist())
    if opt.save_pred_disps:
        np.save(output_dir / 'disps_nuscenes.npy', np.concatenate(predictions))
    print(f'Results saved to {output_dir}')
    return mean


if __name__ == '__main__':
    evaluate(Options().parse())
