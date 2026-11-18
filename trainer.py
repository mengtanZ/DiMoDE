# Adapted for the DiMoDE release; upstream attribution and terms: THIRD_PARTY_NOTICES.md.
from __future__ import absolute_import, division, print_function

import os
import time
import matplotlib

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")

import torch.optim as optim
from kornia.geometry import depth_to_normals
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

import cv2
from tqdm import tqdm

from utils.layers import *

import datasets
import networks
from networks.RAFT.raft import RAFT
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

from losses.normal_ranking_loss import EdgeguidedNormalRankingLoss

import re

import random
import numpy as np

from utils.utils import *
from utils.runtime import get_device

seed = 640
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.start_epoch = 0
        self.total_target_epochs = self.opt.num_epochs

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = get_device(self.opt)

        self.zero_tensor = (torch.tensor(0.0, device=self.device, requires_grad=False),
                            torch.tensor(0.0, device=self.device, requires_grad=False),
                            torch.tensor(0.0, device=self.device, requires_grad=False))

        self.num_scales = len(self.opt.scales)
        self.frame_ids = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.LiteMono(
            model=self.opt.model, drop_path_rate=self.opt.drop_path,
            width=self.opt.width, height=self.opt.height)
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models_pose["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames, pretrained_path=self.opt.pose_pretrain)
        self.models_pose["pose"] = networks.PoseDecoder(
            self.models_pose["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.models_pose["pose_encoder"].to(self.device)
        self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())
        self.models_pose["pose"].to(self.device)
        self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())
        self.models["flow"] = torch.nn.DataParallel(
            RAFT(self.opt), device_ids=[] if self.opt.no_cuda else [self.opt.device_id])
        self.models["flow"].to(self.device)
        self.parameters_to_train += list(self.models["flow"].parameters())
        self.models["flow"].load_state_dict(torch.load(self.opt.flow_weights, map_location=self.device), strict=True)

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3],
                                                    weight_decay=self.opt.weight_decay)

        self.model_lr_scheduler = ChainedScheduler(
            self.model_optimizer,
            T_0=int(self.opt.lr[2]),
            T_mul=1,
            eta_min=self.opt.lr[1],
            last_epoch=-1,
            max_lr=self.opt.lr[0],
            warmup_steps=0,
            gamma=0.9
        )
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.opt.lr[5]),
            T_mul=1,
            eta_min=self.opt.lr[4],
            last_epoch=-1,
            max_lr=self.opt.lr[3],
            warmup_steps=0,
            gamma=0.9
        )

        if self.opt.mypretrain is not None:
            self.load_pretrain()

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.train_posenet == True:
            for param in self.models["encoder"].parameters():
                param.requires_grad = False
            for param in self.models["depth"].parameters():
                param.requires_grad = False
            for param in self.models["flow"].parameters():
                param.requires_grad = False

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        datasets_dict = {"kitti_odom": datasets.KITTIOdomDataset,
                         "nuscenes": datasets.PreprocessedDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(self.opt.split_dir, self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        remain_epochs = max(self.total_target_epochs - self.start_epoch, 0)
        self.num_total_steps = num_train_samples // self.opt.batch_size * remain_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, psedo_depth=self.opt.psedo_depth)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, psedo_depth=self.opt.psedo_depth)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        splits_dir = self.opt.split_dir
        self.eval_datasets = [] if self.opt.eval_dataset == "none" else [self.opt.eval_dataset]
        if "kitti_odom" in self.eval_datasets:
            kitti_odom_eval_data_path = self.opt.eval_data_path or self.opt.data_path
            assert os.path.isdir(kitti_odom_eval_data_path), \
                "Cannot find KITTI odometry data path for evaluation: {}".format(kitti_odom_eval_data_path)
            self.kitti_odom_loaders = {}
            for sequence_id in [9, 10]:
                filenames = readlines(os.path.join(
                    splits_dir, "kitti_odom", "test_files_{:02d}.txt".format(sequence_id)))
                print("Evaluation on KITTI odometry sequence {:02d} with {} image pairs".format(
                    sequence_id, len(filenames)))
                odom_dataset = datasets.KITTIOdomDataset(
                    kitti_odom_eval_data_path, filenames, self.opt.height, self.opt.width,
                    [0, 1], 4, is_train=False, img_ext='.png')
                self.kitti_odom_loaders[sequence_id] = DataLoader(
                    odom_dataset, batch_size=1, shuffle=False,
                    num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.pixel2flow = Pixel2Flow(self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)
        self.rectify_pixel = Rectify_Pixel(self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)
        self.backproject_depth_ps = BackprojectDepth_Ps(self.opt.batch_size, self.opt.height, self.opt.width).to(
            self.device)
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()
        for m in self.models_pose.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
        for m in self.models_pose.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = self.start_epoch
        self.step = 0
        self.start_time = time.time()
        if self.start_epoch >= self.total_target_epochs:
            print("No training will run: start_epoch ({}) >= num_epochs ({})".format(
                self.start_epoch, self.total_target_epochs))
            return

        for self.epoch in range(self.start_epoch, self.total_target_epochs):
            self.run_epoch()
            self.run_eval("epoch_{}".format(self.epoch))
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_eval(self, eval_tag):
        try:
            for eval_dataset in self.eval_datasets:
                if eval_dataset == "kitti_odom":
                    self.test_kitti_odom(eval_tag)
        finally:
            torch.cuda.empty_cache()
            self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        self.model_lr_scheduler.step(self.epoch)
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step(self.epoch)

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Non-finite loss at epoch {self.epoch}, step {self.step}")
            losses["loss"].backward()
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data, losses["D_loss"].cpu().data,
                              losses["t3_loss"].cpu().data, losses["p_loss"].cpu().data)

                self.log("train", inputs, outputs, losses)

            self.step += 1
            if self.opt.eval_frequency > 0 and self.step % self.opt.eval_frequency == 0:
                del inputs, outputs, losses
                torch.cuda.empty_cache()
                self.run_eval("step_{}".format(self.step))

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        outputs.update(self.predict_poses(inputs, features))

        with torch.no_grad():
            features_1 = self.models["encoder"](inputs["color_aug", 1, 0])
            outputs_1 = self.models["depth"](features_1)
        outputs[("disp", 1, 0)] = outputs_1[("disp", 0)].detach()

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        outputs["flow"] = self.models["flow"](inputs[("color", 0, 0)] * 255, inputs[("color", 1, 0)] * 255, iters=12,
                                              test_mode=True)[1]

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models_pose["pose"](pose_inputs)

                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

                mask_1 = torch.tensor([0, 0, 1], dtype=translation.dtype, device=translation.device)
                mask_2 = torch.tensor([1, 1, 0], dtype=translation.dtype, device=translation.device)
                res1_translation = translation[:, 0] - translation[:, 0] * mask_1 # tangential translation
                res2_translation = translation[:, 0] - translation[:, 0] * mask_2 # radial translation

                outputs[("cam_T_cam_remove_rt12", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], res1_translation[:, 0], invert=(f_i < 0))
                outputs[("cam_T_cam_remove_rt3", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], res2_translation[:, 0], invert=(f_i < 0))

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                    res1_T = outputs[("cam_T_cam_remove_rt12", 0, frame_id)]  # R, t1,2 , w/o t3
                    res2_T = outputs[("cam_T_cam_remove_rt3", 0, frame_id)]  # R, t3, w/o t1,2

                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                if frame_id == 1 and scale == 0:

                    disp_1 = outputs[("disp", 1, 0)]
                    disp_1 = F.interpolate(
                        disp_1, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    _, depth_1 = disp_to_depth(disp_1, self.opt.min_depth, self.opt.max_depth)
                    outputs[("depth", 1, 0)] = depth_1

                    outputs[("sample", frame_id, 0)] = pix_coords
                    valid_warp_mask = (
                        (outputs[("sample", frame_id, 0)][..., 0] >= -1.0) &
                        (outputs[("sample", frame_id, 0)][..., 0] <= 1.0) &
                        (outputs[("sample", frame_id, 0)][..., 1] >= -1.0) &
                        (outputs[("sample", frame_id, 0)][..., 1] <= 1.0)
                    ).unsqueeze(1).float()
                    outputs[("valid_warp_mask", frame_id, 0)] = valid_warp_mask

                    outputs[("depth_s2t_raw", frame_id, 0)] = F.grid_sample(
                        outputs[("depth", 1, 0)],
                        outputs[("sample", frame_id, 0)],
                        padding_mode="border", align_corners=True)
                    outputs[("depth_s2t", frame_id, 0)] = outputs[("depth_s2t_raw", frame_id, 0)]

                    flow = outputs["flow"].clone().detach()

                    cam_points_s = self.backproject_depth_ps(
                        outputs[("depth_s2t_raw", frame_id, 0)],
                        inputs[("inv_K", source_scale)],
                        flow
                    )
                    assert outputs[("depth_s2t_raw", frame_id, 0)].shape == outputs[("depth", 1, 0)].shape
                    assert flow.dim() == 4 and flow.size(1) == 2
                    assert cam_points_s.dim() == 3 and cam_points_s.size(1) == 4

                    cam_T_cam = outputs[("cam_T_cam", 0, frame_id)]  # t->s
                    R_t2s = cam_T_cam[:, :3, :3]
                    t_t2s = cam_T_cam[:, :3, 3:4]
                    R_s2t = R_t2s.transpose(1, 2)
                    t_s2t = -torch.matmul(R_s2t, t_t2s)

                    t_rad = torch.zeros_like(t_s2t)
                    t_rad[:, 2:3, :] = t_s2t[:, 2:3, :]
                    t_tan = t_s2t - t_rad

                    points_s = cam_points_s[:, :3, :]
                    points_derot = torch.matmul(R_s2t, points_s)
                    points_pla = points_derot + t_tan  # plane alignment: remove t_rad, keep t_tan
                    points_axi = points_derot + t_rad  # axis alignment: remove t_tan, keep t_rad

                    ones = torch.ones_like(cam_points_s[:, 3:4, :])
                    cam_points_pla = torch.cat([points_pla, ones], 1)
                    cam_points_axi = torch.cat([points_axi, ones], 1)

                    identity_T = torch.eye(4, device=cam_points_s.device, dtype=cam_points_s.dtype).unsqueeze(0)
                    identity_T = identity_T.repeat(cam_points_s.size(0), 1, 1)

                    pix_coords_pla = self.project_3d[source_scale](
                        cam_points_pla, inputs[("K", source_scale)], identity_T)
                    pix_coords_axi = self.project_3d[source_scale](
                        cam_points_axi, inputs[("K", source_scale)], identity_T)
                    assert pix_coords_pla.dim() == 4 and pix_coords_pla.size(-1) == 2
                    assert pix_coords_axi.dim() == 4 and pix_coords_axi.size(-1) == 2

                    valid_pla_mask = (
                        (pix_coords_pla[..., 0] >= -1.0) &
                        (pix_coords_pla[..., 0] <= 1.0) &
                        (pix_coords_pla[..., 1] >= -1.0) &
                        (pix_coords_pla[..., 1] <= 1.0)
                    ).unsqueeze(1).float()
                    valid_axi_mask = (
                        (pix_coords_axi[..., 0] >= -1.0) &
                        (pix_coords_axi[..., 0] <= 1.0) &
                        (pix_coords_axi[..., 1] >= -1.0) &
                        (pix_coords_axi[..., 1] <= 1.0)
                    ).unsqueeze(1).float()

                    outputs[("sample_pla", frame_id, 0)] = pix_coords_pla
                    outputs[("sample_axi", frame_id, 0)] = pix_coords_axi
                    outputs[("valid_pla_mask", frame_id, 0)] = valid_pla_mask
                    outputs[("valid_axi_mask", frame_id, 0)] = valid_axi_mask

                    final_pla_mask = outputs[("valid_warp_mask", frame_id, 0)] * valid_pla_mask
                    final_axi_mask = outputs[("valid_warp_mask", frame_id, 0)] * valid_axi_mask
                    outputs[("final_pla_mask", frame_id, 0)] = final_pla_mask
                    outputs[("final_axi_mask", frame_id, 0)] = final_axi_mask
                    outputs["flow_coa"] = self.pixel2flow(pix_coords_pla)
                    outputs["flow_cop"] = self.pixel2flow(pix_coords_axi)

                    if self.opt.version == 'v4':
                        rigid_pix_coords = pix_coords.clone()
                        outputs[("rig_flow")] = self.pixel2flow(rigid_pix_coords)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).to(self.device))
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss

            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        depth = outputs[("depth", 0, 0)]
        flow = outputs["flow"]
        t = -outputs[("cam_T_cam", 0, 1)][:, 0:3, 3]

        if self.opt.version == 'v1':
            model_loss, loss_D, loss_t, loss_p = self.compute_model_loss_v1(depth, t, flow, outputs["flow_coa"],
                                                                            outputs["flow_cop"])
            losses["D_loss"] = loss_D
            losses["t3_loss"] = loss_t
            losses["p_loss"] = loss_p

        total_loss += self.opt.model_weights * model_loss

        if self.epoch < 1:
            total_loss -= self.opt.model_weights * model_loss

        if self.opt.psedo_depth != "none":
            if not inputs["psedo_depth"].shape == outputs[("depth", 0, 0)].shape:
                inputs["psedo_depth"] = F.interpolate(
                    inputs["psedo_depth"], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            tgt_normal = depth_to_normals(outputs[("depth", 0, 0)], inputs[('K', 0)][:, :3, :3])
            tgt_pseudo_normal = depth_to_normals(inputs['psedo_depth'], inputs[('K', 0)][:, :3, :3])
            loss_normal = (tgt_normal - tgt_pseudo_normal).abs().mean()
            total_loss += self.opt.psedo_weights * loss_normal  # 0.1 for kitti and 0.01 for ddad
            if self.opt.normal_ranking_loss:  # for ddad exp
                normal_ranking_loss = EdgeguidedNormalRankingLoss().to(self.device)
                loss_normalrank = normal_ranking_loss(inputs['psedo_depth'], outputs[("depth", 0, 0)], tgt_normal,
                                                      tgt_pseudo_normal)
                total_loss += self.opt.psedo_weights * loss_normalrank

        losses["model_loss"] = model_loss
        losses["loss"] = total_loss

        return losses

    def compute_model_loss_v1(self, D, t, F, F_coa, F_cop):

        B, _, height, width = D.size()
        center_y = (height - 1) / 2
        center_x = (width - 1) / 2

        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing='ij')
        y_coords_centered = y_coords - center_y
        x_coords_centered = x_coords - center_x

        pix = torch.stack([x_coords_centered, y_coords_centered], axis=0)
        p = pix.unsqueeze(0).expand(B, -1, -1, -1)

        inner_p_F = compute_inner_product(p, F)
        inner_p_p = compute_inner_product(p, p)
        unit_rec1_F = compute_unit_vector(F_coa)
        unit_rec2_F = compute_unit_vector(F_cop)
        unit_p = compute_unit_vector(p)

        cos_pF = compute_inner_product(unit_p, unit_rec1_F)
        cos_pF = cos_pF.mean()
        var_p = torch.var(unit_rec2_F[:, 0, :, :], dim=(1, 2))

        t = t[:, 2:3, None, None].expand(-1, -1, height, width)

        pro_D = t * (inner_p_p + inner_p_F) / (inner_p_F + 1e-5)
        pro_tz = D * inner_p_F / (inner_p_F + inner_p_p + 1e-5)

        loss_1 = torch.abs((pro_D - D) / (D + 1e-5))
        loss_2 = torch.abs((pro_tz - t) / (t + 1e-5))
        loss_3 = torch.acos(cos_pF)
        loss_4 = var_p.mean()

        if self.epoch < 15:
            loss_1 = loss_1[loss_1 < 1].mean()
            loss_2 = loss_2[loss_2 < 1].mean()
            loss = (
                self.opt.weights[0] * loss_1
                + self.opt.weights[2] * loss_3
                + self.opt.weights[3] * loss_4)
        else:
            mask = (loss_1 < 1) & (loss_2 < 1)
            loss_1 = loss_1[mask].mean()
            loss_2 = loss_2[mask].mean()

            loss = (
                self.opt.weights[0] * loss_1
                + self.opt.weights[1] * loss_2
                + self.opt.weights[2] * loss_3
                + self.opt.weights[3] * loss_4)

        return loss, loss_1, loss_2, loss_3 + loss_4

    def test_kitti_odom(self, eval_tag):
        self.set_eval()

        result_dir = os.path.join("eval", self.opt.model_name, eval_tag)
        os.makedirs(result_dir, exist_ok=True)
        result_txt = os.path.join(result_dir, "result.txt")
        if os.path.isfile(result_txt):
            os.remove(result_txt)

        with torch.inference_mode():
            for sequence_id, dataloader in self.kitti_odom_loaders.items():
                global_pose = np.eye(4)
                poses = [global_pose[0:3, :].reshape(1, 12)]
                total_time = 0

                print("-> Computing KITTI odometry pose predictions for sequence {:02d}".format(sequence_id))
                for inputs in tqdm(dataloader):
                    for key, ipt in inputs.items():
                        inputs[key] = ipt.to(self.device)

                    start_time = time.time()
                    if self.opt.posenet == "scipad":
                        axisangle, translation = self.models_pose["posenet"](
                            inputs[("color_aug", 0, 0)], inputs[("color_aug", 1, 0)])
                    else:
                        all_color_aug = torch.cat(
                            [inputs[("color_aug", i, 0)] for i in [0, 1]], 1)
                        pose_features = [self.models_pose["pose_encoder"](all_color_aug)]
                        axisangle, translation = self.models_pose["pose"](pose_features)
                    total_time += time.time() - start_time

                    pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
                    pose_mat = pose.squeeze(0).cpu().numpy()
                    global_pose = global_pose @ np.linalg.inv(pose_mat)
                    poses.append(global_pose[0:3, :].reshape(1, 12))

                poses = np.concatenate(poses, axis=0)
                np.savetxt(
                    os.path.join(result_dir, "{:02d}.txt".format(sequence_id)),
                    poses,
                    delimiter=' ',
                    fmt='%1.8e')

                num_iters = max(len(dataloader), 1)
                avg_time = total_time / num_iters
                print("Sequence {:02d} average inference time per frame: {:.6f} seconds".format(
                    sequence_id, avg_time))
                if avg_time > 0:
                    print("Sequence {:02d} estimated FPS: {:.2f}".format(sequence_id, 1.0 / avg_time))

        gt_dir = self.opt.gt_path or os.path.join(
            self.opt.eval_data_path or self.opt.data_path, "poses")
        from utils.kitti_odom_eval.kitti_odometry import KittiEvalOdom
        KittiEvalOdom().eval(gt_dir, result_dir, alignment="7dof", seqs=[9, 10])

    def log_time(self, batch_idx, duration, loss, D_loss, t_loss, p_loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | loss_D: {:.5f} | loss_t3: {:.5f} | loss_p: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss, D_loss, t_loss, p_loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        to_save["start_epoch"] = self.start_epoch
        to_save["total_target_epochs"] = self.total_target_epochs

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, folder_name=None):
        """Save model weights to disk
        """
        if folder_name is None:
            folder_name = "weights_{}".format(self.epoch)
        save_folder = os.path.join(self.log_path, "models", folder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        model_dict = self.models["encoder"].state_dict()
        pretrained_dict = torch.load(path, map_location=self.device)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        folder_name = os.path.basename(os.path.normpath(self.opt.load_weights_folder))
        matched = re.fullmatch(r"weights_(\d+)", folder_name)
        if self.opt.resume_epoch >= 0:
            self.start_epoch = self.opt.resume_epoch
            print("Resume epoch set from option: {}".format(self.start_epoch))
        elif matched is not None:
            self.start_epoch = int(matched.group(1)) + 1
            print("Resume epoch inferred from folder name: {}".format(self.start_epoch))
        else:
            raise ValueError(
                "Cannot infer resume epoch from load_weights_folder '{}'. "
                "Expected folder name like 'weights_<N>', or pass --resume_epoch explicitly.".format(
                    self.opt.load_weights_folder)
            )

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

            if n in ['pose_encoder', 'pose']:
                model_dict = self.models_pose[n].state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models_pose[n].load_state_dict(model_dict)
            else:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        if self.opt.train_posenet == True:
            print("do not load Adam weights")
        else:
            optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
            optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
            if os.path.isfile(optimizer_pose_load_path) and os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
                self.model_optimizer.load_state_dict(optimizer_dict)
                optimizer_pose_dict = torch.load(optimizer_pose_load_path, map_location=self.device)
                self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")
