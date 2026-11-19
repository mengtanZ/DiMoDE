"""Options for the released nuScenes and KITTI odometry experiments."""
import argparse
from pathlib import Path


class Options:
    def __init__(self):
        p = self.parser = argparse.ArgumentParser(description="DiMoDE options")
        p.add_argument('--data_path', required=True)
        p.add_argument('--dataset', choices=['nuscenes', 'kitti_odom'], default='nuscenes')
        p.add_argument('--split', choices=['nuscenes', 'kitti_odom'], default='nuscenes')
        p.add_argument('--split_dir', default=str(Path(__file__).resolve().parent / 'splits'))
        p.add_argument('--log_dir', default='tmp')
        p.add_argument('--model_name', default='dimode-nuscenes')
        p.add_argument('--device_id', type=int, default=0)
        p.add_argument('--no_cuda', action='store_true')
        p.add_argument('--num_workers', type=int, default=12)
        p.add_argument('--model', choices=['lite-mono', 'lite-mono-8m'], default='lite-mono')
        p.add_argument('--num_layers', type=int, choices=[18], default=18)
        p.add_argument('--height', type=int, default=384)
        p.add_argument('--width', type=int, default=640)
        p.add_argument('--png', action='store_true')
        p.add_argument('--scales', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2])
        p.add_argument('--frame_ids', nargs='+', type=int, default=[0, -1, 1])
        p.add_argument('--min_depth', type=float, default=0.1)
        p.add_argument('--max_depth', type=float, default=80.0)
        p.add_argument('--num_epochs', type=int, default=30)
        p.add_argument('--batch_size', type=int, default=6)
        p.add_argument('--lr', nargs=6, type=float, default=[1e-4, 5e-6, 31, 1e-4, 1e-5, 31],
                       help='depth/flow max_lr min_lr period, then pose max_lr min_lr period')
        p.add_argument('--weight_decay', type=float, default=1e-2)
        p.add_argument('--drop_path', type=float, default=0.2)
        p.add_argument('--disparity_smoothness', type=float, default=1e-3)
        p.add_argument('--v1_multiscale', action='store_true')
        p.add_argument('--avg_reprojection', action='store_true')
        p.add_argument('--disable_automasking', action='store_true')
        p.add_argument('--no_ssim', action='store_true')
        p.add_argument('--weights_init', choices=['pretrained', 'scratch'], default='pretrained')
        p.add_argument('--mypretrain', help='Lite-Mono ImageNet checkpoint')
        p.add_argument('--pose_pretrain', default=str(Path(__file__).resolve().parent / 'weights/resnet18-f37072fd.pth'))
        p.add_argument('--flow_weights', default=str(Path(__file__).resolve().parent / 'weights/raft-kitti.pth'))
        p.add_argument('--mixed_precision', action='store_true')
        p.add_argument('--version', choices=['v1'], default='v1')
        p.add_argument('--model_weights', type=float, default=0.01)
        p.add_argument('--weights', nargs=4, type=float, default=[1.0, 0.25, 0.1, 0.05])
        p.add_argument('--psedo_depth', default='none', choices=['none', 'leres_depth', 'psedo_depth'],
                       help='optional pseudo-depth subdirectory; original spelling retained')
        p.add_argument('--psedo_weights', type=float, default=0.01)
        p.add_argument('--normal_ranking_loss', action='store_true')
        p.add_argument('--train_posenet', action='store_true')
        p.add_argument('--load_weights_folder')
        p.add_argument('--resume_epoch', type=int, default=-1,
                       help='zero-based epoch; overrides weights_<N> folder inference')
        p.add_argument('--models_to_load', nargs='+', default=['encoder', 'depth', 'pose_encoder', 'pose'],
                       choices=['encoder', 'depth', 'pose_encoder', 'pose', 'flow'])
        p.add_argument('--log_frequency', type=int, default=250)
        p.add_argument('--save_frequency', type=int, default=1)
        p.add_argument('--eval_frequency', type=int, default=0)
        p.add_argument('--eval_dataset', choices=['none', 'kitti_odom'], default='none')
        p.add_argument('--eval_data_path', help='KITTI odometry root for evaluation during training')
        p.add_argument('--eval_split', choices=['nuscenes', 'odom_9', 'odom_10'], default='nuscenes')
        p.add_argument('--eval_batch_size', type=int, default=12)
        p.add_argument('--eval_out_dir', help='defaults to eval/<model_name>')
        p.add_argument('--gt_path', help='nuScenes point-depth directory or KITTI poses directory')
        p.add_argument('--gt_height', type=int, default=900)
        p.add_argument('--gt_width', type=int, default=1600)
        p.add_argument('--disable_median_scaling', action='store_true')
        p.add_argument('--pred_depth_scale_factor', type=float, default=1.0)
        p.add_argument('--save_pred_disps', action='store_true')
        p.add_argument('--post_process', action='store_true')
        p.set_defaults(use_stereo=False, predictive_mask=False, pose_model_input='pairs',
                       pose_model_type='separate_resnet', posenet='mono2', version='v1', small=False)

    def parse(self):
        opt = self.parser.parse_args()
        if opt.height % 32 or opt.width % 32:
            self.parser.error('height and width must be multiples of 32')
        if opt.batch_size < 1 or opt.eval_batch_size < 1 or opt.num_workers < 0:
            self.parser.error('batch sizes must be positive and num_workers nonnegative')
        if opt.save_frequency < 1 or opt.log_frequency < 1:
            self.parser.error('save_frequency and log_frequency must be positive')
        if opt.frame_ids != [0, -1, 1] or 0 not in opt.scales:
            self.parser.error('released training requires frame_ids 0 -1 1 and scale 0')
        if opt.normal_ranking_loss and opt.psedo_depth == 'none':
            self.parser.error('normal_ranking_loss requires pseudo-depth maps')
        return opt
