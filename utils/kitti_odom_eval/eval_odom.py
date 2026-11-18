# Copyright (C) Huangying Zhan 2019. All rights reserved.
# Modified for DiMoDE: explicit data/output paths and a guarded CLI entry point.
import argparse
from pathlib import Path
from kitti_odometry import KittiEvalOdom


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KITTI odometry trajectory evaluation')
    parser.add_argument('--result', required=True, help='directory containing 09.txt and 10.txt')
    parser.add_argument('--gt', required=True, help='official KITTI poses directory')
    parser.add_argument('--align', choices=['scale', 'scale_7dof', '7dof', '6dof'], default='7dof')
    parser.add_argument('--seqs', nargs='+', type=int, choices=[9, 10], default=[9, 10])
    args = parser.parse_args()
    for seq in args.seqs:
        for directory in [args.gt, args.result]:
            file = Path(directory) / f'{seq:02d}.txt'
            if not file.is_file():
                raise FileNotFoundError(file)
    KittiEvalOdom().eval(args.gt, args.result, alignment=args.align, seqs=args.seqs)
