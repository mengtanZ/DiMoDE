"""Train DiMoDE on nuScenes or KITTI odometry."""
from .options import Options


if __name__ == "__main__":
    opts = Options().parse()
    from .trainer import Trainer
    Trainer(opts).train()
