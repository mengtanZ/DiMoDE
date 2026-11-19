"""KITTI odometry color-image loader (Monodepth2-style split format)."""
import os
import numpy as np
from PIL import Image
from .mono_dataset import MonoDataset


class KITTIOdomDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = np.array([[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        return False

    def get_image_path(self, folder, frame_index, side):
        return os.path.join(self.data_path, "sequences", f"{int(folder):02d}",
                            f"image_{self.side_map[side]}", f"{frame_index:06d}{self.img_ext}")

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        return color.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if do_flip else color
