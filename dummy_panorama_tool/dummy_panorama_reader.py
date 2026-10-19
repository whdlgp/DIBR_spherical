import cv2
import numpy as np
from pathlib import Path

from util.show import depth_to_show, show


class DummyPanoramaReader:
    def __init__(self, data_dir: Path, image_name="room_rgb.png", depth_name="room_depth.png"):
        data_dir = Path(data_dir)
        self.data_path = str(data_dir)

        self.im = self._read_image(data_dir / image_name)
        self.depth = self._read_depth(data_dir / depth_name)

        if self.im.shape[:2] != self.depth.shape:
            raise ValueError(f"image {self.im.shape[:2]} and depth {self.depth.shape} sizes differ")

        self.height, self.width = self.depth.shape

    def resize(self, scale):
        if scale <= 0:
            raise ValueError("scale must be positive")

        width = round(self.width * scale)
        height = round(self.height * scale)
        size = (width, height)

        self.im = cv2.resize(self.im, size, interpolation=cv2.INTER_AREA)
        self.depth = cv2.resize(self.depth, size, interpolation=cv2.INTER_NEAREST)

        self.width = width
        self.height = height

    def print_info(self):
        print(f"Path of data: {self.data_path}")
        print(f"Resolution (WxH): {self.width} x {self.height}")
        print(f"im: shape={self.im.shape}, dtype={self.im.dtype}")

        valid = self.depth[np.isfinite(self.depth) & (self.depth > 0)]
        print(f"depth: shape={self.depth.shape}, dtype={self.depth.dtype}")
        print(f"depth range [mm]: {valid.min()} ~ {valid.max()}")
        print(f"depth valid ratio: {valid.size / self.depth.size:.4f}")

    def show_data(self, show_width=1280):
        show_img = cv2.vconcat([self.im, depth_to_show(self.depth)])
        show(show_img, show_width, "Panorama Data")

    @staticmethod
    def _read_image(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(path)

        return image

    @staticmethod
    def _read_depth(path):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if depth is None:
            raise FileNotFoundError(path)

        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError(f"depth must be 1-channel uint16 PNG, got {depth.dtype} {depth.shape}")

        return depth.astype(np.float32)
