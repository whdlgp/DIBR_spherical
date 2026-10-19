from pathlib import Path
import numpy as np

from dibr import make_rt, create_dibr
from util.show import show
from dummy_panorama_tool.dummy_panorama_reader import DummyPanoramaReader


if __name__ == "__main__":
    # Create DIBR backend
    dibr = create_dibr()

    # Read input data
    src = DummyPanoramaReader(Path("dummy_panorama"))

    # Destination camera
    camera_dst = np.array([
        [600.0,   0.0, 399.5],
        [  0.0, 600.0, 299.5],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float32)
    camera_dst_size = (800, 600)

    # Build mesh
    mesh = dibr.build(src.depth, src.im)

    # Render from target camera pose
    R, t = make_rt(0, 0, 0, 1000, 0, 0)
    result = dibr.to_plane(mesh, camera_dst, camera_dst_size, R, t)

    # Display result
    show(result, show_width=800, title="DIBR Result")