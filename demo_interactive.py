from pathlib import Path
import numpy as np

from dibr import create_dibr
from dummy_panorama_tool.dummy_panorama_reader import DummyPanoramaReader
from util.interactive_viewer import InteractiveViewer


if __name__ == "__main__":
    # DIBR input data
    data_dir = Path("dummy_panorama")

    # Interactive viewer parameter
    fps = 30

    # Create DIBR backend
    dibr = create_dibr()

    # Read input data
    src = DummyPanoramaReader(data_dir)

    camera_dst = np.array([
        [600.0,   0.0, 399.5],
        [  0.0, 600.0, 299.5],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float32)
    camera_dst_size = (800, 600)

    # Build mesh once
    mesh = dibr.build(src.depth, src.im)

    # DIBR callback for interactive viewer
    def render_function(R, t):
        return dibr.to_plane(mesh, camera_dst, camera_dst_size, R, t)

    # Make interactive viewer
    viewer = InteractiveViewer(render_function, camera_dst_size, fps=fps)
    viewer.run()
