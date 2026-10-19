from pathlib import Path
import numpy as np

from dibr import make_rt, create_dibr
from util.visualizer import ERPAnimationVisualizer
from dummy_panorama_tool.dummy_panorama_reader import DummyPanoramaReader


if __name__ == "__main__":
    # DIBR input data
    data_dir = Path("dummy_panorama")
    output_path = Path("output/erp_camera_walk.webp")

    # Animation Visualizer parameter
    fps = 10
    duration = 18
    move_distance = 1500.0

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

    # DIBR callback for visualizer
    def render_function(tx, ty, tz, yaw):
        R, t = make_rt(0, yaw, 0, tx, ty, tz)
        result = dibr.to_plane(mesh, camera_dst, camera_dst_size, R, t)
        return {"DIBR RESULT": result}

    # Make visualized video(webp)
    visualizer = ERPAnimationVisualizer(render_function, output_path, camera_dst_size, fps=fps, duration=duration, move_distance=move_distance)
    visualizer.save()