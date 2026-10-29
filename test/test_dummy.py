from pathlib import Path
import numpy as np

from dibr import create_dibr
from util.show import show
from dummy_panorama_tool.dummy_panorama_creater import RenderConfig, load_scenes
from dummy_panorama_tool.dummy_panorama_reader import DummyPanoramaReader


if __name__ == "__main__":
    # Scenes (same file used to render the dummy panoramas)
    config = RenderConfig()
    scenes = {name: (R, t) for name, R, t in load_scenes(config.scene_file)}

    # Source scene: origin (R = I, t = 0)
    R_src, t_src = scenes.pop("origin")
    assert np.allclose(R_src, np.eye(3)) and np.allclose(t_src, 0)

    # Create DIBR backend
    dibr = create_dibr()

    # Read input data
    src = DummyPanoramaReader(config.output_dir / "origin")

    # t must be in the same unit as depth (mm or m)
    unit = 1000.0 if src.depth.max() > 100 else 1.0
    print("depth unit:", "mm" if unit == 1000.0 else "m")

    # Build mesh once
    mesh = dibr.build(src.depth, src.im)

    # Render each camera and compare with the ground truth
    for name, (R, t) in scenes.items():
        result = dibr.to_erp(mesh, R, t * unit)
        truth = DummyPanoramaReader(config.output_dir / name)

        comparison = np.vstack((truth.im, result))
        show(comparison, show_width=500, title=f"{name}: ground truth (top) / DIBR (bottom)")