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

    # Build mesh
    mesh = dibr.build(src.depth, src.im)

    # Render ERP image
    R, t = make_rt(0, 0, 0, 1000, 0, 0)
    result = dibr.to_erp(mesh, R, t)

    # Display results
    comparison = np.vstack((src.im, result))
    show(comparison, show_width=900, title="DIBR Comparison")