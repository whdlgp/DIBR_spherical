
import shutil
import tomllib
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from matplotlib import colormaps

from .render_base import RenderSettings, create_renderer
from .scene_builder import create_scene


@dataclass
class RenderConfig:
    """Dummy panorama generation parameters.

    Attributes:
        scene_file: camera poses (TOML). Copied from scenes_example.toml if it does not exist.
        backend: render backend name, or names in priority order.
        output_dir: output directory. Each camera is saved in a subdirectory of its name.
        depth_scale: depth PNG scale, stored value = depth [m] * depth_scale (uint16).
        settings: render parameters (resolution and shading constants).
    """
    scene_file: Path = Path(__file__).parent / "scenes.toml"
    backend: tuple = ("wgpu", "numpy")
    output_dir: Path = Path("dummy_panorama")
    depth_scale: float = 1000.0
    settings: RenderSettings = field(default_factory=RenderSettings)


def make_rt(theta_x: float, theta_y: float, theta_z: float, tx: float, ty: float, tz: float) -> tuple[np.ndarray, np.ndarray]:
    """Make rotation matrix and translation vector.

    Args:
        theta_x: rotation about the x axis (radians).
        theta_y: rotation about the y axis (radians).
        theta_z: rotation about the z axis (radians).
        tx: translation along x.
        ty: translation along y.
        tz: translation along z.

    Returns:
        R (3, 3) and t (3, 1).
    """
    rot_x = np.array([
        [1, 0              ,  0              ],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x),  np.cos(theta_x)]
    ])

    rot_y = np.array([
        [ np.cos(theta_y), 0, np.sin(theta_y)],
        [0               , 1, 0              ],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    rot_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0              ,  0              , 1]
    ])

    R = rot_z @ rot_y @ rot_x
    t = np.array([[tx], [ty], [tz]], dtype=np.float64)

    return R, t


SCENE_KEYS = ("theta_x_deg", "theta_y_deg", "theta_z_deg", "tx", "ty", "tz")


def load_scenes(path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load camera poses from a TOML file. A missing file is copied from <name>_example.toml next to it.

    Args:
        path: TOML file of [[scene]] tables, see scenes_example.toml.

    Returns:
        List of (name, R, t), made by make_rt.
    """
    path = Path(path)
    example = path.with_name(f"{path.stem}_example{path.suffix}")
    if not path.exists():
        shutil.copyfile(example, path)
        print(f"{path.name} not found, copied from {example.name}")
    with open(path, "rb") as file:
        data = tomllib.load(file)
    unknown = set(data) - {"scene"}
    if unknown:
        raise ValueError(f"Unknown top-level keys in {path}: {sorted(unknown)}")

    scenes = []
    names = set()
    for index, entry in enumerate(data.get("scene", [])):
        entry = dict(entry)
        name = entry.pop("name", None)
        if not isinstance(name, str) or not name or name == ".." or Path(name).name != name:
            raise ValueError(f"scene #{index}: 'name' must be a plain folder name, got {name!r}")
        if name in names:
            raise ValueError(f"scene '{name}': duplicate name")
        unknown = set(entry) - set(SCENE_KEYS)
        if unknown:
            raise ValueError(f"scene '{name}': unknown keys {sorted(unknown)}")

        values = [entry.get(key, 0.0) for key in SCENE_KEYS]
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            raise ValueError(f"scene '{name}': values must be numbers")
        R, t = make_rt(*np.deg2rad(values[:3]), *values[3:])
        names.add(name)
        scenes.append((name, R, t))

    if not scenes:
        raise ValueError(f"No [[scene]] in {path}")

    return scenes


def save_outputs(rgb: np.ndarray, depth: np.ndarray, output_dir: Path, depth_scale: float) -> None:
    """Save room_rgb.png, room_depth.png (uint16) and room_depth_vis.png.

    Args:
        rgb: ERP image in [0, 1], shape (H, W, 3).
        depth: ERP depth map (radius, meters), shape (H, W).
        output_dir: output directory.
        depth_scale: stored depth value = depth [m] * depth_scale.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(output_dir / "room_rgb.png")
    depth_uint16 = np.clip(np.round(depth * depth_scale), 0, 65535).astype(np.uint16)
    Image.fromarray(depth_uint16).save(output_dir / "room_depth.png")

    normalized_depth = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-9)
    try:
        depth_visualization = (colormaps["turbo"](1 - normalized_depth)[..., :3] * 255).astype(np.uint8)
    except ImportError:
        depth_visualization = np.repeat((255 * (1 - normalized_depth))[..., None].astype(np.uint8), 3, axis=2)
    Image.fromarray(depth_visualization).save(output_dir / "room_depth_vis.png")
    print("depth range [m]:", float(depth.min()), float(depth.max()))


def main() -> None:
    config = RenderConfig(backend="numpy")
    scene = create_scene()
    cameras = load_scenes(config.scene_file)
    for _, _, t in cameras:
        scene.check_camera(t)

    renderer = create_renderer(config.backend)
    print(f"Renderer backend: {type(renderer).__name__}")
    render_scene = renderer.build(scene, config.settings)
    for name, R, t in cameras:
        rgb, depth = renderer.to_erp(render_scene, R, t)
        save_outputs(rgb, depth, config.output_dir / name, config.depth_scale)


if __name__ == "__main__":
    main()
