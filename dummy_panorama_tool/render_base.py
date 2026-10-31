import importlib
import warnings
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from .scene_builder import Scene


@dataclass
class RenderSettings:
    """Render parameters.

    Attributes:
        width: dst ERP image width.
        height: dst ERP image height.
        gamma: output gamma.
        ambient: ambient light level.
        direct_light: direct light gain.
        light_intensity: point light intensity.
        light_falloff: point light quadratic falloff.
        shadow_bias: shadow ray origin offset along the surface normal.
    """
    width: int = 2048
    height: int = 1024
    gamma: float = 1.8
    ambient: float = 0.22
    direct_light: float = 1.15
    light_intensity: float = 1.25
    light_falloff: float = 0.05
    shadow_bias: float = 0.002


class RenderScene:
    """Scene prepared by a backend (the counterpart of Mesh in DIBR).

    Each backend defines its own subclass. A render scene can only be rendered by the backend that built it.
    """


class Renderer(ABC):
    """Common interface of the dummy panorama render backends."""

    @abstractmethod
    def build(self, scene: Scene, settings: RenderSettings) -> RenderScene:
        """Scene → backend render scene. Run once, then render with any R|t.

        Args:
            scene: room with primitives and a light.
            settings: render parameters.

        Returns:
            Render scene.
        """
        raise NotImplementedError

    @abstractmethod
    def to_erp(self, render_scene: RenderScene, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Render scene → ERP image and depth map. The camera must be inside the room (see Scene.check_camera).

        Args:
            render_scene: render scene from build.
            R: rotation matrix (world to camera), shape (3, 3).
            t: camera position in the room, shape (3, 1).

        Returns:
            rgb: ERP image in [0, 1], shape (H, W, 3).
            depth: ERP depth map (radius, meters, float32), shape (H, W).
        """
        raise NotImplementedError


def create_renderer(backend: str | Sequence[str] = ("wgpu", "numpy")) -> Renderer:
    """Create a render backend.

    A backend is the module render_<name>.py of this package, which defines a subclass of Renderer.

    Args:
        backend: Backend name, or names in priority order. With several names, an unavailable backend is skipped with a warning.

    Returns:
        Render backend.
    """
    names = [backend] if isinstance(backend, str) else list(backend)
    for i, name in enumerate(names):
        try:
            module = importlib.import_module(f".render_{name}", __package__)
            cls = next(c for c in vars(module).values() if isinstance(c, type) and issubclass(c, Renderer) and c.__module__ == module.__name__)
            return cls()
        except (ImportError, RuntimeError) as e:
            if i == len(names) - 1:
                raise
            warnings.warn(f"Backend '{name}' is unavailable ({e}), trying '{names[i + 1]}'.")
