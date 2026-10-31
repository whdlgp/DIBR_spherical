import numpy as np
from abc import ABC, abstractmethod


class Mesh:
    """Triangle mesh of an ERP image (one vertex per pixel).

    Each backend defines its own subclass. A mesh can only be rendered by the backend that built it.
    """


class SphericalDIBR(ABC):
    """Common interface of the spherical DIBR backends."""

    @abstractmethod
    def build(self, depth: np.ndarray, image: np.ndarray, max_tri_stretch: float = 0.0) -> Mesh:
        """src ERP image → triangle mesh. Run once, then render with any R|t.

        Args:
            depth: src ERP depth map (radius), shape (H, W).
            image: src ERP image, shape (H, W, C).
            max_tri_stretch: Maximum allowed triangle stretch ratio. 0 disables filtering.

        Returns:
            Triangle mesh.
        """
        raise NotImplementedError

    @abstractmethod
    def to_erp(self, mesh: Mesh, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Mesh → dst ERP image. dst has the same size as src.

        Args:
            mesh: triangle mesh from build.
            R: rotation matrix, shape (3, 3).
            t: translation vector, shape (3, 1).

        Returns:
            dst ERP image, shape (H, W, C).
        """
        raise NotImplementedError

    @abstractmethod
    def to_plane(self, mesh: Mesh, camera: np.ndarray, camera_dst_size: tuple[int, int], R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Mesh → dst plane image.

        Args:
            mesh: triangle mesh from build.
            camera: dst camera intrinsic matrix, shape (3, 3).
            camera_dst_size: dst image size as (width, height).
            R: rotation matrix, shape (3, 3).
            t: translation vector, shape (3, 1).

        Returns:
            dst plane image, shape (height, width, C).
        """
        raise NotImplementedError


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
