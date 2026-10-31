import numpy as np
from dataclasses import dataclass

from .base import Mesh, SphericalDIBR


@dataclass
class NumpyMesh(Mesh):
    """Triangle mesh of an ERP image (one vertex per pixel).

    Attributes:
        world_coord: vertex 3D coordinates, shape (3, H * W).
        faces: triangle vertex indices, shape (F, 3).
        color: vertex colors, shape (H * W, C).
        height: src ERP image height.
        width: src ERP image width.
    """
    world_coord: np.ndarray
    faces: np.ndarray
    color: np.ndarray
    height: int
    width: int


def unproject_erp(depth: np.ndarray) -> np.ndarray:
    """ERP image pixels → 3D coordinates.

    Args:
        depth: src ERP depth map (radius), shape (H, W).

    Returns:
        3D coordinates, shape (3, H * W).
    """
    height, width = depth.shape

    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    lon, lat = np.meshgrid((u - 0.5) * 2 * np.pi, (0.5 - v) * np.pi)

    world_coord = np.empty((3, height * width), dtype=depth.dtype)
    world_coord[0] = (np.cos(lat) * np.sin(lon) * depth).reshape(-1)
    world_coord[1] = (-np.sin(lat) * depth).reshape(-1)
    world_coord[2] = (np.cos(lat) * np.cos(lon) * depth).reshape(-1)

    return world_coord


def make_faces(depth: np.ndarray) -> np.ndarray:
    """Connect each pixel with its neighbors by two right triangles per cell.

    The last column is connected to the first column (ERP wrap-around).
    A triangle with a non-finite or non-positive depth vertex is degenerated
    (all three indices set to its first vertex).

    Args:
        depth: src ERP depth map (radius), shape (H, W).

    Returns:
        Triangle vertex indices (int32), shape (2 * (H - 1) * W, 3).
    """
    height, width = depth.shape

    valid = np.isfinite(depth) & (depth > 0)

    row, col = np.meshgrid(np.arange(height - 1), np.arange(width), indexing="ij")
    col_next = (col + 1) % width

    a = (row * width + col).reshape(-1)
    b = (row * width + col_next).reshape(-1)
    c = ((row + 1) * width + col).reshape(-1)
    d = ((row + 1) * width + col_next).reshape(-1)

    valid = valid.reshape(-1)
    ok = np.stack([valid[a] & valid[c] & valid[b], valid[b] & valid[c] & valid[d]], axis=1).reshape(-1)

    faces = np.stack([a, c, b, b, c, d], axis=1).reshape(-1, 3)
    faces[~ok] = faces[~ok][:, :1]

    return faces.astype(np.int32)


def make_color(image: np.ndarray) -> np.ndarray:
    """Make vertex colors from an image (one vertex per pixel).

    Args:
        image: src image, shape (H, W, C).

    Returns:
        Vertex colors, shape (H * W, C).
    """
    return image.reshape(-1, image.shape[2])


def filter_faces(world_coord: np.ndarray, faces: np.ndarray, max_tri_stretch: float) -> np.ndarray:
    """Remove triangles exceeding the maximum stretch ratio.

    Args:
        world_coord: vertex 3D coordinates, shape (3, N).
        faces: triangle vertex indices, shape (F, 3).
        max_tri_stretch: maximum allowed triangle stretch ratio.

    Returns:
        Filtered triangle vertex indices, shape (F, 3).
    """
    tri = world_coord[:, faces].transpose(1, 2, 0)

    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 1]

    longest = np.maximum.reduce([
        np.linalg.norm(e0, axis=1),
        np.linalg.norm(e1, axis=1),
        np.linalg.norm(e2, axis=1)
    ])

    area2 = np.linalg.norm(np.cross(e0, e1), axis=1)
    stretch = longest ** 2 / np.maximum(area2, 1e-8)

    bad = (area2 <= 1e-8) | (stretch > max_tri_stretch)
    faces[bad] = faces[bad, :1]

    return faces


def apply_rt(world_coord: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply R|t to 3D coordinates.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        R: rotation matrix, shape (3, 3).
        t: translation vector, shape (3, 1).

    Returns:
        Transformed 3D coordinates, shape (3, N).
    """
    return R @ (world_coord - t)


def project_erp(world_coord: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D coordinates → ERP pixel coordinates.

    Unlike a point splat, no point is dropped and coordinates are not
    rounded. Points with a non-finite or zero radius get depth 0.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        height: dst ERP image height.
        width: dst ERP image width.

    Returns:
        x: pixel x coordinates (float), shape (N,).
        y: pixel y coordinates (float), shape (N,).
        depth: depth (radius) of each point, shape (N,).
    """
    radius = np.linalg.norm(world_coord, axis=0)

    valid = np.isfinite(radius) & (radius > 0)
    world_coord = np.where(valid, world_coord, 0)
    radius = np.where(valid, radius, 1)

    lon = np.arctan2(world_coord[0], world_coord[2])
    lat = np.arcsin(np.clip(-world_coord[1] / radius, -1, 1))

    x = (lon / (2 * np.pi) + 0.5) * width - 0.5
    y = (0.5 - lat / np.pi) * height - 0.5

    return x, y, np.where(valid, radius, 0)


def project_plane(world_coord: np.ndarray, camera: np.ndarray, min_depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D coordinates → plane (pinhole) pixel coordinates.

    Unlike a point splat, no point is dropped and coordinates are not
    rounded. Points with a non-finite coordinate or z <= min_depth get depth 0.

    Args:
        world_coord: 3D coordinates, shape (3, N).
        camera: camera intrinsic matrix, shape (3, 3).
        min_depth: lower bound of z (exclusive).

    Returns:
        x: pixel x coordinates (float), shape (N,).
        y: pixel y coordinates (float), shape (N,).
        depth: depth (z) of each point, shape (N,).
    """
    fx = camera[0, 0]
    fy = camera[1, 1]
    ox = camera[0, 2]
    oy = camera[1, 2]

    valid = np.isfinite(world_coord).all(axis=0) & (world_coord[2] > min_depth)
    world_coord = np.where(valid, world_coord, 1)

    depth = world_coord[2]
    x = world_coord[0] * fx / depth + ox
    y = world_coord[1] * fy / depth + oy

    return x, y, np.where(valid, depth, 0)


def rasterize(x: np.ndarray, y: np.ndarray, depth: np.ndarray, faces: np.ndarray, height: int, width: int, wrap: bool) -> tuple[np.ndarray, np.ndarray]:
    """Find the nearest triangle of each dst pixel (z-buffer).

    Pixel centers are at integer coordinates. Triangles with a vertex of
    depth <= 0 are dropped. Depth is interpolated linearly in screen space.

    Args:
        x: vertex pixel x coordinates, shape (N,).
        y: vertex pixel y coordinates, shape (N,).
        depth: vertex depths, shape (N,).
        faces: triangle vertex indices, shape (F, 3).
        height: dst image height.
        width: dst image width.
        wrap: Wrap x around the image width (ERP). Triangles crossing the seam are unwrapped.

    Returns:
        face_idx: index of the nearest triangle of each pixel (-1 if none), shape (height, width).
        weight: barycentric weights in that triangle, shape (height, width, 3).
    """
    tri_x = x[faces]
    tri_y = y[faces]
    tri_z = depth[faces]

    valid = (tri_z > 0).all(axis=1) & np.isfinite(tri_x).all(axis=1) & np.isfinite(tri_y).all(axis=1)
    face_id = np.flatnonzero(valid)
    tri_x, tri_y, tri_z = tri_x[valid], tri_y[valid], tri_z[valid]

    if wrap:
        seam = tri_x.max(axis=1) - tri_x.min(axis=1) > width / 2
        tri_x = np.where(seam[:, None] & (tri_x < width / 2), tri_x + width, tri_x)

    x_limit = 2 * width - 1 if wrap else width - 1
    x_min = np.clip(np.ceil(tri_x.min(axis=1)), 0, x_limit + 1).astype(np.int64)
    x_max = np.clip(np.floor(tri_x.max(axis=1)), -1, x_limit).astype(np.int64)
    y_min = np.clip(np.ceil(tri_y.min(axis=1)), 0, height).astype(np.int64)
    y_max = np.clip(np.floor(tri_y.max(axis=1)), -1, height - 1).astype(np.int64)

    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1
    det = (tri_y[:, 1] - tri_y[:, 2]) * (tri_x[:, 0] - tri_x[:, 2]) + (tri_x[:, 2] - tri_x[:, 1]) * (tri_y[:, 0] - tri_y[:, 2])

    keep = (box_w > 0) & (box_h > 0) & (np.abs(det) > 1e-12)
    face_id, tri_x, tri_y, tri_z, det = face_id[keep], tri_x[keep], tri_y[keep], tri_z[keep], det[keep]
    x_min, y_min, box_w, box_h = x_min[keep], y_min[keep], box_w[keep], box_h[keep]

    zbuf = np.full(height * width, np.inf)
    face_idx = np.full(height * width, -1, dtype=np.int32)
    weight = np.zeros((height * width, 3), dtype=np.float32)

    eps = 1e-9
    area = box_w * box_h
    ends = np.cumsum(area)

    start = 0
    while start < len(face_id):
        base = ends[start - 1] if start > 0 else 0
        stop = max(int(np.searchsorted(ends, base + (1 << 20), side="right")), start + 1)
        part = slice(start, stop)

        count = area[part]
        tri = np.repeat(np.arange(stop - start), count)
        local = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)

        px = x_min[part][tri] + local % box_w[part][tri]
        py = y_min[part][tri] + local // box_w[part][tri]

        vx = tri_x[part][tri]
        vy = tri_y[part][tri]
        vz = tri_z[part][tri]
        inv_det = 1 / det[part][tri]

        l0 = ((vy[:, 1] - vy[:, 2]) * (px - vx[:, 2]) + (vx[:, 2] - vx[:, 1]) * (py - vy[:, 2])) * inv_det
        l1 = ((vy[:, 2] - vy[:, 0]) * (px - vx[:, 2]) + (vx[:, 0] - vx[:, 2]) * (py - vy[:, 2])) * inv_det
        l2 = 1 - l0 - l1

        inside = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
        tri, px, py, vz = tri[inside], px[inside], py[inside], vz[inside]
        l0, l1, l2 = l0[inside], l1[inside], l2[inside]

        target = py * width + px % width
        frag_depth = l0 * vz[:, 0] + l1 * vz[:, 1] + l2 * vz[:, 2]

        np.minimum.at(zbuf, target, frag_depth)
        winner = frag_depth == zbuf[target]

        face_idx[target[winner]] = face_id[part][tri[winner]]
        weight[target[winner]] = np.stack([l0[winner], l1[winner], l2[winner]], axis=1)

        start = stop

    return face_idx.reshape(height, width), weight.reshape(height, width, 3)


def render_image(color: np.ndarray, faces: np.ndarray, face_idx: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Interpolate vertex colors at dst pixels.

    Args:
        color: vertex colors, shape (N, C).
        faces: triangle vertex indices, shape (F, 3).
        face_idx: index of the nearest triangle of each pixel (-1 if none), shape (H, W).
        weight: barycentric weights in that triangle, shape (H, W, 3).

    Returns:
        dst image, shape (H, W, C).
    """
    height, width = face_idx.shape

    hit = face_idx >= 0
    face = faces[face_idx[hit]]

    image = np.zeros((height, width, color.shape[1]), dtype=color.dtype)
    image[hit] = np.round(np.einsum("nk,nkc->nc", weight[hit], color[face])).astype(color.dtype)

    return image


class NumpyDIBR(SphericalDIBR):
    """Spherical DIBR by numpy array operations."""

    def build(self, depth: np.ndarray, image: np.ndarray, max_tri_stretch: float = 0.0) -> NumpyMesh:
        world_coord = unproject_erp(depth)
        faces = make_faces(depth)

        if max_tri_stretch > 0:
            faces = filter_faces(world_coord, faces, max_tri_stretch)

        color = make_color(image)

        return NumpyMesh(world_coord, faces, color, depth.shape[0], depth.shape[1])

    def to_erp(self, mesh: NumpyMesh, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check(mesh)

        world_coord = apply_rt(mesh.world_coord, R, t)
        x, y, depth = project_erp(world_coord, mesh.height, mesh.width)
        face_idx, weight = rasterize(x, y, depth, mesh.faces, mesh.height, mesh.width, True)

        return render_image(mesh.color, mesh.faces, face_idx, weight)

    def to_plane(self, mesh: NumpyMesh, camera: np.ndarray, camera_dst_size: tuple[int, int], R: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check(mesh)
        width, height = camera_dst_size

        world_coord = apply_rt(mesh.world_coord, R, t)
        x, y, depth = project_plane(world_coord, camera, 1e-6)
        face_idx, weight = rasterize(x, y, depth, mesh.faces, height, width, False)

        return render_image(mesh.color, mesh.faces, face_idx, weight)

    @staticmethod
    def _check(mesh: Mesh) -> None:
        if not isinstance(mesh, NumpyMesh):
            raise TypeError(f"Expected NumpyMesh, got {type(mesh).__name__}.")
