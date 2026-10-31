import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from .render_base import RenderScene, RenderSettings, Renderer
from .scene_builder import Scene


@dataclass
class NumpyPrimitive:
    intersect: object
    material: object
    specular: float = 0.0
    shininess: float = 1.0


@dataclass
class NumpyRenderScene(RenderScene):
    """Scene prepared for the numpy backend.

    Attributes:
        primitives: intersect and material functions of each primitive.
        light: point light position, shape (3,).
        room_min: room lower corner, shape (3,).
        room_max: room upper corner, shape (3,).
        settings: render parameters.
    """
    primitives: list
    light: np.ndarray
    room_min: np.ndarray
    room_max: np.ndarray
    settings: RenderSettings


def vec(*values):
    return np.array(values, dtype=float)


def hash_noise(x):
    value = np.sin(x * 12.9898) * 43758.5453
    return value - np.floor(value)


def value_noise(points):
    indices = np.floor(points)
    fraction = points - indices
    fraction = fraction * fraction * (3 - 2 * fraction)

    def corner(a, b, c):
        return hash_noise(indices[:, 0] + a + 57 * (indices[:, 1] + b) + 113 * (indices[:, 2] + c))

    x, y, z = fraction.T
    c00 = corner(0, 0, 0) * (1 - x) + corner(1, 0, 0) * x
    c10 = corner(0, 1, 0) * (1 - x) + corner(1, 1, 0) * x
    c01 = corner(0, 0, 1) * (1 - x) + corner(1, 0, 1) * x
    c11 = corner(0, 1, 1) * (1 - x) + corner(1, 1, 1) * x

    return (c00 * (1 - y) + c10 * y) * (1 - z) + (c01 * (1 - y) + c11 * y) * z


def fbm(points, octaves=3):
    result, amplitude = 0.0, 0.5

    for _ in range(octaves):
        result += amplitude * value_noise(points)
        points = points * 2.03
        amplitude *= 0.5

    return result / (1 - 0.5 ** octaves)


def safe_divisor(values):
    return np.where(np.abs(values) < 1e-8, 1e-8, values)


def hit_box(origins, directions, minimum, maximum):
    directions = safe_divisor(directions)
    t1, t2 = (minimum - origins) / directions, (maximum - origins) / directions
    near, far = np.minimum(t1, t2), np.maximum(t1, t2)
    t_min, t_max = near.max(axis=1), far.min(axis=1)
    valid = (t_max >= t_min) & (t_min > 1e-4)

    axes = near.argmax(axis=1)
    indices = np.arange(len(directions))
    normals = np.zeros_like(directions)
    normals[indices, axes] = -np.sign(directions[indices, axes])

    return np.where(valid, t_min, np.inf), normals


def hit_sphere(origins, directions, center, radius):
    offset = origins - center
    b = (directions * offset).sum(axis=1)
    discriminant = b * b - ((offset * offset).sum(axis=1) - radius * radius)
    distance = -b - np.sqrt(np.maximum(discriminant, 0))
    valid = (discriminant > 0) & (distance > 1e-4)
    normals = (origins + distance[:, None] * directions - center) / radius

    return np.where(valid, distance, np.inf), normals


def hit_cylinder(origins, directions, center_x, center_z, radius, y_min, y_max):
    ox, oz = origins[:, 0] - center_x, origins[:, 2] - center_z
    dx, dy, dz = directions.T

    a = dx * dx + dz * dz + 1e-12
    b = ox * dx + oz * dz
    c = ox * ox + oz * oz - radius * radius
    discriminant = b * b - a * c

    distance = (-b - np.sqrt(np.maximum(discriminant, 0))) / a
    y = origins[:, 1] + distance * dy
    valid = (discriminant > 0) & (distance > 1e-4) & (y >= y_min) & (y <= y_max)

    best = np.where(valid, distance, np.inf)
    normals = np.stack([ox + distance * dx, np.zeros_like(distance), oz + distance * dz], axis=1) / radius

    for cap_y, sign in ((y_max, 1.0), (y_min, -1.0)):
        cap_distance = (cap_y - origins[:, 1]) / safe_divisor(dy)
        x, z = ox + cap_distance * dx, oz + cap_distance * dz
        valid_cap = (cap_distance > 1e-4) & (x * x + z * z <= radius * radius) & (cap_distance < best)
        best = np.where(valid_cap, cap_distance, best)
        normals = np.where(valid_cap[:, None], vec(0, sign, 0), normals)

    return best, normals


def solid(color, noise=0.0, frequency=40.0):
    base_color = vec(*color)

    def material(points, normals):
        factor = 1 + noise * (value_noise(points * frequency) - 0.5) * 2
        return base_color * factor[:, None], np.zeros(len(points))

    return material


def wood_floor(points):
    x, z = points[:, 0], points[:, 2]
    plank_width = 0.18
    plank_id = np.floor(z / plank_width)
    offset = hash_noise(plank_id * 3.1) * 2.0
    segment = np.floor((x + offset) / 1.4)
    tone = 0.75 + 0.3 * hash_noise(plank_id * 7.7 + segment * 13.3)

    grain = fbm(np.stack([x * 2, z * 40, plank_id], axis=1))
    color = vec(0.62, 0.40, 0.20) * (tone * (0.75 + 0.5 * grain))[:, None]
    gap = (z / plank_width - plank_id < 0.03) | (((x + offset) / 1.4 - segment) * 1.4 < 0.012)

    return color * np.where(gap, 0.35, 1.0)[:, None]


def room_material(points, normals):
    count = len(points)
    x, y, z = points.T
    floor, ceiling = normals[:, 1] < -0.5, normals[:, 1] > 0.5
    wall = ~(floor | ceiling)
    noise = fbm(points * 3.0, 2)

    plaster = np.where((np.abs(normals[:, 0]) > 0.5)[:, None], vec(0.70, 0.78, 0.68), vec(0.90, 0.85, 0.74))
    plaster *= (0.9 + 0.15 * noise)[:, None]

    color = np.where(ceiling[:, None], vec(0.95, 0.95, 0.92) * (0.97 + 0.05 * noise)[:, None], plaster)
    color = np.where(floor[:, None], wood_floor(points), color)
    emission = np.zeros(count)

    baseboard = wall & (y > 1.38)
    color = np.where(baseboard[:, None], vec(0.92, 0.92, 0.9) * np.where(y > 1.385, 1.0, 0.6)[:, None], color)

    back_wall = normals[:, 2] < -0.5
    window = back_wall & (np.abs(x) < 1.2) & (y < 0.4) & (y > -1.0)
    frame = window & ((np.abs(x) > 1.12) | (y > 0.33) | (y < -0.93))
    mullion = window & ((np.abs(x) < 0.03) | (np.abs(np.abs(x) - 0.6) < 0.025) | (np.abs(y + 0.3) < 0.03))
    glass = window & ~frame & ~mullion

    t = np.clip((0.4 - y) / 1.4, 0, 1)
    sky = np.stack([0.80 - 0.25 * t, 0.88 - 0.13 * t, np.ones(count)], axis=1)
    sky += (0.18 * fbm(np.stack([x * 2, y * 3, np.zeros(count)], axis=1)))[:, None]

    color = np.where((frame | mullion)[:, None], vec(0.95, 0.95, 0.93), color)
    color = np.where(glass[:, None], np.clip(sky, 0, 1), color)
    emission = np.where(glass, 1.0, emission)

    front_wall = normals[:, 2] > 0.5
    door = front_wall & (np.abs(x) < 0.55) & (y > -0.5)
    u, v = (x + 0.55) / 1.1, (1.5 - y) / 2.0
    panel = (u > 0.12) & (u < 0.88) & (((v > 0.08) & (v < 0.45)) | ((v > 0.55) & (v < 0.92)))
    grain = (0.9 + 0.2 * value_noise(np.stack([x * 4, y * 30, z * 0], axis=1)))[:, None]
    door_color = np.where(panel[:, None], vec(0.45, 0.28, 0.14), vec(0.30, 0.18, 0.09)) * grain
    color = np.where(door[:, None], door_color, color)

    knob = door & ((x - 0.4) ** 2 + (y - 0.5) ** 2 < 0.04 ** 2)
    color = np.where(knob[:, None], vec(0.85, 0.65, 0.2), color)

    for z_min, z_max in ((-3.0, -1.5), (1.0, 2.5)):
        picture = (normals[:, 0] < -0.5) & (z > z_min) & (z < z_max) & (y < 0.2) & (y > -0.7)
        border = picture & ((z < z_min + 0.05) | (z > z_max - 0.05) | (y > 0.15) | (y < -0.65))
        u, v = (z - z_min) / (z_max - z_min), (0.2 - y) / 0.9
        phase = z_min * 2.0

        artwork = 0.5 + 0.5 * np.sin(np.stack([u * 9 + v * 3 + phase, v * 7 - u * 4 + 2 + phase, u * 5 + v * 8 + 4], axis=1))
        artwork *= (0.75 + 0.35 * fbm(np.stack([u * 6, v * 6, np.full(count, phase)], axis=1)))[:, None]

        color = np.where(picture[:, None], artwork, color)
        color = np.where(border[:, None], vec(0.25, 0.15, 0.07), color)

    return color, emission


def rug_material(points, normals):
    u, v = np.abs(points[:, 0]) / 1.6, np.abs(points[:, 2] - 1.8) / 1.5
    radius = np.maximum(u, v)
    band = (np.floor(radius * 8) % 2).astype(bool)
    color = np.where(band[:, None], vec(0.55, 0.12, 0.12), vec(0.90, 0.80, 0.55))
    color = np.where((radius > 0.88)[:, None], vec(0.10, 0.12, 0.35), color)
    return color * (0.85 + 0.3 * value_noise(points * 60))[:, None], np.zeros(len(points))


def wood_material(base):
    base_color = vec(*base)

    def material(points, normals):
        grain = fbm(np.stack([points[:, 0] * 3, points[:, 2] * 30, points[:, 1] * 30], axis=1))
        return base_color * (0.7 + 0.6 * grain)[:, None], np.zeros(len(points))

    return material


def marble_material(points, normals):
    value = 0.82 + 0.15 * np.sin(-points[:, 1] * 6 + 8 * fbm(points * 2))
    return np.stack([value, value, value * 1.03], axis=1), np.zeros(len(points))


def ball_material(points, normals):
    angle = np.arctan2(normals[:, 0], normals[:, 2])
    index = (np.floor((angle + np.pi) / (2 * np.pi) * 6) % 6).astype(int)
    palette = np.array([[0.9, 0.15, 0.15], [1, 1, 1], [0.1, 0.4, 0.9], [1, 1, 1], [0.95, 0.8, 0.1], [1, 1, 1]])
    color = np.where((np.abs(normals[:, 1]) > 0.92)[:, None], vec(1, 1, 1), palette[index])
    return color, np.zeros(len(points))


def crate_material(points, normals):
    y = -points[:, 1]
    row = np.floor(y / 0.12)
    fraction = y / 0.12 - row
    tone = 0.75 + 0.35 * hash_noise(row * 3.1)
    grain = 0.85 + 0.3 * value_noise(np.stack([points[:, 0] * 3, y * 25, points[:, 2] * 3], axis=1))
    color = vec(0.65, 0.47, 0.25) * (tone * grain)[:, None]
    return color * np.where(fraction < 0.06, 0.4, 1.0)[:, None], np.zeros(len(points))


def shelf_material(points, normals):
    count = len(points)
    wood = vec(0.35, 0.22, 0.10) * (0.8 + 0.4 * value_noise(points * 20))[:, None]
    y = 1.5 - points[:, 1]
    row = np.floor(y / 0.42)
    fraction = y / 0.42 - row
    book_id = np.floor(points[:, 2] / 0.05) + row * 31
    height = 0.55 + 0.4 * hash_noise(book_id * 1.7)
    in_book = (fraction > 0.07) & ((fraction - 0.07) / 0.93 < height)

    book_color = 0.2 + 0.7 * np.stack([hash_noise(book_id * 2.3), hash_noise(book_id * 5.1), hash_noise(book_id * 9.7)], axis=1)
    front = np.where((fraction < 0.07)[:, None], wood, np.where(in_book[:, None], book_color, vec(0.12, 0.08, 0.05)))

    return np.where((normals[:, 0] > 0.5)[:, None], front, wood), np.zeros(count)


def shade_material(points, normals):
    return vec(1.0, 0.88, 0.55) * (0.85 - 0.15 * normals[:, 1])[:, None], np.ones(len(points))


FIXED_MATERIALS = {"rug": rug_material, "marble": marble_material, "ball": ball_material, "crate": crate_material, "shelf": shelf_material, "shade": shade_material}


def build_intersect(kind, shape):
    if kind == "box":
        minimum, maximum = vec(*shape[:3]), vec(*shape[3:])
        return lambda o, d: hit_box(o, d, minimum, maximum)
    if kind == "sphere":
        center, radius = vec(*shape[:3]), shape[3]
        return lambda o, d: hit_sphere(o, d, center, radius)
    if kind == "cylinder":
        return lambda o, d: hit_cylinder(o, d, *shape)
    raise ValueError(f"Unknown primitive kind '{kind}'.")


def build_material(material):
    name, *params = material
    if name == "solid":
        return solid(params[:3], *params[3:])
    if name == "wood":
        return wood_material(params)
    if name in FIXED_MATERIALS:
        return FIXED_MATERIALS[name]
    raise ValueError(f"Unknown material '{name}'.")


def render_rays(rs, camera, directions):
    count = len(directions)
    origins = np.broadcast_to(camera, (count, 3))
    distance, normals = hit_room(rs, origins, directions)
    primitive_ids = np.zeros(count, dtype=int)

    for index, primitive in enumerate(rs.primitives, 1):
        hit_distance, hit_normals = primitive.intersect(origins, directions)
        closer = hit_distance < distance
        distance = np.where(closer, hit_distance, distance)
        normals = np.where(closer[:, None], hit_normals, normals)
        primitive_ids = np.where(closer, index, primitive_ids)

    points = origins + distance[:, None] * directions
    color = np.zeros((count, 3))
    emission = np.zeros(count)
    specular = np.zeros(count)
    shininess = np.ones(count)

    for index in range(len(rs.primitives) + 1):
        mask = primitive_ids == index

        if not mask.any():
            continue

        if index == 0:
            color[mask], emission[mask] = room_material(points[mask], normals[mask])
        else:
            primitive = rs.primitives[index - 1]
            color[mask], emission[mask] = primitive.material(points[mask], normals[mask])
            specular[mask] = primitive.specular
            shininess[mask] = primitive.shininess

    floor_mask = (primitive_ids == 0) & (normals[:, 1] < -0.5)
    specular[floor_mask] = 0.12
    shininess[floor_mask] = 25

    light_vector = rs.light - points
    light_distance = np.linalg.norm(light_vector, axis=1)
    light_direction = light_vector / np.maximum(light_distance[:, None], 1e-9)
    diffuse = np.clip((normals * light_direction).sum(axis=1), 0, None)

    indices = np.where(diffuse > 0)[0]
    shadow_origins = points[indices] + normals[indices] * rs.settings.shadow_bias
    shadow_directions = light_direction[indices]
    shadow_distances = light_distance[indices]
    blocked = np.zeros(len(indices), dtype=bool)

    for primitive in rs.primitives:
        blocked |= primitive.intersect(shadow_origins, shadow_directions)[0] < shadow_distances

    shadow = np.ones(count)
    shadow[indices[blocked]] = 0

    attenuation = rs.settings.light_intensity / (1 + rs.settings.light_falloff * light_distance**2)
    half_vector = light_direction - directions
    half_vector /= np.linalg.norm(half_vector, axis=1, keepdims=True) + 1e-9
    highlight = specular * np.clip((normals * half_vector).sum(axis=1), 0, None)**shininess
    ambient = rs.settings.ambient + 0.12 * (0.5 - 0.5 * normals[:, 1])

    result = color * (ambient + rs.settings.direct_light * diffuse * attenuation * shadow)[:, None]
    result += (highlight * shadow * attenuation)[:, None]
    result = result * (1 - emission[:, None]) + color * emission[:, None]

    return np.clip(result, 0, 1)**(1 / rs.settings.gamma), distance


def hit_room(rs, origins, directions):
    dd = safe_divisor(directions)
    distances = np.maximum((rs.room_min - origins) / dd, (rs.room_max - origins) / dd)
    axes = distances.argmin(axis=1)
    indices = np.arange(len(directions))
    normals = np.zeros_like(directions)
    normals[indices, axes] = -np.sign(directions[indices, axes])
    return distances[indices, axes], normals


class NumpyRenderer(Renderer):
    """Dummy panorama renderer by numpy array operations (CPU ray tracing).

    Args:
        chunk_size: number of rays traced at once (memory vs. speed).
    """

    def __init__(self, chunk_size: int = 250_000) -> None:
        self.chunk_size = chunk_size

    def build(self, scene: Scene, settings: RenderSettings) -> NumpyRenderScene:
        primitives = [NumpyPrimitive(build_intersect(p.kind, p.shape), build_material(p.material), p.specular, p.shininess) for p in scene.primitives]
        return NumpyRenderScene(primitives, np.asarray(scene.light, dtype=float), np.asarray(scene.room_min, dtype=float), np.asarray(scene.room_max, dtype=float), settings)

    def to_erp(self, render_scene: NumpyRenderScene, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check(render_scene)
        settings = render_scene.settings
        camera = np.asarray(t, dtype=float).reshape(3)
        u = (np.arange(settings.width) + 0.5) / settings.width
        v = (np.arange(settings.height) + 0.5) / settings.height
        longitude, latitude = np.meshgrid((u - 0.5) * 2 * np.pi, (0.5 - v) * np.pi)

        directions = np.stack([np.cos(latitude) * np.sin(longitude), -np.sin(latitude), np.cos(latitude) * np.cos(longitude)], axis=-1).reshape(-1, 3)
        directions = directions @ np.asarray(R, dtype=float)
        rgb = np.zeros((len(directions), 3))
        depth = np.zeros(len(directions))

        for start in tqdm(range(0, len(directions), self.chunk_size), desc="Rendering", unit="chunk"):
            end = min(start + self.chunk_size, len(directions))
            rgb[start:end], depth[start:end] = render_rays(render_scene, camera, directions[start:end])

        return rgb.reshape(settings.height, settings.width, 3), depth.reshape(settings.height, settings.width).astype(np.float32)

    @staticmethod
    def _check(render_scene: RenderScene) -> None:
        if not isinstance(render_scene, NumpyRenderScene):
            raise TypeError(f"Expected NumpyRenderScene, got {type(render_scene).__name__}.")
