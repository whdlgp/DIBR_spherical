
from dataclasses import dataclass, field, replace
from pathlib import Path
import shutil
import tomllib
from tqdm import tqdm

import numpy as np
from PIL import Image
from matplotlib import colormaps


@dataclass
class RenderConfig:
    width: int = 2048
    height: int = 1024
    scene_file: Path = Path(__file__).parent / "scenes.toml"
    light: tuple = (0.0, -1.25, 0.5)
    room_min: tuple = (-3.5, -1.5, -4.5)
    room_max: tuple = (3.5, 1.5, 4.5)
    chunk_size: int = 250_000
    output_dir: Path = Path("dummy_panorama")
    gamma: float = 1.8
    ambient: float = 0.22
    direct_light: float = 1.15
    light_intensity: float = 1.25
    light_falloff: float = 0.05
    shadow_bias: float = 0.002
    depth_scale: float = 1000.0


@dataclass
class Primitive:
    intersect: object
    material: object
    specular: float = 0.0
    shininess: float = 1.0


@dataclass
class Scene:
    primitives: list = field(default_factory=list)

    def add_box(self, minimum, maximum, material, specular=0.0, shininess=1.0):
        minimum, maximum = vec(*minimum), vec(*maximum)
        self.primitives.append(Primitive(
            intersect=lambda o, d: hit_box(o, d, minimum, maximum),
            material=material,
            specular=specular,
            shininess=shininess,
        ))

    def add_sphere(self, center, radius, material, specular=0.0, shininess=1.0):
        center = vec(*center)
        self.primitives.append(Primitive(
            intersect=lambda o, d: hit_sphere(o, d, center, radius),
            material=material,
            specular=specular,
            shininess=shininess,
        ))

    def add_cylinder(self, center_x, center_z, radius, y_min, y_max, material, specular=0.0, shininess=1.0):
        self.primitives.append(Primitive(
            intersect=lambda o, d: hit_cylinder(o, d, center_x, center_z, radius, y_min, y_max),
            material=material,
            specular=specular,
            shininess=shininess,
        ))


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


def create_scene():
    scene = Scene()
    floor = 1.5

    scene.add_box((-1.6, floor - 0.02, 0.3), (1.6, floor, 3.3), rug_material)
    scene.add_box((-0.8, 0.75, 1.0), (0.8, 0.81, 2.0), wood_material((0.45, 0.28, 0.14)), 0.2, 30)

    for x in (-0.78, 0.72):
        for z in (1.02, 1.92):
            scene.add_box((x, 0.81, z), (x + 0.06, floor, z + 0.06), wood_material((0.40, 0.25, 0.12)))

    scene.add_sphere((0.25, 0.61, 1.55), 0.14, solid((0.85, 0.1, 0.1)), 0.7, 80)
    scene.add_cylinder(-0.35, 1.4, 0.07, 0.45, 0.75, solid((0.15, 0.35, 0.8)), 0.5, 50)
    scene.add_box((-0.1, 0.70, 1.2), (0.15, 0.75, 1.45), solid((0.15, 0.55, 0.25), 0.1))

    sofa = solid((0.15, 0.5, 0.55), 0.15)
    scene.add_box((2.2, 1.12, 0.4), (3.5, floor, 3.1), sofa)
    scene.add_box((3.15, 0.5, 0.4), (3.5, 1.12, 3.1), sofa)
    scene.add_box((2.2, 0.85, 0.4), (3.15, 1.12, 0.6), sofa)
    scene.add_box((2.2, 0.85, 2.9), (3.15, 1.12, 3.1), sofa)
    scene.add_box((2.4, 1.0, 0.6), (3.15, 1.12, 2.9), solid((0.2, 0.62, 0.66), 0.15))

    scene.add_box((-3.5, -0.9, 0.5), (-3.0, floor, 3.0), shelf_material)
    scene.add_cylinder(-2.2, 3.4, 0.28, -1.5, floor, marble_material, 0.3, 40)
    scene.add_sphere((-1.3, 1.1, 0.4), 0.4, ball_material, 0.5, 60)

    scene.add_box((1.8, 0.9, -1.5), (2.4, floor, -0.9), crate_material)
    scene.add_box((1.9, 0.5, -1.4), (2.3, 0.9, -1.0), crate_material)

    scene.add_cylinder(-2.6, -1.0, 0.025, -0.5, floor, solid((0.15, 0.15, 0.17)), 0.5, 40)
    scene.add_sphere((-2.6, -0.7, -1.0), 0.25, shade_material)
    scene.add_box((-2.9, floor - 0.03, -1.2), (-2.3, floor, -0.8), solid((0.2, 0.2, 0.22)))

    return scene


def make_rt(theta_x, theta_y, theta_z, tx, ty, tz):
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


def load_scenes(path):
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


class Renderer:
    def __init__(self, config, scene, R, t):
        self.config = config
        self.scene = scene
        self.rotation = np.asarray(R, dtype=float)
        self.camera = np.asarray(t, dtype=float).reshape(3)
        self.light = np.asarray(config.light, dtype=float)
        self.room_min = np.asarray(config.room_min, dtype=float)
        self.room_max = np.asarray(config.room_max, dtype=float)

        if ((self.camera <= self.room_min) | (self.camera >= self.room_max)).any():
            raise ValueError(f"Camera {self.camera.tolist()} is outside the room.")

    def render_rays(self, directions):
        count = len(directions)
        origins = np.broadcast_to(self.camera, (count, 3))
        distance, normals = self.hit_room(origins, directions)
        primitive_ids = np.zeros(count, dtype=int)

        for index, primitive in enumerate(self.scene.primitives, 1):
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

        for index in range(len(self.scene.primitives) + 1):
            mask = primitive_ids == index

            if not mask.any():
                continue

            if index == 0:
                color[mask], emission[mask] = room_material(points[mask], normals[mask])
            else:
                primitive = self.scene.primitives[index - 1]
                color[mask], emission[mask] = primitive.material(points[mask], normals[mask])
                specular[mask] = primitive.specular
                shininess[mask] = primitive.shininess

        floor_mask = (primitive_ids == 0) & (normals[:, 1] < -0.5)
        specular[floor_mask] = 0.12
        shininess[floor_mask] = 25

        light_vector = self.light - points
        light_distance = np.linalg.norm(light_vector, axis=1)
        light_direction = light_vector / np.maximum(light_distance[:, None], 1e-9)
        diffuse = np.clip((normals * light_direction).sum(axis=1), 0, None)

        indices = np.where(diffuse > 0)[0]
        shadow_origins = points[indices] + normals[indices] * self.config.shadow_bias
        shadow_directions = light_direction[indices]
        shadow_distances = light_distance[indices]
        blocked = np.zeros(len(indices), dtype=bool)

        for primitive in self.scene.primitives:
            blocked |= primitive.intersect(shadow_origins, shadow_directions)[0] < shadow_distances

        shadow = np.ones(count)
        shadow[indices[blocked]] = 0

        attenuation = self.config.light_intensity / (1 + self.config.light_falloff * light_distance**2)
        half_vector = light_direction - directions
        half_vector /= np.linalg.norm(half_vector, axis=1, keepdims=True) + 1e-9
        highlight = specular * np.clip((normals * half_vector).sum(axis=1), 0, None)**shininess
        ambient = self.config.ambient + 0.12 * (0.5 - 0.5 * normals[:, 1])

        result = color * (ambient + self.config.direct_light * diffuse * attenuation * shadow)[:, None]
        result += (highlight * shadow * attenuation)[:, None]
        result = result * (1 - emission[:, None]) + color * emission[:, None]

        return np.clip(result, 0, 1)**(1 / self.config.gamma), distance

    def hit_room(self, origins, directions):
        dd = safe_divisor(directions)
        distances = np.maximum((self.room_min - origins) / dd, (self.room_max - origins) / dd)
        axes = distances.argmin(axis=1)
        indices = np.arange(len(directions))
        normals = np.zeros_like(directions)
        normals[indices, axes] = -np.sign(directions[indices, axes])
        return distances[indices, axes], normals

    def render(self):
        config = self.config
        u = (np.arange(config.width) + 0.5) / config.width
        v = (np.arange(config.height) + 0.5) / config.height
        longitude, latitude = np.meshgrid((u - 0.5) * 2 * np.pi, (0.5 - v) * np.pi)

        directions = np.stack([np.cos(latitude) * np.sin(longitude), -np.sin(latitude), np.cos(latitude) * np.cos(longitude)], axis=-1).reshape(-1, 3)
        directions = directions @ self.rotation
        rgb = np.zeros((len(directions), 3))
        depth = np.zeros(len(directions))

        for start in tqdm(range(0, len(directions), config.chunk_size), desc="Rendering", unit="chunk"):
            end = min(start + config.chunk_size, len(directions))
            rgb[start:end], depth[start:end] = self.render_rays(directions[start:end])

        rgb = rgb.reshape(config.height, config.width, 3)
        depth = depth.reshape(config.height, config.width).astype(np.float32)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray((rgb * 255).astype(np.uint8)).save(config.output_dir / "room_rgb.png")
        depth_uint16 = np.clip(np.round(depth * config.depth_scale), 0, 65535).astype(np.uint16)
        Image.fromarray(depth_uint16).save(config.output_dir / "room_depth.png")

        normalized_depth = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-9)

        try:
            depth_visualization = (colormaps["turbo"](1 - normalized_depth)[..., :3] * 255).astype(np.uint8)
        except ImportError:
            depth_visualization = np.repeat((255 * (1 - normalized_depth))[..., None].astype(np.uint8), 3, axis=2)

        Image.fromarray(depth_visualization).save(config.output_dir / "room_depth_vis.png")
        print("depth range [m]:", float(depth.min()), float(depth.max()))


def main():
    config = RenderConfig()
    scene = create_scene()

    for name, R, t in load_scenes(config.scene_file):
        Renderer(replace(config, output_dir=config.output_dir / name), scene, R, t).render()


if __name__ == "__main__":
    main()
