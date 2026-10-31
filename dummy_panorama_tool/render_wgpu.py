import numpy as np
import wgpu
from dataclasses import dataclass

from .render_base import RenderScene, RenderSettings, Renderer
from .scene_builder import Scene

KIND_IDS = {"box": 0, "sphere": 1, "cylinder": 2}
MATERIAL_IDS = {"rug": 0, "solid": 1, "wood": 2, "marble": 3, "ball": 4, "crate": 5, "shelf": 6, "shade": 7}

SHADER = """
const PI: f32 = 3.14159265358979;
const INF: f32 = 1e30;
const EPS: f32 = 1e-4;

struct Primitive {
    kind: u32,
    material: u32,
    pad0: u32,
    pad1: u32,
    a: vec4<f32>,
    b: vec4<f32>,
    mat: vec4<f32>,
    surf: vec4<f32>,
}

struct Params {
    rot: array<vec4<f32>, 3>,
    camera: vec4<f32>,
    light: vec4<f32>,
    room_min: vec4<f32>,
    room_max: vec4<f32>,
    shade: vec4<f32>,
    misc: vec4<f32>,
    grid: vec4<u32>,
}

struct Hit {
    t: f32,
    n: vec3<f32>,
}

struct Surface {
    color: vec3<f32>,
    emission: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> primitives: array<Primitive>;
@group(0) @binding(2) var<storage, read_write> rgb: array<f32>;
@group(0) @binding(3) var<storage, read_write> depth: array<f32>;

fn safe_div(v: vec3<f32>) -> vec3<f32> {
    return select(v, vec3<f32>(1e-8), abs(v) < vec3<f32>(1e-8));
}

fn safe_div1(x: f32) -> f32 {
    return select(x, 1e-8, abs(x) < 1e-8);
}

fn safe_pow(x: f32, y: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    return pow(x, y);
}

fn axis_normal(axis: u32, s: vec3<f32>) -> vec3<f32> {
    if (axis == 0u) {
        return vec3<f32>(s.x, 0.0, 0.0);
    }
    if (axis == 1u) {
        return vec3<f32>(0.0, s.y, 0.0);
    }
    return vec3<f32>(0.0, 0.0, s.z);
}

fn hash_noise(x: f32) -> f32 {
    let value = sin(x * 12.9898) * 43758.5453;
    return value - floor(value);
}

fn corner(i: vec3<f32>, a: f32, b: f32, c: f32) -> f32 {
    return hash_noise(i.x + a + 57.0 * (i.y + b) + 113.0 * (i.z + c));
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    var f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    let c00 = corner(i, 0.0, 0.0, 0.0) * (1.0 - f.x) + corner(i, 1.0, 0.0, 0.0) * f.x;
    let c10 = corner(i, 0.0, 1.0, 0.0) * (1.0 - f.x) + corner(i, 1.0, 1.0, 0.0) * f.x;
    let c01 = corner(i, 0.0, 0.0, 1.0) * (1.0 - f.x) + corner(i, 1.0, 0.0, 1.0) * f.x;
    let c11 = corner(i, 0.0, 1.0, 1.0) * (1.0 - f.x) + corner(i, 1.0, 1.0, 1.0) * f.x;
    return (c00 * (1.0 - f.y) + c10 * f.y) * (1.0 - f.z) + (c01 * (1.0 - f.y) + c11 * f.y) * f.z;
}

fn fbm(p0: vec3<f32>, octaves: u32) -> f32 {
    var p = p0;
    var result = 0.0;
    var amplitude = 0.5;
    for (var k = 0u; k < octaves; k++) {
        result += amplitude * value_noise(p);
        p = p * 2.03;
        amplitude *= 0.5;
    }
    return result / (1.0 - pow(0.5, f32(octaves)));
}

fn hit_box(o: vec3<f32>, d0: vec3<f32>, minimum: vec3<f32>, maximum: vec3<f32>) -> Hit {
    let d = safe_div(d0);
    let t1 = (minimum - o) / d;
    let t2 = (maximum - o) / d;
    let near = min(t1, t2);
    let far = max(t1, t2);
    let t_min = max(near.x, max(near.y, near.z));
    let t_max = min(far.x, min(far.y, far.z));
    var axis = 0u;
    var best = near.x;
    if (near.y > best) {
        axis = 1u;
        best = near.y;
    }
    if (near.z > best) {
        axis = 2u;
    }
    return Hit(select(INF, t_min, t_max >= t_min && t_min > EPS), axis_normal(axis, -sign(d)));
}

fn hit_sphere(o: vec3<f32>, d: vec3<f32>, center: vec3<f32>, radius: f32) -> Hit {
    let offset = o - center;
    let b = dot(d, offset);
    let discriminant = b * b - (dot(offset, offset) - radius * radius);
    let dist = -b - sqrt(max(discriminant, 0.0));
    return Hit(select(INF, dist, discriminant > 0.0 && dist > EPS), (o + dist * d - center) / radius);
}

fn hit_cylinder(o: vec3<f32>, d: vec3<f32>, center_x: f32, center_z: f32, radius: f32, y_min: f32, y_max: f32) -> Hit {
    let ox = o.x - center_x;
    let oz = o.z - center_z;
    let a = d.x * d.x + d.z * d.z + 1e-12;
    let b = ox * d.x + oz * d.z;
    let c = ox * ox + oz * oz - radius * radius;
    let discriminant = b * b - a * c;
    let dist = (-b - sqrt(max(discriminant, 0.0))) / a;
    let y = o.y + dist * d.y;
    var best = select(INF, dist, discriminant > 0.0 && dist > EPS && y >= y_min && y <= y_max);
    var n = vec3<f32>(ox + dist * d.x, 0.0, oz + dist * d.z) / radius;
    for (var k = 0u; k < 2u; k++) {
        var cap_y = y_max;
        var cap_sign = 1.0;
        if (k == 1u) {
            cap_y = y_min;
            cap_sign = -1.0;
        }
        let cap_dist = (cap_y - o.y) / safe_div1(d.y);
        let x = ox + cap_dist * d.x;
        let z = oz + cap_dist * d.z;
        if (cap_dist > EPS && x * x + z * z <= radius * radius && cap_dist < best) {
            best = cap_dist;
            n = vec3<f32>(0.0, cap_sign, 0.0);
        }
    }
    return Hit(best, n);
}

fn intersect(prim: Primitive, o: vec3<f32>, d: vec3<f32>) -> Hit {
    if (prim.kind == 0u) {
        return hit_box(o, d, prim.a.xyz, prim.b.xyz);
    }
    if (prim.kind == 1u) {
        return hit_sphere(o, d, prim.a.xyz, prim.a.w);
    }
    return hit_cylinder(o, d, prim.a.x, prim.a.y, prim.a.z, prim.b.x, prim.b.y);
}

fn hit_room(o: vec3<f32>, d: vec3<f32>) -> Hit {
    let dd = safe_div(d);
    let far = max((params.room_min.xyz - o) / dd, (params.room_max.xyz - o) / dd);
    var axis = 0u;
    var best = far.x;
    if (far.y < best) {
        axis = 1u;
        best = far.y;
    }
    if (far.z < best) {
        axis = 2u;
        best = far.z;
    }
    return Hit(best, axis_normal(axis, -sign(d)));
}

fn wood_floor(p: vec3<f32>) -> vec3<f32> {
    let plank_width = 0.18;
    let plank_id = floor(p.z / plank_width);
    let offset = hash_noise(plank_id * 3.1) * 2.0;
    let segment = floor((p.x + offset) / 1.4);
    let tone = 0.75 + 0.3 * hash_noise(plank_id * 7.7 + segment * 13.3);
    let grain = fbm(vec3<f32>(p.x * 2.0, p.z * 40.0, plank_id), 3u);
    var color = vec3<f32>(0.62, 0.40, 0.20) * (tone * (0.75 + 0.5 * grain));
    if (p.z / plank_width - plank_id < 0.03 || ((p.x + offset) / 1.4 - segment) * 1.4 < 0.012) {
        color = color * 0.35;
    }
    return color;
}

fn room_surface(p: vec3<f32>, n: vec3<f32>) -> Surface {
    let x = p.x;
    let y = p.y;
    let z = p.z;
    let is_floor = n.y < -0.5;
    let is_ceiling = n.y > 0.5;
    var color = vec3<f32>(0.0);
    var emission = 0.0;
    if (is_floor) {
        color = wood_floor(p);
    } else {
        let noise = fbm(p * 3.0, 2u);
        if (is_ceiling) {
            color = vec3<f32>(0.95, 0.95, 0.92) * (0.97 + 0.05 * noise);
        } else {
            var plaster = vec3<f32>(0.90, 0.85, 0.74);
            if (abs(n.x) > 0.5) {
                plaster = vec3<f32>(0.70, 0.78, 0.68);
            }
            color = plaster * (0.9 + 0.15 * noise);
        }
    }
    if (!is_floor && !is_ceiling && y > 1.38) {
        var tone = 0.6;
        if (y > 1.385) {
            tone = 1.0;
        }
        color = vec3<f32>(0.92, 0.92, 0.9) * tone;
    }
    let window = n.z < -0.5 && abs(x) < 1.2 && y < 0.4 && y > -1.0;
    let frame = window && (abs(x) > 1.12 || y > 0.33 || y < -0.93);
    let mullion = window && (abs(x) < 0.03 || abs(abs(x) - 0.6) < 0.025 || abs(y + 0.3) < 0.03);
    if (frame || mullion) {
        color = vec3<f32>(0.95, 0.95, 0.93);
    }
    if (window && !frame && !mullion) {
        let t = clamp((0.4 - y) / 1.4, 0.0, 1.0);
        var sky = vec3<f32>(0.80 - 0.25 * t, 0.88 - 0.13 * t, 1.0);
        sky = sky + vec3<f32>(0.18 * fbm(vec3<f32>(x * 2.0, y * 3.0, 0.0), 3u));
        color = clamp(sky, vec3<f32>(0.0), vec3<f32>(1.0));
        emission = 1.0;
    }
    if (n.z > 0.5 && abs(x) < 0.55 && y > -0.5) {
        let u = (x + 0.55) / 1.1;
        let v = (1.5 - y) / 2.0;
        let panel = u > 0.12 && u < 0.88 && ((v > 0.08 && v < 0.45) || (v > 0.55 && v < 0.92));
        let grain = 0.9 + 0.2 * value_noise(vec3<f32>(x * 4.0, y * 30.0, z * 0.0));
        var door_color = vec3<f32>(0.30, 0.18, 0.09);
        if (panel) {
            door_color = vec3<f32>(0.45, 0.28, 0.14);
        }
        color = door_color * grain;
        if ((x - 0.4) * (x - 0.4) + (y - 0.5) * (y - 0.5) < 0.04 * 0.04) {
            color = vec3<f32>(0.85, 0.65, 0.2);
        }
    }
    for (var k = 0u; k < 2u; k++) {
        var z_min = -3.0;
        var z_max = -1.5;
        if (k == 1u) {
            z_min = 1.0;
            z_max = 2.5;
        }
        if (n.x < -0.5 && z > z_min && z < z_max && y < 0.2 && y > -0.7) {
            let u = (z - z_min) / (z_max - z_min);
            let v = (0.2 - y) / 0.9;
            let phase = z_min * 2.0;
            var artwork = 0.5 + 0.5 * sin(vec3<f32>(u * 9.0 + v * 3.0 + phase, v * 7.0 - u * 4.0 + 2.0 + phase, u * 5.0 + v * 8.0 + 4.0));
            artwork = artwork * (0.75 + 0.35 * fbm(vec3<f32>(u * 6.0, v * 6.0, phase), 3u));
            color = artwork;
            if (z < z_min + 0.05 || z > z_max - 0.05 || y > 0.15 || y < -0.65) {
                color = vec3<f32>(0.25, 0.15, 0.07);
            }
        }
    }
    return Surface(color, emission);
}

fn primitive_surface(prim: Primitive, p: vec3<f32>, n: vec3<f32>) -> Surface {
    var color = vec3<f32>(0.0);
    var emission = 0.0;
    switch (prim.material) {
        case 0u: {
            let radius = max(abs(p.x) / 1.6, abs(p.z - 1.8) / 1.5);
            var base = vec3<f32>(0.90, 0.80, 0.55);
            if (floor(radius * 8.0) % 2.0 != 0.0) {
                base = vec3<f32>(0.55, 0.12, 0.12);
            }
            if (radius > 0.88) {
                base = vec3<f32>(0.10, 0.12, 0.35);
            }
            color = base * (0.85 + 0.3 * value_noise(p * 60.0));
        }
        case 1u: {
            color = prim.mat.xyz * (1.0 + prim.mat.w * (value_noise(p * prim.surf.z) - 0.5) * 2.0);
        }
        case 2u: {
            color = prim.mat.xyz * (0.7 + 0.6 * fbm(vec3<f32>(p.x * 3.0, p.z * 30.0, p.y * 30.0), 3u));
        }
        case 3u: {
            let value = 0.82 + 0.15 * sin(-p.y * 6.0 + 8.0 * fbm(p * 2.0, 3u));
            color = vec3<f32>(value, value, value * 1.03);
        }
        case 4u: {
            let index = u32(floor((atan2(n.x, n.z) + PI) / (2.0 * PI) * 6.0)) % 6u;
            var palette = array<vec3<f32>, 6>(
                vec3<f32>(0.9, 0.15, 0.15),
                vec3<f32>(1.0, 1.0, 1.0),
                vec3<f32>(0.1, 0.4, 0.9),
                vec3<f32>(1.0, 1.0, 1.0),
                vec3<f32>(0.95, 0.8, 0.1),
                vec3<f32>(1.0, 1.0, 1.0),
            );
            color = palette[index];
            if (abs(n.y) > 0.92) {
                color = vec3<f32>(1.0, 1.0, 1.0);
            }
        }
        case 5u: {
            let y = -p.y;
            let row = floor(y / 0.12);
            let tone = 0.75 + 0.35 * hash_noise(row * 3.1);
            let grain = 0.85 + 0.3 * value_noise(vec3<f32>(p.x * 3.0, y * 25.0, p.z * 3.0));
            color = vec3<f32>(0.65, 0.47, 0.25) * (tone * grain);
            if (y / 0.12 - row < 0.06) {
                color = color * 0.4;
            }
        }
        case 6u: {
            let wood = vec3<f32>(0.35, 0.22, 0.10) * (0.8 + 0.4 * value_noise(p * 20.0));
            let y = 1.5 - p.y;
            let row = floor(y / 0.42);
            let fraction = y / 0.42 - row;
            let book_id = floor(p.z / 0.05) + row * 31.0;
            let book_height = 0.55 + 0.4 * hash_noise(book_id * 1.7);
            let book_color = 0.2 + 0.7 * vec3<f32>(hash_noise(book_id * 2.3), hash_noise(book_id * 5.1), hash_noise(book_id * 9.7));
            var front = vec3<f32>(0.12, 0.08, 0.05);
            if (fraction > 0.07 && (fraction - 0.07) / 0.93 < book_height) {
                front = book_color;
            }
            if (fraction < 0.07) {
                front = wood;
            }
            color = wood;
            if (n.x > 0.5) {
                color = front;
            }
        }
        case 7u: {
            color = vec3<f32>(1.0, 0.88, 0.55) * (0.85 - 0.15 * n.y);
            emission = 1.0;
        }
        default: {
        }
    }
    return Surface(color, emission);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = params.grid.x;
    let height = params.grid.y;
    let count = params.grid.z;
    if (id.x >= width || id.y >= height) {
        return;
    }

    let index = id.y * width + id.x;
    let lon = ((f32(id.x) + 0.5) / f32(width) - 0.5) * 2.0 * PI;
    let lat = (0.5 - (f32(id.y) + 0.5) / f32(height)) * PI;
    let dir = vec3<f32>(cos(lat) * sin(lon), -sin(lat), cos(lat) * cos(lon));
    let d = params.rot[0].xyz * dir.x + params.rot[1].xyz * dir.y + params.rot[2].xyz * dir.z;
    let o = params.camera.xyz;

    var hit = hit_room(o, d);
    var primitive_id = 0u;
    for (var i = 0u; i < count; i++) {
        let h = intersect(primitives[i], o, d);
        if (h.t < hit.t) {
            hit = h;
            primitive_id = i + 1u;
        }
    }
    let dist = hit.t;
    let n = hit.n;
    let p = o + dist * d;

    var surface: Surface;
    var specular = 0.0;
    var shininess = 1.0;
    if (primitive_id == 0u) {
        surface = room_surface(p, n);
        if (n.y < -0.5) {
            specular = 0.12;
            shininess = 25.0;
        }
    } else {
        let prim = primitives[primitive_id - 1u];
        surface = primitive_surface(prim, p, n);
        specular = prim.surf.x;
        shininess = prim.surf.y;
    }

    let light_vector = params.light.xyz - p;
    let light_dist = length(light_vector);
    let l = light_vector / max(light_dist, 1e-9);
    let diffuse = max(dot(n, l), 0.0);
    var shadow = 1.0;
    if (diffuse > 0.0) {
        let shadow_origin = p + n * params.misc.x;
        for (var i = 0u; i < count; i++) {
            if (intersect(primitives[i], shadow_origin, l).t < light_dist) {
                shadow = 0.0;
                break;
            }
        }
    }

    let attenuation = params.shade.z / (1.0 + params.shade.w * light_dist * light_dist);
    var half_vector = l - d;
    half_vector = half_vector / (length(half_vector) + 1e-9);
    let highlight = specular * safe_pow(max(dot(n, half_vector), 0.0), shininess);
    let ambient = params.shade.x + 0.12 * (0.5 - 0.5 * n.y);
    var result = surface.color * (ambient + params.shade.y * diffuse * attenuation * shadow);
    result = result + vec3<f32>(highlight * shadow * attenuation);
    result = result * (1.0 - surface.emission) + surface.color * surface.emission;
    result = clamp(result, vec3<f32>(0.0), vec3<f32>(1.0));

    let inv_gamma = 1.0 / params.misc.y;
    rgb[index * 3u] = safe_pow(result.x, inv_gamma);
    rgb[index * 3u + 1u] = safe_pow(result.y, inv_gamma);
    rgb[index * 3u + 2u] = safe_pow(result.z, inv_gamma);
    depth[index] = dist;
}
"""


@dataclass
class WgpuRenderScene(RenderScene):
    """Scene on the GPU.

    Attributes:
        primitives: packed primitives (20 float32 each), shape (max(P, 1), 20).
        count: number of primitives P.
        light: point light position (x, y, z).
        room_min: room lower corner (x, y, z).
        room_max: room upper corner (x, y, z).
        settings: render parameters.
        rgb: output colors (float32), shape (H * W * 3,).
        depth: output depth (float32), shape (H * W,).
    """
    primitives: wgpu.GPUBuffer
    count: int
    light: tuple
    room_min: tuple
    room_max: tuple
    settings: RenderSettings
    rgb: wgpu.GPUBuffer
    depth: wgpu.GPUBuffer


class WgpuRenderer(Renderer):
    """Dummy panorama renderer on the GPU by a wgpu compute shader (one thread per ERP pixel).

    Textures are procedural and use float32 and the GPU's sin, so they can differ slightly from the numpy backend. Geometry (depth) matches it.
    Each instance requests its own GPU device, so create only one per process.
    """

    def __init__(self) -> None:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No wgpu adapter found.")
        limits = {key: value for key, value in adapter.limits.items() if key.replace("-", "_") in ["max_buffer_size", "max_storage_buffer_binding_size"]}

        self.device = adapter.request_device_sync(required_limits=limits)
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.device.create_shader_module(code=SHADER), "entry_point": "main"},
        )

    def build(self, scene: Scene, settings: RenderSettings) -> WgpuRenderScene:
        pixels = settings.width * settings.height
        primitives = self.device.create_buffer_with_data(data=self._pack_primitives(scene), usage=wgpu.BufferUsage.STORAGE)
        return WgpuRenderScene(primitives, len(scene.primitives), scene.light, scene.room_min, scene.room_max, settings, self._storage(pixels * 3 * 4), self._storage(pixels * 4))

    def to_erp(self, render_scene: WgpuRenderScene, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check(render_scene)
        height, width = render_scene.settings.height, render_scene.settings.width

        self._compute([self._pack_params(render_scene, R, t), render_scene.primitives, render_scene.rgb, render_scene.depth], (height, width))

        rgb = np.frombuffer(self.device.queue.read_buffer(render_scene.rgb), dtype=np.float32).reshape(height, width, 3).copy()
        depth = np.frombuffer(self.device.queue.read_buffer(render_scene.depth), dtype=np.float32).reshape(height, width).copy()
        return rgb, depth

    @staticmethod
    def _check(render_scene: RenderScene) -> None:
        if not isinstance(render_scene, WgpuRenderScene):
            raise TypeError(f"Expected WgpuRenderScene, got {type(render_scene).__name__}.")

    @staticmethod
    def _pack_primitives(scene: Scene) -> np.ndarray:
        """Pack primitives as 20 float32 each: kind, material, 2 pads, a, b, material params 0-3, (specular, shininess, material params 4-5)."""
        data = np.zeros((max(len(scene.primitives), 1), 20), dtype=np.float32)
        ids = data.view(np.uint32)
        for i, primitive in enumerate(scene.primitives):
            name, *params = primitive.material
            if primitive.kind not in KIND_IDS:
                raise ValueError(f"Unknown primitive kind '{primitive.kind}'.")
            if name not in MATERIAL_IDS:
                raise ValueError(f"Unknown material '{name}'.")
            if len(params) > 6:
                raise ValueError(f"Material '{name}' has too many parameters ({len(params)} > 6).")

            ids[i, 0] = KIND_IDS[primitive.kind]
            ids[i, 1] = MATERIAL_IDS[name]
            shape = primitive.shape
            if primitive.kind == "box":
                data[i, 4:7] = shape[:3]
                data[i, 8:11] = shape[3:6]
            elif primitive.kind == "sphere":
                data[i, 4:8] = shape[:4]
            else:
                data[i, 4:7] = shape[:3]
                data[i, 8:10] = shape[3:5]

            params = params + [0.0] * (6 - len(params))
            data[i, 12:16] = params[:4]
            data[i, 16] = primitive.specular
            data[i, 17] = primitive.shininess
            data[i, 18:20] = params[4:6]

        return data

    @staticmethod
    def _pack_params(render_scene: WgpuRenderScene, R: np.ndarray, t: np.ndarray) -> bytes:
        """Pack the uniform block: R rows, camera, light, room bounds, shading constants, grid size."""
        settings = render_scene.settings
        rot = np.zeros((3, 4), dtype=np.float32)
        rot[:, :3] = R
        floats = np.concatenate([
            rot.reshape(-1),
            [*np.asarray(t, dtype=np.float64).reshape(3), 0.0],
            [*render_scene.light, 0.0],
            [*render_scene.room_min, 0.0],
            [*render_scene.room_max, 0.0],
            [settings.ambient, settings.direct_light, settings.light_intensity, settings.light_falloff],
            [settings.shadow_bias, settings.gamma, 0.0, 0.0],
        ]).astype(np.float32)
        grid = np.array([settings.width, settings.height, render_scene.count, 0], dtype=np.uint32)
        return floats.tobytes() + grid.tobytes()

    def _compute(self, resources: list[wgpu.GPUBuffer | bytes], shape: tuple[int, int]) -> None:
        """Run the render pipeline over an image (height, width)."""
        groups = ((shape[1] + 7) // 8, (shape[0] + 7) // 8)

        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, self._bind(resources))
        compute_pass.dispatch_workgroups(*groups)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])

    def _bind(self, resources: list[wgpu.GPUBuffer | bytes]) -> wgpu.GPUBindGroup:
        """Bind resources to group 0 of the pipeline, in binding order. Bytes become uniform buffers."""
        entries = []
        for i, resource in enumerate(resources):
            if isinstance(resource, bytes):
                resource = self.device.create_buffer_with_data(data=resource, usage=wgpu.BufferUsage.UNIFORM)
            entries.append({"binding": i, "resource": {"buffer": resource, "offset": 0, "size": resource.size}})

        return self.device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=entries)

    def _storage(self, size: int) -> wgpu.GPUBuffer:
        """Make an uninitialized storage buffer of the given size in bytes (readable back to the CPU)."""
        return self.device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
