import numpy as np
import wgpu
from dataclasses import dataclass

from .base import Mesh, SphericalDIBR

SHADERS = {
    "unproject_erp": """
const PI: f32 = 3.14159265358979;

@group(0) @binding(0) var<storage, read> depth: array<f32>;
@group(0) @binding(1) var<storage, read_write> world_coord: array<f32>;
@group(0) @binding(2) var<uniform> grid: vec4<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = grid.x;
    let height = grid.y;
    if (id.x >= width || id.y >= height) {
        return;
    }

    let index = id.y * width + id.x;
    let count = width * height;
    let lon = ((f32(id.x) + 0.5) / f32(width) - 0.5) * 2.0 * PI;
    let lat = (0.5 - (f32(id.y) + 0.5) / f32(height)) * PI;

    world_coord[index] = cos(lat) * sin(lon) * depth[index];
    world_coord[count + index] = -sin(lat) * depth[index];
    world_coord[2u * count + index] = cos(lat) * cos(lon) * depth[index];
}
""",
    "make_faces": """
@group(0) @binding(0) var<storage, read> depth: array<f32>;
@group(0) @binding(1) var<storage, read_write> faces: array<u32>;
@group(0) @binding(2) var<uniform> grid: vec4<u32>;

fn is_valid(index: u32) -> bool {
    return depth[index] > 0.0 && depth[index] < 1e38;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = grid.x;
    let height = grid.y;
    if (id.x >= width || id.y + 1u >= height) {
        return;
    }

    let col_next = (id.x + 1u) % width;
    let a = id.y * width + id.x;
    let b = id.y * width + col_next;
    let c = (id.y + 1u) * width + id.x;
    let d = (id.y + 1u) * width + col_next;

    var tri0 = vec3<u32>(a, c, b);
    if (!(is_valid(a) && is_valid(c) && is_valid(b))) {
        tri0 = vec3<u32>(a, a, a);
    }

    var tri1 = vec3<u32>(b, c, d);
    if (!(is_valid(b) && is_valid(c) && is_valid(d))) {
        tri1 = vec3<u32>(b, b, b);
    }

    let base = a * 6u;
    faces[base] = tri0.x;
    faces[base + 1u] = tri0.y;
    faces[base + 2u] = tri0.z;
    faces[base + 3u] = tri1.x;
    faces[base + 4u] = tri1.y;
    faces[base + 5u] = tri1.z;
}
""",
    "apply_rt": """
struct Params {
    rot: array<vec4<f32>, 3>,
    t: vec4<f32>,
    grid: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> world_coord: array<f32>;
@group(0) @binding(1) var<storage, read_write> world_coord_out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let count = params.grid.x;
    let index = id.x + id.y * groups.x * 64u;
    if (index >= count) {
        return;
    }

    let d = vec3<f32>(world_coord[index], world_coord[count + index], world_coord[2u * count + index]) - params.t.xyz;

    world_coord_out[index] = dot(params.rot[0].xyz, d);
    world_coord_out[count + index] = dot(params.rot[1].xyz, d);
    world_coord_out[2u * count + index] = dot(params.rot[2].xyz, d);
}
""",
    "project_erp": """
const PI: f32 = 3.14159265358979;

struct Params {
    dst: vec4<f32>,
    grid: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> world_coord: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> depth: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let count = params.grid.x;
    let index = id.x + id.y * groups.x * 64u;
    if (index >= count) {
        return;
    }

    let v = vec3<f32>(world_coord[index], world_coord[count + index], world_coord[2u * count + index]);
    let radius = length(v);

    var px = 0.0;
    var py = 0.0;
    var pd = 0.0;

    if (radius > 0.0 && radius < 1e38) {
        let lon = atan2(v.x, v.z);
        let lat = asin(clamp(-v.y / radius, -1.0, 1.0));
        px = (lon / (2.0 * PI) + 0.5) * params.dst.x - 0.5;
        py = (0.5 - lat / PI) * params.dst.y - 0.5;
        pd = radius;
    }

    x[index] = px;
    y[index] = py;
    depth[index] = pd;
}
""",
    "project_plane": """
const MIN_DEPTH: f32 = 1e-6;

struct Params {
    camera: vec4<f32>,
    grid: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> world_coord: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> depth: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let count = params.grid.x;
    let index = id.x + id.y * groups.x * 64u;
    if (index >= count) {
        return;
    }

    let v = vec3<f32>(world_coord[index], world_coord[count + index], world_coord[2u * count + index]);

    var px = 0.0;
    var py = 0.0;
    var pd = 0.0;

    if (v.z > MIN_DEPTH && v.z < 1e38) {
        px = v.x * params.camera.x / v.z + params.camera.z;
        py = v.y * params.camera.y / v.z + params.camera.w;
        pd = v.z;
    }

    x[index] = px;
    y[index] = py;
    depth[index] = pd;
}
""",
    "render_image": """
@group(0) @binding(0) var<storage, read> color: array<u32>;
@group(0) @binding(1) var<storage, read> faces: array<u32>;
@group(0) @binding(2) var face_idx: texture_2d<u32>;
@group(0) @binding(3) var weight: texture_2d<u32>;
@group(0) @binding(4) var<storage, read_write> image: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(face_idx);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let pixel = vec2<i32>(i32(id.x), i32(id.y));
    let index = id.y * size.x + id.x;
    let face = textureLoad(face_idx, pixel, 0).x;
    if (face == 0u) {
        image[index] = 0u;
        return;
    }

    let w = textureLoad(weight, pixel, 0);
    let w0 = bitcast<f32>(w.x);
    let w1 = bitcast<f32>(w.y);
    let w2 = 1.0 - w0 - w1;
    let base = (face - 1u) * 3u;

    image[index] = pack4x8unorm(
        w0 * unpack4x8unorm(color[faces[base]]) +
        w1 * unpack4x8unorm(color[faces[base + 1u]]) +
        w2 * unpack4x8unorm(color[faces[base + 2u]])
    );
}
""",
    "rasterize": """
struct Params {
    dst: vec4<f32>,
}

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) weight: vec3<f32>,
    @location(1) @interpolate(flat) face: u32,
}

struct FsOut {
    @location(0) face_idx: u32,
    @location(1) weight: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> faces: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> y: array<f32>;
@group(0) @binding(4) var<storage, read> depth: array<f32>;

fn load_vertex(index: u32) -> vec3<f32> {
    return vec3<f32>(x[index], y[index], depth[index]);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> VsOut {
    let tri = vertex_id / 3u;
    let corner = vertex_id % 3u;
    let width = params.dst.x;
    let wrap = params.dst.z > 0.5;

    var p = array<vec3<f32>, 3>(
        load_vertex(faces[tri * 3u]),
        load_vertex(faces[tri * 3u + 1u]),
        load_vertex(faces[tri * 3u + 2u])
    );
    let valid = p[0].z > 0.0 && p[1].z > 0.0 && p[2].z > 0.0;

    var seam = false;
    if (valid && wrap) {
        let lo = min(p[0].x, min(p[1].x, p[2].x));
        let hi = max(p[0].x, max(p[1].x, p[2].x));
        seam = hi - lo > 0.5 * width;
        if (seam) {
            for (var k = 0u; k < 3u; k++) {
                if (p[k].x < 0.5 * width) {
                    p[k].x += width;
                }
            }
        }
    }

    var vs: VsOut;
    if (!valid || (instance_id == 1u && !seam)) {
        vs.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        vs.weight = vec3<f32>(0.0, 0.0, 0.0);
        vs.face = 0u;
        return vs;
    }

    var shift = 0.0;
    if (instance_id == 1u) {
        shift = -width;
    }

    let q = p[corner];
    vs.position = vec4<f32>(
        (q.x + shift + 0.5) / width * 2.0 - 1.0,
        1.0 - (q.y + 0.5) / params.dst.y * 2.0,
        q.z / (1.0 + q.z),
        1.0
    );
    vs.weight = vec3<f32>(0.0, 0.0, 0.0);
    vs.weight[corner] = 1.0;
    vs.face = tri + 1u;

    return vs;
}

@fragment
fn fs_main(vs: VsOut) -> FsOut {
    var fs: FsOut;
    fs.face_idx = vs.face;
    fs.weight = vec2<u32>(bitcast<u32>(vs.weight.x), bitcast<u32>(vs.weight.y));

    return fs;
}
""",
}

@dataclass
class WgpuMesh(Mesh):
    """Triangle mesh of an ERP image on the GPU (one vertex per pixel).

    Attributes:
        world_coord: vertex 3D coordinates (float32, planar x, y, z), shape (3, H * W).
        faces: triangle vertex indices (uint32), shape (F, 3).
        color: vertex colors packed as RGBA8, shape (H * W,).
        height: src ERP image height.
        width: src ERP image width.
    """
    world_coord: wgpu.GPUBuffer
    faces: wgpu.GPUBuffer
    color: wgpu.GPUBuffer
    height: int
    width: int



class WgpuDIBR(SphericalDIBR):
    """Spherical DIBR on the GPU by wgpu (compute shaders and the hardware rasterizer).

    Each instance requests its own GPU device, so create only one per process.
    """

    def __init__(self) -> None:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        limits = {key: value for key, value in adapter.limits.items() if key.replace("-", "_") in ["max_buffer_size", "max_storage_buffer_binding_size"]}

        self.device = adapter.request_device_sync(required_limits=limits)
        self.pipelines = {}
        self.targets = {}

        for name in ["unproject_erp", "make_faces", "apply_rt", "project_erp", "project_plane", "render_image"]:
            self.pipelines[name] = self.device.create_compute_pipeline(
                layout="auto",
                compute={"module": self.device.create_shader_module(code=SHADERS[name]), "entry_point": "main"},
            )

        module = self.device.create_shader_module(code=SHADERS["rasterize"])
        self.pipelines["rasterize"] = self.device.create_render_pipeline(
            layout="auto",
            vertex={"module": module, "entry_point": "vs_main"},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
            depth_stencil={"format": wgpu.TextureFormat.depth32float, "depth_write_enabled": True, "depth_compare": wgpu.CompareFunction.less},
            fragment={"module": module, "entry_point": "fs_main", "targets": [{"format": wgpu.TextureFormat.r32uint}, {"format": wgpu.TextureFormat.rg32uint}]},
        )

    def build(self, depth: np.ndarray, image: np.ndarray) -> WgpuMesh:
        """src ERP image → triangle mesh on the GPU.

        Args:
            depth: src ERP depth map (radius), shape (H, W).
            image: src ERP image (uint8, 3 channels), shape (H, W, 3).

        Returns:
            Triangle mesh.
        """
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be uint8 with 3 channels, got {image.dtype} {image.shape}")

        height, width = depth.shape
        depth_buffer = self.device.create_buffer_with_data(data=np.ascontiguousarray(depth, dtype=np.float32), usage=wgpu.BufferUsage.STORAGE)
        grid = np.array([width, height, 0, 0], dtype=np.uint32).tobytes()

        world_coord = self._storage(3 * height * width * 4)
        self._compute("unproject_erp", [depth_buffer, world_coord, grid], (height, width))

        faces = self._storage(2 * (height - 1) * width * 3 * 4)
        self._compute("make_faces", [depth_buffer, faces, grid], (height, width))

        color = np.full((height * width, 4), 255, dtype=np.uint8)
        color[:, :3] = image.reshape(-1, 3)
        color = self.device.create_buffer_with_data(data=color, usage=wgpu.BufferUsage.STORAGE)

        return WgpuMesh(world_coord, faces, color, height, width)

    def to_erp(self, mesh: WgpuMesh, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self._render(mesh, R, t, "project_erp", [mesh.width, mesh.height, 0, 0], mesh.height, mesh.width, True)

    def to_plane(self, mesh: WgpuMesh, camera: np.ndarray, camera_dst_size: tuple[int, int], R: np.ndarray, t: np.ndarray) -> np.ndarray:
        width, height = camera_dst_size
        intrinsics = [camera[0, 0], camera[1, 1], camera[0, 2], camera[1, 2]]

        return self._render(mesh, R, t, "project_plane", intrinsics, height, width, False)

    def _render(self, mesh: WgpuMesh, R: np.ndarray, t: np.ndarray, project: str, project_params: list[float], height: int, width: int, wrap: bool) -> np.ndarray:
        """Apply R|t, project, rasterize and shade the mesh.

        Args:
            mesh: triangle mesh from build.
            R: rotation matrix, shape (3, 3).
            t: translation vector, shape (3, 1).
            project: projection pipeline ("project_erp" or "project_plane").
            project_params: (dst width, dst height, 0, 0) for ERP, (fx, fy, ox, oy) of the dst camera for plane.
            height: dst image height.
            width: dst image width.
            wrap: Wrap x around the image width (ERP).

        Returns:
            dst image (uint8, 3 channels), shape (height, width, 3).
        """
        if not isinstance(mesh, WgpuMesh):
            raise TypeError(f"Expected WgpuMesh, got {type(mesh).__name__}.")

        count = mesh.world_coord.size // 12

        rot = np.zeros((3, 4), dtype=np.float32)
        rot[:, :3] = R
        rt = np.concatenate([rot.reshape(-1), [*t.reshape(-1), 0]]).astype(np.float32).tobytes() + np.array([count, 0, 0, 0], dtype=np.uint32).tobytes()

        world_coord = self._storage(mesh.world_coord.size)
        self._compute("apply_rt", [mesh.world_coord, world_coord, rt], (count,))

        params = np.array(project_params, dtype=np.float32).tobytes() + np.array([count, 0, 0, 0], dtype=np.uint32).tobytes()

        x, y, depth = self._storage(count * 4), self._storage(count * 4), self._storage(count * 4)
        self._compute(project, [world_coord, x, y, depth, params], (count,))

        face_idx, weight = self._rasterize(x, y, depth, mesh.faces, height, width, wrap)

        image = self._storage(height * width * 4, readback=True)
        self._compute("render_image", [mesh.color, mesh.faces, face_idx, weight, image], (height, width))

        return np.frombuffer(self.device.queue.read_buffer(image), dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()

    def _rasterize(self, x: wgpu.GPUBuffer, y: wgpu.GPUBuffer, depth: wgpu.GPUBuffer, faces: wgpu.GPUBuffer, height: int, width: int, wrap: bool) -> tuple[wgpu.GPUTexture, wgpu.GPUTexture]:
        """Find the nearest triangle of each dst pixel by the hardware z-buffer.

        Returns:
            face_idx: index + 1 of the nearest triangle of each pixel (0 if none), shape (height, width).
            weight: barycentric weights w0, w1 (float32 bits) in that triangle, shape (height, width, 2).
        """
        if (height, width) not in self.targets:
            size = (width, height, 1)
            usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING

            self.targets[(height, width)] = (
                self.device.create_texture(size=size, format=wgpu.TextureFormat.r32uint, usage=usage),
                self.device.create_texture(size=size, format=wgpu.TextureFormat.rg32uint, usage=usage),
                self.device.create_texture(size=size, format=wgpu.TextureFormat.depth32float, usage=wgpu.TextureUsage.RENDER_ATTACHMENT),
            )

        face_idx, weight, zbuffer = self.targets[(height, width)]

        pipeline = self.pipelines["rasterize"]
        params = np.array([width, height, 1 if wrap else 0, 0], dtype=np.float32).tobytes()
        clear = {"load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 0)}

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[{"view": face_idx.create_view(), **clear}, {"view": weight.create_view(), **clear}],
            depth_stencil_attachment={"view": zbuffer.create_view(), "depth_clear_value": 1.0, "depth_load_op": wgpu.LoadOp.clear, "depth_store_op": wgpu.StoreOp.discard},
        )
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, self._bind(pipeline, [params, faces, x, y, depth]))
        render_pass.draw(faces.size // 4, 2 if wrap else 1)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])

        return face_idx, weight

    def _compute(self, name: str, resources: list[wgpu.GPUBuffer | wgpu.GPUTexture | bytes], shape: tuple[int, ...]) -> None:
        """Run a compute pipeline over an image (height, width) or over N items (count,)."""
        pipeline = self.pipelines[name]

        if len(shape) == 2:
            groups = ((shape[1] + 7) // 8, (shape[0] + 7) // 8)
        else:
            x = min((shape[0] + 63) // 64, 65535)
            groups = (x, (shape[0] + 64 * x - 1) // (64 * x))

        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, self._bind(pipeline, resources))
        compute_pass.dispatch_workgroups(*groups)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])

    def _bind(self, pipeline: wgpu.GPUComputePipeline | wgpu.GPURenderPipeline, resources: list[wgpu.GPUBuffer | wgpu.GPUTexture | bytes]) -> wgpu.GPUBindGroup:
        """Bind resources to group 0 of the pipeline, in binding order. Bytes become uniform buffers."""
        entries = []
        for i, resource in enumerate(resources):
            if isinstance(resource, wgpu.GPUTexture):
                entries.append({"binding": i, "resource": resource.create_view()})
                continue

            if isinstance(resource, bytes):
                resource = self.device.create_buffer_with_data(data=resource, usage=wgpu.BufferUsage.UNIFORM)

            entries.append({"binding": i, "resource": {"buffer": resource, "offset": 0, "size": resource.size}})

        return self.device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)

    def _storage(self, size: int, readback: bool = False) -> wgpu.GPUBuffer:
        """Make an uninitialized storage buffer of the given size in bytes."""
        usage = wgpu.BufferUsage.STORAGE | (wgpu.BufferUsage.COPY_SRC if readback else 0)

        return self.device.create_buffer(size=size, usage=usage)
