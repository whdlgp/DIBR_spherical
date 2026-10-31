import shutil
import tomllib
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Primitive:
    """Primitive of a scene. It holds no renderer-specific data, each backend builds its own from it.

    Attributes:
        kind: "box", "sphere" or "cylinder".
        shape: box (min_x, min_y, min_z, max_x, max_y, max_z), sphere (x, y, z, radius), cylinder (center_x, center_z, radius, y_min, y_max).
        material: (name, *params). Make it with solid, wood_material or one of the *_material constants.
        specular: specular strength.
        shininess: specular exponent.
    """
    kind: str
    shape: tuple
    material: tuple
    specular: float = 0.0
    shininess: float = 1.0


@dataclass
class Scene:
    """Closed box room with primitives and a point light. Coordinates: x right, y down, z forward (meters).

    Attributes:
        light: point light position (x, y, z).
        room_min: room lower corner (x, y, z).
        room_max: room upper corner (x, y, z).
        primitives: objects in the room.
    """
    light: tuple
    room_min: tuple
    room_max: tuple
    primitives: list = field(default_factory=list)

    def add_box(self, minimum: tuple, maximum: tuple, material: tuple, specular: float = 0.0, shininess: float = 1.0) -> None:
        self.primitives.append(Primitive("box", (*minimum, *maximum), material, specular, shininess))

    def add_sphere(self, center: tuple, radius: float, material: tuple, specular: float = 0.0, shininess: float = 1.0) -> None:
        self.primitives.append(Primitive("sphere", (*center, radius), material, specular, shininess))

    def add_cylinder(self, center_x: float, center_z: float, radius: float, y_min: float, y_max: float, material: tuple, specular: float = 0.0, shininess: float = 1.0) -> None:
        """Add a vertical cylinder (axis along y)."""
        self.primitives.append(Primitive("cylinder", (center_x, center_z, radius, y_min, y_max), material, specular, shininess))

    def check_camera(self, t: np.ndarray) -> None:
        """Check that a camera position is strictly inside the room.

        Args:
            t: camera position, shape (3, 1).

        Raises:
            ValueError: the camera is outside the room.
        """
        camera = np.asarray(t, dtype=float).reshape(3)
        if ((camera <= np.asarray(self.room_min)) | (camera >= np.asarray(self.room_max))).any():
            raise ValueError(f"Camera {camera.tolist()} is outside the room.")


rug_material = ("rug",)
marble_material = ("marble",)
ball_material = ("ball",)
crate_material = ("crate",)
shelf_material = ("shelf",)
shade_material = ("shade",)


def solid(color: tuple, noise: float = 0.0, frequency: float = 40.0) -> tuple:
    """Single color material with value noise."""
    return ("solid", *color, noise, frequency)


def wood_material(base: tuple) -> tuple:
    """Wood grain material."""
    return ("wood", *base)


OBJECT_KEYS = {"box": ("min", "max"), "sphere": ("center", "radius"), "cylinder": ("center_x", "center_z", "radius", "y_min", "y_max")}
MATERIALS = {"rug": rug_material, "marble": marble_material, "ball": ball_material, "crate": crate_material, "shelf": shelf_material, "shade": shade_material}


def make_material(spec: str | dict) -> tuple:
    """Make a material from a TOML value: a built-in name, or a table of solid() or wood_material() arguments plus name."""
    if isinstance(spec, str):
        if spec not in MATERIALS:
            raise ValueError(f"Unknown material '{spec}'. Built-in materials: {sorted(MATERIALS)}.")
        return MATERIALS[spec]
    params = dict(spec)
    name = params.pop("name", None)
    factories = {"solid": solid, "wood": wood_material}
    if name not in factories:
        raise ValueError(f"Unknown material table name {name!r}. Use one of {sorted(factories)}.")
    return factories[name](**params)


def create_scene(path: Path) -> Scene:
    """Make a scene from a TOML file (see scene.toml).

    Args:
        path: TOML file with room, light and object entries.

    Returns:
        Scene.
    """
    path = Path(path)
    example = path.with_name(f"{path.stem}_example{path.suffix}")
    if not path.exists():
        shutil.copyfile(example, path)
        print(f"{path.name} not found, copied from {example.name}")
    with open(path, "rb") as file:
        data = tomllib.load(file)
    unknown = set(data) - {"room", "light", "object"}
    if unknown:
        raise ValueError(f"Unknown top-level keys in {path}: {sorted(unknown)}")

    scene = Scene(light=tuple(data["light"]["position"]), room_min=tuple(data["room"]["min"]), room_max=tuple(data["room"]["max"]))
    for i, obj in enumerate(data.get("object", [])):
        kind = obj.get("kind")
        if kind not in OBJECT_KEYS:
            raise ValueError(f"object #{i}: unknown kind {kind!r}, expected one of {sorted(OBJECT_KEYS)}")
        keys = set(obj) - {"kind", "specular", "shininess"}
        required = {"material", *OBJECT_KEYS[kind]}
        if keys != required:
            raise ValueError(f"object #{i} ({kind}): expected keys {sorted(required)}, got {sorted(keys)}")
        getattr(scene, f"add_{kind}")(*[obj[key] for key in OBJECT_KEYS[kind]], make_material(obj["material"]), obj.get("specular", 0.0), obj.get("shininess", 1.0))

    return scene
