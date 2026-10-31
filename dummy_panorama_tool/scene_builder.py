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


def create_scene() -> Scene:
    """Make the dummy room: rug, table, sofa, shelf, marble column, ball, crates and a floor lamp."""
    scene = Scene(light=(0.0, -1.25, 0.5), room_min=(-3.5, -1.5, -4.5), room_max=(3.5, 1.5, 4.5))
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
