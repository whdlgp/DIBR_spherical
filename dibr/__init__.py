import importlib
import warnings
from collections.abc import Sequence
 
from .base import Mesh, SphericalDIBR, make_rt
 
 
def create_dibr(backend: str | Sequence[str] = ("wgpu", "numpy")) -> SphericalDIBR:
    """Create a DIBR backend.
 
    A backend is the module backend_<name>.py of this package, which defines a subclass of SphericalDIBR.
 
    Args:
        backend: Backend name, or names in priority order. With several names, an unavailable backend is skipped with a warning.
 
    Returns:
        DIBR backend.
    """
    names = [backend] if isinstance(backend, str) else list(backend)
 
    for i, name in enumerate(names):
        try:
            module = importlib.import_module(f".backend_{name}", __package__)
            cls = next(c for c in vars(module).values() if isinstance(c, type) and issubclass(c, SphericalDIBR) and c.__module__ == module.__name__)
 
            return cls()
        except (ImportError, RuntimeError) as e:
            if i == len(names) - 1:
                raise
            warnings.warn(f"Backend '{name}' is unavailable ({e}), trying '{names[i + 1]}'.")