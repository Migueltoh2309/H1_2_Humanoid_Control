"""Configuración común de los tests (corren sin ROS: módulos puros)."""
import os
import sys

import pytest

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

_MJCF_CANDIDATES = [
    os.environ.get("H12_MJCF_PATH", ""),
    os.path.join(_PKG_ROOT, "..", "h1_2_utec", "h1_2_description", "mjcf",
                 "h1_2_scene_qp_reachable.xml"),
]
# El paquete suele estar enlazado (symlink) desde ~/humanoid_ws/src a este repo, y
# entonces _PKG_ROOT/.. es el repo, sin h1_2_utec: probar tambien el share instalado.
try:
    from ament_index_python.packages import get_package_share_directory
    _MJCF_CANDIDATES.append(os.path.join(get_package_share_directory("h1_2_description"),
                                         "mjcf", "h1_2_scene_qp_reachable.xml"))
except Exception:
    pass


@pytest.fixture(scope="session")
def mjcf_path():
    for c in _MJCF_CANDIDATES:
        if c and os.path.isfile(c):
            return c
    pytest.skip("MJCF del H1-2 no encontrado; exporta H12_MJCF_PATH")
