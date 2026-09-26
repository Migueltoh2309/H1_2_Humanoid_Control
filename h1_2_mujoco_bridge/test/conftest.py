"""Configuración común de los tests (corren sin ROS: módulos puros)."""
import os
import sys

import pytest

# El paquete importable está un nivel arriba de test/
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

_MJCF_CANDIDATES = [
    os.environ.get("H12_MJCF_PATH", ""),
    "/home/claude/h1_2_mujoco_bridge_modificado/mjcf/h1_2_scene_qp_reachable.xml",
    os.path.join(_PKG_ROOT, "..", "h1_2_mujoco_bridge", "mjcf",
                 "h1_2_scene_qp_reachable.xml"),
]


@pytest.fixture(scope="session")
def mjcf_path():
    for c in _MJCF_CANDIDATES:
        if c and os.path.isfile(c):
            return c
    pytest.skip("MJCF del H1-2 no encontrado; exporta H12_MJCF_PATH")


class FakeMotorCmd:
    def __init__(self):
        self.mode = 0
        self.q = self.dq = self.tau = self.kp = self.kd = 0.0
        self.reserve = 0


class FakeLowCmd:
    def __init__(self):
        self.mode_pr = 0
        self.mode_machine = 0
        self.motor_cmd = [FakeMotorCmd() for _ in range(35)]
        self.reserve = [0, 0, 0, 0]
        self.crc = 0


@pytest.fixture
def fake_lowcmd():
    return FakeLowCmd()
