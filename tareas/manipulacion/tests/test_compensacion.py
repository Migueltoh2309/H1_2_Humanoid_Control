"""Compensación del error del brazo y del cuerpo (6.3.4): la corrección por la IMU con el pivote
y el modelo de gravedad con su lazo."""
import math
import os
import sys
import tempfile

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sim"))
from manip.compensacion import por_cuerpo, rot_rpy  # noqa: E402


def test_por_cuerpo():
    piv = np.array([0.02, 0.0, -0.56])
    p = np.array([0.40, 0.10, -0.05])
    R0 = rot_rpy([0.0, 0.0, 0.3])
    assert np.allclose(por_cuerpo(p, R0, R0, piv), p)                  # sin giro, nada
    assert np.allclose(por_cuerpo(piv, R0, rot_rpy([0.02, 0.03, 0.3]), piv), piv)   # el pivote no se mueve
    # el torso se inclina hacia delante (pitch +): un punto delante y abajo se ve más cerca del torso en x
    p1 = por_cuerpo(p, rot_rpy([0, 0, 0]), rot_rpy([0, math.radians(1.0), 0]), piv)
    assert abs(np.linalg.norm(p1 - piv) - np.linalg.norm(p - piv)) < 1e-9  # sólido rígido
    assert 0.005 < np.linalg.norm(p1 - p) < 0.02                         # ~1 cm por grado a 0.6 m


def test_rot_rpy_ortonormal():
    R = rot_rpy([0.1, -0.2, 0.7])
    assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-9


def test_compensador_topes():
    pytest.importorskip("mujoco")
    from construir_estacion import construir
    from manip.compensacion import Compensador
    from manip.config import cargar, cargar_objetos
    xml, _ = construir(0, 0, cargar(), cargar_objetos(), salida=tempfile.mkdtemp())
    c = Compensador(xml, kp=100.0, tope_lazo=0.05, tope_total=0.1)
    q = np.array([-0.3, 0.3, 0.0, 0.6, 0.0, 0.0, 0.0])
    g = c.gravedad("izq", q)
    assert np.abs(g).max() > 0.5                                           # el brazo pesa
    d0 = c.desvio("izq", q)
    assert np.allclose(d0, np.clip(g / 100.0, -0.1, 0.1))
    c.poner_carga("izq", 0.25, np.array([0.22, -0.06, 0.0]))
    assert np.abs(c.desvio("izq", q)).max() >= np.abs(d0).max()              # con carga, más
    for _ in range(10):
        c.corregir("izq", q, q - 0.02)                                      # 0.02 corto cada vez
    assert np.allclose(c.lazo["izq"], 0.05)                                 # con tope
