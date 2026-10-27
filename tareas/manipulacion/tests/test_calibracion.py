"""El ajuste AX = YB de la calibración encuentra una cámara y unos marcadores conocidos a
partir de sus esquinas en la imagen (datos sintéticos con ruido)."""
import math
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from manip import calibracion as K_  # noqa: E402
from manip import cinematica as C  # noqa: E402
from manip.config import cargar  # noqa: E402

pytest.importorskip("scipy")
yaml = pytest.importorskip("yaml")
K = np.array([[378.8, 0, 320.0], [0, 378.8, 240.0], [0, 0, 1.0]])
LADO = 0.04


def _datos(rng, ruido_px=0.3, error_q=0.0):
    cfg = cargar()
    X_nom = K_.camara_nominal(cfg)
    dX = K_.T_de(K_.rotvec_a_R(np.radians([1.5, -1.0, 0.8])), np.array([0.008, -0.005, 0.01]))
    X = X_nom @ dX
    Y_nom = {lado: K_.marcador_nominal(lado) for lado in C.LADOS}
    Y = {lado: Y_nom[lado] @ K_.T_de(K_.rotvec_a_R(np.radians([3, -2, 4])), np.array([0.004, 0.002, -0.003]))
         for lado in C.LADOS}
    poses = yaml.safe_load(open(os.path.join(RAIZ, "config", "poses.yaml")))["calibracion"]
    obj_h = np.column_stack([K_.ESQUINAS_ARUCO * LADO, np.ones(4)])
    muestras = []
    for lado in C.LADOS:
        for q in poses[lado]:
            q = np.array(q)
            q_med = q + rng.normal(0, error_q, 7)       # el brazo no está donde dicen las juntas
            T = C.fk(lado, q_med) @ Y[lado]
            P = (K_.inv(X) @ T @ obj_h.T).T[:, :3]
            esq = K_.proyectar(K, P) + rng.normal(0, ruido_px, (4, 2))
            muestras.append({"lado": lado, "q_medida": q, "q_consigna": q, "esquinas": esq,
                             "T_cam_marcador": K_.inv(X) @ T})
    return X_nom, Y_nom, X, Y, muestras


def test_recupera_camara_y_marcadores():
    rng = np.random.default_rng(1)
    X_nom, Y_nom, X, Y, mu = _datos(rng)
    Xe, Ye, inf = K_.resolver(mu, K, LADO, X_nom, Y_nom)
    assert inf["convergio"]
    assert np.linalg.norm(Xe[:3, 3] - X[:3, 3]) < 0.002          # < 2 mm
    assert K_.ang_deg(Xe[:3, :3], X[:3, :3]) < 0.15              # < 0.15 grados
    for lado in C.LADOS:
        assert np.linalg.norm(Ye[lado][:3, 3] - Y[lado][:3, 3]) < 0.003
    assert inf["reproy_px_rms"] < 0.6


def test_juntas_erroneas_aparecen_en_el_residuo():
    """Si las juntas no dicen dónde está la mano (brazo ~5 grados corto, 6.1.4), el residuo
    sube: la calibración no puede absorber un error que cambia de pose a pose."""
    rng = np.random.default_rng(2)
    X_nom, Y_nom, X, Y, mu = _datos(rng, error_q=math.radians(2.0))
    Xe, Ye, inf = K_.resolver(mu, K, LADO, X_nom, Y_nom)
    assert inf["reproy_px_rms"] > 2.0


def test_detecta_marcador_dibujado():
    cv2 = pytest.importorskip("cv2")
    sys.path.insert(0, os.path.join(RAIZ, "sim"))
    from construir_estacion import celdas_aruco
    img = np.full((480, 640), 200, np.uint8)
    cel = celdas_aruco(10)
    img[200:296, 280:376] = np.kron(np.where(cel, 0, 255), np.ones((16, 16))).astype(np.uint8)
    det = K_.detectar(img, K, LADO)
    assert 10 in det and det[10]["reproy_px"] < 0.5
    # 96 px de marcador de 4 cm con f = 378.8 -> a ~16 cm
    assert abs(det[10]["T_cam_marcador"][2, 3] - 378.8 * LADO / 96 * (6 / 6)) < 0.01
    assert cv2 is not None
