"""Nivel 1: el detector de dianas sobre una IR1 sintética (tablero claro, discos oscuros de 2 cm,
puntos del emisor) recupera los centros con la pose de cámara nominal."""
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
cv2 = pytest.importorskip("cv2")


def test_detecta_dianas():
    from manip import calibracion as CAL
    from manip.config import cargar
    from manip.dianas import detectar, rayos_a_plano
    from manip.estacion import Estacion
    from manip.tipos import Fotograma
    cfg = cargar()
    est = Estacion(cfg)
    T = CAL.camara_nominal(cfg, ir=True)
    ci = cfg["camara"]["ir"]
    K = np.array([[ci["fx"], 0, ci["ancho"] / 2], [0, ci["fy"], ci["alto"] / 2], [0, 0, 1.0]])
    z = est.z_mesa - est.torso_z
    plano = np.array([0, 0, 1.0, -z])
    dianas = np.array([[0.36, 0.08], [0.42, -0.10], [0.31, -0.02], [0.40, 0.14]])
    h, w = ci["alto"], ci["ancho"]
    img = np.full((h, w), 200, np.uint8)
    Ti = np.linalg.inv(T)
    for x, y in dianas:
        pc = Ti[:3, :3] @ np.array([x, y, z]) + Ti[:3, 3]
        u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
        r = K[0, 0] * 0.01 / pc[2]
        cv2.circle(img, (int(round(u)), int(round(v))), int(round(r)), 110, -1)
    rng = np.random.default_rng(0)
    pts = rng.integers(0, [w, h], size=(3000, 2))
    img[pts[:, 1], pts[:, 0]] = 255                                          # puntos del emisor
    f = Fotograma(t=0.0, color=None, prof=np.zeros((h, w), np.float32), K_color=K, K_prof=K, ir=img)
    det = detectar(f, T, plano, est.zona)
    assert len(det) == 4
    for x, y in dianas:
        e = min(np.hypot(p[0] - x, p[1] - y) for p, _, _ in det)
        assert e < 0.004, e
    # y el rayo del centro de la imagen cae en la mesa
    assert abs(rayos_a_plano(np.array([w / 2]), np.array([h / 2]), K, T, plano)[0][2] - z) < 1e-9


def test_detecta_cruz():
    from manip import calibracion as CAL
    from manip.config import cargar
    from manip.dianas import detectar_cruz
    from manip.estacion import Estacion
    from manip.tipos import Fotograma
    cfg = cargar()
    est = Estacion(cfg)
    T = CAL.camara_nominal(cfg, ir=True)
    ci = cfg["camara"]["ir"]
    K = np.array([[ci["fx"], 0, ci["ancho"] / 2], [0, ci["fy"], ci["alto"] / 2], [0, 0, 1.0]])
    z = est.z_mesa - est.torso_z
    plano = np.array([0, 0, 1.0, -z])
    c = np.array([est.tablero.cx + 0.012, est.tablero.cy - 0.008])     # no exactamente en el centro
    h, w = ci["alto"], ci["ancho"]
    img = np.full((h, w), 200, np.uint8)
    Ti = np.linalg.inv(T)

    def px(x, y):
        pc = Ti[:3, :3] @ np.array([x, y, z]) + Ti[:3, 3]
        return int(round(K[0, 0] * pc[0] / pc[2] + K[0, 2])), int(round(K[1, 1] * pc[1] / pc[2] + K[1, 2]))
    for d in ((0.05, 0), (0, 0.05)):
        a, b = px(c[0] - d[0], c[1] - d[1]), px(c[0] + d[0], c[1] + d[1])
        cv2.line(img, a, b, 20, 4)
    p = detectar_cruz(Fotograma(0.0, None, np.zeros((h, w), np.float32), K, K, ir=img), T, plano,
                      (est.tablero.cx, est.tablero.cy))
    assert p is not None and np.hypot(p[0] - c[0], p[1] - c[1]) < 0.004
