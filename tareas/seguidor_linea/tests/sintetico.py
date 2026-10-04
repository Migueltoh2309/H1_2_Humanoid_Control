"""Imagenes IR sinteticas para las pruebas: cada pixel se proyecta al suelo con la
geometria de la camara y se pinta suelo claro o cinta oscura. Sin MuJoCo ni ROS."""
import math

import numpy as np

from seguidor.geometria import ModeloCamara


def camara():
    return ModeloCamara(378.8, 378.8, 320.0, 240.0, 640, 480, altura=1.703, inclinacion=0.8859, x=0.111, y=0.0325)


def recta(x0, y0, rumbo, largo, paso=0.01):
    s = np.arange(0, largo, paso)
    return np.column_stack([x0 + s * math.cos(rumbo), y0 + s * math.sin(rumbo)])


def arco(x0, y0, rumbo, radio, angulo, paso=0.01):
    n = max(2, int(abs(angulo) * radio / paso))
    th = rumbo + np.linspace(0, angulo, n)
    cx, cy = x0 - math.copysign(radio, angulo) * math.sin(rumbo), y0 + math.copysign(radio, angulo) * math.cos(rumbo)
    return np.column_stack([cx + math.copysign(radio, angulo) * np.sin(th), cy - math.copysign(radio, angulo) * np.cos(th)])


def imagen(cam, trazos, ancho=0.05, suelo=210, cinta=10, sombra=None, emisor=False, ruido=3.0, semilla=0,
           roll=0.0, pitch=0.0):
    """trazos: lista de polilineas (N x 2) en el marco del robot, cada una con su
    ancho (tupla (P, ancho)) o con `ancho`. sombra: (x0, x1, y0, y1, factor)."""
    v, u = np.mgrid[0:cam.alto, 0:cam.ancho]
    G = cam.pixel_a_suelo(u.ravel(), v.ravel(), roll, pitch)
    X, Y = G[:, 0], G[:, 1]
    img = np.full(X.shape, float(suelo))
    img[np.isnan(X)] = 40.0                      # por encima del horizonte
    ok = ~np.isnan(X)
    for tr in trazos:
        P, w = tr if isinstance(tr, tuple) else (tr, ancho)
        d = np.full(X.shape, np.inf)
        for i in range(0, len(P), 200):
            B = P[i:i + 200]
            d[ok] = np.minimum(d[ok], np.sqrt((X[ok, None] - B[None, :, 0]) ** 2 + (Y[ok, None] - B[None, :, 1]) ** 2).min(1))
        val = cinta if w >= 0.03 else 150        # una junta del suelo es estrecha y poco oscura
        img[d < w / 2] = val
    if sombra is not None:
        x0, x1, y0, y1, k = sombra
        img[ok & (X > x0) & (X < x1) & (Y > y0) & (Y < y1)] *= k
    rng = np.random.default_rng(semilla)
    img = img.reshape(cam.alto, cam.ancho)
    if emisor:
        p = np.zeros_like(img)
        ys, xs = rng.integers(0, cam.alto, 6000), rng.integers(0, cam.ancho, 6000)
        p[ys, xs] = rng.uniform(0.5, 1.0, 6000)
        img = img * (1 + 0.8 * p)
    img = img + rng.normal(0, ruido, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
