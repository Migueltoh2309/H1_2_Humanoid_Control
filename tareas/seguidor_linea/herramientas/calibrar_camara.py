#!/usr/bin/env python3
"""Altura e inclinacion de la camara con el plano del suelo (reto, 6.1.2):
"la profundidad de una imagen de suelo vacio es un plano; ajustarlo da altura e
inclinacion sin cinta metrica".

  1. Cada pixel de profundidad -> punto 3D en el marco optico (x der, y abajo, z adelante).
  2. RANSAC de un plano (lo que no es suelo -pies, manos, una persona- queda fuera)
     + minimos cuadrados con los inliers.
  3. Normal n hacia arriba (hacia la camara) y distancia d:
        altura      = d
        inclinacion = asin(-n . z_optico)        (bajo la horizontal)
        alabeo      = atan2(n . x_optico, -n . y_optico)

Fuentes:
  --sim                el simulador (/camera/depth/image_raw + camera_info)
  (por defecto)        la D435i en el PC2 (sudo, pyrealsense2)
Promedia --n fotogramas. Escribe datos/calibracion.json y el trozo de config para
copiar en config/seguidor.yaml. En el simulador la verdad es la del generador de
pistas (50.8 grados, 1.703 m + 4 mm de la marcha cinematica) para validar el metodo.

OJO: en el simulador la profundidad es la de la camara de COLOR (alineada al RGB),
que va 15 mm a la derecha de la IR izquierda, con la misma orientacion: altura e
inclinacion valen para las dos. En la D435i real la profundidad esta en el marco
de la IR izquierda.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))


def nube(prof, K, paso=4, zmax=6.0):
    fx, fy, cx, cy = K
    h, w = prof.shape
    v, u = np.mgrid[0:h:paso, 0:w:paso]
    z = prof[::paso, ::paso]
    ok = np.isfinite(z) & (z > 0.2) & (z < zmax)
    u, v, z = u[ok], v[ok], z[ok]
    return np.column_stack([(u - cx) / fx * z, (v - cy) / fy * z, z])


def plano_ransac(P, iters=300, tol=0.01, semilla=0):
    rng = np.random.default_rng(semilla)
    mejor, n_mejor = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        ok = np.abs(P @ n + d) < tol
        if ok.sum() > n_mejor:
            mejor, n_mejor = ok, ok.sum()
    Q = P[mejor]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)    # sin U: con 19 200 puntos serian 3 GB
    n = vt[2]
    d = -n @ c
    if d < 0:                       # que apunte hacia la camara (origen)
        n, d = -n, -d
    return n, d, mejor.mean(), float(np.sqrt(np.mean((Q @ n + d) ** 2)))


def geometria(n, d):
    return d, math.asin(float(np.clip(-n[2], -1, 1))), math.atan2(n[0], -n[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--n", type=int, default=10, help="fotogramas a promediar")
    ap.add_argument("--salida", default=os.path.join(AQUI, "..", "datos", "calibracion.json"))
    a = ap.parse_args()
    if a.sim:
        from seguidor.fuentes import FuenteROS
        f = FuenteROS(ir="/camera/depth/image_raw", info="/camera/depth/camera_info", verdad=False)
        def leer():
            fr = f.leer(3.0)
            return None if fr is None else (fr.ir, fr.K)
    else:
        from seguidor.fuentes import FuenteRealSense
        f = FuenteRealSense(profundidad=True)
        def leer():
            fr = f.leer(3.0)
            return None if fr is None or fr.profundidad is None else (fr.profundidad, fr.K_prof)
    res = []
    while len(res) < a.n:
        r = leer()
        if r is None:
            sys.exit("no llegan fotogramas de profundidad")
        prof, K = r
        P = nube(prof, K)
        n, d, frac, rms = plano_ransac(P, semilla=len(res))
        h, inc, alab = geometria(n, d)
        res.append((h, inc, alab, frac, rms))
        print(f"  fotograma {len(res)}: altura {h:.4f} m, inclinacion {math.degrees(inc):.2f} grados, "
              f"alabeo {math.degrees(alab):+.2f} grados ({frac * 100:.0f} % suelo, residuo {rms * 1000:.1f} mm)")
        time.sleep(0.1)
    R = np.array(res)
    m, s = R.mean(0), R.std(0)
    out = {"altura": round(float(m[0]), 4), "inclinacion": round(float(m[1]), 5), "alabeo": round(float(m[2]), 5),
           "inclinacion_deg": round(math.degrees(m[1]), 3), "alabeo_deg": round(math.degrees(m[2]), 3),
           "dispersion": {"altura_mm": round(float(s[0]) * 1000, 2), "inclinacion_deg": round(math.degrees(s[1]), 3)},
           "fotogramas": a.n, "fuente": "simulador" if a.sim else "D435i", "fecha": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs(os.path.dirname(os.path.abspath(a.salida)), exist_ok=True)
    with open(a.salida, "w") as fj:
        json.dump(out, fj, indent=2)
    print(f"\naltura {out['altura']} m (+-{out['dispersion']['altura_mm']} mm), inclinacion {out['inclinacion_deg']} "
          f"grados (+-{out['dispersion']['inclinacion_deg']}), alabeo {out['alabeo_deg']} grados -> {a.salida}")
    print("\nPara config/seguidor.yaml (camara:):")
    print(f"  altura: {out['altura']}\n  inclinacion: {out['inclinacion']}\n  alabeo: {out['alabeo']}")
    if hasattr(f, "cerrar"):
        f.cerrar()


if __name__ == "__main__":
    main()
