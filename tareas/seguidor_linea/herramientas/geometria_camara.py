#!/usr/bin/env python3
"""Respuestas numericas a 6.1.1 y 6.1.3 del reto a partir de los intrinsecos
(sec. 3) y de la altura/inclinacion (config o datos/calibracion.json):

  * campo de vision vertical de color y de IR, y cual ve mas suelo;
  * distancia a la que empieza a verse el suelo (zona ciega) y hasta donde llega,
    en el eje de la imagen;
  * tamano en el suelo de un pixel cerca y lejos (resolucion util).

    python3 herramientas/geometria_camara.py [--calibracion datos/calibracion.json]
"""
import argparse
import json
import math
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))

from seguidor import config                      # noqa: E402
from seguidor.geometria import ModeloCamara      # noqa: E402

COLOR = (604.1, 603.6, 321.3, 236.4)             # reto, sec. 3
IR = (378.8, 378.8, 321.5, 236.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibracion", default=os.path.join(AQUI, "..", "datos", "calibracion.json"))
    a = ap.parse_args()
    cfg = config.cargar()
    origen = "config/seguidor.yaml"
    if os.path.isfile(a.calibracion):
        cal = json.load(open(a.calibracion))
        for k in ("altura", "inclinacion", "alabeo"):
            cfg["camara"][k] = cal[k]
        origen = f"{a.calibracion} ({cal.get('fuente')}, {cal.get('fecha')})"
    c = cfg["camara"]
    print(f"Montaje ({origen}): altura {c['altura']:.3f} m, inclinacion {math.degrees(c['inclinacion']):.1f} grados, "
          f"camara {c['x']:.3f} m delante del centro del robot\n")
    for nombre, K in (("color", COLOR), ("IR", IR)):
        cam = ModeloCamara.desde_config(cfg, K=K)
        cerca, lejos = cam.alcance_suelo()
        # tamano de un pixel en el suelo en las filas de abajo y de arriba (centro de la imagen)
        def px_suelo(v):
            a_, b_ = cam.pixel_a_suelo([cam.cx, cam.cx + 1], [v, v])
            c_, d_ = cam.pixel_a_suelo([cam.cx, cam.cx], [v, v + 1])
            return abs(a_[1] - b_[1]), abs(c_[0] - d_[0])
        abajo, arriba = px_suelo(cam.alto - 2), px_suelo(max(2, cam.cy - cam.fy * math.tan(cam.inclinacion - 0.15)))
        print(f"{nombre:5s} fx={K[0]:.1f}: FOV vertical {math.degrees(cam.fov_vertical()):.1f} grados, horizontal "
              f"{math.degrees(cam.fov_horizontal()):.1f} grados")
        print(f"       suelo visible en el eje de la imagen: de {cerca:.2f} m a {lejos:.2f} m del centro del robot")
        print(f"       un pixel en el suelo: {abajo[0] * 100:.2f} x {abajo[1] * 100:.2f} cm abajo, "
              f"{arriba[0] * 100:.2f} x {arriba[1] * 100:.2f} cm cerca del horizonte util\n")
    print("La IR ve mas suelo (FOV vertical 64.7 frente a 43.4 grados): zona ciega mas corta y mas alcance.\n"
          "La zona ciega REAL es mayor: los pies y las manos tapan la parte de abajo de la imagen "
          "(config percepcion.mascara_robot). El alcance UTIL lo limita la resolucion: la cinta de 5 cm "
          "debe ocupar >= 2-3 px de ancho.")


if __name__ == "__main__":
    main()
