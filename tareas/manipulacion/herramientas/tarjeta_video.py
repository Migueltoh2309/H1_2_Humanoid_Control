#!/usr/bin/env python3
"""Un video hecho solo de tarjetas (resultados, graficas), 1280x720 H.264.
Cada elemento: un .png (imagen con el titulo) o un .txt (primera linea = titulo).

    python3 herramientas/tarjeta_video.py --salida videos/cierre.mp4 --titulo "Cierre" resumen.txt grafica.png
"""
import argparse
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from video_comun import EscritorVideo, tarjeta, tarjeta_imagen   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("elementos", nargs="+")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--titulo", default="")
    ap.add_argument("--segundos", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    vid = EscritorVideo(a.salida, a.fps)
    for e in a.elementos:
        if e.endswith(".png"):
            c = tarjeta_imagen(e, a.titulo)
        else:
            lineas = open(e, encoding="utf-8").read().splitlines()
            c = tarjeta(lineas[0] if lineas else a.titulo, lineas[1:])
        vid.escribir(c, veces=int(a.segundos * a.fps))
    print(f"video: {a.salida} ({vid.cerrar():.0f} s)")


if __name__ == "__main__":
    main()
