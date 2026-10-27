#!/usr/bin/env python3
"""Reproduce en MuJoCo una tirada registrada (hito 3: "las trayectorias se calculan, se registran y
se reproducen en MuJoCo, sin enviarse al robot"). Cinemático: pone en el modelo de la estación las
juntas DESEADAS del brazo (o las medidas, --medidas) y los ángulos de la mano del lazo.csv, y graba
un vídeo desde una cámara exterior con el estado del supervisor sobreimpreso.

    MUJOCO_GL=egl python3 herramientas/reproducir.py datos/tiradas/nivel2_<fecha> [--fps 10] [--medidas]
"""
import argparse
import csv
import json
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sim"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tirada")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--medidas", action="store_true", help="juntas medidas en vez de las deseadas")
    ap.add_argument("--ancho", type=int, default=960)
    ap.add_argument("--alto", type=int, default=600)
    a = ap.parse_args()
    import cv2
    import mujoco
    from construir_estacion import construir
    from manip import cinematica as C
    from manip.config import cargar, cargar_objetos
    from manip.modelo import Modelo
    meta = json.load(open(os.path.join(a.tirada, "eventos.json")))["meta"] \
        if os.path.isfile(os.path.join(a.tirada, "eventos.json")) else {}
    esc = meta.get("escena", {})
    nivel, sem = esc.get("nivel", 0), esc.get("semilla") or 0
    xml, _ = construir(nivel if nivel in (0, 1, 2, 3, 4) else 0, sem, cargar(), cargar_objetos(),
                       salida=tempfile.mkdtemp())
    mod = Modelo(xml)
    m, d = mod.m, mod.d
    filas = list(csv.DictReader(open(os.path.join(a.tirada, "lazo.csv"))))
    pre = "q_" if a.medidas else "qd_"
    cols = {"izq": [f"{pre}i{k}" for k in range(7)], "der": [f"{pre}d{k}" for k in range(7)]}
    r = mujoco.Renderer(m, a.alto, a.ancho)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.35, 0.0, 1.0]
    cam.distance, cam.azimuth, cam.elevation = 1.9, 155, -28
    salida = os.path.join(a.tirada, "reproduccion_medidas.mp4" if a.medidas else "reproduccion.mp4")
    vw = cv2.VideoWriter(salida, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.ancho, a.alto))
    t_sig, n = 0.0, 0
    for f in filas:
        t = float(f["t"])
        if t < t_sig:
            continue
        t_sig = t + 1.0 / a.fps
        try:
            q = {lado: np.array([float(f[c]) for c in cols[lado]]) for lado in C.LADOS}
        except ValueError:
            continue
        for lado in C.LADOS:
            ang = [f.get(f"{lado}_a{k}") for k in range(6)]
            if all(v not in (None, "", "nan") for v in ang):
                mod.poner_mano(lado, [float(v) for v in ang])
        mod.poner(q["izq"], q["der"])
        r.update_scene(d, cam)
        img = r.render()[..., ::-1].copy()
        cv2.putText(img, f"t {t:6.1f} s  {f['estado']} {f['objeto']}  peso {float(f['peso']):.2f}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
        n += 1
    vw.release()
    r.close()
    print(f"{n} cuadros -> {salida}")


if __name__ == "__main__":
    main()
