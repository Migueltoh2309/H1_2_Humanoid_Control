#!/usr/bin/env python3
"""Evalua la PERCEPCION sobre un conjunto de datos grabado (reto, hito 2), sin robot:

  * porcentaje de fotogramas con deteccion (y con confianza sobre el umbral);
  * tiempo por fotograma (media y p95; en el PC2 sera mayor: medirlo alli);
  * umbral FIJO frente a ADAPTATIVO (6.2.2), dentro y fuera de la zona en sombra;
  * con verdad de terreno (datasets del simulador): distancia de cada punto
    detectado a la linea real (cm), y falsos positivos (puntos a > 10 cm);
  * video con la linea detectada superpuesta (--video).

    python3 herramientas/evaluar_percepcion.py datos/dataset_n3_XXXX [--video] [--umbral fijo|adaptativo|ambos]
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))

from seguidor import config                          # noqa: E402
from seguidor.fuentes import FuenteDataset           # noqa: E402
from seguidor.percepcion import Percepcion           # noqa: E402


def en_sombra(pista, X, Y):
    s = pista.get("sombra") if pista else None
    if not s:
        return np.zeros(len(X), bool)
    c, si = math.cos(s["rumbo"]), math.sin(s["rumbo"])
    dx, dy = X - s["x"], Y - s["y"]
    u, v = dx * c + dy * si, -dx * si + dy * c
    return (np.abs(u) < s["largo"] / 2) & (np.abs(v) < s["ancho"] / 2)


def distancia_a_linea(pista, X, Y):
    L = np.array(pista["linea"])[:, :2]
    d = np.full(len(X), np.inf)
    for i in range(0, len(L), 400):                 # por bloques: memoria acotada
        B = L[i:i + 400]
        d = np.minimum(d, np.sqrt((X[:, None] - B[None, :, 0]) ** 2 + (Y[:, None] - B[None, :, 1]) ** 2).min(1))
    return d


def evaluar(carpeta, umbral, pista, video=None):
    cfg = config.cargar()
    cfg["percepcion"]["umbral"] = umbral
    conf_min = cfg["estimacion"]["confianza_min"]
    per = Percepcion(cfg)
    ds = FuenteDataset(carpeta)
    r0 = p0 = None
    n = det = buenas = 0
    ms, dist_sombra, dist_luz = [], [], []
    escritor = None
    while True:
        f = ds.leer()
        if f is None:
            break
        fila = f.fila
        roll, pitch = float(fila["roll"]), float(fila["pitch"])
        if r0 is None:
            r0, p0 = roll, pitch          # referencia: el primer fotograma (robot quieto de pie)
        med = per.procesar(f, roll - r0, pitch - p0)
        n += 1
        det += med.valida
        buenas += med.valida and med.confianza >= conf_min
        ms.append(med.ms)
        if pista and fila.get("gt_x") and med.valida and len(med.puntos):
            gx, gy, gw = float(fila["gt_x"]), float(fila["gt_y"]), float(fila["gt_yaw"])
            P = med.puntos
            X = gx + P[:, 0] * math.cos(gw) - P[:, 1] * math.sin(gw)
            Y = gy + P[:, 0] * math.sin(gw) + P[:, 1] * math.cos(gw)
            d = distancia_a_linea(pista, X, Y)
            sm = en_sombra(pista, X, Y)
            dist_sombra += d[sm].tolist()
            dist_luz += d[~sm].tolist()
        if video:
            img = cv2.cvtColor(f.ir, cv2.COLOR_GRAY2BGR)
            if len(med.puntos):
                uv = per.cam.suelo_a_pixel(med.puntos[:, 0], med.puntos[:, 1], roll - r0, pitch - p0)
                for u, v in uv:
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        cv2.circle(img, (int(u), int(v)), 3, (0, 255, 0), -1)
            txt = (f"{umbral} conf {med.confianza:.2f} desp {med.desplazamiento * 100:+.1f} cm ang "
                   f"{math.degrees(med.angulo):+.1f}" if med.valida else f"{umbral}: sin linea")
            if med.barra is not None:
                txt += f" | barra {med.barra:.2f} m"
            if med.esquina is not None:
                txt += f" | esquina {med.esquina[0]:.2f} m {'izq' if med.esquina[1] > 0 else 'der'}"
            cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            bev, _, mk = per.ultimo_bev
            b = cv2.cvtColor(np.clip(bev, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            b[mk] = (0, 0, 255)
            b = cv2.resize(b, (int(b.shape[1] * img.shape[0] / b.shape[0]), img.shape[0]))
            cuadro = np.hstack([img, b])
            if escritor is None:
                escritor = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (cuadro.shape[1], cuadro.shape[0]))
            escritor.write(cuadro)
    if escritor is not None:
        escritor.release()
    def est(d):
        d = np.array(d) * 100
        if len(d) == 0:
            return None
        return {"n": int(len(d)), "media_cm": round(float(d.mean()), 2), "p95_cm": round(float(np.percentile(d, 95)), 2),
                "pct_falsos_10cm": round(100.0 * float((d > 10).mean()), 2)}
    return {"umbral": umbral, "fotogramas": n, "pct_deteccion": round(100.0 * det / max(1, n), 1),
            "pct_confianza": round(100.0 * buenas / max(1, n), 1),
            "ms_media": round(float(np.mean(ms)), 2) if ms else None,
            "ms_p95": round(float(np.percentile(ms, 95)), 2) if ms else None,
            "puntos_luz": est(dist_luz), "puntos_sombra": est(dist_sombra)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta")
    ap.add_argument("--umbral", choices=["fijo", "adaptativo", "ambos"], default="ambos")
    ap.add_argument("--pista", default=None, help="json de la pista (verdad); por defecto el del nivel")
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    pista = None
    ruta = a.pista
    if ruta is None:
        nombre = os.path.basename(os.path.normpath(a.carpeta))
        if nombre.startswith("dataset_n") and nombre[9].isdigit():
            ruta = os.path.join(AQUI, "..", "sim", "pistas", f"pista_nivel{nombre[9]}.json")
    if ruta and os.path.isfile(ruta):
        pista = json.load(open(ruta))
    umbrales = ["fijo", "adaptativo"] if a.umbral == "ambos" else [a.umbral]
    res = []
    for u in umbrales:
        video = os.path.join(a.carpeta, f"percepcion_{u}.mp4") if a.video else None
        r = evaluar(a.carpeta, u, pista, video)
        res.append(r)
        print(f"\n[{u}] {r['fotogramas']} fotogramas: deteccion {r['pct_deteccion']} %, confianza sobre umbral "
              f"{r['pct_confianza']} %, {r['ms_media']} ms/fotograma (p95 {r['ms_p95']})")
        for zona in ("puntos_luz", "puntos_sombra"):
            z = r[zona]
            if z:
                print(f"    {zona:14s} {z['n']:6d} puntos: distancia a la linea real media {z['media_cm']} cm, "
                      f"p95 {z['p95_cm']} cm, falsos (> 10 cm) {z['pct_falsos_10cm']} %")
        if video:
            print(f"    video: {video}")
    json.dump(res, open(os.path.join(a.carpeta, "evaluacion_percepcion.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
