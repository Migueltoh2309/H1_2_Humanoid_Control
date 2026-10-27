#!/usr/bin/env python3
"""Video del hito 2 (percepcion sin robot, sobre un conjunto de datos grabado): la
misma percepcion con umbral ADAPTATIVO (izquierda) y FIJO (derecha), fotograma a
fotograma, con la linea detectada superpuesta, la mascara de la vista cenital, el
porcentaje de fotogramas con deteccion y el tiempo por fotograma. Los resultados
(y la salida de pytest si se da --pytest) van a <salida>_resultados.txt / .json;
con --tarjetas, tambien como tarjetas dentro del video.

    python3 herramientas/video_percepcion.py datos/dataset_n3_XXXX --salida videos/hito2.mp4 [--pytest log]
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
sys.path.insert(0, AQUI)
from evaluar_percepcion import distancia_a_linea, en_sombra       # noqa: E402
from seguidor import config                                       # noqa: E402
from seguidor.fuentes import FuenteDataset                        # noqa: E402
from seguidor.percepcion import Percepcion                        # noqa: E402
from video_comun import ANCHO, ALTO, GRIS, EscritorVideo, Lienzo, tarjeta   # noqa: E402

TIT_H = 48
PW, PH = 640, 480


def panel(f, med, per, r, p, nombre):
    img = cv2.cvtColor(f.ir, cv2.COLOR_GRAY2RGB)
    if len(med.puntos):
        uv = per.cam.suelo_a_pixel(med.puntos[:, 0], med.puntos[:, 1], r, p)
        for u, v in uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (int(u), int(v)), 4, (40, 230, 60), -1)
    bev, _, mk = per.ultimo_bev
    b = cv2.cvtColor(np.clip(bev, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    b[mk] = (230, 40, 40)
    b = cv2.flip(cv2.rotate(b, cv2.ROTATE_90_COUNTERCLOCKWISE), 1) if b.shape[0] < b.shape[1] else b
    esc = 190 / b.shape[0]
    b = cv2.resize(b, (int(b.shape[1] * esc), 190), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (PW, PH))
    img[8:8 + b.shape[0], PW - 8 - b.shape[1]:PW - 8] = b
    cv2.rectangle(img, (PW - 9 - b.shape[1], 7), (PW - 7, 9 + b.shape[0]), (255, 255, 255), 1)
    return img


class Cuenta:
    def __init__(self):
        self.n = self.det = self.buenas = 0
        self.ms, self.luz, self.sombra = [], [], []

    def sumar(self, med, conf_min):
        self.n += 1
        self.det += med.valida
        self.buenas += med.valida and med.confianza >= conf_min
        self.ms.append(med.ms)

    def resumen(self):
        ms = np.array(self.ms) if self.ms else np.zeros(1)
        return (f"detección {100 * self.det / max(1, self.n):.0f} %  ·  confianza sobre umbral "
                f"{100 * self.buenas / max(1, self.n):.0f} %  ·  {ms.mean():.1f} ms/fotograma (p95 "
                f"{np.percentile(ms, 95):.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--pytest", default=None, help="fichero con la salida de pytest para la tarjeta final")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--tarjetas", action="store_true",
                    help="tarjetas de presentacion y de resultados dentro del video (por defecto, solo en el .txt)")
    a = ap.parse_args()
    nombre = os.path.basename(os.path.normpath(a.carpeta))
    pista = None
    if nombre.startswith("dataset_n") and nombre[9].isdigit():
        ruta = os.path.join(AQUI, "..", "sim", "pistas", f"pista_nivel{nombre[9]}.json")
        if os.path.isfile(ruta):
            pista = json.load(open(ruta))
    cfgs = {}
    for u in ("adaptativo", "fijo"):
        c = config.cargar()
        c["percepcion"]["umbral"] = u
        cfgs[u] = c
    conf_min = cfgs["adaptativo"]["estimacion"]["confianza_min"]
    pers = {u: Percepcion(c) for u, c in cfgs.items()}
    cuentas = {u: Cuenta() for u in cfgs}
    ds = FuenteDataset(a.carpeta)
    vid = EscritorVideo(a.salida, a.fps)
    if a.tarjetas:
        vid.escribir(tarjeta("Hito 2 · Percepción sin robot", [
            "La percepción, desarrollada y probada en el PC sobre un recorrido grabado",
            f"({nombre}: {len(ds)} fotogramas IR + IMU con marca de tiempo, nivel con zona en sombra).",
            "",
            "Izquierda: umbral ADAPTATIVO (contraste relativo al fondo local).",
            "Derecha: umbral FIJO (nivel de gris), para comparar (reto 6.2.2).",
            "Verde: puntos de la línea detectada, reproyectados sobre la imagen.",
            "Recuadro: vista cenital; en rojo, lo que el umbral considera cinta.",
        ], "Entregable: vídeo con la línea detectada, % de fotogramas con detección y tiempo por fotograma"),
            veces=5 * a.fps)
    r0 = p0 = None
    while True:
        f = ds.leer()
        if f is None:
            break
        fila = f.fila
        r, p = float(fila["roll"]), float(fila["pitch"])
        if r0 is None:
            r0, p0 = r, p
        L = Lienzo()
        L.rect(0, 0, ANCHO, TIT_H, (16, 18, 22))
        L.texto(16, 12, "Hito 2 · Percepción sobre el conjunto de datos", 22, negrita=True)
        L.texto(ANCHO - 16, 14, f"fotograma {cuentas['fijo'].n + 1}/{len(ds)}", 18, GRIS, mono=True, ancla="ra")
        for k, u in enumerate(("adaptativo", "fijo")):
            med = pers[u].procesar(f, r - r0, p - p0)
            cuentas[u].sumar(med, conf_min)
            c = cuentas[u]
            if pista and fila.get("gt_x") and med.valida and len(med.puntos):
                gx, gy, gw = float(fila["gt_x"]), float(fila["gt_y"]), float(fila["gt_yaw"])
                P = med.puntos
                X = gx + P[:, 0] * math.cos(gw) - P[:, 1] * math.sin(gw)
                Y = gy + P[:, 0] * math.sin(gw) + P[:, 1] * math.cos(gw)
                d = distancia_a_linea(pista, X, Y)
                sm = en_sombra(pista, X, Y)
                c.sombra += d[sm].tolist()
                c.luz += d[~sm].tolist()
            x0 = k * PW
            L.pegar(x0, TIT_H, panel(f, med, pers[u], r - r0, p - p0, u))
            L.rect(x0, TIT_H, x0 + 230, TIT_H + 30, (0, 0, 0), alfa=0.6)
            L.texto(x0 + 10, TIT_H + 5, f"umbral {u.upper()}", 18, negrita=True)
            y = TIT_H + PH + 10
            if med.valida:
                txt = (f"conf {med.confianza:.2f}  desp {med.desplazamiento * 100:+.1f} cm  "
                       f"ang {math.degrees(med.angulo):+.1f}°  {med.ms:.1f} ms")
                col = (120, 230, 140) if med.confianza >= conf_min else (240, 200, 90)
            else:
                txt, col = f"sin línea  ({med.ms:.1f} ms)", (240, 110, 110)
            if med.barra is not None:
                txt += f"  · barra a {med.barra:.2f} m"
            L.texto(x0 + 14, y, txt, 17, col, mono=True)
            L.texto(x0 + 14, y + 32, "Acumulado:", 15, GRIS)
            L.texto(x0 + 14, y + 54, c.resumen(), 15)
            if c.luz or c.sombra:
                def cm(v):
                    return f"{100 * np.mean(v):.1f} cm" if v else "—"
                def fal(v):
                    return f"{100 * np.mean(np.array(v) > 0.10):.0f} %" if v else "—"
                L.texto(x0 + 14, y + 80, f"error a la cinta real: luz {cm(c.luz)}, sombra {cm(c.sombra)}", 15)
                L.texto(x0 + 14, y + 102, f"puntos falsos (> 10 cm): luz {fal(c.luz)}, sombra {fal(c.sombra)}", 15)
        sombra_aqui = False
        if pista and fila.get("gt_x"):
            gx, gy = float(fila["gt_x"]), float(fila["gt_y"])
            gw = float(fila["gt_yaw"])
            sombra_aqui = bool(en_sombra(pista, np.array([gx + 1.2 * math.cos(gw)]),
                                         np.array([gy + 1.2 * math.sin(gw)]))[0])
        if sombra_aqui:
            L.rect(ANCHO // 2 - 110, TIT_H + 8, ANCHO // 2 + 110, TIT_H + 40, (230, 160, 30))
            L.texto(ANCHO // 2, TIT_H + 13, "ZONA EN SOMBRA", 18, negrita=True, ancla="ma")
        L.rect(PW - 1, TIT_H, PW + 1, ALTO, (60, 60, 70))
        vid.escribir(L.final())

    lineas = []
    res = {}
    for u in ("adaptativo", "fijo"):
        c = cuentas[u]
        ms = np.array(c.ms)
        res[u] = {"fotogramas": c.n, "pct_deteccion": round(100 * c.det / max(1, c.n), 1),
                  "pct_confianza": round(100 * c.buenas / max(1, c.n), 1),
                  "ms_media": round(float(ms.mean()), 2), "ms_p95": round(float(np.percentile(ms, 95)), 2)}
        lineas.append(f"Umbral {u}: {c.resumen()}")
        for zona, v in (("luz", c.luz), ("sombra", c.sombra)):
            if v:
                d = np.array(v) * 100
                lineas.append(f"  {zona:7s} {len(d):6d} puntos: error medio {d.mean():5.1f} cm, p95 "
                              f"{np.percentile(d, 95):5.1f} cm, falsos (> 10 cm) {100 * (d > 10).mean():5.1f} %")
        lineas.append("")
    lineas.append("Tiempo medido en el PC de desarrollo: en el PC2 del robot hay que volver a medirlo.")
    if a.pytest and os.path.isfile(a.pytest):
        lineas += ["", "Pruebas automáticas (python3 -m pytest tests -q):"]
        lineas += ["  " + s for s in open(a.pytest).read().strip().splitlines()[-2:]]
    if a.tarjetas:
        vid.escribir(tarjeta("Hito 2 · Resultados", lineas, nombre), veces=9 * a.fps)
    s = vid.cerrar()
    base = os.path.splitext(a.salida)[0]
    json.dump(res, open(base + "_resultados.json", "w"), indent=2)
    with open(base + "_resultados.txt", "w", encoding="utf-8") as ft:
        ft.write(f"Hito 2 · Resultados ({nombre})\n\n" + "\n".join(lineas) + "\n")
    print("\n".join(lineas))
    print(f"video: {a.salida} ({s:.0f} s)")


if __name__ == "__main__":
    main()
