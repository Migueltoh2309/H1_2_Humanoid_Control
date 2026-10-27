"""Bloque REGISTRO (sec. 5): todo lo necesario para reconstruir una tirada después.

  datos/tiradas/nivelN_<fecha>/
    lazo.csv      a la frecuencia del brazo (50 Hz): t, estado del supervisor, objeto, peso de
                  arm_sdk, q medida / deseada / publicada de las 15 juntas, IMU (rpy), y de
                  cada mano los 6 ángulos y las 6 fuerzas (último sondeo)
    eventos.json  eventos con su instante (estados, percepciones, planes, agarres, paradas) y
                  las métricas de la sección 9 (por objeto y de la tirada)
    *.png         la imagen de la cámara en cada observación y en cada agarre

Un hilo propio escribe el CSV; el supervisor solo llama a `evento`, `objeto` y `imagen`.
"""
import csv
import json
import os
import threading
import time

import numpy as np

from .robot_io import BRAZOS

CAB_J = ["cintura"] + [f"{ld}{k}" for ld in ("i", "d") for k in range(7)]


class Registro:
    def __init__(self, raiz, nivel, estado, brazo, hz=50.0, extra=None):
        self.dir = os.path.join(raiz, f"nivel{nivel}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.dir, exist_ok=True)
        self.estado, self.brazo, self.hz = estado, brazo, hz
        self.t0 = time.monotonic()
        self.eventos, self.objetos = [], []
        self.meta = {"nivel": nivel, "inicio": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}
        self.sup_estado, self.sup_objeto = "ESPERA", ""
        self.manos = {}                  # lado -> (angulos, fuerzas) del último sondeo
        self.fuerza_max = {}             # lado -> 6, máximo de la tirada (gf)
        self.cerrojo = threading.Lock()
        self.fh = open(os.path.join(self.dir, "lazo.csv"), "w", newline="")
        self.w = csv.writer(self.fh)
        cab = ["t", "estado", "objeto", "peso"] + [f"q_{c}" for c in CAB_J] + [f"qd_{c}" for c in CAB_J] + \
              [f"qp_{c}" for c in CAB_J] + ["roll", "pitch", "yaw"]
        for lado in ("izq", "der"):
            cab += [f"{lado}_a{k}" for k in range(6)] + [f"{lado}_f{k}" for k in range(6)]
        self.w.writerow(cab)
        self.vivo = True
        self.hilo = threading.Thread(target=self._bucle, daemon=True, name="registro")
        self.hilo.start()

    def t(self):
        return round(time.monotonic() - self.t0, 3)

    # ------------------------------------------------------------ desde el supervisor
    def estado_sup(self, estado, objeto=""):
        with self.cerrojo:
            cambia = estado != self.sup_estado
            self.sup_estado, self.sup_objeto = estado, objeto
        if cambia:
            self.evento("estado", estado=estado, objeto=objeto)

    def mano(self, lado, angulos, fuerzas):
        with self.cerrojo:
            self.manos[lado] = (np.asarray(angulos, float), np.asarray(fuerzas, float))
            f = np.nan_to_num(np.asarray(fuerzas, float))
            self.fuerza_max[lado] = np.maximum(self.fuerza_max.get(lado, np.zeros(6)), f)

    def evento(self, tipo, **datos):
        e = {"t": self.t(), "tipo": tipo, **{k: _json(v) for k, v in datos.items()}}
        with self.cerrojo:
            self.eventos.append(e)
        return e

    def objeto(self, **m):
        """Métricas de un objeto (sec. 9): clase, mano, intentos, cogido a la primera,
        colocado, tiempo, motivo si falló."""
        with self.cerrojo:
            self.objetos.append({k: _json(v) for k, v in m.items()})

    def imagen(self, nombre, img_rgb):
        if img_rgb is None:
            return None
        import cv2
        ruta = os.path.join(self.dir, f"{self.t():07.1f}_{nombre}.png")
        cv2.imwrite(ruta, np.asarray(img_rgb)[..., ::-1])
        return ruta

    # ------------------------------------------------------------ lazo
    def _bucle(self):
        periodo = 1.0 / self.hz
        t_sig = time.monotonic()
        while self.vivo:
            try:
                self._fila()
            except Exception:
                pass                       # el registro nunca para la tirada
            t_sig += periodo
            time.sleep(max(0.0, t_sig - time.monotonic()))

    def _fila(self):
        m, _ = self.estado.ultimo()
        if m is None:
            return
        q = self.estado.q()
        with self.brazo.cerrojo:
            qd = dict(self.brazo.obj)
            peso = self.brazo.peso
        qp = self.brazo.ultima_publicada
        with self.cerrojo:
            est, obj = self.sup_estado, self.sup_objeto
            manos = dict(self.manos)
        fila = [self.t(), est, obj, round(peso, 4)]
        fila += [round(float(q[j]), 5) for j in BRAZOS]
        fila += [round(float(qd[j]), 5) for j in BRAZOS]
        fila += [round(float(qp.get(j, np.nan)), 5) for j in BRAZOS]
        fila += [round(float(v), 5) for v in self.estado.rpy()]
        for lado in ("izq", "der"):
            a, f = manos.get(lado, (np.full(6, np.nan), np.full(6, np.nan)))
            fila += [round(float(v), 1) for v in a] + [round(float(v), 1) for v in f]
        self.w.writerow(fila)

    def cerrar(self, metricas=None):
        self.vivo = False
        self.hilo.join(timeout=1.0)
        self.fh.close()
        with self.cerrojo:
            d = {"meta": self.meta, "metricas": _json(metricas or {}), "objetos": self.objetos,
                 "fuerza_max_gf": {k: v.tolist() for k, v in self.fuerza_max.items()}, "eventos": self.eventos}
        with open(os.path.join(self.dir, "eventos.json"), "w") as fh:
            json.dump(d, fh, indent=1)
        return self.dir


def _json(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    return v
