#!/usr/bin/env python3
"""Respuesta a un escalon de vyaw (reto, 6.3.1): cuanto tarda el robot en empezar a
girar y en alcanzar la velocidad pedida. Se mide con el giroscopo de la IMU
(rt/lowstate) a ~200 Hz, igual que se midio la deriva en cuadrado.py.

Ajusta un modelo de primer orden con retardo  w(t) = K * w_c * (1 - exp(-(t - T) / tau))
y escribe los numeros para config/seguidor.yaml (estimacion.retardo_marcha, tau_marcha).

    python3 herramientas/escalon_vyaw.py --sim                  # valida el metodo: el sim
                                                                 # tiene retardo 0.30 s y tau 0.35 s
    ~/teleop_venv/bin/python herramientas/escalon_vyaw.py --iface eth0 --vyaw 0.3   # FSM 201, L2+B a mano

MUEVE EL ROBOT (gira sobre el sitio). Robot de pie en FSM 201, espacio libre.
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from seguidor.robot_io import FuenteImu, iniciar_dds   # noqa: E402


def ajustar(t, w, wc):
    """(retardo, tau, ganancia) por busqueda en rejilla del retardo y minimos
    cuadrados en tau (pocos parametros y robusto al ruido del balanceo)."""
    mejor = None
    for T in np.arange(0.0, 1.0, 0.01):
        for tau in np.arange(0.05, 1.5, 0.01):
            m = np.where(t > T, 1 - np.exp(-(t - T) / tau), 0.0)
            K = float(np.dot(m, w) / max(np.dot(m, m), 1e-9)) / wc
            e = float(np.mean((K * wc * m - w) ** 2))
            if mejor is None or e < mejor[0]:
                mejor = (e, T, tau, K)
    return mejor[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--vyaw", type=float, default=0.3)
    ap.add_argument("--segundos", type=float, default=4.0)
    ap.add_argument("--repeticiones", type=int, default=3)
    a = ap.parse_args()
    iniciar_dds(a.sim, a.iface)
    imu = FuenteImu()
    if not imu.esperar(5.0):
        sys.exit("No llega rt/lowstate")
    from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient
    c = LocoClient()
    c.SetTimeout(5.0)
    c.Init()
    if not a.sim and input("Gira sobre el sitio. Escribe GIRAR: ").strip() != "GIRAR":
        sys.exit("no confirmado")
    resultados = []
    for rep in range(a.repeticiones):
        signo = 1 if rep % 2 == 0 else -1
        muestras, vivo = [], {"si": True}

        def registrar():
            while vivo["si"]:
                u = imu.leer()
                muestras.append((time.monotonic(), u.gyro[2]))
                time.sleep(0.005)
        hilo = threading.Thread(target=registrar, daemon=True)
        hilo.start()
        time.sleep(0.5)
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.segundos:
            c.Move(0.0, 0.0, signo * a.vyaw)
            time.sleep(0.05)
        c.StopMove()
        time.sleep(1.5)
        vivo["si"] = False
        hilo.join()
        M = np.array(muestras)
        sel = (M[:, 0] >= t0) & (M[:, 0] <= t0 + a.segundos)
        T, tau, K = ajustar(M[sel, 0] - t0, signo * M[sel, 1], a.vyaw)
        resultados.append((T, tau, K))
        print(f"  escalon {rep + 1} ({'+' if signo > 0 else '-'}{a.vyaw} rad/s): retardo {T:.2f} s, tau {tau:.2f} s, "
              f"ganancia {K:.2f} (al 95 % en {T + 3 * tau:.2f} s)")
    R = np.array(resultados)
    out = {"retardo_s": round(float(R[:, 0].mean()), 3), "tau_s": round(float(R[:, 1].mean()), 3),
           "ganancia": round(float(R[:, 2].mean()), 3), "vyaw": a.vyaw, "fuente": "simulador" if a.sim else "robot"}
    ruta = os.path.join(AQUI, "..", "datos", "escalon_vyaw.json")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    json.dump(out, open(ruta, "w"), indent=2)
    print(f"\nmedia: retardo {out['retardo_s']} s, tau {out['tau_s']} s, ganancia {out['ganancia']} -> {ruta}")
    print(f"config/seguidor.yaml (estimacion:):\n  retardo_marcha: {out['retardo_s']}\n  tau_marcha: {out['tau_s']}")


if __name__ == "__main__":
    main()
