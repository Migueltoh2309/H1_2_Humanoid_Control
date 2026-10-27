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


def media_centrada(t, w, ventana):
    """Media movil centrada en el tiempo (muestras irregulares)."""
    cs = np.concatenate([[0.0], np.cumsum(w)])
    i0 = np.searchsorted(t, t - ventana / 2)
    i1 = np.searchsorted(t, t + ventana / 2)
    return (cs[i1] - cs[i0]) / np.maximum(i1 - i0, 1)


def ajustar(t, w, wc, ventana=0.0, t_fin=None):
    """(retardo, tau, ganancia) por busqueda en rejilla del retardo y minimos
    cuadrados en tau (pocos parametros y robusto al ruido del balanceo).

    `w` puede venir suavizado con una media centrada de `ventana` s: entonces el modelo
    se suaviza IGUAL antes de comparar. Si no, la media (que adelanta la subida
    ventana/2) sesgaba el retardo hacia 0. Se ajusta en [0, t_fin - ventana/2]: mas
    alla, la media ya incluye la frenada tras el StopMove."""
    t_fin = t[-1] if t_fin is None else t_fin
    sel = (t >= 0.0) & (t <= t_fin - ventana / 2)
    ws = w[sel]
    mejor = None
    for T in np.arange(0.0, 1.0, 0.01):
        for tau in np.arange(0.05, 1.5, 0.01):
            m = np.where(t > T, 1 - np.exp(-np.maximum(t - T, 0.0) / tau), 0.0)
            if ventana > 0:
                m = media_centrada(t, m, ventana)
            m = m[sel]
            K = float(np.dot(m, ws) / max(np.dot(m, m), 1e-9)) / wc
            e = float(np.mean((K * wc * m - ws) ** 2))
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
    ap.add_argument("--suavizar", type=float, default=0.8,
                    help="s de media movil CENTRADA (sin retardo) sobre el giroscopo: un periodo de la marcha "
                         "(0.8 s en la politica de unitree_rl_gym) quita la oscilacion de cada paso; 0 = sin suavizar")
    ap.add_argument("--png", action="store_true", help="grafica de la respuesta medida y del modelo ajustado")
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
    resultados, curvas = [], []
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
        crudo = M.copy()
        if a.suavizar > 0:
            # con los pasos el giroscopo oscila y el ajuste de primer orden salia con tau de 0.05 s
            M[:, 1] = media_centrada(M[:, 0], M[:, 1], a.suavizar)
        T, tau, K = ajustar(M[:, 0] - t0, signo * M[:, 1], a.vyaw, a.suavizar, a.segundos)
        resultados.append((T, tau, K))
        curvas.append((M[:, 0] - t0, signo * M[:, 1], T, tau, K, signo, signo * crudo[:, 1]))
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
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5.2))
        for k, (t, w, T, tau, K, signo, w_crudo) in enumerate(curvas):
            c_ = f"C{k}"
            ax.plot(t, w_crudo, c_, alpha=0.18, lw=0.8)
            ax.plot(t, w, c_, alpha=0.85, label=f"escalon {k + 1} ({'+' if signo > 0 else '-'}{a.vyaw}): giroscopo "
                    f"(signo corregido; claro = crudo, oscuro = media de {a.suavizar} s centrada)")
            tm = np.linspace(0, a.segundos, 200)
            ax.plot(tm, np.where(tm > T, K * a.vyaw * (1 - np.exp(-(tm - T) / max(tau, 1e-3))), 0.0), c_ + "--",
                    label=f"modelo: retardo {T:.2f} s, tau {tau:.2f} s, ganancia {K:.2f}")
        ax.step([-0.5, 0, a.segundos, a.segundos + 1.5], [0, a.vyaw, 0, 0], "k", where="post", lw=1.2,
                label=f"vyaw pedido ({a.vyaw} rad/s)")
        ax.set_xlabel("t desde el Move (s)")
        ax.set_ylabel("vyaw (rad/s)")
        ax.set_title(f"Respuesta a escalones de vyaw ({'simulador' if a.sim else 'robot'}): retardo "
                     f"{out['retardo_s']:.2f} s, tau {out['tau_s']:.2f} s, ganancia {out['ganancia']:.2f}\n"
                     f"(modelo y medida con la misma media centrada de {a.suavizar} s)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        png = ruta[:-5] + ".png"
        fig.savefig(png, dpi=100)
        print(f"grafica: {png}")


if __name__ == "__main__":
    main()
