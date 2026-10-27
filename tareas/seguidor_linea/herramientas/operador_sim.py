#!/usr/bin/env python3
"""SOLO SIMULADOR: un "operador" que lleva el robot por la pista con LocoClient.Move,
en lugar de una persona con wasd.py / teleop.py. Conoce la pista (el json del
generador) y la pose real de la pelvis (/sim/ground_truth/pelvis), como un operador
que ve la cinta: NO usa la camara ni nada del seguidor.

Sirve para grabar los recorridos del conjunto de datos (grabar_dataset.py) y para el
hito 3 (seguidor.py --simulacro: el seguidor calcula sus ordenes mientras el
operador lleva el robot), sin tener que conducir a mano.

    python3 herramientas/operador_sim.py --nivel 2 [--vx 0.25] [--espera 3]
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from seguidor.robot_io import iniciar_dds   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nivel", required=True)
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--vyaw-max", type=float, default=0.4)
    ap.add_argument("--mirada", type=float, default=0.6, help="m: punto de la cinta al que apunta")
    ap.add_argument("--espera", type=float, default=0.0, help="s quieto antes de arrancar")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--t-max", type=float, default=120.0)
    a = ap.parse_args()
    pista = json.load(open(os.path.join(AQUI, "..", "sim", "pistas", f"pista_nivel{a.nivel}.json")))
    L = np.array(pista["linea"])[:, :2]

    import rclpy
    from geometry_msgs.msg import PoseStamped
    rclpy.init()
    nodo = rclpy.create_node("operador_sim")
    pose = {}

    def al_verdad(m):
        q = m.pose.orientation
        pose["v"] = (m.pose.position.x, m.pose.position.y,
                     math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
    nodo.create_subscription(PoseStamped, "/sim/ground_truth/pelvis", al_verdad, 10)
    iniciar_dds(True)
    from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient
    c = LocoClient()
    c.SetTimeout(5.0)
    c.Init()

    fin = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: fin.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: fin.__setitem__("v", True))
    t0 = time.monotonic()
    while "v" not in pose and time.monotonic() - t0 < 5.0:
        rclpy.spin_once(nodo, timeout_sec=0.1)
    if "v" not in pose:
        sys.exit("no llega /sim/ground_truth/pelvis (¿simulador arrancado?)")
    print(f"operador: nivel {a.nivel}, vx {a.vx} m/s (sigue la cinta real; no usa la camara)", flush=True)
    while time.monotonic() - t0 < a.espera and not fin["v"]:
        rclpy.spin_once(nodo, timeout_sec=0.05)
    i, t_imp = 0, 0.0
    try:
        while not fin["v"] and time.monotonic() - t0 < a.t_max:
            t = time.monotonic()
            rclpy.spin_once(nodo, timeout_sec=0.0)
            x, y, yaw = pose["v"]
            # punto de la cinta mas cercano, buscando solo hacia delante (no salta a otro tramo)
            ventana = L[i:i + 200]
            i += int(np.argmin(np.hypot(ventana[:, 0] - x, ventana[:, 1] - y)))
            if i >= len(L) - 5:
                print("operador: final de la cinta", flush=True)
                break
            d = np.hypot(L[i:, 0] - x, L[i:, 1] - y)
            j = i + int(np.argmax(d >= a.mirada)) if (d >= a.mirada).any() else len(L) - 1
            dx, dy = L[j, 0] - x, L[j, 1] - y
            xr, yr = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
            kappa = 2.0 * yr / max(xr * xr + yr * yr, 1e-3)
            vx = a.vx / (1.0 + 1.5 * abs(kappa))
            if xr < 0.1:                       # el punto se ha quedado al lado o detras: girar en el sitio
                vx = 0.0
            w = max(-a.vyaw_max, min(a.vyaw_max, (vx if vx > 0 else 0.3) * kappa if vx > 0
                                     else math.copysign(0.3, yr)))
            c.Move(vx, 0.0, w)
            if t - t_imp > 2.0:
                print(f"operador: vx {vx:+.2f}  vyaw {w:+.2f}  ({i * 100 // len(L)} % de la cinta)", flush=True)
                t_imp = t
            time.sleep(max(0.0, 1.0 / a.hz - (time.monotonic() - t)))
    finally:
        c.StopMove()
        print("operador: StopMove", flush=True)
        nodo.destroy_node()
        rclpy.shutdown()
        os._exit(0)


if __name__ == "__main__":
    main()
