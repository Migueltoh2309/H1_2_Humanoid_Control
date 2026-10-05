#!/usr/bin/env python3
"""factor_vx = velocidad real / velocidad pedida (reto, 6.4.4: sin odometria, la
distancia se estima con vx * t; es el --factor de cuadrado.py).

Manda Move(vx, 0, 0) durante --segundos y mide cuanto avanzo:
  * robot real: lo mides con cinta y lo escribes;
  * simulador (--sim): con la verdad de terreno (/sim/ground_truth/pelvis).
El primer y el ultimo tramo (arranque y frenada) se descuentan con el modelo de la
marcha de la config (retardo_marcha). Repite con --repeticiones y promedia.

    python3 herramientas/calibrar_vx.py --sim --vx 0.3
    ~/teleop_venv/bin/python herramientas/calibrar_vx.py --iface eth0 --vx 0.2   # FSM 201, pista libre, L2+B

MUEVE EL ROBOT hacia delante vx * segundos (por defecto 0.3 * 6 = 1.8 m).
"""
import argparse
import json
import math
import os
import sys
import threading
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from seguidor import config                          # noqa: E402
from seguidor.robot_io import iniciar_dds            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--segundos", type=float, default=6.0)
    ap.add_argument("--repeticiones", type=int, default=2)
    a = ap.parse_args()
    cfg = config.cargar()
    retardo = cfg["estimacion"].get("retardo_marcha", 0.3)
    iniciar_dds(a.sim, a.iface)
    from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient
    c = LocoClient()
    c.SetTimeout(5.0)
    c.Init()
    gt = {}
    if a.sim:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        rclpy.init()
        n = rclpy.create_node("calibrar_vx")
        n.create_subscription(PoseStamped, "/sim/ground_truth/pelvis",
                              lambda m: gt.__setitem__("p", (m.pose.position.x, m.pose.position.y)), 10)
        threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
        time.sleep(1.5)
    elif input("Avanza en linea recta. Escribe AVANZAR: ").strip() != "AVANZAR":
        sys.exit("no confirmado")
    factores = []
    for r in range(a.repeticiones):
        p0 = gt.get("p")
        t0 = time.time()
        while time.time() - t0 < a.segundos:
            c.Move(a.vx, 0.0, 0.0)
            time.sleep(0.05)
        c.StopMove()
        time.sleep(2.0)                              # que acabe de frenar
        if a.sim:
            p1 = gt["p"]
            d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        else:
            d = float(input(f"  tirada {r + 1}: metros recorridos (cinta): "))
        # con retardo puro + primer orden, lo que se pierde al arrancar se recupera al frenar:
        # la distancia de un escalon de duracion T es ~ v * T (el retardo solo desplaza)
        f = d / (a.vx * a.segundos)
        factores.append(f)
        print(f"  tirada {r + 1}: {d:.3f} m en {a.segundos} s a {a.vx} m/s pedidos -> factor {f:.3f}")
    f = sum(factores) / len(factores)
    out = {"factor_vx": round(f, 3), "vx": a.vx, "segundos": a.segundos, "fuente": "simulador" if a.sim else "robot",
           "retardo_supuesto": retardo}
    ruta = os.path.join(AQUI, "..", "datos", "factor_vx.json")
    json.dump(out, open(ruta, "w"), indent=2)
    print(f"\nfactor_vx = {f:.3f} -> {ruta}\nconfig/seguidor.yaml (estimacion:):  factor_vx: {f:.3f}", flush=True)
    os._exit(0)                                     # sin esperar a los hilos de rclpy / DDS


if __name__ == "__main__":
    main()
