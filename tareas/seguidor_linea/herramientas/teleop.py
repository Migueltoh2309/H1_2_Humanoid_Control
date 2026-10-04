#!/usr/bin/env python3
"""Teleoperacion con el teclado por LocoClient, para llevar el robot por la pista
mientras se graba el conjunto de datos (reto, hito "Conjunto de datos").

Mismos patrones que Codigos/wasd.py (Move con duration = 1 s: si esto muere, el robot
para solo en 1 s; sin tecla durante --soltar s, velocidad a cero), pero con --sim
para el simulador (dominio DDS 1 en lo). wasd.py usa siempre el dominio 0 y los
ficheros de ~/robotics40 no se modifican (sec. 10).

    python3 herramientas/teleop.py --sim            # en una terminal (necesita teclado)
    python3 herramientas/teleop.py --iface eth0     # robot real, en el PC2 por ssh -t

Teclas: w/s adelante/atras, a/d lateral, q/e giro, espacio parar, +/- escala, x salir.
"""
import argparse
import os
import select
import sys
import termios
import time
import tty

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from seguidor.robot_io import iniciar_dds   # noqa: E402

MAPA = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0), "q": (0, 0, 1), "e": (0, 0, -1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--vy", type=float, default=0.15)
    ap.add_argument("--vyaw", type=float, default=0.35)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--soltar", type=float, default=0.4)
    a = ap.parse_args()
    iniciar_dds(a.sim, a.iface)
    from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient
    c = LocoClient()
    c.SetTimeout(5.0)
    c.Init()
    escala, v, t_tecla = 1.0, (0.0, 0.0, 0.0), 0.0
    viejo = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    print(__doc__.split("Teclas:")[1])
    try:
        while True:
            t = time.time()
            if select.select([sys.stdin], [], [], 1.0 / a.hz)[0]:
                k = sys.stdin.read(1)
                if k == "x":
                    break
                if k == " ":
                    v = (0.0, 0.0, 0.0)
                elif k in "+-":
                    escala = min(1.5, max(0.25, escala + (0.25 if k == "+" else -0.25)))
                    print(f"\rescala {escala:.2f}   ", end="", flush=True)
                elif k in MAPA:
                    m = MAPA[k]
                    v = (m[0] * a.vx * escala, m[1] * a.vy * escala, m[2] * a.vyaw * escala)
                    t_tecla = t
            if t - t_tecla > a.soltar:
                v = (0.0, 0.0, 0.0)
            c.Move(*v)
            print(f"\rvx {v[0]:+.2f} vy {v[1]:+.2f} vyaw {v[2]:+.2f}   ", end="", flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, viejo)
        c.StopMove()
        print("\nStopMove")


if __name__ == "__main__":
    main()
