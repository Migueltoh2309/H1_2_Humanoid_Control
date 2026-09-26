#!/usr/bin/env python3
"""Demo 1 — La IK con límites vs la IK original. NO necesita ROS ni el bridge.

Genera objetivos alcanzables por construcción (xd = FK(q_ref) con q_ref
dentro de los rangos del MJCF), así que para todos existe al menos una
solución válida. Mide cuántas soluciones respetan los límites mecánicos.

Uso:
    python3 demos/benchmark_ik.py [N]      # N objetivos por brazo (def. 40)
"""
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h1_2_algoritms.fk_functions import (          # noqa: E402
    fkine_arm_left_unitree, fkine_arm_right_unitree, TF2xyzquat)
from h1_2_algoritms.ik_functions import (          # noqa: E402
    ik_dls_step, ik_pseudo_step, ik_solve_limited, pose_error)
from h1_2_algoritms import joint_limits as JL      # noqa: E402


def solve_original(fk, xd, q0, method, iters=500):
    """IK original: sin ninguna noción de límites articulares."""
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        qn = (ik_dls_step(fk, TF2xyzquat, q, xd, lamb=0.05) if method == "dls"
              else ik_pseudo_step(fk, TF2xyzquat, q, xd, alpha=0.3))
        if np.any(~np.isfinite(qn)):
            return q
        q = qn
    return q


def main():
    n_targets = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    print()
    print("#" * 78)
    print("#  DEMO 1 — CINEMÁTICA INVERSA: RESPETO DE LOS LÍMITES ARTICULARES")
    print("#" * 78)
    print(f"\n{n_targets} objetivos por brazo, todos alcanzables por construcción")
    print("(se generan como xd = FK(q_ref) con q_ref dentro de rango, así que")
    print(" para cada uno existe garantizadamente una solución admisible).\n")

    total = {"orig": 0, "nueva": 0, "n": 0}

    for side, fk in [("left", fkine_arm_left_unitree),
                     ("right", fkine_arm_right_unitree)]:
        lo, hi = JL.get_limits(side)
        rng = np.random.default_rng(11)
        targets = [(TF2xyzquat(fk(qr)), qr) for qr in
                   (rng.uniform(lo + 0.1, hi - 0.1) for _ in range(n_targets))]

        print("=" * 78)
        print(f"BRAZO {'IZQUIERDO' if side == 'left' else 'DERECHO'}")
        print("=" * 78)
        print(f"{'método':<34}{'converge':>11}{'y dentro de límites':>22}"
              f"{'t/obj':>10}")
        print("-" * 78)

        for label, kind, start in [
            ("ORIGINAL  DLS          (q0=0)", ("old", "dls"), "zero"),
            ("ORIGINAL  pseudoinversa(q0=0)", ("old", "pseudo"), "zero"),
            ("ORIGINAL  DLS   (q0 cercano)", ("old", "dls"), "near"),
            ("NUEVA  con límites     (q0=0)", ("new", None), "zero"),
            ("NUEVA  con límites (q0 cercano)", ("new", None), "near"),
        ]:
            kindn, method = kind
            conv = inlim = 0
            t0 = time.time()
            for xd, q_ref in targets:
                q0 = (np.zeros(7) if start == "zero"
                      else q_ref + rng.normal(0, 0.15, 7))
                if kindn == "old":
                    q = solve_original(fk, xd, q0, method)
                    e = pose_error(xd, TF2xyzquat(fk(q)))
                    ok = (np.linalg.norm(e[:3]) < 1e-4
                          and np.linalg.norm(e[3:]) < 1e-4)
                else:
                    r = ik_solve_limited(fk, TF2xyzquat, q0, xd, lo, hi)
                    q, ok = r["q"], r["ok"]
                conv += ok
                inlim += (ok and JL.within_limits(q, side))
            dt = (time.time() - t0) / len(targets) * 1000

            pct = 100.0 * inlim / len(targets)
            bar = "#" * int(pct / 4)
            print(f"{label:<34}{conv:>7}/{len(targets):<3}"
                  f"{inlim:>14}/{len(targets):<4} {pct:5.1f}% {dt:>6.0f}ms")
            print(f"{'':34}{'':11}{bar}")

            if start == "zero" and kindn == "old" and method == "dls":
                total["orig"] += inlim
                total["n"] += len(targets)
            if start == "zero" and kindn == "new":
                total["nueva"] += inlim
        print()

    print("=" * 78)
    print("RESUMEN (arranque en frío, q0 = 0, ambos brazos)")
    print("=" * 78)
    print(f"  IK ORIGINAL (DLS) : {total['orig']:>3}/{total['n']} soluciones usables "
          f"({100.0*total['orig']/total['n']:.0f}%)")
    print(f"  IK NUEVA          : {total['nueva']:>3}/{total['n']} soluciones usables "
          f"({100.0*total['nueva']/total['n']:.0f}%)")
    print()
    print("  Toda solución que converge está dentro de los límites por")
    print("  construcción: el algoritmo satura q en cada iteración y bloquea")
    print("  las articulaciones que empujan contra su tope.")
    print()


if __name__ == '__main__':
    main()
