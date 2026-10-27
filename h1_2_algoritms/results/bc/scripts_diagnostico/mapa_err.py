"""Barrido IK del agarre (brazo derecho): error de posición/orientación en una rejilla (x, y)."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.expanduser("~/Github/H1_2_Humanoid_Control/h1_2_algoritms/demos"))
import numpy as np
from multiprocessing import Pool
import visual_servoing_grasp as V
from h1_2_algoritms.vision import grasp_geometry as GG
from h1_2_algoritms.movimiento.ik_functions import ik_solve_limited
from h1_2_algoritms.aprendizaje.escena_faja import BELT_TOP, FRUIT_R
z = BELT_TOP + FRUIT_R - GG.TORSO_Z
xs = np.round(np.arange(0.30, 0.4801, 0.01), 2); ys = np.round(np.arange(-0.35, 0.1001, 0.01), 2)
s = "right"; lo, hi = GG.JL.get_limits(s); qh, _ = GG.solve_arm_ik(s, V.HOME[s])
def row(x):
    out = []
    for y in ys:
        r = ik_solve_limited(GG.FK[s], GG.TF2xyzquat, qh, GG.grasp_wrist_target(s, np.array([x, y, z])), lo, hi, restarts=4, max_iters=200)
        out.append((r["e_pos"], np.degrees(2 * np.arcsin(min(1, r["e_ori"])))))
    return out
if __name__ == "__main__":
    with Pool(8) as p: E = np.array(p.map(row, xs))
    np.savez(os.path.join(os.path.dirname(__file__), "mapa_err.npz"), xs=xs, ys=ys, E=E)
    print("ok", E.shape)
