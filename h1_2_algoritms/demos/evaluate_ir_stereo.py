#!/usr/bin/env python3
"""¿Detecta mejor el par IR de la RealSense que la profundidad? Comparación
de los cuatro estimadores de `fruit_localization.py` contra la posición real
de la mandarina en MuJoCo.

Sin ROS (mismo patrón que evaluate_perception.py): MuJoCo directo, escena
`h1_2_scene_surgery_table_hands.xml` (con el par IR `robot_ir_left/right`).

Condiciones:
  ideal     imágenes y profundidad perfectas del render (valida la
            GEOMETRÍA de cada método: con datos perfectos, ¿el método mide
            el centro o no?)
  ruido     profundidad con el modelo de ruido D435 (σ_z ∝ z², correlado,
            huecos en bordes) + ruido gaussiano en RGB/IR. Mismo ruido para
            todos los métodos, varias semillas por posición.
  oclusion  como 'ruido' pero con la mano derecha en pre-agarre justo
            encima/delante de la fruta (el caso real del final del
            servoing: la mano tapa parte de la fruta).

Uso:
    MUJOCO_GL=egl python3 demos/evaluate_ir_stereo.py
    MUJOCO_GL=egl python3 demos/evaluate_ir_stereo.py --seeds 5 --n-y 11

Guarda resultados_percepcion/ir_stereo.csv y un resumen en pantalla.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import mujoco
import numpy as np

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)

from h1_2_algoritms.fruit_localization import (   # noqa: E402
    Intrinsics, segment_orange, locate_median, locate_median_radius, locate_sphere,
    locate_stereo_ir, realsense_depth_noise, image_noise)

MJCF = os.path.normpath(os.path.join(PKG_DIR, "..", "h1_2_utec", "h1_2_description", "mjcf",
                                     "h1_2_scene_surgery_table_hands.xml"))
RESULTS_DIR = os.path.join(PKG_DIR, "h1_2_algoritms", "resultados_percepcion")
OPT_FLIP = np.diag([1.0, -1.0, -1.0])   # óptico ROS <-> cámara MuJoCo
METHODS = ["median", "median+R", "sphere", "stereo_ir"]

# Pre-agarre de la mano derecha sobre una fruta en (0.34, -0.10) del mundo
# (sale de demos/visual_servoing_grasp.py; basta con que tape la fruta).
Q_OCCLUDE_RIGHT = None


class Rig:
    def __init__(self, mjcf):
        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.cam = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, n)
                    for n in ("robot_rgbd_camera", "robot_ir_left", "robot_ir_right")}
        self.r_rgb = mujoco.Renderer(self.m, 480, 640)
        self.r_dep = mujoco.Renderer(self.m, 480, 640)
        self.r_dep.enable_depth_rendering()
        self.r_ir = mujoco.Renderer(self.m, 480, 848)
        self.K_rgb = Intrinsics.from_fovy(640, 480, self.m.cam_fovy[self.cam["robot_rgbd_camera"]])
        self.K_ir = Intrinsics.from_fovy(848, 480, self.m.cam_fovy[self.cam["robot_ir_left"]])
        pl = self.m.cam_pos[self.cam["robot_ir_left"]]
        pr = self.m.cam_pos[self.cam["robot_ir_right"]]
        self.baseline = float(np.linalg.norm(pr - pl))
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "mandarina_free")
        self.fruit_adr = self.m.jnt_qposadr[j]
        self.fruit_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "mandarina")

    def pose(self, name):
        """(R, p) del frame ÓPTICO de la cámara en mundo."""
        c = self.cam[name]
        return self.d.cam_xmat[c].reshape(3, 3) @ OPT_FLIP, self.d.cam_xpos[c].copy()

    def set_fruit(self, p):
        self.d.qpos[self.fruit_adr:self.fruit_adr + 3] = p
        self.d.qpos[self.fruit_adr + 3:self.fruit_adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)

    def render(self):
        self.r_rgb.update_scene(self.d, camera=self.cam["robot_rgbd_camera"])
        rgb = self.r_rgb.render().copy()
        self.r_dep.update_scene(self.d, camera=self.cam["robot_rgbd_camera"])
        depth = self.r_dep.render().copy()
        irs = []
        for n in ("robot_ir_left", "robot_ir_right"):
            self.r_ir.update_scene(self.d, camera=self.cam[n])
            irs.append(cv2.cvtColor(self.r_ir.render(), cv2.COLOR_RGB2GRAY))
        return rgb, depth, irs[0], irs[1]


def estimate_all(rig, rgb, depth, irL, irR):
    """Devuelve {método: centro en mundo o None} y tiempos [ms]."""
    out, ms = {}, {}
    R_rgb, p_rgb = rig.pose("robot_rgbd_camera")
    t0 = time.perf_counter()
    mask = segment_orange(rgb)
    t_seg = (time.perf_counter() - t0) * 1e3
    if mask is None:
        return {k: None for k in METHODS}, {k: t_seg for k in METHODS}
    for name, fn in (("median", locate_median), ("median+R", locate_median_radius),
                     ("sphere", locate_sphere)):
        t0 = time.perf_counter()
        c = fn(mask, depth, rig.K_rgb)
        ms[name] = t_seg + (time.perf_counter() - t0) * 1e3
        out[name] = None if c is None else R_rgb @ c + p_rgb
    # Estéreo IR: prior = esfera RGB-D (la más fiel), llevada al IR izq.
    t0 = time.perf_counter()
    prior_w = out["sphere"] if out["sphere"] is not None else out["median+R"]
    R_l, p_l = rig.pose("robot_ir_left")
    c = None
    if prior_w is not None:
        c = locate_stereo_ir(irL, irR, rig.K_ir, rig.baseline, R_l.T @ (prior_w - p_l))
    ms["stereo_ir"] = ms["sphere"] + (time.perf_counter() - t0) * 1e3
    out["stereo_ir"] = None if c is None else R_l @ c + p_l
    return out, ms


def set_occluding_arm(rig, fruit_w):
    """Mano derecha en pre-agarre sobre la fruta (IK del paquete), para el
    caso con oclusión."""
    from h1_2_algoritms.grasp_geometry import pregrasp_wrist_target, solve_arm_ik, TORSO_Z
    xd = pregrasp_wrist_target("right", fruit_w - np.array([0, 0, TORSO_Z]), approach=0.10)
    q, ok = solve_arm_ik("right", xd)
    names = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
             "right_wrist_yaw_joint"]
    for n, v in zip(names, q):
        rig.d.qpos[rig.m.jnt_qposadr[mujoco.mj_name2id(rig.m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
    mujoco.mj_forward(rig.m, rig.d)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-y", type=int, default=9)
    ap.add_argument("--conditions", nargs="+", default=["ideal", "ruido", "oclusion"])
    args = ap.parse_args()

    rig = Rig(MJCF)
    q0 = rig.d.qpos.copy()
    z_fruit = 0.8655
    ys = np.linspace(-0.40, 0.40, args.n_y)
    xs = [0.30, 0.34, 0.38]
    rows = []
    for cond in args.conditions:
        seeds = [None] if cond == "ideal" else list(range(args.seeds))
        for x in xs:
            for y in ys:
                rig.d.qpos[:] = q0
                p_true = np.array([x, y, z_fruit])
                rig.set_fruit(p_true)
                if cond == "oclusion":
                    if y > 0.05:          # la derecha no llega bien al lado izq.
                        continue
                    set_occluding_arm(rig, p_true)
                rgb0, dep0, irL0, irR0 = rig.render()
                for s in seeds:
                    if s is None:
                        rgb, dep, irL, irR = rgb0, dep0, irL0, irR0
                    else:
                        rng = np.random.default_rng(1000 * s + int(1e3 * (x + 3 * y)) % 997)
                        dep = realsense_depth_noise(dep0, rng, rig.K_ir.fx, rig.baseline)
                        rgb = image_noise(rgb0, rng)
                        irL, irR = image_noise(irL0, rng), image_noise(irR0, rng)
                    est, ms = estimate_all(rig, rgb, dep, irL, irR)
                    for meth in METHODS:
                        e = None if est[meth] is None else est[meth] - p_true
                        rows.append({"cond": cond, "x": x, "y": y, "seed": s, "method": meth,
                                     "ok": e is not None,
                                     "ex": np.nan if e is None else e[0],
                                     "ey": np.nan if e is None else e[1],
                                     "ez": np.nan if e is None else e[2],
                                     "e3d": np.nan if e is None else np.linalg.norm(e),
                                     "ms": ms.get(meth, np.nan)})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "ir_stereo.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'cond':9s} {'método':10s} {'recall':>6s} {'e3d media':>9s} {'mediana':>8s} "
          f"{'p95':>6s} {'|ex|':>6s} {'|ey|':>6s} {'|ez|':>6s} {'ms':>6s}   [mm]")
    for cond in args.conditions:
        for meth in METHODS:
            r = [x for x in rows if x["cond"] == cond and x["method"] == meth]
            if not r:
                continue
            ok = [x for x in r if x["ok"]]
            if not ok:
                print(f"{cond:9s} {meth:10s} {0:6.0%}   (sin detecciones)")
                continue
            e = np.array([x["e3d"] for x in ok]) * 1e3
            ab = lambda k: np.mean([abs(x[k]) for x in ok]) * 1e3
            print(f"{cond:9s} {meth:10s} {len(ok) / len(r):6.0%} {e.mean():9.1f} {np.median(e):8.1f} "
                  f"{np.percentile(e, 95):6.1f} {ab('ex'):6.1f} {ab('ey'):6.1f} {ab('ez'):6.1f} "
                  f"{np.mean([x['ms'] for x in ok]):6.1f}")
    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
