#!/usr/bin/env python3
"""Visual servoing (PBVS, cámara fija en el torso) + agarre de la mandarina,
en lazo cerrado contra MuJoCo con física real. Ver VISUAL_SERVOING_PLAN.md.

Lazo (todo en torso_link):

    cámara RGB-D/IR (15 Hz, render real con el brazo en escena)
      -> fruit_localization (median | median+R | sphere | stereo_ir | gt)
      -> FruitTracker: Kalman de velocidad constante (posición + velocidad)
      -> máquina de estados: APPROACH -> DESCEND -> CLOSE -> LIFT
      -> grasp_geometry: centro de fruta predicho -> pose objetivo de muñeca
      -> BimanualAvoidanceController (el MISMO QP de las capas 1-2: dampers
         brazo-brazo, torso y nube de la cámara con ObstacleMemory)
      -> q_ref -> PD + compensación de gravedad (igual que el bridge) -> MuJoCo

Por qué PBVS y no IBVS: la cámara está en el torso (eye-to-hand) y la tarea
del QP ya es una pose cartesiana; el error "pose deseada de la muñeca
(función de la fruta vista) − pose actual (cinemática)" se cierra en cada
ciclo con la estimación NUEVA de la fruta. Es la misma interfaz que usará
la primitiva aprendida (LfD): la DMP entrega la trayectoria, la percepción
entrega la meta g — ver el plan, sección 7.

Agarre: dedos Inspire con física de contacto real. Un contacto rígido
esfera-malla no sostiene la fruta al levantarla (medido; ver el plan,
§4), así que se usa una ASISTENCIA: un weld mano-fruta que se activa solo
cuando la mano, al cerrar, ya toca la fruta con ≥2 grupos de dedos/palma y
la fruta está a < 2 cm del punto de agarre esperado. Si el servoing deja la
mano mal puesta no hay contacto y el agarre falla igual: la métrica sigue
midiendo la precisión del servoing. `--no-assist` lo desactiva.

Uso:
    MUJOCO_GL=egl python3 demos/visual_servoing_grasp.py                 # batería completa
    MUJOCO_GL=egl python3 demos/visual_servoing_grasp.py --quick
    MUJOCO_GL=egl python3 demos/visual_servoing_grasp.py --video --only static:sphere:-0.15
    MUJOCO_GL=egl python3 demos/visual_servoing_grasp.py --only moving:sphere:0.05 --video

Resultados en results/servoing/ (resumen.csv + un CSV por
ensayo con la traza completa, útil también como "demostración" robot para
comparar contra LfD).
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import cv2
import mujoco
import numpy as np

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "..", "h1_2_mujoco_sim_bridge"))

from h1_2_mujoco_sim_bridge.mujoco_sim import MujocoSim             # noqa: E402
from h1_2_mujoco_sim_bridge.pd_controller import JointPD             # noqa: E402
from h1_2_mujoco_sim_bridge.sim_bridge_node import DEFAULT_JOINT_NAMES, DEFAULT_GAINS  # noqa: E402

from h1_2_algoritms.vision import grasp_geometry as GG                      # noqa: E402
from h1_2_algoritms.bimanual.bimanual_avoidance import BimanualAvoidanceController  # noqa: E402
from h1_2_algoritms.vision.visual_servoing import FruitTracker, GraspServoing  # noqa: E402
from h1_2_algoritms.bimanual.collision_model import arm_frames                # noqa: E402
from h1_2_algoritms.bimanual.depth_obstacles import obstacles_from_depth, ObstacleMemory  # noqa: E402
from h1_2_algoritms.movimiento.fk_functions import TF2xyzquat                   # noqa: E402
from h1_2_algoritms.vision.fruit_localization import (                      # noqa: E402
    Intrinsics, segment_orange, locate_median, locate_median_radius, locate_sphere,
    locate_stereo_ir, realsense_depth_noise, image_noise, MANDARINA_RADIUS)

from ament_index_python.packages import get_package_share_directory  # noqa: E402
MJCF = os.path.join(get_package_share_directory("h1_2_scenes"), "mjcf",
                    "h1_2_scene_surgery_table_hands.xml")
RESULTS_DIR = os.path.join(PKG_DIR, "results", "servoing")
OPT_FLIP = np.diag([1.0, -1.0, -1.0])
ARM_JOINTS = {s: [f"{s}_{j}" for j in ("shoulder_pitch_joint", "shoulder_roll_joint",
                                       "shoulder_yaw_joint", "elbow_joint", "wrist_roll_joint",
                                       "wrist_pitch_joint", "wrist_yaw_joint")]
              for s in ("left", "right")}
Q_FWD = np.array([1.0, 0.0, 0.0, 0.0])
HOME = {"left": np.array([0.25, 0.25, 0.10, *Q_FWD]),
        "right": np.array([0.25, -0.25, 0.10, *Q_FWD])}
FRUIT_Z_WORLD = 0.8255 + MANDARINA_RADIUS
CARRIER_Y0 = -0.03          # y del carro en la escena (qpos = y - CARRIER_Y0)


# =====================================================================
# Cámaras
# =====================================================================
class Cameras:
    def __init__(self, m, d, noise=True, seed=0):
        self.m, self.d = m, d
        self.cid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n)
                    for n in ("robot_rgbd_camera", "robot_ir_left", "robot_ir_right")}
        self.r_rgb = mujoco.Renderer(m, 480, 640)
        self.r_dep = mujoco.Renderer(m, 480, 640)
        self.r_dep.enable_depth_rendering()
        self.r_ir = None
        self.K = Intrinsics.from_fovy(640, 480, m.cam_fovy[self.cid["robot_rgbd_camera"]])
        self.K_ir = Intrinsics.from_fovy(848, 480, m.cam_fovy[self.cid["robot_ir_left"]])
        self.baseline = float(np.linalg.norm(m.cam_pos[self.cid["robot_ir_right"]] -
                                             m.cam_pos[self.cid["robot_ir_left"]]))
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.R_tw = d.xmat[tb].reshape(3, 3).T
        self.p_tw = -self.R_tw @ d.xpos[tb]

    def T_torso(self, name):
        """(R, p) del frame óptico en torso_link (fijo: la base está soldada)."""
        c = self.cid[name]
        R_w = self.d.cam_xmat[c].reshape(3, 3) @ OPT_FLIP
        return self.R_tw @ R_w, self.R_tw @ self.d.cam_xpos[c] + self.p_tw

    def grab(self, want_ir=False):
        self.r_rgb.update_scene(self.d, camera=self.cid["robot_rgbd_camera"])
        rgb = self.r_rgb.render().copy()
        self.r_dep.update_scene(self.d, camera=self.cid["robot_rgbd_camera"])
        depth = self.r_dep.render().copy()
        irL = irR = None
        if want_ir:
            if self.r_ir is None:
                self.r_ir = mujoco.Renderer(self.m, 480, 848)
            ims = []
            for n in ("robot_ir_left", "robot_ir_right"):
                self.r_ir.update_scene(self.d, camera=self.cid[n])
                ims.append(cv2.cvtColor(self.r_ir.render(), cv2.COLOR_RGB2GRAY))
            irL, irR = ims
        if self.noise:
            depth_n = realsense_depth_noise(depth, self.rng, self.K_ir.fx, self.baseline)
            rgb = image_noise(rgb, self.rng)
            if want_ir:
                irL, irR = image_noise(irL, self.rng), image_noise(irR, self.rng)
        else:
            depth_n = depth
        return rgb, depth, depth_n, irL, irR


def measure(cams, method, rgb, depth, irL, irR, gt_torso):
    """Centro de la fruta en torso_link según `method`, o None (no vista /
    ocluida). Rechaza máscaras con < 35% del área esperada: con la mano
    encima, una silueta parcial sesga el centroide (y la esfera) hacia el
    lado visible; mejor no medir y dejar que el Kalman prediga."""
    if method == "gt":
        return gt_torso.copy()
    R, p = cams.T_torso("robot_rgbd_camera")
    mask = segment_orange(rgb)
    if mask is None:
        return None
    fn = {"median": locate_median, "median+R": locate_median_radius}.get(method, locate_sphere)
    c = fn(mask, depth, cams.K)
    if c is None:
        return None
    area_exp = np.pi * (cams.K.fx * MANDARINA_RADIUS / c[2]) ** 2
    if mask.sum() / 255 < 0.35 * area_exp:
        return None
    if method == "stereo_ir":
        Rl, pl = cams.T_torso("robot_ir_left")
        prior_l = Rl.T @ (R @ c + p - pl)
        cl = locate_stereo_ir(irL, irR, cams.K_ir, cams.baseline, prior_l)
        return None if cl is None else Rl @ cl + pl
    return R @ c + p


# =====================================================================
# Ensayo
# =====================================================================
def run_trial(scenario, method, param, assist=True, noise=True, seed=0, video=None,
              t_max=None, log_dir=None):
    """scenario 'static': param = y de la fruta (mundo, x = 0.34).
    scenario 'moving': param = velocidad de la faja [m/s] (+y), la fruta
    arranca en y = -0.50."""
    sim = MujocoSim(MJCF, DEFAULT_JOINT_NAMES, joint_armature=0.01, joint_damping=0.05)
    m, d = sim.model, sim.data
    kp = np.array([DEFAULT_GAINS[n][0] for n in DEFAULT_JOINT_NAMES])
    kd = np.array([DEFAULT_GAINS[n][1] for n in DEFAULT_JOINT_NAMES])
    tm = np.array([DEFAULT_GAINS[n][2] for n in DEFAULT_JOINT_NAMES])
    pd = JointPD(sim.mapping, kp, kd, tm, np.zeros(len(DEFAULT_JOINT_NAMES)))
    idx = {s: [sim.mapping.index[n] for n in ARM_JOINTS[s]] for s in ("left", "right")}
    qadr = {s: [m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS[s]] for s in ("left", "right")}
    hand_ids = {s: [m.actuator(n).id for n in GG.hand_actuator_names(s)] for s in ("left", "right")}
    fj = m.jnt_qposadr[m.joint("mandarina_free").id]
    fv = m.jnt_dofadr[m.joint("mandarina_free").id]
    cj = m.jnt_qposadr[m.joint("belt_carrier_joint").id]
    cv = m.jnt_dofadr[m.joint("belt_carrier_joint").id]
    conv = m.actuator("conveyor_motor").id
    fruit_bid = m.body("mandarina").id

    # ---- estado inicial ----
    if scenario == "static":
        y0, v_belt = float(param), 0.0
    else:
        y0, v_belt = -0.50, float(param)
    d.qpos[fj:fj + 3] = [0.34, y0, FRUIT_Z_WORLD]
    d.qpos[fj + 3:fj + 7] = [1, 0, 0, 0]
    d.qpos[cj] = y0 - CARRIER_Y0
    d.qvel[cv] = v_belt
    d.qvel[fv + 1] = v_belt
    d.ctrl[conv] = v_belt
    q_ref = np.zeros(14)
    for k, s in enumerate(("left", "right")):
        q, ok = GG.solve_arm_ik(s, HOME[s])
        q_ref[7 * k:7 * k + 7] = q
        d.qpos[qadr[s]] = q
        pd.q_des[idx[s]] = q
        d.ctrl[hand_ids[s]] = GG.HAND_PRESHAPE
    mujoco.mj_forward(m, d)

    cams = Cameras(m, d, noise=noise, seed=seed)
    ctrl = BimanualAvoidanceController(dt=0.05)
    memory = ObstacleMemory(voxel=0.03)
    tracker = FruitTracker()
    want_ir = method == "stereo_ir"

    dt_phys = m.opt.timestep
    n_ctrl = int(round(0.05 / dt_phys))        # QP a 20 Hz
    n_cam = int(round((1 / 15.0) / dt_phys))    # cámara a 15 Hz
    t_max = t_max or (12.0 if scenario == "static" else 22.0)
    n_steps = int(t_max / dt_phys)

    fsm = GraspServoing(HOME)
    phase, side = fsm.phase, None
    obstacles = np.zeros((0, 3))
    pending = []        # (t_disponible, medida, t_captura)
    rows = []
    events = {}
    welded = False
    frames = []
    vid_r = None
    if video:
        vid_r = mujoco.Renderer(m, 480, 640)
        vcam = mujoco.MjvCamera()
        vcam.lookat[:] = [0.36, 0.0, 0.88]
        vcam.distance, vcam.azimuth, vcam.elevation = 1.1, 150, -30

    def gt_torso():
        return d.xpos[fruit_bid] - [0, 0, GG.TORSO_Z]

    for step in range(n_steps):
        t = step * dt_phys

        # ---------------- cámara ----------------
        if step % n_cam == 0:
            rgb, depth_clean, depth, irL, irR = cams.grab(want_ir)
            t0 = time.perf_counter()
            z = measure(cams, method, rgb, depth, irL, irR, gt_torso())
            proc = time.perf_counter() - t0
            # Latencia: la medida llega después de procesar (+ 1 cuadro de
            # transporte/sincronización, como en ROS).
            pending.append((t + proc + 1 / 15.0, z, t))
            # Obstáculos del mismo cuadro (con la q de ESTE instante).
            q_now = np.hstack([sim.get_q_dq()[0][idx[s]] for s in ("left", "right")])
            fl, fr = arm_frames("left", q_now[:7]), arm_frames("right", q_now[7:])
            est, _ = tracker.predict(t)
            excl = [est] if est is not None else None
            P = obstacles_from_depth(depth, (cams.K.fx, cams.K.fy, cams.K.cx, cams.K.cy),
                                     _T(cams.T_torso("robot_rgbd_camera")), fl, fr,
                                     exclude_centers=excl, exclude_radius=0.12)
            obstacles = memory.update(P, depth, (cams.K.fx, cams.K.fy, cams.K.cx, cams.K.cy),
                                      _T(cams.T_torso("robot_rgbd_camera")), t)
            if excl is not None and len(obstacles):
                obstacles = obstacles[np.linalg.norm(obstacles - est, axis=1) > 0.12]
            if vid_r is not None:
                vid_r.update_scene(d, vcam)
                ext = vid_r.render()
                cam_small = cv2.resize(rgb, (213, 160))
                ext = ext.copy()
                ext[0:160, 640 - 213:640] = cam_small
                cv2.putText(ext, f"{method} t={t:4.1f}s {phase}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
                frames.append(ext)
        while pending and pending[0][0] <= t:
            _, z, tc = pending.pop(0)
            if z is not None:
                fsm.observe(tc, z)
                if fsm.tracking():
                    tracker.update(z, tc)

        # ---------------- control (20 Hz) ----------------
        if step % n_ctrl == 0:
            q_meas = np.hstack([sim.get_q_dq()[0][idx[s]] for s in ("left", "right")])
            est, vel = tracker.predict(t)
            out = fsm.step(t, q_meas, est, vel, tracker.n_updates)
            targets, ff, phase, side = out["targets"], out["x_dot_ff"], out["phase"], out["side"]
            if side is not None:
                hand_cmd = out["hand"][side]
            if "t_close" in fsm.events and "err_close_mm" not in events:
                # Error del punto de agarre contra la fruta REAL al empezar a cerrar.
                qc = fsm.events["q_close"]
                events["err_close_mm"] = 1e3 * float(np.linalg.norm(GG.wrist_to_fruit(side, qc) - gt_torso()))
            if phase in ("MISSED", "FAILED"):
                break
            dq, diag = ctrl.step(q_ref, targets, obstacles=obstacles, x_dot_ff=ff)
            if os.environ.get("VS_DEBUG") and step % (n_ctrl * 10) == 0:
                close = sorted(diag["env"], key=lambda e: e[1])[:3]
                print(f"t={t:.1f}", [(n, round(dd, 3), np.round(pt, 3)) for n, dd, pt in close])
            q_ref = q_ref + dq * ctrl.dt
            rows.append({
                "t": round(t, 3), "phase": phase, "side": side or "",
                **{f"gt_{a}": v for a, v in zip("xyz", gt_torso())},
                **{f"est_{a}": (v if est is not None else np.nan) for a, v in
                   zip("xyz", est if est is not None else [np.nan] * 3)},
                **{f"vest_{a}": (v if vel is not None else np.nan) for a, v in
                   zip("xyz", vel if vel is not None else [np.nan] * 3)},
                **{f"wr_{s[0]}_{a}": v for s in ("left", "right") for a, v in
                   zip("xyz", TF2xyzquat(GG.FK[s](q_meas[:7] if s == "left" else q_meas[7:]))[0:3])},
                "n_obs": len(obstacles), "d_env_min": diag["d_env_min"],
                "d_arms_min": diag["d_arms_min"], "welded": int(welded),
            })
            if phase == "DONE":
                break

        # ---------------- mano ----------------
        if side is not None:
            d.ctrl[hand_ids[side]] = hand_cmd
            if assist and phase in ("CLOSE", "LIFT") and not welded:
                welded = _try_weld(m, d, side, fruit_bid, idx, sim)
                if welded:
                    events["t_weld"] = t
        # ---------------- PD + física ----------------
        # Interpolación lineal de q_ref entre ciclos del QP (como el nodo:
        # el PD recibe la referencia a la tasa de control).
        pd.q_des[idx["left"]] = q_ref[:7]
        pd.q_des[idx["right"]] = q_ref[7:]
        sim.step(pd)
        if d.xpos[fruit_bid][2] < 0.5:        # se cayó de la faja
            events["dropped"] = True
            if phase not in ("CLOSE", "LIFT"):
                break

    # ---------------- resultado ----------------
    lift = d.xpos[fruit_bid][2] - FRUIT_Z_WORLD
    success = bool(phase == "DONE" and lift > 0.08)
    est_err = [np.linalg.norm([r["est_x"] - r["gt_x"], r["est_y"] - r["gt_y"], r["est_z"] - r["gt_z"]])
               for r in rows if r["phase"] in ("APPROACH", "DESCEND") and np.isfinite(r["est_x"])]
    res = {
        "scenario": scenario, "method": method, "param": param, "assist": assist, "noise": noise,
        "seed": seed, "side": side or "", "success": success,
        "t_grasp": round(fsm.events.get("t_close", np.nan) - fsm.events.get("t_start", np.nan), 2),
        "err_close_mm": round(events.get("err_close_mm", np.nan), 1),
        "est_err_mm_mean": round(1e3 * float(np.mean(est_err)), 1) if est_err else np.nan,
        "lift_mm": round(1e3 * lift, 1),
        "welded": welded, "dropped": bool(events.get("dropped", False)),
        "d_arms_min_mm": round(1e3 * min(r["d_arms_min"] for r in rows), 1) if rows else np.nan,
        "final_phase": phase,
    }
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        name = f"{scenario}_{method}_{param}_s{seed}{'' if assist else '_noassist'}.csv"
        with open(os.path.join(log_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if video and frames:
        vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
        for fr_ in frames:
            vw.write(fr_[:, :, ::-1])
        vw.release()
    return res


def _T(Rp):
    T = np.eye(4)
    T[0:3, 0:3], T[0:3, 3] = Rp
    return T


def _try_weld(m, d, side, fruit_bid, idx, sim):
    groups = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if fruit_bid not in (b1, b2):
            continue
        other = m.body(b2 if b1 == fruit_bid else b1).name
        for g in ("thumb", "index", "middle", "ring", "pinky", "wrist_yaw"):
            if g in other and other.startswith(("L_", "R_", side)):
                groups.add(g)
    if len(groups) < 2:
        return False
    q = sim.get_q_dq()[0][idx[side]]
    expected = GG.wrist_to_fruit(side, q)
    if np.linalg.norm(expected - (d.xpos[fruit_bid] - [0, 0, GG.TORSO_Z])) > 0.02:
        return False
    eid = m.equality(f"{side}_grasp_assist").id
    b1 = m.body(f"{side}_wrist_yaw_link").id
    R1 = d.xmat[b1].reshape(3, 3)
    q1i, rq = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(q1i, d.xquat[b1])
    mujoco.mju_mulQuat(rq, q1i, d.xquat[fruit_bid])
    m.eq_data[eid][0:3] = 0.0
    m.eq_data[eid][3:6] = R1.T @ (d.xpos[fruit_bid] - d.xpos[b1])
    m.eq_data[eid][6:10] = rq
    m.eq_data[eid][10] = 1.0
    d.eq_active[eid] = 1
    return True


def _job(a):
    scenario, method, param, assist, seed = a
    try:
        return run_trial(scenario, method, param, assist=assist, seed=seed,
                         log_dir=os.path.join(RESULTS_DIR, "trazas"))
    except Exception as e:     # un ensayo roto no tumba la batería
        return {"scenario": scenario, "method": method, "param": param, "assist": assist,
                "seed": seed, "success": False, "final_phase": f"ERROR {e!r}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["gt", "median", "median+R", "sphere", "stereo_ir"])
    ap.add_argument("--ys", nargs="+", type=float, default=[-0.25, -0.15, -0.05, 0.05, 0.15, 0.25])
    ap.add_argument("--speeds", nargs="+", type=float, default=[0.03, 0.05, 0.07])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--no-assist", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None, help="scenario:method:param, p.ej. static:sphere:-0.15")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    assist = not args.no_assist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.only:
        sc, me, pa = args.only.split(":")
        vid = os.path.join(RESULTS_DIR, f"{sc}_{me}_{pa}.mp4") if args.video else None
        r = run_trial(sc, me, float(pa), assist=assist, video=vid,
                      log_dir=os.path.join(RESULTS_DIR, "trazas"))
        print(r)
        if vid:
            print("video:", vid)
        return

    if args.quick:
        args.methods, args.ys, args.speeds = ["gt", "median", "sphere"], [-0.15, 0.15], [0.05]
    jobs = [("static", me, y, assist, s) for me in args.methods for y in args.ys for s in range(args.seeds)]
    jobs += [("moving", me, v, assist, s) for me in args.methods for v in args.speeds for s in range(args.seeds)]
    t0 = time.time()
    with Pool(args.workers) as P:
        res = P.map(_job, jobs)
    path = os.path.join(RESULTS_DIR, "resumen.csv" if assist else "resumen_noassist.csv")
    keys = sorted({k for r in res for k in r.keys()}, key=lambda k: list(res[0].keys()).index(k)
                  if k in res[0] else 99)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(res)
    print(f"{len(jobs)} ensayos en {time.time() - t0:.0f} s -> {path}\n")
    print(f"{'escenario':8s} {'método':10s} {'éxito':>6s} {'err cierre':>10s} {'err estim':>9s} "
          f"{'t agarre':>8s}  [mm, s]")
    for sc in ("static", "moving"):
        for me in args.methods:
            r = [x for x in res if x["scenario"] == sc and x["method"] == me]
            if not r:
                continue
            ok = [x for x in r if x.get("success")]
            f = lambda k: np.nanmean([x.get(k, np.nan) for x in r])
            print(f"{sc:8s} {me:10s} {len(ok)}/{len(r):<4d} {f('err_close_mm'):10.1f} "
                  f"{f('est_err_mm_mean'):9.1f} {f('t_grasp'):8.2f}")
    for x in res:
        if not x.get("success"):
            print("  falla:", {k: x.get(k) for k in ("scenario", "method", "param", "final_phase",
                                                      "err_close_mm", "welded", "dropped")})


if __name__ == "__main__":
    main()
