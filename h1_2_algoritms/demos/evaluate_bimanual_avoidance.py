#!/usr/bin/env python3
"""Evaluación del QP bimanual con evasión de colisiones, contra MuJoCo.

NO usa ROS/el bridge (mismo patrón que evaluate_perception.py y
benchmark_ik.py): instancia MuJoCo directo, renderiza la cámara de
profundidad del torso con el mismo modelo y corre EXACTAMENTE el mismo
controlador que el nodo en vivo (`bimanual_avoidance.BimanualAvoidanceController`
+ `depth_obstacles.obstacles_from_depth`).

La verdad-terreno de colisión NO sale del modelo de cápsulas del
controlador (sería circular), sino de MuJoCo:
  * contactos con penetración (d.contact, dist < 0) sobre las mallas reales;
  * distancia brazo-brazo mínima entre vértices de las mallas de colisión;
  * holgura mano-obstáculo analítica (poste: distancia al eje menos radio;
    faja: altura sobre su superficie).

Seguimiento cinemático: q de referencia -> qpos directo (el bridge corre
PD + compensación de gravedad con error de seguimiento ~0, ver README raíz).
La cámara se renderiza cada `--cam-every` ciclos y la nube se calcula con la
q de ESE instante, así que la latencia de la cámara queda modelada.

Escenarios (objetivos de la muñeca en torso_link):
  shared    ZONA COMPARTIDA: las dos manos deben ir al mismo punto (p.ej. la
            mandarina en el centro de la faja), quedarse 1 s ("agarrar") y
            volver a casa. A la vez es imposible; en secuencia sí. Es el
            caso que la capa 1 sola no resuelve (bloqueo) y la capa 2 sí.
  converge  las dos manos a casi el mismo punto delante del pecho
            (objetivos cruzados 6 cm). OJO: objetivos INALCANZABLES, cada
            uno por separado (con la mano horizontal la muñeca no baja de
            z ~ 0.08) y además incompatibles entre sí (los antebrazos
            quedarían ~12 cm uno dentro del otro). Lo correcto es detenerse
            sin chocar y decir por qué (la capa 3 lo diagnostica).
  swap      cada mano va al lado contrario; igual que converge.
  table     las manos bajan apuntando hacia la faja, con objetivos por
            debajo de su superficie: choque con el entorno, que el modelo
            cinemático no conoce -> solo la cámara lo evita.
  circles   las dos manos giran en círculos de 8 cm en contrafase que se
            cruzan en el centro (el escenario por defecto del nodo ROS).
  obstacle  una BANDEJA horizontal (24x24x2 cm, obstáculo desconocido para el
            modelo) justo encima de la mano derecha, que debe subir a un
            punto por encima de ella. El camino recto choca de frente con la
            placa: la capa 1 se traba debajo (mínimo local) y hace falta la
            capa 3 para rodearla. (Contra obstáculos pequeños y convexos,
            como una esfera, la capa 1 ya resbala por el costado y llega.)

Modos por escenario:
  off          QP bimanual sin restricciones de distancia (línea base)
  self         + autocolisión desde la cinemática (sin cámara)
  self+frame   + obstáculos del ÚLTIMO cuadro de profundidad (ablación:
               muestra el problema de la oclusión)
  self+cam     + obstáculos con memoria de vóxeles (ObstacleMemory)
  full         self+cam + capa 2 (bimanual_coordinator: prioridad y retirada)
  full+plan    full + capa 3 (bimanual_planner: RRT-Connect cuando no se avanza)

Uso:
    python3 demos/evaluate_bimanual_avoidance.py
    python3 demos/evaluate_bimanual_avoidance.py --scenarios converge table --seconds 6
    MUJOCO_GL=egl python3 demos/evaluate_bimanual_avoidance.py   # sin pantalla

Guarda CSVs (y PNG si hay matplotlib) en
results/bimanual/.
"""
import argparse
import csv
import os
import sys
import time

import mujoco
import numpy as np
from scipy.spatial import cKDTree

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)

from h1_2_algoritms.bimanual.bimanual_avoidance import BimanualAvoidanceController  # noqa: E402
from h1_2_algoritms.bimanual.bimanual_coordinator import BimanualCoordinator, TaskSequence  # noqa: E402
from h1_2_algoritms.bimanual.collision_model import arm_frames                      # noqa: E402
from h1_2_algoritms.bimanual.depth_obstacles import (                               # noqa: E402
    obstacles_from_depth, camera_pose_from_mjcf_xyaxes, intrinsics_from_fovy, ObstacleMemory)
from h1_2_algoritms.bimanual.bimanual_planner import GlobalPlannerLayer                   # noqa: E402
from h1_2_algoritms.movimiento.fk_functions import fkine_arm_left_unitree, fkine_arm_right_unitree  # noqa: E402
from h1_2_algoritms.movimiento.ik_functions import ik_solve_limited                  # noqa: E402
from h1_2_algoritms.movimiento.fk_functions import TF2xyzquat                         # noqa: E402
from h1_2_algoritms.movimiento import joint_limits as JL                              # noqa: E402

from ament_index_python.packages import get_package_share_directory  # noqa: E402
DEFAULT_MJCF = os.path.join(get_package_share_directory("h1_2_scenes"), "mjcf",
                            "h1_2_scene_surgery_table.xml")
RESULTS_DIR = os.path.join(PKG_DIR, "results", "bimanual")
CAMERA_NAME = "robot_rgbd_camera"
WIDTH, HEIGHT = 640, 480

Q_FWD = np.array([1.0, 0.0, 0.0, 0.0])               # mano hacia adelante
Q_DOWN = np.array([0.70710678, 0.0, 0.70710678, 0.0])  # Ry(+90°): mano hacia abajo

# Inicio común: manos delante, cada una de su lado.
HOME = {"left": np.array([0.30, 0.20, 0.05, *Q_FWD]),
        "right": np.array([0.30, -0.20, 0.05, *Q_FWD])}

SCENARIOS = {
    "converge": {"left": np.array([0.35, -0.03, 0.05, *Q_FWD]),
                 "right": np.array([0.35, 0.03, 0.05, *Q_FWD])},
    "swap": {"left": np.array([0.30, -0.12, 0.08, *Q_FWD]),
             "right": np.array([0.30, 0.12, 0.08, *Q_FWD])},
    "table": {"left": np.array([0.30, 0.15, -0.06, *Q_DOWN]),
              "right": np.array([0.30, -0.15, -0.06, *Q_DOWN])},
    "obstacle": {"left": HOME["left"].copy(),
                 "right": np.array([0.34, -0.22, 0.35, *Q_FWD])},
}


def circles_targets(t, t_home=1.0, omega=0.8):
    # Igual que QP_bimanual_avoidance._targets (sin la fase de home larga).
    if t < t_home:
        return HOME
    out = {}
    for side, c, ph in (("left", np.array([0.33, 0.03, 0.05]), 0.0),
                        ("right", np.array([0.33, -0.03, 0.05]), np.pi)):
        a = omega * (t - t_home) + ph
        out[side] = np.array([c[0], c[1] + 0.08 * np.cos(a), c[2] + 0.08 * np.sin(a), *Q_FWD])
    return out


SCENARIOS["circles"] = circles_targets

# Zona compartida: mismo punto para las dos manos (alcanzable por cada una
# sola con la mano horizontal), 1 s de "agarre" y regreso a casa.
SHARED_POINT = np.array([0.34, 0.0, 0.15, *Q_FWD])
SCENARIOS["shared"] = {"seq": {s: [(SHARED_POINT, 1.0), (HOME[s], 0.0)] for s in ("left", "right")}}

MODES = ["off", "self", "self+frame", "self+cam", "full", "full+plan"]
TOL_POS = 0.02    # [m] "llegó" (tarea completada)

# Bandeja del escenario "obstacle", en torso_link (caja alineada con los
# ejes): centro y semi-lados. Elegida con una búsqueda sistemática para que
# (1) la postura objetivo sea alcanzable sin colisión, (2) la capa 1 sola se
# trabe y (3) el planificador encuentre camino.
TRAY_CENTER = np.array([0.44, -0.22, 0.19])
TRAY_HALF = np.array([0.12, 0.12, 0.01])

# Superficie de la faja (README de h1_2_mujoco_sim_bridge): z mundo 0.824.
BELT_TOP_Z_WORLD = 0.824

ARM_LINKS = ["shoulder_roll_link", "shoulder_yaw_link", "elbow_link",
             "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link"]
JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
          "wrist_roll", "wrist_pitch", "wrist_yaw"]


# ============================================================ modelo

def build_model(mjcf_path, with_tray):
    spec = mujoco.MjSpec.from_file(mjcf_path)
    if with_tray:
        # torso_link está en (0, 0, z_pelvis) del mundo sin rotación con la
        # cintura en 0, que es como queda durante toda la evaluación.
        m0 = spec.compile()
        d0 = mujoco.MjData(m0)
        mujoco.mj_forward(m0, d0)
        z_torso = d0.xpos[m0.body("torso_link").id][2]
        body = spec.worldbody.add_body(
            name="obstaculo_bandeja",
            pos=[TRAY_CENTER[0], TRAY_CENTER[1], z_torso + TRAY_CENTER[2]])
        body.add_geom(name="obstaculo_bandeja_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=TRAY_HALF.tolist(), rgba=[0.2, 0.6, 1.0, 1.0])
    return spec.compile()


class Scene:
    def __init__(self, m, has_tray=False):
        self.m = m
        self.has_tray = has_tray
        self.d = mujoco.MjData(m)
        mujoco.mj_forward(m, self.d)
        self.adr = {s: [m.jnt_qposadr[m.joint(f"{s}_{j}_joint").id] for j in JOINTS]
                    for s in ("left", "right")}
        self.arm_geoms = {s: {g for g in range(m.ngeom) if m.geom_contype[g]
                              and m.body(m.geom_bodyid[g]).name in [f"{s}_{l}" for l in ARM_LINKS]}
                          for s in ("left", "right")}
        self.robot_body_ids = np.array(sorted(self._robot_bodies()))
        self.geoms_left = np.array(sorted(self.arm_geoms["left"]))
        self.geoms_right = np.array(sorted(self.arm_geoms["right"]))
        self.tid = m.body("torso_link").id
        # vértices de colisión de cada brazo (distancia real brazo-brazo)
        self.verts = {}
        for s in ("left", "right"):
            for g in self.arm_geoms[s]:
                mid = m.geom_dataid[g]
                a = m.mesh_vertadr[mid]
                self.verts[g] = m.mesh_vert[a:a + m.mesh_vertnum[mid]][::4].copy()
        self.renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
        self.renderer.enable_depth_rendering()
        cid = m.cam(CAMERA_NAME).id
        self.K = intrinsics_from_fovy(WIDTH, HEIGHT, m.cam_fovy[cid])
        self.T_torso_cam = camera_pose_from_mjcf_xyaxes()

    def _robot_bodies(self):
        # Base fija: todos los cuerpos del robot cuelgan de pelvis.
        root = self.m.body_rootid[self.m.body("pelvis").id]
        return {b for b in range(self.m.nbody) if self.m.body_rootid[b] == root}

    def set_q(self, q):
        self.d.qpos[self.adr["left"]] = q[:7]
        self.d.qpos[self.adr["right"]] = q[7:]
        mujoco.mj_forward(self.m, self.d)

    def depth(self):
        self.renderer.update_scene(self.d, camera=CAMERA_NAME)
        return self.renderer.render()

    def to_torso(self, P):
        R = self.d.xmat[self.tid].reshape(3, 3)
        return (P - self.d.xpos[self.tid]) @ R

    def arm_vertices(self, side):
        return np.vstack([(self.d.geom_xmat[g].reshape(3, 3) @ self.verts[g].T).T + self.d.geom_xpos[g]
                          for g in self.arm_geoms[side]])

    def ground_truth(self):
        m, d = self.m, self.d
        # Vectorizado: iterar d.contact en Python cuesta ~1 s por paso con
        # las mallas de la mesa/faja.
        n = d.ncon
        g = d.contact.geom[:n]
        dist = d.contact.dist[:n]
        is_l = np.isin(g, self.geoms_left)
        is_r = np.isin(g, self.geoms_right)
        is_arm = is_l | is_r
        is_robot = np.isin(m.geom_bodyid[g], self.robot_body_ids)
        pen_mask = dist < 0
        arm_arm = pen_mask & ((is_l[:, 0] & is_r[:, 1]) | (is_r[:, 0] & is_l[:, 1]))
        one_arm = pen_mask & (is_arm[:, 0] ^ is_arm[:, 1])
        other_is_robot = np.where(is_arm[:, 0], is_robot[:, 1], is_robot[:, 0])
        arm_torso = one_arm & other_is_robot
        arm_env = one_arm & ~other_is_robot
        n_arm_arm, n_arm_torso, n_arm_env = int(arm_arm.sum()), int(arm_torso.sum()), int(arm_env.sum())
        hit = arm_arm | arm_torso | arm_env
        pen = float(dist[hit].min()) if hit.any() else 0.0
        VL, VR = self.arm_vertices("left"), self.arm_vertices("right")
        # distance_upper_bound: con los brazos lejos la consulta completa tarda
        # ~1 s; más allá de 0.25 m el valor exacto no interesa.
        d_arms = 0.0 if n_arm_arm else float(min(
            cKDTree(VR).query(VL, k=1, distance_upper_bound=0.25)[0].min(), 0.25))
        V = self.to_torso(np.vstack((VL, VR)))
        # Distancia de las mallas de los brazos a la bandeja (caja alineada):
        # norma del exceso sobre los semi-lados; 0 si el vértice está dentro.
        d_obst = np.nan
        if self.has_tray:
            ex = np.maximum(np.abs(V - TRAY_CENTER) - TRAY_HALF, 0.0)
            d_obst = float(np.linalg.norm(ex, axis=1).min())
        belt_top = BELT_TOP_Z_WORLD - self.d.xpos[self.tid][2]
        over_belt = (V[:, 0] > 0.26) & (V[:, 0] < 0.53) & (np.abs(V[:, 1]) < 0.55)
        d_belt = float(np.min(V[over_belt, 2] - belt_top, initial=np.inf))
        return {"n_arm_arm": n_arm_arm, "n_arm_env": n_arm_env, "n_arm_torso": n_arm_torso,
                "pen": pen, "d_arms": d_arms, "d_obst": d_obst, "d_belt": d_belt}

    def close(self):
        self.renderer.close()


# ============================================================ corrida

def home_configuration():
    q = []
    for side, fk in (("left", fkine_arm_left_unitree), ("right", fkine_arm_right_unitree)):
        lo, hi = JL.get_limits(side)
        sol = ik_solve_limited(fk, TF2xyzquat, np.zeros(7), HOME[side], lo, hi)
        if not sol["ok"]:
            raise RuntimeError(f"IK de la pose inicial ({side}) no convergió: {sol}")
        q.append(sol["q"])
    return np.hstack(q)


def run(scene, scenario, mode, seconds, dt, cam_every, on_step=None):
    """Una corrida. on_step(ctx), opcional, se llama al final de cada ciclo
    con el estado completo (lo usa demos/record_bimanual_videos.py para
    grabar exactamente lo mismo que se evalúa)."""
    use_cam = mode in ("self+frame", "self+cam", "full", "full+plan")
    ctrl = BimanualAvoidanceController(
        dt=dt, avoid_self=(mode != "off"), avoid_obstacles=use_cam)
    memory = ObstacleMemory() if mode in ("self+cam", "full", "full+plan") else None
    coord = BimanualCoordinator(dt, d_influence=ctrl.d_i, tol_pos=TOL_POS) \
        if mode in ("full", "full+plan") else None
    planner = GlobalPlannerLayer(dt, ctrl.q_min, ctrl.q_max, tol_pos=TOL_POS) \
        if mode == "full+plan" else None
    t_plan = 0.0
    spec = SCENARIOS[scenario]
    # Tarea: secuencia por brazo (un objetivo fijo = secuencia de 1 paso).
    # Los objetivos que se mueven (circles) no son "completables".
    tasks = None
    if not callable(spec):
        steps = spec["seq"] if "seq" in spec else {s: [(spec[s], 0.0)] for s in ("left", "right")}
        tasks = {s: TaskSequence(steps[s], tol_pos=TOL_POS) for s in ("left", "right")}
    q = home_configuration()
    scene.set_q(q)
    obstacles = None
    rows = []
    t_qp = []
    d_self = np.inf
    for k in range(int(round(seconds / dt))):
        t = k * dt
        if use_cam and k % cam_every == 0:
            depth = scene.depth()
            obstacles = obstacles_from_depth(
                depth, scene.K, scene.T_torso_cam,
                arm_frames("left", q[:7]), arm_frames("right", q[7:]))
            if memory is not None:
                obstacles = memory.update(obstacles, depth, scene.K, scene.T_torso_cam, k * dt)
        t0 = time.perf_counter()
        goals = spec(t) if tasks is None else {s: tasks[s].goal() for s in tasks}
        plan_busy = planner is not None and planner.active
        if coord is not None and not plan_busy:
            tg, task_err = coord.step(t, q, goals, d_self)
        else:
            tg, task_err = goals, BimanualCoordinator.task_error(q, goals)
        if tasks is not None:
            for s in tasks:
                tasks[s].update(t, task_err[s])
        jt = None
        dt_plan = 0.0
        if planner is not None:
            l2_busy = coord is not None and coord.state == "yield"
            t1 = time.perf_counter()
            jt = planner.step(t, q, goals, task_err, obstacles, l2_busy=l2_busy)
            dt_plan = time.perf_counter() - t1
            t_plan += dt_plan
            if planner.active and coord is not None:
                coord.reset()       # la capa 3 manda; la 2 vuelve a empezar después
        dq, diag = ctrl.step(q, tg, obstacles, joint_targets=jt)
        d_self = diag["d_arms_min"]
        t_qp.append(time.perf_counter() - t0 - dt_plan)   # sin el tiempo de planificar
        q = q + dq * dt
        scene.set_q(q)
        gt = scene.ground_truth()
        if on_step is not None:
            on_step({"k": k, "t": (k + 1) * dt, "q": q, "goals": goals, "targets": tg,
                     "gt": gt, "diag": diag, "obstacles": obstacles, "coord": coord,
                     "planner": planner, "tasks": tasks, "mode": mode, "scenario": scenario})
        rows.append({
            "t": (k + 1) * dt, "mode": mode, "scenario": scenario,
            "e_left": task_err["left"], "e_right": task_err["right"],
            "step_left": -1 if tasks is None else tasks["left"].i,
            "step_right": -1 if tasks is None else tasks["right"].i,
            "coord_state": "" if coord is None else coord.state,
            "plan_active": int(planner is not None and planner.plan is not None),
            "d_self_model": diag["d_self_min"], "d_env_model": diag["d_env_min"],
            "n_constraints": diag["n_constraints"], "qp_ok": int(diag["ok"]),
            "n_obstacle_pts": 0 if obstacles is None else len(obstacles),
            **gt,
        })
    events = ([] if coord is None else [(t_, "[capa 2] " + m_) for t_, m_ in coord.events]) + \
             ([] if planner is None else [(t_, "[capa 3] " + m_) for t_, m_ in planner.events])
    info = {"tasks": tasks, "events": sorted(events, key=lambda e: e[0]), "t_plan": t_plan,
            "plans": [] if planner is None else planner.plans}
    return rows, np.array(t_qp), info


def summarize(rows, t_qp, info):
    r = rows
    tasks = info["tasks"]
    if tasks is None:
        exito, t_fin = "-", np.nan
    else:
        exito = f"{sum(x.done for x in tasks.values())}/2"
        t_fin = max(x.t_done for x in tasks.values()) if all(x.done for x in tasks.values()) else np.nan
    col = lambda k: np.array([x[k] for x in r])  # noqa: E731
    return {
        "scenario": r[0]["scenario"], "mode": r[0]["mode"],
        "tareas_completadas": exito,
        "t_completar_s": t_fin,
        "eventos_capas23": len(info["events"]),
        "t_planificacion_s": info["t_plan"],
        "pasos_con_choque_brazos": int(np.sum(col("n_arm_arm") > 0)),
        "pasos_con_choque_entorno": int(np.sum(col("n_arm_env") > 0)),
        "pasos_con_choque_torso": int(np.sum(col("n_arm_torso") > 0)),
        "penetracion_max_mm": -1000 * col("pen").min(),
        "d_brazos_min_mm": 1000 * col("d_arms").min(),
        "d_bandeja_min_mm": 1000 * np.min(col("d_obst")),
        "d_faja_min_mm": 1000 * col("d_belt").min(),
        "err_final_izq_mm": 1000 * r[-1]["e_left"],
        "err_final_der_mm": 1000 * r[-1]["e_right"],
        "qp_fallos": int(np.sum(col("qp_ok") == 0)),
        "qp_ms_medio": 1000 * t_qp.mean(),
        "qp_ms_max": 1000 * t_qp.max(),
    }


def plot(all_rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    scenarios = list(dict.fromkeys(r[0]["scenario"] for r in all_rows))
    metric = {"converge": ("d_arms", "distancia brazo-brazo [mm]"),
              "shared": ("d_arms", "distancia brazo-brazo [mm]"),
              "swap": ("d_arms", "distancia brazo-brazo [mm]"),
              "circles": ("d_arms", "distancia brazo-brazo [mm]"),
              "table": ("d_belt", "altura de la mano sobre la faja [mm]"),
              "obstacle": ("d_obst", "distancia brazo-bandeja [mm]")}
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8, 2.8 * len(scenarios)), squeeze=False)
    for ax, sc in zip(axes[:, 0], scenarios):
        key, label = metric[sc]
        for rows in all_rows:
            if rows[0]["scenario"] != sc:
                continue
            t = [x["t"] for x in rows]
            ax.plot(t, [1000 * x[key] for x in rows], label=rows[0]["mode"])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(sc)
        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=0.05, help="periodo del QP (def. 20 Hz, como QP_whole_body)")
    ap.add_argument("--cam-every", type=int, default=3,
                    help="render de profundidad cada N ciclos (def. 3 -> 6.7 Hz, como el bridge)")
    ap.add_argument("--mjcf", default=DEFAULT_MJCF)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"MJCF: {args.mjcf}")
    summaries, all_rows = [], []
    for scenario in args.scenarios:
        scene = Scene(build_model(args.mjcf, with_tray=(scenario == "obstacle")),
                      has_tray=(scenario == "obstacle"))
        for mode in args.modes:
            rows, t_qp, info = run(scene, scenario, mode, args.seconds, args.dt, args.cam_every)
            all_rows.append(rows)
            s = summarize(rows, t_qp, info)
            summaries.append(s)
            print(f"  {scenario:9s} {mode:10s} tareas={s['tareas_completadas']:>3s}"
                  f" t={s['t_completar_s']:5.1f}s  choques(brazos/entorno/torso)="
                  f"{s['pasos_con_choque_brazos']}/{s['pasos_con_choque_entorno']}/{s['pasos_con_choque_torso']}"
                  f"  d_brazos_min={s['d_brazos_min_mm']:.0f}mm  d_bandeja_min={s['d_bandeja_min_mm']:.0f}mm"
                  f"  d_faja_min={s['d_faja_min_mm']:.0f}mm  err_final=({s['err_final_izq_mm']:.0f},"
                  f"{s['err_final_der_mm']:.0f})mm  QP {s['qp_ms_medio']:.1f}ms")
            for t_ev, msg in info["events"]:
                print(f"      t={t_ev:5.2f}s  {msg}")
        scene.close()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    detail = os.path.join(RESULTS_DIR, f"bimanual_detalle_{'_'.join(args.scenarios)}_{stamp}.csv")
    with open(detail, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0][0].keys()))
        w.writeheader()
        for rows in all_rows:
            w.writerows(rows)
    summary = os.path.join(RESULTS_DIR, f"bimanual_resumen_{'_'.join(args.scenarios)}_{stamp}.csv")
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    png = plot(all_rows, os.path.join(RESULTS_DIR, f"bimanual_distancias_{'_'.join(args.scenarios)}_{stamp}.png"))
    print(f"\nCSV: {detail}\n     {summary}" + (f"\nPNG: {png}" if png else ""))


if __name__ == "__main__":
    main()
