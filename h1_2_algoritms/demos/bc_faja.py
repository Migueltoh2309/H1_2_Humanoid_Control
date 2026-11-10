#!/usr/bin/env python3
"""Fase B del LfD: política de alto nivel sobre la biblioteca de habilidades,
aprendida por behavior cloning (docs/LFD_BC.md).

Episodio: llegan `n` frutas por la faja (posiciones, separaciones y
velocidad al azar); los dos brazos trabajan a la vez con las habilidades
agarrar (DMP de la fase A) / depositar / volver, y la política de alto nivel
decide, cada vez que un brazo queda libre, qué fruta toma o si espera.
Todo con física (MuJoCo), percepción real (esfera + ruido D435, varias
frutas, pistas de Kalman) y el QP bimanual con dampers.

    generar   episodios con el EXPERTO al mando -> demostraciones (estado,
              acción) en results/bc/datos/
    entrenar  BC (+ rondas de DAgger: corre la política aprendida, el
              experto etiqueta los estados que visita, se reentrena)
    evaluar   FIFO vs. experto vs. BC en episodios NO vistos
              -> results/bc/evaluacion.csv

Uso:
    MUJOCO_GL=egl python3 demos/bc_faja.py generar --episodios 40
    MUJOCO_GL=egl python3 demos/bc_faja.py entrenar --dagger 2
    MUJOCO_GL=egl python3 demos/bc_faja.py evaluar --episodios 20
    MUJOCO_GL=egl python3 demos/bc_faja.py video --politica bc --seed 1000
"""
import os
# Un hilo de BLAS/OpenMP por proceso, ANTES de importar numpy: con varios
# procesos en paralelo el número de hilos cambiaba el orden de las sumas, y
# el episodio (contactos: caótico) divergía; la misma semilla daba 1 o 7
# frutas en serie o en el pool. Así cada semilla es reproducible.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse                                                               # noqa: E402
import copy                                                                   # noqa: E402
import csv                                                                    # noqa: E402
import glob                                                                   # noqa: E402
import pickle                                                                 # noqa: E402
import sys                                                                    # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402

import mujoco                                                                 # noqa: E402
import numpy as np                                                            # noqa: E402

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import visual_servoing_grasp as V                                              # noqa: E402
from h1_2_mujoco_sim_bridge.mujoco_sim import MujocoSim                      # noqa: E402
from h1_2_mujoco_sim_bridge.pd_controller import JointPD                      # noqa: E402
from h1_2_algoritms.vision import grasp_geometry as GG                         # noqa: E402
from h1_2_algoritms.vision.multi_tracking import MultiFruitTracker, detect_fruits, on_belt  # noqa: E402
from h1_2_algoritms.bimanual.bimanual_avoidance import BimanualAvoidanceController  # noqa: E402
from h1_2_algoritms.bimanual.collision_model import arm_frames                # noqa: E402
from h1_2_algoritms.bimanual.depth_obstacles import obstacles_from_depth, ObstacleMemory  # noqa: E402
from h1_2_algoritms.aprendizaje.dmp import DMP                                # noqa: E402
from h1_2_algoritms.aprendizaje import escena_faja as EF                    # noqa: E402
from h1_2_algoritms.aprendizaje.escena_faja import build_xml, FajaMulti      # noqa: E402
from h1_2_algoritms.aprendizaje.habilidades import ArmExecutor, DEPOSIT_SHORT  # noqa: E402
from h1_2_algoritms.aprendizaje.alto_nivel import (                          # noqa: E402
    FifoPolicy, ExpertPolicy, BCPolicy, features, reach_time)

OUT = os.path.join(PKG_DIR, "results", "bc")
DATA = os.path.join(OUT, "datos")
DMP_PATH = os.path.join(PKG_DIR, "results", "dmp", "dmp_alcance.npz")
NET_PATH = os.path.join(OUT, "politica_bc.pt")
SIDES = ("left", "right")
_XML = {}


# Escenarios de llegada: (rango de velocidad de la faja [m/s], media de la
# separación exponencial entre frutas [s], número de frutas, rango de x [m]).
#   normal    el de la primera evaluación: con la faja lenta cualquier regla
#             alcanza casi todo (FIFO 86/96 = experto 86/96 con v < 0.045).
#   exigente  las frutas llegan más rápido de lo que los dos brazos las
#             atienden (ciclo ~9 s por brazo): la asignación importa.
#   alcanzable  las llegadas del exigente, con las frutas en la franja de la
#             faja que los brazos alcanzan en TODA su ventana en y. Barrido
#             de IK del agarre: con x = 0.30 la ventana entera está a < 3 mm;
#             con x = 0.38 quedan ~15 cm; con x ≥ 0.40 ningún punto a < 11 mm.
#             Con x ~ U(0.30, 0.40), ~30% de las frutas quedan fuera de ese
#             criterio de IK (en el sim, el agarre asistido igual recoge
#             muchas: LFD_BC §6.5).
#             La faja va de x = 0.28 a 0.49 (escena_faja.BELT_X) y la fruta
#             tiene r = 4 cm: con x < 0.30 se cae por el borde. Franja útil:
#             x ∈ (0.30, 0.37).
ESCENARIOS = {"normal": ((0.03, 0.07), 3.0, 8, (0.30, 0.40)),
              "exigente": ((0.05, 0.08), 2.0, 10, (0.30, 0.40)),
              "alcanzable": ((0.05, 0.08), 2.0, 10, (0.30, 0.37)),
              # barrido de velocidad (mesa nueva): llegadas del exigente, faja más rápida
              "rapido": ((0.08, 0.10), 2.0, 10, (0.30, 0.40)),
              "muy_rapido": ((0.10, 0.12), 2.0, 10, (0.30, 0.40)),
              "extremo": ((0.12, 0.15), 2.0, 10, (0.30, 0.40))}
ESCENARIO = "normal"           # lo fija main() antes de abrir el pool (los procesos lo heredan)
# Perfiles de tiempos de las habilidades (main: --perfil): parámetros de
# ArmExecutor y de su GraspServoing, escala de τ del alcance y el modelo de
# tiempos medido con ese perfil ("timing": main() actualiza
# habilidades.TIMING en el lugar, y así lo ven el experto y las
# características).
#   base    el de las evaluaciones normal / exigente: ciclo 8.6 s.
#   rapido  sostener 0.6 s tras cerrar (antes 0.8), levantar en 0.6 s sin
#           pausa (antes 1.0 + 0.3), abrir a 0.25 rad del depósito (antes
#           0.08) y alcance con τ = 0.6·τ_demo (antes 0.7): ciclo 7.25 s, sin
#           más fallos (FIFO, 16 episodios exigentes: 65 -> 77 frutas de 160).
#           Cerrar los dedos más rápido (0.4 s) ya suelta frutas, y τ = 0.5
#           no acorta más el alcance (la mano tarda en asentarse).
PERFILES = {
    "base": {},
    "rapido": {"reach_tau_scale": 0.6, "open_tol": 0.25,
               "grasp_kwargs": {"close_hold": 0.6, "lift_time": 0.6, "lift_hold": 0.0},
               "timing": {"reach_extra": 0.50, "close_lift": 1.2, "show_verify": 0.85, "deposit": 1.3,
                          "home": 0.9}},
    # base + verificación por la fuerza de los dedos (FajaMulti.grip_force,
    # en el robot FORCE_ACT de la Inspire): sin la pose de mostrar a la
    # cámara (-0.9 s), pero el depósito sale de la fruta levantada (+0.45 s).
    "tactil": {"tactil": True, "timing": {"show_verify": 0.1, "deposit": 1.95}},
    # tactil + cerrar con la mano asentada (< 0.05 m/s relativos a la
    # fruta: los agarres que se escapaban cerraban a 0.075) + depósito corto
    # (habilidades.DEPOSIT_SHORT, dedos al frente: depositar 1.05 s, volver
    # 0.50): ciclo 6.8 s (experto, 16 episodios exigentes: 94 -> 109 de 160).
    "corto": {"tactil": True, "deposit_pose": DEPOSIT_SHORT,
              "grasp_kwargs": {"close_max_vrel": 0.05},
              "timing": {"show_verify": 0.05, "deposit": 1.05, "home": 0.5}},
    # corto + pose de ESPERA en pre-agarre (home_pose: 12 cm sobre la faja,
    # ya orientada como el agarre; desde V.HOME la mano giraba 1.3 rad) +
    # alcance con τ = 0.5·τ_demo + cierre adaptativo (levanta apenas los
    # dedos sienten la fruta) y subida de 6 cm en 0.5 s: ciclo 5.0 s
    # (experto, 16 episodios exigentes: corto 105 -> 119 de 160; τ = 0.4
    # empeora: 96). Velocidad articular pico medida 3.9 rad/s (URDF: 9-31).
    # Con ciclos de 5 s la regla anti-interferencia del experto (2 s, 20 cm)
    # sobra: con 1 s y 15 cm, 302 -> 324 de 480 (48 episodios de desarrollo,
    # semillas 5000-5047; 10 cm o 0.5 s no agregan; ventanas 4 cm más
    # anchas: 159 -> 112 de 240).
    "espera": {"tactil": True, "experto": {"sep_t": 1.0, "sep_y": 0.15}, "deposit_pose": DEPOSIT_SHORT, "espera": (0.34, -0.10, 0.12),
               "reach_tau_scale": 0.5,
               "grasp_kwargs": {"close_max_vrel": 0.05, "close_touch": True, "lift_height": 0.06,
                                "lift_time": 0.5},
               "timing": {"reach_extra": 0.37, "close_lift": 1.35, "show_verify": 0.05, "deposit": 1.15,
                          "home": 0.4}},
    # espera + brazos amortiguados: el PD articular (kd de DEFAULT_GAINS,
    # ζ ≈ 0.2 en el hombro con armature 0.01) dejaba la muñeca oscilando
    # 4-5 cm durante ~1 s al volver (33 cm recorridos en el segundo
    # siguiente; kd x4: 10 cm, aceleración p99 9.8 -> 6.3 m/s²). Además la
    # referencia se interpola en cada paso de física (antes, escalones de
    # 50 ms) y volver termina con el brazo asentado. Cuesta ~0.6 s de ciclo
    # (depositar 1.15 -> 1.50 s, volver 0.40 -> 0.65; experto, 12 episodios).
    # suave_kp2 (además kp x2) recupera la mitad (1.25 / 0.50 s) pero
    # vuelve a vibrar (aceleración p99 4.5 -> 6.1-7.6 m/s²).
    "suave": {"tactil": True, "experto": {"sep_t": 1.0, "sep_y": 0.15}, "deposit_pose": DEPOSIT_SHORT,
              "espera": (0.34, -0.10, 0.12), "reach_tau_scale": 0.5,
              "kd_escala": 4.0, "interp": True, "home_vel_tol": 0.5,
              "grasp_kwargs": {"close_max_vrel": 0.05, "close_touch": True, "lift_height": 0.06,
                               "lift_time": 0.5},
              "timing": {"reach_extra": 0.37, "close_lift": 1.35, "show_verify": 0.05, "deposit": 1.50,
                         "home": 0.65}},
}
# suave con el QP y las habilidades a 50 / 100 Hz (la cámara sigue a 15 Hz)
PERFILES["suave_50"] = {**PERFILES["suave"], "ctrl_hz": 50}
PERFILES["suave_100"] = {**PERFILES["suave"], "ctrl_hz": 100}
PERFILES["suave_kp2"] = {**PERFILES["suave"], "kp_escala": 2.0,
                         "timing": {**PERFILES["suave"]["timing"], "deposit": 1.25, "home": 0.50}}
# Acortar el ciclo (mesa nueva, 2026-11-09): sufijos combinables sobre suave
# (suave_D, suave_DR, suave_DRA, ...). Los tiempos se recalibran con
# `calibrar` y se pasan con --timing.
#   D  depósito más cerca de la faja (y = 0.30 en vez de 0.40; a −30°, que
#      es el que tiene IK ahí con los límites del robot) y soltar en
#      movimiento: abre a 0.30 rad de la postura de depósito, en 0.3 s.
#   R  volver sin esperar a que el brazo se asiente: libre a 0.25 rad de
#      la espera y con |q̇| < 1.5 rad/s (suave: 0.08 rad, 0.5 rad/s).
#   A  alcance anticipado: meta en el punto de intercepción
#      (GraspServoing.reach_intercept) y el planificador deja empezar con la
#      fruta hasta 25 cm antes de la ventana (alto_nivel.TRACK_BAND; 8 cm).
_VARIANTES = {
    "D": {"deposit_pose": ((0.20, -0.30, -0.05 + GG.DZ_MESA), -30.0, 0.0), "open_tol": 0.30, "open_time": 0.30},
    "R": {"joint_tol": 0.25, "home_vel_tol": 1.5},
    "A": {"track_band": 0.25, "grasp_kwargs": {"reach_intercept": True}},
}
for _suf in ("D", "R", "A", "DR", "DA", "RA", "DRA"):
    _p = dict(PERFILES["suave"])
    for _c in _suf:
        for _k, _v in _VARIANTES[_c].items():
            _p[_k] = {**_p.get(_k, {}), **_v} if _k == "grasp_kwargs" else _v
    PERFILES[f"suave_{_suf}"] = _p
PERFIL = "base"                # el del resultado principal (LFD_BC §6.3)
Z_BELT_FRUIT = EF.BELT_TOP + EF.FRUIT_R - GG.TORSO_Z   # z (torso_link) del centro de una fruta en la faja
# Modo ROBOT (main: --robot): los comandos van por el camino del H1-2 real
# (aprendizaje/comandos_robot.py, docs/LFD_ROBOT.md): agente arm_sdk emulado
# (límites, rampa de velocidad, kp/kd del robot, 100 Hz), motor PD puro con la
# gravedad en tau_ff desde el PC y manos Inspire 0-1000 a SPEED_SET.
# None = el sim de la tesis (PD ideal con compensación de gravedad).
ROBOT = None
DROP_DEBUG = bool(os.environ.get("BC_DROP_DEBUG"))
TRACE = [float(x) for x in os.environ["BC_TRACE"].split(",")] if os.environ.get("BC_TRACE") else None
LOST_TIMEOUT = None            # main: --lost-timeout (GraspServoing; None = sin él, salvo en --robot: 1 s)
KP_JOINT = 6.0                 # servo articular de depositar / volver (3: depositar 1.80 s, volver 1.20; 6: 1.45, 0.90)


def home_pose(espera=None):
    """Pose de los brazos libres. Por omisión V.HOME (mano al frente, 26 cm
    sobre la fruta). `espera` = (x, y, approach) del perfil: pre-agarre del
    brazo derecho sobre una fruta en (x, y) de la faja, retrocedido
    `approach` m por la normal de la palma (izquierda reflejada en y). Ya
    está orientada como el agarre: el alcance no tiene que girar la mano."""
    if espera is None:
        return V.HOME
    x, y, approach = espera
    return {s: GG.pregrasp_wrist_target(s, np.array([x, y if s == "right" else -y, Z_BELT_FRUIT]),
                                        approach=approach) for s in SIDES}


def episode_params(seed, escenario=None):
    """Escenario del episodio: velocidad de la faja, llegadas y posiciones."""
    (v_lo, v_hi), gap_mean, n, (x_lo, x_hi) = ESCENARIOS[escenario or ESCENARIO]
    rng = np.random.default_rng(seed)
    v = float(rng.uniform(v_lo, v_hi))
    gaps = [max(0.15 / v, rng.exponential(gap_mean)) for _ in range(n)]
    times = 1.0 + np.cumsum([0.0] + gaps[:-1])
    xs = rng.uniform(x_lo, x_hi, n) - EF.BELT_DX     # misma posición relativa sobre la faja
    return v, times, xs


def make_policy(name):
    if name == "fifo":
        return FifoPolicy()
    if name == "experto":
        return ExpertPolicy(**PERFILES[PERFIL].get("experto", {}))
    if name == "bc":
        return BCPolicy.load(NET_PATH)
    raise ValueError(name)


# =====================================================================
# Episodio
# =====================================================================
def run_episode(policy, seed, labeler=None, perception="sphere", reach_tau_scale=0.7,
                video=None, decision_dt=0.25):
    v_belt, times, xs = episode_params(seed)
    n = len(times)
    if n not in _XML:
        _XML[n] = build_xml(V.MJCF, n)
    sim = MujocoSim(_XML[n], V.DEFAULT_JOINT_NAMES, joint_armature=0.01, joint_damping=0.05)
    m, d = sim.model, sim.data
    G, N = V.DEFAULT_GAINS, V.DEFAULT_JOINT_NAMES
    pd = JointPD(sim.mapping, np.array([G[k][0] for k in N]), np.array([G[k][1] for k in N]),
                 np.array([G[k][2] for k in N]), np.zeros(len(N)))
    idx = {s: [sim.mapping.index[k] for k in V.ARM_JOINTS[s]] for s in SIDES}
    qadr = {s: [m.jnt_qposadr[m.joint(k).id] for k in V.ARM_JOINTS[s]] for s in SIDES}
    hand_ids = {s: [m.actuator(k).id for k in GG.hand_actuator_names(s)] for s in SIDES}
    faja = FajaMulti(m, d, n, v_belt)
    q_ref = np.zeros(14)
    home = home_pose(PERFILES[PERFIL].get("espera"))
    for k, s in enumerate(SIDES):
        q, _ = GG.solve_arm_ik(s, home[s])
        q_ref[7 * k:7 * k + 7] = q
        d.qpos[qadr[s]] = q
        pd.q_des[idx[s]] = q
        d.ctrl[hand_ids[s]] = GG.HAND_PRESHAPE
    q_ref0 = q_ref.copy()
    mujoco.mj_forward(m, d)

    cams = V.Cameras(m, d, noise=True, seed=seed)
    Kc = (cams.K.fx, cams.K.fy, cams.K.cx, cams.K.cy)
    T_cam = V._T(cams.T_torso("robot_rgbd_camera"))
    R_cam, p_cam = cams.T_torso("robot_rgbd_camera")
    ctrl_hz = PERFILES[PERFIL].get("ctrl_hz", 20)
    ctrl = BimanualAvoidanceController(dt=1.0 / ctrl_hz)
    ctrl.Kp_joint = KP_JOINT
    if ROBOT:      # el QP no pide más velocidad articular que la rampa del agente
        ctrl.dq_max = np.full(14, min(2.0, ROBOT["vel_max"]))
        ctrl.dq_min = -ctrl.dq_max
    memory = ObstacleMemory(voxel=0.03)
    tracker = MultiFruitTracker()
    dmp = DMP.load(DMP_PATH)
    prof = dict(PERFILES[PERFIL])
    tau = dmp.tau_demo * prof.pop("reach_tau_scale", reach_tau_scale)
    prof.pop("timing", None)
    prof.pop("espera", None)
    prof.pop("experto", None)
    prof.pop("ctrl_hz", None)
    prof.pop("track_band", None)           # main() lo pone en alto_nivel.TRACK_BAND
    pd.kd[idx["left"] + idx["right"]] *= prof.pop("kd_escala", 1.0)   # amortiguamiento de los brazos
    pd.kp[idx["left"] + idx["right"]] *= prof.pop("kp_escala", 1.0)
    interp = prof.pop("interp", False)     # referencia en rampa entre ciclos de control
    if prof.pop("tactil", False):       # verificación por la fuerza de los dedos, sin mostrar a la cámara
        prof["grasp_kwargs"] = {**prof.get("grasp_kwargs", {}), "touch": faja.grip_force}
        # touch_hits cuenta ciclos de control: misma duración (150 ms) a cualquier frecuencia
        prof["grasp_kwargs"].setdefault("touch_hits", max(3, round(3 * ctrl_hz / 20)))
    # un agarre no puede quedarse esperando una pista que ya no existe (en el robot, siempre)
    lost = LOST_TIMEOUT if LOST_TIMEOUT is not None else (1.0 if ROBOT else None)
    if lost is not None:
        prof["grasp_kwargs"] = {**prof.get("grasp_kwargs", {}), "lost_timeout": lost}
    TR = reach_time(tau)
    ex = {s: ArmExecutor(s, home, dmp, tau, **prof) for s in SIDES}

    dt_phys = m.opt.timestep
    rob = ROBOT
    if rob:
        from h1_2_algoritms.aprendizaje import comandos_robot as CR
        arm_idx = idx["left"] + idx["right"]
        i_torso = sim.mapping.index.get("torso_joint")
        agente = CR.AgenteEmulado(hz=rob["hz"], vel_max=rob["vel_max"], tau_max=rob["tau_max"],
                                  kp_escala=rob["kp_escala"], kd_escala=rob["kd_escala"])
        q27 = np.zeros(CR.J.N)
        q27[CR.IDX_DDS] = q_ref
        if i_torso is not None:
            q27[CR.J.INDICE["torso_joint"]] = sim.get_q_dq()[0][i_torso]
        agente.iniciar(q27)
        manos = CR.ManosInspire(rob["vel_mano"])
        for s in SIDES:
            manos.iniciar(s, GG.HAND_PRESHAPE)
        n_ag = max(1, int(round(1.0 / rob["hz"] / dt_phys)))
        reg = CR.Registro()
        tau_cmd = np.zeros(14)
        sel15 = ([i_torso] if i_torso is not None else [arm_idx[0]]) + arm_idx
    n_ctrl = int(round(1.0 / ctrl_hz / dt_phys))
    n_cam = int(round((1 / 15.0) / dt_phys))
    t_max = times[-1] + 1.10 / v_belt + 15.0
    assigned = {}            # pista -> brazo
    misses = {}              # pista -> (n abortos, t del último)
    samples = []
    t_dec = {s: -np.inf for s in SIDES}
    pending = []
    obstacles = np.zeros((0, 3))
    targets = {s: home[s].copy() for s in SIDES}
    ff = {s: None for s in SIDES}
    hands = {s: GG.HAND_PRESHAPE.copy() for s in SIDES}
    drop_log = []
    counts = {"decisiones": 0, "agarrar": 0, "cierra": 0, "verificado": 0, "fallo": 0, "perdida": 0,
              "suelta": 0}
    d_arms_min = np.inf
    next_spawn = 0
    frames = []
    vid_r = None
    if video:
        vid_r = mujoco.Renderer(m, 480, 640)
        vcam = mujoco.MjvCamera()
        vcam.lookat[:] = [0.36, 0.0, 1.02]
        vcam.distance, vcam.azimuth, vcam.elevation = 1.4, 150, -35

    def reaching_points():
        """Puntos de agarre de las manos que están alcanzando (tapan la fruta)."""
        q = sim.get_q_dq()[0]
        return [GG.wrist_to_fruit(s, q[idx[s]]) for s in SIDES
                if ex[s].busy and ex[s].phase in ("REACH", "APPROACH", "DESCEND")]

    def hand_points():
        out = []
        for s in SIDES:
            if ex[s].busy and ex[s].phase not in ("REACH", "APPROACH", "DESCEND"):
                q = sim.get_q_dq()[0][idx[s]]
                out.append(GG.wrist_to_fruit(s, q))
        return out

    t = 0.0
    for step in range(int(t_max / dt_phys)):
        t = step * dt_phys
        while next_spawn < n and times[next_spawn] <= t:
            faja.spawn(next_spawn, xs[next_spawn], t)
            next_spawn += 1

        # ---------------- cámara (15 Hz, con latencia) ----------------
        if step % n_cam == 0:
            rgb, _, depth, _, _ = cams.grab()
            if perception == "gt":
                dets = list(faja.visible_truth().values())
            else:
                dets = detect_fruits(rgb, depth, cams.K, R_cam, p_cam)
            pending.append((t + 1 / 15.0, dets, t))
            q_now = np.hstack([sim.get_q_dq()[0][idx[s]] for s in SIDES])
            fl, fr = arm_frames("left", q_now[:7]), arm_frames("right", q_now[7:])
            excl = [p for p, _ in tracker.estimates(t).values()] + hand_points()
            P = obstacles_from_depth(depth, Kc, T_cam, fl, fr, exclude_centers=excl or None,
                                     exclude_radius=0.12)
            obstacles = memory.update(P, depth, Kc, T_cam, t)
            if excl and len(obstacles):
                dmin = np.min(np.linalg.norm(obstacles[:, None, :] - np.array(excl)[None], axis=2), axis=1)
                obstacles = obstacles[dmin > 0.12]
            if vid_r is not None:
                vid_r.update_scene(d, vcam)
                fr_ = vid_r.render().copy()
                import cv2
                fr_[0:160, 640 - 213:640] = cv2.resize(rgb, (213, 160))
                txt = f"{policy.name} t={t:4.1f}s  L:{ex['left'].phase}  R:{ex['right'].phase}  " \
                      f"ok={counts['suelta']}"
                cv2.putText(fr_, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                frames.append(fr_)
        while pending and pending[0][0] <= t:
            _, dets, tc = pending.pop(0)
            for s in SIDES:
                for z in dets:
                    ex[s].observe(tc, z)
            # Solo frutas APOYADAS en la faja: una detección a > 5 cm sobre la
            # superficie es la fruta que lleva una mano (la exclusión por
            # radio alrededor de la mano no basta: sujeta, queda a ~8 cm del
            # punto nominal) y contaminaría las pistas de la faja.
            dets_belt = [z for z in dets if z[2] < Z_BELT_FRUIT + 0.05]
            tracker.update(dets_belt, tc, exclude=hand_points(), occluders=reaching_points())

        # ---------------- control (20 Hz) ----------------
        if step % n_ctrl == 0:
            q_meas = np.hstack([sim.get_q_dq()[0][idx[s]] for s in SIDES])
            est = tracker.estimates(t)
            for s in SIDES:
                e = est.get(ex[s].track) if ex[s].busy else None
                targets[s], ff[s], hands[s], evs = ex[s].step(t, q_meas, *(e if e else (None, None)))
                for ev in evs:
                    counts[ev] = counts.get(ev, 0) + 1
                    tid = ex[s].track
                    if DROP_DEBUG:
                        why = ex[s].fsm.events.get("why", "") if ex[s].fsm else ""
                        drop_log.append((t, f"{s} {ev} track={tid} held={faja.held[s]} why={why}"))
                    if ev == "cierra":
                        tracker.remove(tid)
                    elif ev == "suelta":
                        faja.release(s, deposited=True)
                    elif ev in ("fallo", "perdida"):
                        faja.release(s, deposited=False)
                        assigned.pop(tid, None)
                        if ev == "perdida":
                            misses[tid] = (misses.get(tid, (0, 0.0))[0] + 1, t)
                    elif ev == "libre":
                        assigned = {k: a for k, a in assigned.items() if a != s}
            est = tracker.estimates(t)
            # ---- decisiones de alto nivel ----
            for s in ("right", "left"):
                if ex[s].busy or t - t_dec[s] < decision_dt:
                    continue
                t_dec[s] = t
                # Una pista abortada espera 1 s antes de volver a ser candidata
                # y se descarta tras 2 abortos (no reintentar en bucle).
                cands = {k: (p.copy(), vv.copy()) for k, (p, vv) in est.items()
                         if k not in assigned and on_belt(p)
                         and not (k in misses and (misses[k][0] >= 2 or t - misses[k][1] < 1.0))}
                if not cands:
                    continue
                state = {"t": t, "tau": tau, "cands": cands, "arms": {}}
                for a in SIDES:
                    e_ = ex[a]
                    reaching = e_.busy and e_.phase in ("REACH", "APPROACH", "DESCEND")
                    state["arms"][a] = {"busy": e_.busy, "remaining": e_.remaining(t),
                                        "target": None if not reaching or e_.target_p is None else e_.target_p.copy(),
                                        "t_grasp": e_.t_skill + TR if reaching else None}
                act = policy.act(state, s)
                counts["decisiones"] += 1
                if labeler is not None:
                    lab = act if labeler is policy else labeler.act(state, s)
                    samples.append({"state": copy.deepcopy(state), "side": s, "label": lab})
                if act is not None:
                    ex[s].pick(t, act)
                    assigned[act] = s
                    counts["agarrar"] += 1
            dq, diag = ctrl.step(q_ref, targets, obstacles=obstacles, x_dot_ff=ff,
                                 joint_targets={s: ex[s].joint_target for s in SIDES})
            q_ref0 = q_ref
            q_ref = q_ref + dq * ctrl.dt
            if rob:
                # la orden del PC: q del QP, dq como feedforward y g(q medida) en tau_ff
                dq14 = dq.copy() if rob["dq"] else None
                tau14 = sim.gravity_torque()[arm_idx] if rob["grav"] else None
                agente.orden(q_ref, dq14, tau14)
                for s in SIDES:
                    manos.mandar(s, hands[s])
                reg.orden_pc(t, q_ref, dq14, tau14,
                             [*CR.rad_a_inspire("left", hands["left"]), *CR.rad_a_inspire("right", hands["right"])])
            d_arms_min = min(d_arms_min, diag["d_arms_min"])
            for s in SIDES:
                if ex[s].skill == "agarrar" and ex[s].phase in ("CLOSE", "LIFT") and faja.held[s] is None:
                    faja.try_weld(s, GG.wrist_to_fruit(s, q_meas[:7] if s == "left" else q_meas[7:]))
            if next_spawn == n and not any(st in ("belt", "parked") or st.startswith("held")
                                           for st in faja.state) and not any(ex[s].busy for s in SIDES):
                break

        if rob:
            if step % n_ag == 0:            # un ciclo del agente: el LowCmd de rt/arm_sdk
                cmd = agente.ciclo()
                qm, dqm = sim.get_q_dq()
                reg.ciclo_agente(t, cmd, qm[sel15], dqm[sel15])
                pd.q_des[arm_idx] = cmd["q"][1:]
                pd.dq_des[arm_idx] = cmd["dq"][1:]
                pd.kp[arm_idx] = cmd["kp"][1:]
                pd.kd[arm_idx] = cmd["kd"][1:]
                tau_cmd = cmd["tau"][1:]
            # motor del robot = PD puro + tau_ff: se descuenta la compensación de
            # gravedad ideal que MujocoSim suma en cada paso
            pd.tau_ff[arm_idx] = tau_cmd - sim.gravity_torque()[arm_idx]
            hr = manos.paso(t, dt_phys)
            for s in SIDES:
                d.ctrl[hand_ids[s]] = hr[s]
        else:
            for s in SIDES:
                d.ctrl[hand_ids[s]] = hands[s]
            q_cmd = q_ref
            if interp:
                q_cmd = q_ref0 + (q_ref - q_ref0) * min(1.0, (step % n_ctrl + 1) / n_ctrl)
            pd.q_des[idx["left"]] = q_cmd[:7]
            pd.q_des[idx["right"]] = q_cmd[7:]
        faja.before_step()
        sim.step(pd)
        if DROP_DEBUG and TRACE and TRACE[1] <= t <= TRACE[2] and step % 25 == 0:
            i_ = int(TRACE[0])
            wl, wr = (faja.d.xpos[faja.wrist[s_]] for s_ in SIDES)
            print(f"  [traza] t={t:.2f} fruta{i_} {faja.state[i_]:8s} p={np.round(faja.d.xpos[faja.bid[i_]], 3)} "
                  f"izq={np.round(wl, 2)} {ex['left'].phase} der={np.round(wr, 2)} {ex['right'].phase} "
                  f"Fizq={faja.grip_force('left'):.1f} Fder={faja.grip_force('right'):.1f}", flush=True)
        if DROP_DEBUG:
            st0 = list(faja.state)
            pos0 = [faja.d.xpos[b].copy() for b in faja.bid]
        for ev_f, i_f in faja.after_step(t):
            if DROP_DEBUG and ev_f == "dropped":
                print(f"[caida] {policy.name} s{seed} t={t:.2f} fruta {i_f} estado_previo={st0[i_f]} "
                      f"pos={np.round(pos0[i_f], 3)} t_spawn={faja.t_spawn[i_f]:.1f} | "
                      + " | ".join(f"{s}: {ex[s].skill}/{ex[s].phase} track={ex[s].track} held={faja.held[s]}"
                                   for s in SIDES), flush=True)
                for tl, e_ in drop_log[-12:]:
                    print(f"    {tl:6.2f} {e_}", flush=True)

    res = {"politica": policy.name, "seed": seed, "v_faja": round(v_belt, 3), "n_frutas": n,
           "depositadas": faja.state.count("deposited"), "perdidas_faja": faja.state.count("lost"),
           "caidas": faja.state.count("dropped"), **counts,
           "t_fin": round(t, 1), "d_brazos_min_mm": round(1e3 * d_arms_min, 1)}
    res["tasa"] = round(res["depositadas"] / n, 3)
    if rob:
        res.update({k: round(v, 4) for k, v in CR.resumen_registro(reg, 1.0 / rob["hz"]).items()})
        res["ordenes_recortadas"] = int(agente.n_recortes)
        res["tau_ff_saturado"] = int(agente.n_tau_sat)
        if rob.get("registro"):
            os.makedirs(rob["registro"], exist_ok=True)
            reg.guardar(os.path.join(rob["registro"], f"comandos_{policy.name}_s{seed}.npz"),
                        hz_agente=rob["hz"], vel_max=rob["vel_max"], tau_max=rob["tau_max"],
                        ctrl_hz=ctrl_hz, perfil=PERFIL, seed=seed)
    res["_eventos"] = {s: list(ex[s].log) for s in SIDES}
    if os.environ.get("BC_DEBUG"):
        for s in SIDES:
            for e in ex[s].log:
                print(s, *e)
            print(s, "fase final:", ex[s].phase, "track", ex[s].track)
    if video and frames:
        import cv2
        vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
        for f in frames:
            vw.write(f[:, :, ::-1])
        vw.release()
    return res, samples


# =====================================================================
# Datos, entrenamiento, evaluación
# =====================================================================
def _job(a):
    pol_name, seed, label, tag, perception = a[:5]
    keep_events = len(a) > 5 and a[5]
    pol = make_policy(pol_name)
    lab = ExpertPolicy(**PERFILES[PERFIL].get("experto", {})) if label else None
    if label and pol_name == "experto":
        lab = pol
    try:
        res, samples = run_episode(pol, seed, labeler=lab, perception=perception)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"politica": pol_name, "seed": seed, "error": repr(e)}
    if label:
        with open(os.path.join(DATA, f"{tag}_s{seed}.pkl"), "wb") as f:
            pickle.dump(samples, f)
    res["muestras"] = len(samples)
    if not keep_events:
        res.pop("_eventos", None)
    return res


def _pool(jobs, workers):
    with Pool(workers, maxtasksperchild=1) as P:
        return P.map(_job, jobs, chunksize=1)


def calibrar(args):
    """Duraciones medidas de las habilidades (mediana sobre episodios con el
    experto), para el modelo de tiempos (habilidades.TIMING)."""
    from h1_2_algoritms.aprendizaje.habilidades import TIMING
    res = _pool([("experto", s, False, "", args.percepcion, True) for s in range(900, 900 + args.episodios)],
                args.workers)
    dur = {"alcance (agarrar->cierra)": [], "cerrar+levantar+mostrar (cierra->verificado)": [],
           "depositar (verificado->suelta)": [], "volver (suelta->libre)": []}
    pairs = [("agarrar", "cierra", 0), ("cierra", "verificado", 1), ("verificado", "suelta", 2), ("suelta", "libre", 3)]
    keys = list(dur)
    for r in res:
        for log in r.get("_eventos", {}).values():
            for (t0, e0, k0, *_), (t1, e1, k1, *_) in zip(log, log[1:]):
                for a, b, j in pairs:
                    if e0 == a and e1 == b and k0 == k1:
                        dur[keys[j]].append(t1 - t0)
    for k, v in dur.items():
        print(f"{k:48s} n={len(v):3d}  mediana {np.median(v):5.2f} s  p90 {np.percentile(v, 90):5.2f} s"
              if v else f"{k}: sin datos")
    print("TIMING actual:", TIMING)


def generar(args):
    os.makedirs(DATA, exist_ok=True)
    seeds = range(args.seed0, args.seed0 + args.episodios)
    t0 = time.time()
    res = _pool([("experto", s, True, "experto", args.percepcion) for s in seeds], args.workers)
    print(f"{len(res)} episodios en {time.time() - t0:.0f} s")
    resumen(res)


def load_samples(tags=None):
    S = []
    for f in sorted(glob.glob(os.path.join(DATA, "*.pkl"))):
        if tags and not any(os.path.basename(f).startswith(t) for t in tags):
            continue
        with open(f, "rb") as fh:
            S += pickle.load(fh)
    return S


def to_tensors(samples):
    out = []
    for smp in samples:
        ids, X, ctx = features(smp["state"], smp["side"])
        y = ids.index(smp["label"]) if smp["label"] in ids else len(ids)
        out.append((X, ctx, y))
    return out


def _batch(data):
    """Lista de (X, ctx, y) -> tensores con relleno; la etiqueta "esperar"
    (y = n_i) se lleva a la última columna (n_max)."""
    import torch
    nmax = max(len(X) for X, _, _ in data)
    B = len(data)
    Xp = torch.zeros(B, nmax, data[0][0].shape[1])
    mask = torch.zeros(B, nmax, dtype=torch.bool)
    C = torch.from_numpy(np.stack([c for _, c, _ in data]))
    Y = torch.zeros(B, dtype=torch.long)
    for i, (X, _, y) in enumerate(data):
        Xp[i, :len(X)] = torch.from_numpy(X)
        mask[i, :len(X)] = True
        Y[i] = y if y < len(X) else nmax
    return Xp, mask, C, Y


def fit(samples, epochs=200, lr=2e-3, seed=0, val_frac=0.15):
    """Cross-entropy sobre {frutas candidatas, esperar}. "Esperar" domina
    las muestras (se decide cada 0.25 s mientras un brazo está libre), así
    que se repondera para que las decisiones de agarre pesen igual."""
    import torch
    from h1_2_algoritms.aprendizaje.alto_nivel import build_net
    torch.manual_seed(seed)
    data = [d for d in to_tensors(samples) if len(d[0]) > 0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data))
    nv = int(len(data) * val_frac)
    val, tr = [data[i] for i in perm[:nv]], [data[i] for i in perm[nv:]]
    Xt, Mt, Ct, Yt = _batch(tr)
    Xv, Mv, Cv, Yv = _batch(val)
    wait_t = Yt == Xt.shape[1]
    n_wait = int(wait_t.sum())
    w = torch.ones(len(Yt))
    if n_wait:
        w[wait_t] = min(1.0, (len(Yt) - n_wait) / n_wait)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    best, best_acc = None, -1.0
    for ep in range(epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(len(Yt)))
        for b in range(0, len(order), 128):
            i = order[b:b + 128]
            lg = net.forward_batch(Xt[i], Mt[i], Ct[i])
            loss = (w[i] * torch.nn.functional.cross_entropy(lg, Yt[i], reduction="none")).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = accuracy(net, (Xv, Mv, Cv, Yv))
        if acc["total"] >= best_acc:
            best_acc, best = acc["total"], copy.deepcopy(net.state_dict())
    net.load_state_dict(best)
    return net, accuracy(net, (Xv, Mv, Cv, Yv)), len(tr), len(val)


def accuracy(net, batch):
    import torch
    X, M, C, Y = batch
    net.eval()
    with torch.no_grad():
        k = net.forward_batch(X, M, C).argmax(dim=1)
    pick = Y < X.shape[1]
    return {"total": float((k == Y).float().mean()),
            "agarrar": float((k[pick] == Y[pick]).float().mean()) if pick.any() else float("nan"),
            "esperar": float((k[~pick] == Y[~pick]).float().mean()) if (~pick).any() else float("nan")}


def entrenar(args):
    import torch
    tags = ["experto"]
    for rnd in range(args.dagger + 1):
        S = load_samples(tags)
        net, acc, ntr, nval = fit(S, epochs=args.epocas)
        torch.save(net.state_dict(), NET_PATH)
        print(f"[ronda {rnd}] {ntr} muestras de entrenamiento, {nval} de validación: "
              f"acierto {100 * acc['total']:.1f}% (agarrar: {100 * acc['agarrar']:.1f}%, "
              f"esperar: {100 * acc['esperar']:.1f}%)")
        if rnd == args.dagger:
            break
        # DAgger: la política aprendida manda, el experto etiqueta.
        seeds = range(args.seed0 + 500 * (rnd + 1), args.seed0 + 500 * (rnd + 1) + args.episodios)
        tag = f"dagger{rnd + 1}"
        res = _pool([("bc", s, True, tag, args.percepcion) for s in seeds], args.workers)
        tags.append(tag)
        resumen(res)


def evaluar(args):
    os.makedirs(OUT, exist_ok=True)
    seeds = range(args.seed, args.seed + args.episodios)   # >= 10000: nunca usadas para entrenar
    jobs = [(p, s, False, "", args.percepcion) for p in args.politicas for s in seeds]
    t0 = time.time()
    res = _pool(jobs, args.workers)
    keys = sorted({k for r in res for k in r}, key=lambda k: list(res[0]).index(k) if k in res[0] else 99)
    with open(os.path.join(OUT, args.out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(res)
    print(f"{len(jobs)} episodios en {time.time() - t0:.0f} s -> results/bc/{args.out}\n")
    resumen(res)


def resumen(res):
    print(f"{'política':9s} {'ep':>3s} {'depositadas':>12s} {'tasa':>6s} {'perdidas':>9s} {'caídas':>7s} "
          f"{'intentos':>9s} {'fallos':>7s} {'pérdidas':>9s} {'d brazos':>9s}")
    for p in dict.fromkeys(r["politica"] for r in res):
        r = [x for x in res if x["politica"] == p and "error" not in x]
        if not r:
            continue

        def S(k):
            return sum(int(x.get(k, 0)) for x in r)
        nf = S("n_frutas")
        print(f"{p:9s} {len(r):3d} {S('depositadas'):5d}/{nf:<6d} {S('depositadas') / nf:6.2f} "
              f"{S('perdidas_faja'):9d} {S('caidas'):7d} {S('agarrar'):9d} {S('fallo'):7d} {S('perdida'):9d} "
              f"{min(float(x['d_brazos_min_mm']) for x in r):9.0f}")
    for x in res:
        if "error" in x:
            print("  ERROR:", x)


def video(args):
    os.makedirs(OUT, exist_ok=True)
    suf = "" if PERFIL == "base" else f"_{PERFIL}"
    path = os.path.join(OUT, f"episodio_{args.politica}{suf}_s{args.seed}.mp4")
    res, _ = run_episode(make_policy(args.politica), args.seed, perception=args.percepcion, video=path)
    print(res)
    print("video:", path)


def main():
    global ESCENARIO, KP_JOINT, DATA, NET_PATH, PERFIL, ROBOT, LOST_TIMEOUT
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["calibrar", "generar", "entrenar", "evaluar", "resumen", "video"])
    ap.add_argument("--episodios", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dagger", type=int, default=2)
    ap.add_argument("--epocas", type=int, default=200)
    ap.add_argument("--politicas", nargs="+", default=["fifo", "experto", "bc"])
    ap.add_argument("--politica", default="bc")
    ap.add_argument("--percepcion", default="sphere", choices=["sphere", "gt"])
    ap.add_argument("--out", default="evaluacion.csv")
    ap.add_argument("--escenario", default="normal", choices=list(ESCENARIOS))
    ap.add_argument("--kp-articular", type=float, default=KP_JOINT)
    ap.add_argument("--perfil", default=PERFIL, choices=list(PERFILES))
    ap.add_argument("--timing", default=None, help="k=v,... sobre habilidades.TIMING (p. ej. deposit=1.2,home=0.4)")
    ap.add_argument("--red", default=None, help="red BC a evaluar (por omisión, la del escenario)")
    ap.add_argument("--mapa-alcance", nargs="?", const="", default=None, metavar="JSON",
                    help="ventanas del planificador dependientes de x (alto_nivel.window); FIFO no cambia. "
                         "Opcional: JSON {'right': [[x, [y_lo, y_hi]], ...], 'xmax': x} en lugar del mapa por omisión")
    ap.add_argument("--faja-dx", type=float, default=0.0,
                    help="acerca la faja al robot (m): estudio de montaje (escena_faja.BELT_DX)")
    rb = ap.add_argument_group("modo robot (comandos del H1-2 real: docs/LFD_ROBOT.md)")
    rb.add_argument("--robot", action="store_true", help="agente arm_sdk emulado + motor PD puro + manos Inspire")
    rb.add_argument("--vel-max", type=float, default=1.0, help="--vel-max del agente (rad/s; el suyo por omisión: 1)")
    rb.add_argument("--hz-agente", type=float, default=100.0, help="--hz del agente (arm_sdk: 100)")
    rb.add_argument("--tau-max", type=float, default=10.0, help="--tau-max del agente (N·m de tau_ff)")
    rb.add_argument("--kd-robot", type=float, default=1.0, help="escala de J.KD (el agente hoy usa 1)")
    rb.add_argument("--kp-robot", type=float, default=1.0, help="escala de J.KP")
    rb.add_argument("--sin-gravedad", action="store_true", help="tau_ff = 0 (sin compensación de gravedad del PC)")
    rb.add_argument("--sin-dq", action="store_true", help="dq = 0 en las órdenes")
    rb.add_argument("--vel-mano", type=int, default=1000, help="SPEED_SET de las Inspire")
    rb.add_argument("--registro", default=None, help="carpeta para guardar el flujo de comandos (.npz por episodio)")
    ap.add_argument("--lost-timeout", type=float, default=None,
                    help="s sin ver la fruta durante el alcance -> abortar (LFD_ROBOT §3; en --robot, 1 por omisión)")
    args = ap.parse_args()
    LOST_TIMEOUT = args.lost_timeout
    if args.robot:
        ROBOT = {"hz": args.hz_agente, "vel_max": args.vel_max, "tau_max": args.tau_max,
                 "kd_escala": args.kd_robot, "kp_escala": args.kp_robot, "grav": not args.sin_gravedad,
                 "dq": not args.sin_dq, "vel_mano": args.vel_mano,
                 "registro": os.path.abspath(args.registro) if args.registro else None}
        from h1_2_algoritms.aprendizaje.comandos_robot import aplicar_limites_robot
        aplicar_limites_robot()            # antes del pool: los procesos lo heredan
    ESCENARIO, KP_JOINT, PERFIL = args.escenario, args.kp_articular, args.perfil
    from h1_2_algoritms.aprendizaje.habilidades import TIMING
    TIMING.update(PERFILES[PERFIL].get("timing", {}))
    if args.timing:                # tiempos recalibrados (k=v,k=v): pisan los del perfil
        TIMING.update({k: float(v) for k, v in (kv.split("=") for kv in args.timing.split(","))})
    from h1_2_algoritms.aprendizaje import alto_nivel
    alto_nivel.TRACK_BAND = PERFILES[PERFIL].get("track_band", alto_nivel.TRACK_BAND)
    if ESCENARIO != "normal":      # datos y red propios del escenario
        DATA = os.path.join(OUT, f"datos_{ESCENARIO}")
        NET_PATH = os.path.join(OUT, f"politica_bc_{ESCENARIO}.pt")
    if PERFIL != "base":           # datos y red propios del perfil (el experto etiqueta con sus tiempos)
        DATA += f"_{PERFIL}"
        NET_PATH = NET_PATH[:-3] + f"_{PERFIL}.pt"
    if args.mapa_alcance is not None:          # antes del pool: los procesos lo heredan
        from h1_2_algoritms.aprendizaje import alto_nivel
        alto_nivel.USE_REACH_MAP = True
        tag = "_mapa"
        if args.mapa_alcance:
            import json
            with open(args.mapa_alcance) as fh:
                m = json.load(fh)
            alto_nivel.REACH_MAP_RIGHT = [(x, tuple(w)) for x, w in m["right"]]
            alto_nivel.REACH_MAP_XMAX = m["xmax"]
            tag += "_" + os.path.splitext(os.path.basename(args.mapa_alcance))[0]
        DATA += tag
        NET_PATH = NET_PATH[:-3] + tag + ".pt"
    if args.faja_dx:               # antes del pool y de build_xml
        EF.BELT_DX = args.faja_dx
        tag = f"_dx{round(args.faja_dx * 100)}"
        DATA += tag
        NET_PATH = NET_PATH[:-3] + tag + ".pt"
    if args.red:
        NET_PATH = os.path.join(OUT, args.red)
    if args.accion == "calibrar":
        calibrar(args)
    elif args.accion == "generar":
        generar(args)
    elif args.accion == "entrenar":
        entrenar(args)
    elif args.accion == "evaluar":
        evaluar(args)
    elif args.accion == "video":
        video(args)
    else:
        with open(os.path.join(OUT, args.out)) as f:
            resumen(list(csv.DictReader(f)))


if __name__ == "__main__":
    main()
