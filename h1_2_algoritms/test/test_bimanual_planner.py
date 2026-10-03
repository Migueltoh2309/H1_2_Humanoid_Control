"""Pruebas de la capa 3 (planificador global), solo cinemática (sin MuJoCo).

Escenario de la bandeja (el mismo de demos/evaluate_bimanual_avoidance.py,
pero con la bandeja dada como nube de puntos en vez de vista por la cámara):
la capa 1 sola se traba debajo; con la capa 3 la mano llega sin que ninguna
postura del camino viole las distancias mínimas de la capa 1.
"""
import numpy as np

from h1_2_algoritms.bimanual.bimanual_avoidance import BimanualAvoidanceController
from h1_2_algoritms.bimanual.bimanual_coordinator import BimanualCoordinator
from h1_2_algoritms.bimanual.bimanual_planner import (
    ArmCollisionChecker, GlobalPlannerLayer, plan_arm, rrt_connect)
from h1_2_algoritms.movimiento.fk_functions import TF2xyzquat, fkine_arm_left_unitree, fkine_arm_right_unitree
from h1_2_algoritms.movimiento.ik_functions import ik_solve_limited
from h1_2_algoritms.movimiento import joint_limits as JL

Q = [1.0, 0.0, 0.0, 0.0]
HOME = {"left": np.array([0.30, 0.20, 0.05, *Q]), "right": np.array([0.30, -0.20, 0.05, *Q])}
GOAL_R = np.array([0.34, -0.22, 0.35, *Q])
TRAY_CENTER = np.array([0.44, -0.22, 0.19])
TRAY_HALF = np.array([0.12, 0.12, 0.01])
DT = 0.05


def tray_points(step=0.02):
    lo, hi = TRAY_CENTER - TRAY_HALF, TRAY_CENTER + TRAY_HALF
    xs, ys = np.arange(lo[0], hi[0] + 1e-9, step), np.arange(lo[1], hi[1] + 1e-9, step)
    return np.array([[x, y, z] for x in xs for y in ys for z in (lo[2], hi[2])])


def home_q():
    q = []
    for side, fk in (("left", fkine_arm_left_unitree), ("right", fkine_arm_right_unitree)):
        lo, hi = JL.get_limits(side)
        q.append(ik_solve_limited(fk, TF2xyzquat, np.zeros(7), HOME[side], lo, hi)["q"])
    return np.hstack(q)


def run(with_planner, seconds=14.0):
    ctrl = BimanualAvoidanceController(dt=DT)
    planner = GlobalPlannerLayer(DT, ctrl.q_min, ctrl.q_max) if with_planner else None
    P = tray_points()
    goals = {"left": HOME["left"], "right": GOAL_R}
    q, d_env_min = home_q(), np.inf
    for k in range(int(seconds / DT)):
        err = BimanualCoordinator.task_error(q, goals)
        jt = planner.step(k * DT, q, goals, err, P) if planner else None
        dq, diag = ctrl.step(q, goals, P, joint_targets=jt)
        q = q + dq * DT
        d_env_min = min(d_env_min, diag["d_env_min"])
    return BimanualCoordinator.task_error(q, goals), d_env_min, planner


def test_layer1_gets_stuck_under_tray():
    err, d_env, _ = run(False, seconds=8.0)
    assert err["right"] > 0.10          # trabada debajo de la bandeja
    assert d_env > 0.04                 # pero segura (d_s del entorno = 0.05)


def test_layer3_reaches_over_tray_safely():
    err, d_env, planner = run(True)
    assert err["right"] < 0.02, planner.events
    assert d_env > 0.04
    assert any(p["status"] == "ok" for p in planner.plans)


def test_planned_path_is_collision_free_with_margins():
    ctrl = BimanualAvoidanceController()
    q = home_q()
    r = plan_arm("right", q, GOAL_R, tray_points(), ctrl.q_min, ctrl.q_max, time_limit=8.0)
    assert r["status"] == "ok"
    ch = ArmCollisionChecker("right", q[:7], tray_points(), ctrl.q_min[7:], ctrl.q_max[7:])
    ch.relax_for_start(q[7:])
    for a, b in zip(r["path"][:-1], r["path"][1:]):
        assert ch.valid(b) and ch.edge_valid(a, b)


def test_goal_inside_obstacle_is_reported_not_attempted():
    ctrl = BimanualAvoidanceController()
    q = home_q()
    inside = np.array([*TRAY_CENTER - np.array([0.10, 0.0, 0.0]), *Q])   # muñeca metida en la bandeja
    r = plan_arm("right", q, inside, tray_points(), ctrl.q_min, ctrl.q_max, time_limit=3.0)
    assert r["status"] == "goal_invalid"


def test_rrt_connect_simple_free_space():
    lo, hi = -np.ones(7), np.ones(7)

    class Free:
        q_min, q_max = lo, hi

        def edge_valid(self, a, b, step=0.04):
            return True

    path = rrt_connect(Free(), np.zeros(7), np.full(7, 0.5))
    assert np.allclose(path[0], 0.0) and np.allclose(path[-1], 0.5)
