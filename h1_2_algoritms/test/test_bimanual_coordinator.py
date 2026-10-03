"""Regresión de la capa 2: en la zona compartida, la capa 1 sola se
bloquea y con el coordinador las dos tareas se completan sin acercarse
por debajo de d_s (solo cinemática, sin MuJoCo)."""
import numpy as np

from h1_2_algoritms.bimanual.bimanual_avoidance import BimanualAvoidanceController
from h1_2_algoritms.bimanual.bimanual_coordinator import BimanualCoordinator, TaskSequence
from h1_2_algoritms.movimiento.fk_functions import TF2xyzquat, fkine_arm_left_unitree, fkine_arm_right_unitree
from h1_2_algoritms.movimiento.ik_functions import ik_solve_limited
from h1_2_algoritms.movimiento import joint_limits as JL

Q = [1.0, 0.0, 0.0, 0.0]
HOME = {"left": np.array([0.30, 0.20, 0.05, *Q]), "right": np.array([0.30, -0.20, 0.05, *Q])}
SHARED = np.array([0.34, 0.0, 0.15, *Q])
DT = 0.05


def home_q():
    q = []
    for side, fk in (("left", fkine_arm_left_unitree), ("right", fkine_arm_right_unitree)):
        lo, hi = JL.get_limits(side)
        q.append(ik_solve_limited(fk, TF2xyzquat, np.zeros(7), HOME[side], lo, hi)["q"])
    return np.hstack(q)


def run_shared(use_coord, seconds=15.0):
    ctrl = BimanualAvoidanceController(dt=DT, avoid_obstacles=False)
    coord = BimanualCoordinator(DT, d_influence=ctrl.d_i)
    tasks = {s: TaskSequence([(SHARED, 1.0), (HOME[s], 0.0)]) for s in ("left", "right")}
    q, d, d_min = home_q(), np.inf, np.inf
    for k in range(int(seconds / DT)):
        t = k * DT
        goals = {s: tasks[s].goal() for s in tasks}
        if use_coord:
            tg, err = coord.step(t, q, goals, d)
        else:
            tg, err = goals, BimanualCoordinator.task_error(q, goals)
        for s in tasks:
            tasks[s].update(t, err[s])
        if all(x.done for x in tasks.values()):
            break
        dq, diag = ctrl.step(q, tg)
        q = q + dq * DT
        d = diag["d_arms_min"]
        d_min = min(d_min, d)
    return tasks, d_min, coord


def test_layer1_alone_deadlocks_in_shared_zone():
    tasks, d_min, _ = run_shared(False, seconds=8.0)
    assert not any(x.done for x in tasks.values())
    assert d_min > 0.025


def test_layer2_completes_shared_zone_safely():
    tasks, d_min, coord = run_shared(True)
    assert all(x.done for x in tasks.values()), coord.events
    assert d_min > 0.025
    assert coord.state != "escalate"
