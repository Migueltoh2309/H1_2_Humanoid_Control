"""Pruebas del modelo de colisión del QP bimanual (sin ROS ni MuJoCo)."""
import numpy as np
import pytest

from h1_2_algoritms.movimiento.fk_functions import fkine_arm_left_unitree, fkine_arm_right_unitree
from h1_2_algoritms.bimanual.collision_model import (
    arm_frames, arm_capsules, point_jacobian, closest_points_segments,
    self_collision_constraints, CAPSULES)
from h1_2_algoritms.movimiento import joint_limits as JL

FK = {"left": fkine_arm_left_unitree, "right": fkine_arm_right_unitree}
RNG = np.random.default_rng(1)


def random_q(side):
    lo, hi = JL.get_limits(side)
    return RNG.uniform(lo, hi)


@pytest.mark.parametrize("side", ["left", "right"])
def test_frame7_matches_fk(side):
    for _ in range(20):
        q = random_q(side)
        assert np.allclose(arm_frames(side, q)[7], FK[side](q), atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
def test_capsule_points_rigid_to_their_link(side):
    # Un extremo definido en el frame k no debe moverse al girar las
    # articulaciones > k (si lo hiciera, el jacobiano con columnas 1..k
    # estaría mal). El extremo del brazo superior es el origen del frame 1
    # expresado en el 3: esto comprueba que está sobre el eje del yaw.
    for _ in range(10):
        q = random_q(side)
        caps = arm_capsules(side, q)
        for c, (_, k, *_rest) in zip(caps, CAPSULES):
            q2 = q.copy()
            q2[k:] = random_q(side)[k:]
            c2 = arm_capsules(side, q2)[[x[0] for x in CAPSULES].index(c["name"])]
            assert np.allclose(c["a"], c2["a"], atol=1e-9)
            assert np.allclose(c["b"], c2["b"], atol=1e-9)
    # y el origen del frame 1 coincide con el extremo 'a' del brazo superior
    q = random_q(side)
    assert np.allclose(arm_frames(side, q)[1][:3, 3], arm_capsules(side, q)[0]["a"], atol=1e-9)


@pytest.mark.parametrize("side", ["left", "right"])
def test_point_jacobian_matches_finite_differences(side):
    h = 1e-7
    for _ in range(10):
        q = random_q(side)
        F = arm_frames(side, q)
        for name, k, a_loc, b_loc, _r in CAPSULES:
            loc = 0.3 * a_loc + 0.7 * b_loc
            p = F[k][:3, :3] @ loc + F[k][:3, 3]
            J = point_jacobian(F, k, p)
            Jn = np.zeros((3, 7))
            for i in range(7):
                dq = np.zeros(7)
                dq[i] = h
                Fh = arm_frames(side, q + dq)
                Jn[:, i] = (Fh[k][:3, :3] @ loc + Fh[k][:3, 3] - p) / h
            assert np.allclose(J, Jn, atol=1e-5), name


def test_closest_points_segments_brute_force():
    for _ in range(200):
        p1, q1, p2, q2 = RNG.normal(size=(4, 3))
        c1, c2 = closest_points_segments(p1, q1, p2, q2)
        s = np.linspace(0, 1, 201)
        A = p1 + s[:, None] * (q1 - p1)
        B = p2 + s[:, None] * (q2 - p2)
        brute = np.min(np.linalg.norm(A[:, None] - B[None], axis=2))
        assert np.linalg.norm(c1 - c2) <= brute + 1e-9
        assert np.linalg.norm(c1 - c2) >= brute - 1e-2


def test_self_collision_row_is_distance_derivative():
    # n^T (J_a dq_a - J_b dq_b) debe ser la derivada de la distancia.
    ql, qr = random_q("left") * 0.3, random_q("right") * 0.3
    ql[:4] = [-1.0, 0.2, 0.0, 1.2]
    qr[:4] = [-1.0, -0.2, 0.0, 1.2]
    Fl, Fr = arm_frames("left", ql), arm_frames("right", qr)
    A, lb, info = self_collision_constraints(Fl, Fr, 0.03, 10.0, 0.4)  # d_i grande: todas activas
    d0 = np.array([d for _, d in info])
    dq = RNG.normal(size=14) * 1e-6
    _, _, info2 = self_collision_constraints(
        arm_frames("left", ql + dq[:7]), arm_frames("right", qr + dq[7:]), 0.03, 10.0, 0.4)
    d1 = np.array([d for _, d in info2])
    assert np.allclose(A @ dq, d1 - d0, atol=1e-9)
