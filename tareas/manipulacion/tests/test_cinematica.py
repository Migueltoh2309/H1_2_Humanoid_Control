"""La cinemática de la tesis, comprobada contra el modelo del simulador (el reto pide
comprobar el marco y el efector del IK, no suponerlos), y la IK del reto."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manip import cinematica as C  # noqa: E402

RNG = np.random.default_rng(0)


def _modelo():
    mujoco = pytest.importorskip("mujoco")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))
    from construir_estacion import ESCENA_ROBOT
    return mujoco, mujoco.MjModel.from_xml_path(ESCENA_ROBOT)


@pytest.mark.parametrize("lado", C.LADOS)
def test_fk_tesis_igual_que_mjcf(lado):
    """Muñeca de la DH de la tesis = *_wrist_yaw_link del MJCF, en torso_link."""
    mujoco, m = _modelo()
    d = mujoco.MjData(m)
    tor = m.body("torso_link").id
    mano = m.body(f"{C.INGLES[lado]}_wrist_yaw_link").id
    lo, hi = C.limites(lado, None)
    for _ in range(50):
        q = RNG.uniform(lo, hi)
        mujoco.mj_resetData(m, d)
        for n, v in zip(C.NOMBRES_JUNTAS[lado], q):
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
        mujoco.mj_kinematics(m, d)
        Rt = d.xmat[tor].reshape(3, 3)
        p = Rt.T @ (d.xpos[mano] - d.xpos[tor])
        R = Rt.T @ d.xmat[mano].reshape(3, 3)
        T = C.fk(lado, q)
        assert np.linalg.norm(T[:3, 3] - p) < 1e-5
        assert np.abs(T[:3, :3] - R).max() < 1e-5


@pytest.mark.parametrize("lado", C.LADOS)
def test_jacobiano_geometrico_igual_que_diferencias(lado):
    lo, hi = C.limites(lado)
    for _ in range(10):
        q = RNG.uniform(lo, hi)
        fr = C.frames(lado, q)
        p = C.punto(lado, q, "palma")
        J = C.jacobiano(fr, p)
        h = 1e-6
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = h
            dp = (C.punto(lado, q + dq, "palma") - p) / h
            assert np.allclose(J[:3, i], dp, atol=1e-4)


@pytest.mark.parametrize("lado", C.LADOS)
def test_ik_llega_y_respeta_limites(lado):
    """Objetivo alcanzable por construcción (FK de una postura válida): la IK llega."""
    lo, hi = C.limites(lado)
    n_ok = 0
    for _ in range(20):
        q_ref = C.postura_inicial(lado) + RNG.uniform(-0.3, 0.3, 7)
        q_ref = np.clip(q_ref, lo, hi)
        T = C.fk(lado, q_ref)
        p = T[:3, 3] + T[:3, :3] @ C.PUNTOS_MANO["palma"][lado]
        q, ep, eo, ok = C.ik(lado, p, T[:3, :3])
        assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)
        n_ok += ok
        if ok:
            assert ep < 0.002 and eo < 2.0
    assert n_ok >= 18


def test_simetria_de_los_brazos():
    """El brazo derecho llega al punto simétrico (y -> -y) que el izquierdo, con la
    orientación simétrica: el mapa de alcance de una mano es el espejo del de la otra."""
    p = np.array([0.30, 0.15, -0.12])
    R = C.rotacion_agarre("izq", "arriba", 0.0, -20.0)
    qi, *_, oki = C.ik("izq", p, R)
    S = np.diag([1.0, -1.0, 1.0])
    qd, *_, okd = C.ik("der", S @ p, C.rotacion_agarre("der", "arriba", 0.0, 20.0))
    assert oki and okd
    # mismo cabeceo de hombro y codo; los giros cambian de signo
    assert abs(qi[0] - qd[0]) < 0.05 and abs(qi[3] - qd[3]) < 0.05


def test_log_rot():
    for _ in range(20):
        v = RNG.normal(size=3)
        v *= RNG.uniform(0.0, 3.0) / np.linalg.norm(v)
        a = np.linalg.norm(v)
        k = v / a
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        assert np.allclose(C.log_rot(R), v, atol=1e-6)


def test_holgura_mesa():
    """Un brazo con la mano por debajo del tablero da holgura negativa; levantado, positiva."""
    q = C.postura_inicial("izq")
    p = C.punto("izq", q, "muneca")
    assert C.holgura_plano("izq", q, p[2] + 0.2, -1.0, 1.0) < 0
    assert C.holgura_plano("izq", q, p[2] - 0.5, -1.0, 1.0) > 0
