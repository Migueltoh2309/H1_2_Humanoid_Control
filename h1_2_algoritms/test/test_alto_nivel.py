"""Política de alto nivel (aprendizaje/alto_nivel.py) y tracker multi-fruta,
sin MuJoCo: estados simbólicos armados a mano."""
import numpy as np

from h1_2_algoritms.aprendizaje.alto_nivel import (
    ExpertPolicy, FifoPolicy, BCPolicy, build_net, features, start_interval, reach_time,
    N_CAND_FEAT, N_CTX_FEAT)
from h1_2_algoritms.vision.multi_tracking import MultiFruitTracker

TAU = 2.9
V = np.array([0.0, 0.05, 0.0])


def arms(left_busy=False, right_busy=False, rem=0.0):
    return {"left": {"busy": left_busy, "remaining": rem if left_busy else 0.0, "target": None, "t_grasp": None},
            "right": {"busy": right_busy, "remaining": rem if right_busy else 0.0, "target": None,
                      "t_grasp": None}}


def fruit(y):
    return (np.array([0.35, y, -0.16]), V.copy())


def test_start_interval_matches_window():
    p, v = fruit(-0.40)
    TR = reach_time(TAU)
    # sin franja: al empezar en a, la fruta llega al borde de entrada de la
    # ventana (con margen) justo al agarrarla
    a, b = start_interval("right", p, v, 0.0, TAU, track_band=1.0)
    assert np.isclose(p[1] + v[1] * (a + TR), -0.28)
    assert np.isclose(p[1] + v[1] * (b + TR), 0.03)
    # con franja: no se empieza hasta que la fruta está a < 8 cm de la ventana
    a, _ = start_interval("right", p, v, 0.0, TAU)
    assert np.isclose(p[1] + v[1] * a, -0.38)
    # fruta quieta fuera de la ventana: intervalo vacío
    a, b = start_interval("left", np.array([0.35, -0.2, -0.16]), np.zeros(3), 0.0, TAU)
    assert a > b


def test_expert_waits_for_fruit_not_yet_reachable():
    st = {"t": 0.0, "tau": TAU, "cands": {7: fruit(-0.55)}, "arms": arms()}
    assert ExpertPolicy().act(st, "right") is None       # todavía no entra a la ventana a tiempo
    st["t"] = 0.0
    st["cands"] = {7: fruit(-0.30)}
    assert ExpertPolicy().act(st, "right") == 7


def test_expert_leaves_downstream_fruit_to_other_arm():
    """Dos frutas: una ya pasando al lado izquierdo, otra entrando por la
    derecha. El brazo derecho debe tomar la que entra (la otra la alcanza
    el izquierdo); FIFO toma la más antigua."""
    st = {"t": 0.0, "tau": TAU, "cands": {1: fruit(-0.12), 2: fruit(-0.33)}, "arms": arms()}
    assert ExpertPolicy().act(st, "right") == 2
    assert ExpertPolicy().act(st, "left") == 1
    assert FifoPolicy().act(st, "right") == 1


def test_expert_skips_infeasible():
    st = {"t": 0.0, "tau": TAU, "cands": {3: fruit(0.29)}, "arms": arms()}
    assert ExpertPolicy().act(st, "right") is None     # ya pasó la ventana derecha
    assert ExpertPolicy().act(st, "left") is None      # y sale de la izquierda antes del agarre


def test_features_are_mirror_symmetric():
    """La misma situación vista desde el otro brazo (reflejada en y y en la
    dirección de la faja) da las mismas características."""
    st_r = {"t": 0.0, "tau": TAU, "cands": {1: fruit(-0.2)}, "arms": arms(left_busy=True, rem=3.0)}
    p, v = fruit(-0.2)
    st_l = {"t": 0.0, "tau": TAU,
            "cands": {1: (p * [1, -1, 1], v * [1, -1, 1])}, "arms": arms(right_busy=True, rem=3.0)}
    _, Xr, cr = features(st_r, "right")
    _, Xl, cl = features(st_l, "left")
    assert Xr.shape == (1, N_CAND_FEAT) and cr.shape == (N_CTX_FEAT,)
    assert np.allclose(Xr, Xl) and np.allclose(cr, cl)


def test_bc_policy_runs_on_any_number_of_fruits():
    pol = BCPolicy(build_net())
    for n in (1, 3, 7):
        st = {"t": 0.0, "tau": TAU, "cands": {k: fruit(-0.5 + 0.1 * k) for k in range(n)}, "arms": arms()}
        a = pol.act(st, "right")
        assert a is None or a in st["cands"]


def test_multi_tracker_keeps_identities():
    tr = MultiFruitTracker()
    ys = np.array([-0.5, -0.3, -0.1])
    ids = None
    for k in range(30):
        t = k / 15.0
        dets = [np.array([0.35, y + 0.05 * t, -0.16]) for y in ys]
        tr.update(dets[::-1] if k % 2 else dets, t)      # el orden de las detecciones no importa
        est = tr.estimates(t)
        if k == 5:
            ids = {tid: p[1] for tid, (p, _) in est.items()}
    est = tr.estimates(t)
    assert len(est) == 3
    for tid, (p, v) in est.items():
        assert abs(p[1] - (ids[tid] + 0.05 * (t - 5 / 15.0))) < 0.01
        assert abs(v[1] - 0.05) < 0.01
