"""DMP de alcance (aprendizaje/dmp.py) y su fase REACH en GraspServoing,
sin MuJoCo: demostraciones sintéticas con forma conocida."""
import numpy as np

from h1_2_algoritms.aprendizaje.dmp import DMP, DMPRunner
from h1_2_algoritms.aprendizaje.demostraciones import to_right
from h1_2_algoritms.vision import grasp_geometry as GG
from h1_2_algoritms.vision.visual_servoing import GraspServoing

DT = 0.01


def synthetic_demo(y0, T=3.0, bump=0.05):
    """Mínimo jerk desde y0 hasta 0 más una joroba en z (el "pasar por
    encima" del pre-agarre), cuyo tamaño depende del inicio: la DMP
    paramétrica tiene que capturar esa dependencia."""
    t = np.arange(0.0, T + 1e-9, DT)
    u = t / T
    mj = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    y = np.outer(1 - mj, y0)
    y[:, 2] += (bump + 0.3 * y0[1]) * np.sin(np.pi * u) ** 2
    yd = np.gradient(y, DT, axis=0)
    return t, y, yd, np.gradient(yd, DT, axis=0)


def demos():
    starts = [np.array([0.02, yy, 0.15]) for yy in (-0.10, -0.05, 0.0, 0.05, 0.10, 0.20)]
    return [synthetic_demo(s) for s in starts]


def test_converges_to_goal():
    m = DMP(parametric=True).fit(demos())
    for y0 in ([0.02, 0.0, 0.15], [0.05, 0.25, 0.10]):
        _, Y, Yd = m.rollout(np.array(y0), t_extra=1.0)
        assert np.linalg.norm(Y[-1]) < 0.003
        assert np.linalg.norm(Yd[-1]) < 0.01


def test_parametric_generalizes_better_than_pooled():
    D = demos()
    held_out = synthetic_demo(np.array([0.02, 0.15, 0.15]))   # inicio no demostrado
    errs = {}
    for par in (False, True):
        m = DMP(parametric=par).fit(D)
        t, Y, _ = m.rollout(held_out[1][0], tau=held_out[0][-1])
        errs[par] = np.linalg.norm(Y - held_out[1][:len(Y)], axis=1).mean()
    assert errs[True] < 0.005
    assert errs[True] < 0.5 * errs[False]


def test_tau_scales_time_not_shape():
    m = DMP(parametric=True).fit(demos())
    y0 = np.array([0.02, 0.05, 0.15])
    _, Y1, _ = m.rollout(y0, tau=3.0)
    _, Y2, _ = m.rollout(y0, tau=1.5)
    assert np.allclose(Y2, Y1[::2][:len(Y2)], atol=0.005)


def test_phase_stopping_slows_phase():
    m = DMP().fit(demos())
    r0, r1 = DMPRunner(m, [0.0, 0.1, 0.1]), DMPRunner(m, [0.0, 0.1, 0.1], alpha_err=50.0)
    for _ in range(50):
        r0.step(DT)
        r1.step(DT, err=0.05)
    assert r1.s > r0.s


def test_save_load(tmp_path):
    m = DMP(parametric=True).fit(demos())
    p = tmp_path / "dmp.npz"
    m.save(p)
    m2 = DMP.load(p)
    y0 = np.array([0.02, 0.07, 0.12])
    assert np.allclose(m.rollout(y0)[1], m2.rollout(y0)[1])


def test_mirror_is_involution():
    v = np.array([0.1, -0.2, 0.3])
    assert np.allclose(to_right("left", to_right("left", v)), v)
    assert np.allclose(to_right("right", v), v)


def test_reach_phase_tracks_moving_goal():
    """GraspServoing con DMP: con un "robot" que sigue la referencia
    perfecto (q por IK de x_ref), la fase REACH llega al agarre de una
    fruta que se mueve y pasa a CLOSE; el feedforward incluye ġ."""
    m = DMP(parametric=True).fit(demos())
    home = {"left": np.array([0.25, 0.25, 0.10, 1, 0, 0, 0.0]),
            "right": np.array([0.25, -0.25, 0.10, 1, 0, 0, 0.0])}
    fsm = GraspServoing(home, reach_dmp=m, reach_tau=2.0)
    q = {s: GG.solve_arm_ik(s, home[s])[0] for s in ("left", "right")}
    v = np.array([0.0, 0.03, 0.0])
    p0 = np.array([0.34, -0.20, -0.163])
    saw_ff = False
    for k in range(int(5.0 / 0.05)):
        t = k * 0.05
        out = fsm.step(t, np.hstack((q["left"], q["right"])), p0 + t * v, v, 10)
        if out["phase"] == "REACH":
            ff = out["x_dot_ff"]["right"]
            saw_ff = saw_ff or ff is not None
            q["right"] = GG.solve_arm_ik("right", out["targets"]["right"], q0=q["right"])[0]
        if out["phase"] == "CLOSE":
            break
    assert saw_ff
    assert out["side"] == "right"
    assert out["phase"] == "CLOSE", out["phase"]
    assert fsm.events["err_close"] < fsm.tol_grasp
