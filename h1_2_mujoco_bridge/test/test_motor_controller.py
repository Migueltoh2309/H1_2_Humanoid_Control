"""Tests de la ley de control PD+ff, saturaciones y rate limiting."""
import numpy as np
import pytest

from h1_2_mujoco_lowlevel_bridge.motor_controller import MotorController
from h1_2_mujoco_lowlevel_bridge.safety_limits import SafetyLimits


def make_limits(n=3, tau_max=10.0, rate=1000.0, dq_max=5.0,
                kp_max=200.0, kd_max=20.0):
    return SafetyLimits(
        q_min=np.full(n, -2.0), q_max=np.full(n, 2.0),
        dq_max=np.full(n, dq_max), tau_max=np.full(n, tau_max),
        kp_max=np.full(n, kp_max), kd_max=np.full(n, kd_max),
        tau_rate_max=np.full(n, rate),
    )


def test_pd_formula_exact():
    """tau = tau_ff + kp(q_des-q) + kd(dq_des-dq), verificado numéricamente."""
    lim = make_limits(rate=1e9)          # rate enorme: no interfiere
    c = MotorController(lim, dt_substep=0.001)
    c.set_command(mode=[1, 1, 1], q_des=[0.5, -0.2, 0.0],
                  dq_des=[0.1, 0.0, -0.3], tau_ff=[0.2, 0.0, 1.0],
                  kp=[10.0, 20.0, 5.0], kd=[1.0, 2.0, 0.5])
    q = np.array([0.4, 0.1, 0.2])
    dq = np.array([0.0, 0.5, 0.1])
    tau = c.compute(q, dq)
    esperado = np.array([
        0.2 + 10.0 * (0.5 - 0.4) + 1.0 * (0.1 - 0.0),
        0.0 + 20.0 * (-0.2 - 0.1) + 2.0 * (0.0 - 0.5),
        1.0 + 5.0 * (0.0 - 0.2) + 0.5 * (-0.3 - 0.1),
    ])
    np.testing.assert_allclose(tau, esperado, rtol=1e-12)


def test_mode_zero_gives_zero_torque():
    lim = make_limits(rate=1e9)
    c = MotorController(lim, dt_substep=0.001)
    c.set_command(mode=[0, 1, 0], q_des=[1.0] * 3, dq_des=[0.0] * 3,
                  tau_ff=[5.0] * 3, kp=[100.0] * 3, kd=[1.0] * 3)
    tau = c.compute(np.zeros(3), np.zeros(3))
    assert tau[0] == 0.0 and tau[2] == 0.0 and tau[1] != 0.0


def test_startup_is_disabled():
    lim = make_limits(rate=1e9)
    c = MotorController(lim, dt_substep=0.001)
    tau = c.compute(np.array([1.0, 1.0, 1.0]), np.zeros(3))
    np.testing.assert_array_equal(tau, np.zeros(3))


def test_torque_saturation():
    lim = make_limits(tau_max=10.0, rate=1e9)
    c = MotorController(lim, dt_substep=0.001)
    c.set_command(mode=[1] * 3, q_des=[100.0] * 3, dq_des=[0.0] * 3,
                  tau_ff=[0.0] * 3, kp=[100.0] * 3, kd=[0.0] * 3)
    tau = c.compute(np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(tau, [10.0] * 3)


def test_rate_limit_step_by_step():
    """|delta_tau| <= tau_rate_max * dt en cada compute()."""
    lim = make_limits(tau_max=100.0, rate=1000.0)
    dt = 0.001
    c = MotorController(lim, dt_substep=dt)
    c.set_command(mode=[1] * 3, q_des=[0.0] * 3, dq_des=[0.0] * 3,
                  tau_ff=[50.0] * 3, kp=[0.0] * 3, kd=[0.0] * 3)
    paso = 1000.0 * dt   # 1 N·m por sub-paso
    tau1 = c.compute(np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(tau1, [paso] * 3)
    tau2 = c.compute(np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(tau2, [2 * paso] * 3)
    for _ in range(100):
        tau = c.compute(np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(tau, [50.0] * 3)   # converge al objetivo


def test_disable_ramps_down_smoothly():
    """Watchdog zero_torque: bajada en rampa, sin salto."""
    lim = make_limits(tau_max=100.0, rate=1000.0)
    dt = 0.001
    c = MotorController(lim, dt_substep=dt)
    c.set_command(mode=[1] * 3, q_des=[0.0] * 3, dq_des=[0.0] * 3,
                  tau_ff=[5.0] * 3, kp=[0.0] * 3, kd=[0.0] * 3)
    for _ in range(10):
        c.compute(np.zeros(3), np.zeros(3))       # llega a 5 N·m
    c.disable_all()
    prev = 5.0
    for _ in range(10):
        tau = c.compute(np.zeros(3), np.zeros(3))
        assert abs(tau[0] - prev) <= 1000.0 * dt + 1e-9
        prev = tau[0]
    assert tau[0] == pytest.approx(0.0, abs=1e-9)


def test_hold_position_clamps_gains():
    lim = make_limits(kp_max=50.0, kd_max=2.0, rate=1e9)
    c = MotorController(lim, dt_substep=0.001)
    c.set_hold(np.array([0.1, 0.2, 0.3]), hold_kp=500.0, hold_kd=100.0)
    assert np.all(c.kp <= 50.0) and np.all(c.kd <= 2.0)
    assert np.all(c.mode == 1)


def test_nonfinite_defense():
    lim = make_limits(rate=1e9)
    c = MotorController(lim, dt_substep=0.001)
    c.set_command(mode=[1] * 3, q_des=[0.0] * 3, dq_des=[0.0] * 3,
                  tau_ff=[np.nan, np.inf, 1.0], kp=[0.0] * 3, kd=[0.0] * 3)
    tau = c.compute(np.zeros(3), np.zeros(3))
    assert np.all(np.isfinite(tau))
    assert tau[0] == 0.0 and tau[1] == 0.0
