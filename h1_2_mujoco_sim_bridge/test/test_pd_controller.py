"""Tests del PD por joint (sin MuJoCo: mapping stub mínimo)."""
import numpy as np
import pytest

from h1_2_mujoco_sim_bridge.pd_controller import JointPD


class StubMapping:
    """Suficiente de JointMapping para JointPD/set_targets_by_name."""

    def __init__(self, names):
        self.joint_names = names
        self.index = {n: i for i, n in enumerate(names)}
        self.q_min = np.full(len(names), -3.14)
        self.q_max = np.full(len(names), 3.14)

    def __len__(self):
        return len(self.joint_names)


@pytest.fixture
def mapping():
    return StubMapping(["j0", "j1", "j2"])


@pytest.fixture
def pd(mapping):
    kp = np.array([10.0, 10.0, 10.0])
    kd = np.array([1.0, 1.0, 1.0])
    tau_max = np.array([5.0, 5.0, 5.0])
    q_init = np.array([0.0, 0.1, 0.2])
    return JointPD(mapping, kp, kd, tau_max, q_init)


def test_initial_targets_equal_q_init(pd):
    assert np.allclose(pd.q_des, [0.0, 0.1, 0.2])
    assert np.allclose(pd.dq_des, 0.0)


def test_partial_update_by_name_leaves_others_untouched(pd):
    pd.set_targets_by_name(["j1"], position=[0.5])
    assert pd.q_des[1] == pytest.approx(0.5)
    assert pd.q_des[0] == pytest.approx(0.0)
    assert pd.q_des[2] == pytest.approx(0.2)


def test_unknown_joint_name_is_ignored_not_raised(pd):
    n = pd.set_targets_by_name(["no_existe"], position=[1.0])
    assert n == 0
    assert np.allclose(pd.q_des, [0.0, 0.1, 0.2])


def test_position_clamped_to_joint_limits(pd):
    pd.set_targets_by_name(["j0"], position=[100.0])
    assert pd.q_des[0] == pytest.approx(3.14)


def test_compute_zero_error_gives_zero_torque(pd):
    tau = pd.compute(q=pd.q_des.copy(), dq=np.zeros(3))
    assert np.allclose(tau, 0.0)


def test_compute_saturates_to_tau_max(pd):
    pd.set_targets_by_name(["j0"], position=[3.0])
    tau = pd.compute(q=np.array([-3.0, 0.1, 0.2]), dq=np.zeros(3))
    assert tau[0] == pytest.approx(5.0)  # 10*(3-(-3))=60 -> saturado a tau_max


def test_effort_feedforward_added(pd):
    pd.set_targets_by_name(["j0"], position=[0.0], effort=[2.0])
    tau = pd.compute(q=np.array([0.0, 0.1, 0.2]), dq=np.zeros(3))
    assert tau[0] == pytest.approx(2.0)


def test_tau_ext_added_before_saturation(pd):
    """La compensación de gravedad entra por tau_ext y se suma ANTES de saturar.

    Si se sumara después, el par podría pasarse de tau_max y el motor
    entregaría algo que el robot real no puede dar.
    """
    pd.set_targets_by_name(["j0"], [0.0])
    # Sin error de posición: el par es exactamente el tau_ext pedido.
    tau = pd.compute(np.array([0.0, 0.1, 0.2]), np.zeros(3),
                     tau_ext=np.array([2.0, 0.0, 0.0]))
    assert np.isclose(tau[0], 2.0)

    # tau_ext enorme: debe quedar saturado a tau_max (5.0), no pasarse.
    tau = pd.compute(np.array([0.0, 0.1, 0.2]), np.zeros(3),
                     tau_ext=np.array([100.0, 0.0, 0.0]))
    assert np.isclose(tau[0], 5.0)


def test_tau_ext_defaults_to_none(pd):
    """Sin tau_ext el comportamiento es el PD de siempre."""
    q = np.array([0.0, 0.0, 0.2])
    assert np.allclose(pd.compute(q, np.zeros(3)),
                       pd.compute(q, np.zeros(3), tau_ext=None))
