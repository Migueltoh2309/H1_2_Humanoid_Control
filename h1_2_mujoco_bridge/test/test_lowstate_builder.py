"""Tests de la construcción de LowState (con dobles duck-typed, sin ROS)."""
import numpy as np

from h1_2_mujoco_lowlevel_bridge.lowstate_builder import (NUM_STATE_MOTORS,
                                                          fill_lowstate)


class FakeIMU:
    def __init__(self):
        self.quaternion = [0.0] * 4
        self.gyroscope = [0.0] * 3
        self.accelerometer = [0.0] * 3
        self.rpy = [0.0] * 3
        self.temperature = 0


class FakeMotorState:
    def __init__(self):
        self.mode = 0
        self.q = self.dq = self.ddq = self.tau_est = 0.0
        self.temperature = [0, 0]
        self.vol = 0.0
        self.sensor = [0, 0]
        self.motorstate = 0
        self.reserve = [0, 0, 0, 0]


class FakeLowState:
    def __init__(self):
        self.version = [0, 0]
        self.mode_pr = 0
        self.mode_machine = 0
        self.tick = 0
        self.imu_state = FakeIMU()
        self.motor_state = [FakeMotorState() for _ in range(NUM_STATE_MOTORS)]
        self.wireless_remote = [0] * 40
        self.reserve = [0, 0, 0, 0]
        self.crc = 0


def make_snap():
    n = 27
    return {
        "q": np.arange(n) * 0.01,
        "dq": np.arange(n) * -0.001,
        "ddq": np.ones(n) * 0.5,
        "tau_est": np.ones(n) * 2.0,
        "mode": np.ones(n, dtype=int),
        "sim_time": 1.234,
    }


def _fill(msg):
    fill_lowstate(
        msg, make_snap(),
        imu_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        imu_rpy=np.array([0.01, 0.02, 0.03]),
        imu_gyro=np.array([0.1, 0.2, 0.3]),
        imu_accel=np.array([0.0, 0.0, 9.81]),
        mode_pr=0, mode_machine=4,
    )


def test_has_35_motor_states_and_sim_values():
    msg = FakeLowState()
    _fill(msg)
    assert len(msg.motor_state) == 35
    assert msg.motor_state[0].q == 0.0
    assert msg.motor_state[19].q == np.float64(19 * 0.01)   # wrist_yaw izq
    assert msg.motor_state[26].tau_est == 2.0
    assert msg.motor_state[26].mode == 1


def test_extra_motors_27_to_34_are_safe_zeros():
    msg = FakeLowState()
    _fill(msg)
    for i in range(27, 35):
        s = msg.motor_state[i]
        assert s.mode == 0 and s.q == 0.0 and s.dq == 0.0 and s.tau_est == 0.0


def test_imu_passthrough_and_conventions():
    msg = FakeLowState()
    _fill(msg)
    assert msg.imu_state.quaternion == [1.0, 0.0, 0.0, 0.0]   # (w,x,y,z)
    assert msg.imu_state.accelerometer[2] == 9.81             # +g en reposo
    assert msg.imu_state.rpy[2] == 0.03


def test_header_and_tick():
    msg = FakeLowState()
    _fill(msg)
    assert msg.mode_machine == 4
    assert msg.mode_pr == 0
    assert msg.tick == 1234            # ms de simulación
    assert msg.crc == 0                # documentado: no se calcula
    assert list(msg.wireless_remote) == [0] * 40
