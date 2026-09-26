"""Tests del mapeo por-nombre contra el MJCF real del proyecto."""
import mujoco
import pytest

from h1_2_mujoco_sim_bridge.joint_mapping import build_joint_mapping

ARMS_ONLY = [
    "torso_joint",
    "left_shoulder_pitch_joint", "left_elbow_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_elbow_joint", "right_wrist_yaw_joint",
]

FULL_27 = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


@pytest.fixture(scope="module")
def model(mjcf_path):
    return mujoco.MjModel.from_xml_path(mjcf_path)


def test_full_mapping_27(model):
    mapping = build_joint_mapping(model, FULL_27)
    assert len(mapping) == 27
    assert mapping.joint_names == FULL_27


def test_subset_mapping_arms_only(model):
    mapping = build_joint_mapping(model, ARMS_ONLY)
    assert len(mapping) == len(ARMS_ONLY)
    assert "left_hip_yaw_joint" not in mapping.index


def test_addresses_unique_and_valid(model):
    mapping = build_joint_mapping(model, FULL_27)
    assert len(set(mapping.qpos_adr.tolist())) == 27
    assert len(set(mapping.qvel_adr.tolist())) == 27
    assert len(set(mapping.act_id.tolist())) == 27
    assert mapping.qpos_adr.max() < model.nq
    assert mapping.qvel_adr.max() < model.nv
    assert mapping.act_id.max() < model.nu


def test_actuator_transmits_to_its_joint(model):
    mapping = build_joint_mapping(model, FULL_27)
    for e in mapping.entries:
        assert int(model.actuator_trnid[e.actuator_id, 0]) == e.joint_id


def test_position_limits_read_from_xml(model):
    mapping = build_joint_mapping(model, FULL_27)
    for e in mapping.entries:
        assert e.q_min < e.q_max


def test_missing_joint_raises_clear_error(model):
    bad = list(FULL_27)
    bad[5] = "joint_inexistente"
    with pytest.raises(RuntimeError, match="joint_inexistente"):
        build_joint_mapping(model, bad)


def test_duplicate_name_raises(model):
    bad = list(FULL_27) + [FULL_27[0]]
    with pytest.raises(RuntimeError, match="repetidos"):
        build_joint_mapping(model, bad)
