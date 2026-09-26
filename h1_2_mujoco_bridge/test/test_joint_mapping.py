"""Tests del mapeo Unitree<->MuJoCo contra el MJCF REAL del proyecto."""
import mujoco
import pytest

from h1_2_mujoco_lowlevel_bridge.joint_mapping import (
    DEFAULT_UNITREE_TO_MUJOCO, NUM_CMD_MOTORS, build_joint_mapping)


@pytest.fixture(scope="module")
def model(mjcf_path):
    return mujoco.MjModel.from_xml_path(mjcf_path)


def test_mapping_has_27_entries(model):
    mapping = build_joint_mapping(model)
    assert len(mapping) == NUM_CMD_MOTORS == 27


def test_indices_are_contiguous_and_ordered(model):
    mapping = build_joint_mapping(model)
    assert [e.unitree_index for e in mapping.entries] == list(range(27))


def test_known_correspondences(model):
    mapping = build_joint_mapping(model)
    assert mapping.entries[0].joint_name == "left_hip_yaw_joint"
    assert mapping.entries[12].joint_name == "torso_joint"       # WAIST_YAW
    assert mapping.entries[19].joint_name == "left_wrist_yaw_joint"
    assert mapping.entries[26].joint_name == "right_wrist_yaw_joint"


def test_addresses_unique_and_valid(model):
    mapping = build_joint_mapping(model)
    assert len(set(mapping.qpos_adr.tolist())) == 27
    assert len(set(mapping.qvel_adr.tolist())) == 27
    assert len(set(mapping.act_id.tolist())) == 27
    assert mapping.qpos_adr.max() < model.nq
    assert mapping.qvel_adr.max() < model.nv
    assert mapping.act_id.max() < model.nu


def test_actuator_transmits_to_its_joint(model):
    mapping = build_joint_mapping(model)
    for e in mapping.entries:
        assert int(model.actuator_trnid[e.actuator_id, 0]) == e.joint_id


def test_position_limits_read_from_xml(model):
    mapping = build_joint_mapping(model)
    for e in mapping.entries:
        assert e.q_min < e.q_max


def test_missing_joint_raises_clear_error(model):
    bad = dict(DEFAULT_UNITREE_TO_MUJOCO)
    bad[5] = "joint_inexistente"
    with pytest.raises(RuntimeError, match="joint_inexistente"):
        build_joint_mapping(model, bad)


def test_incomplete_mapping_raises(model):
    bad = dict(DEFAULT_UNITREE_TO_MUJOCO)
    del bad[10]
    with pytest.raises(RuntimeError, match="idx 10"):
        build_joint_mapping(model, bad)
