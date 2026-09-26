"""Tests de los overrides en memoria del simulador (timestep, armature,
damping y elevación de la base)."""
import mujoco
import numpy as np
import pytest

from h1_2_mujoco_lowlevel_bridge.joint_mapping import build_joint_mapping
from h1_2_mujoco_lowlevel_bridge.mujoco_simulator import MujocoSimulator


def make(mjcf_path, **kw):
    return MujocoSimulator(
        mjcf_path, mapping_builder=lambda m: build_joint_mapping(m), **kw
    )


def _ankle_z(sim):
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY,
                            "left_ankle_roll_link")
    return float(sim.data.xpos[bid][2])


def test_base_height_offset_raises_robot(mjcf_path):
    z0 = _ankle_z(make(mjcf_path))
    z1 = _ankle_z(make(mjcf_path, base_height_offset=0.30))
    assert z1 - z0 == pytest.approx(0.30, abs=1e-9)


def test_default_keeps_scene_as_is(mjcf_path):
    sim = make(mjcf_path)
    assert sim.base_height_offset == 0.0


def test_missing_base_body_raises(mjcf_path):
    with pytest.raises(RuntimeError, match="cuerpo_inexistente"):
        make(mjcf_path, base_height_offset=0.1, base_body="cuerpo_inexistente")


def test_timestep_armature_damping_overrides(mjcf_path):
    sim = make(mjcf_path, timestep=0.001,
               joint_armature=0.01, joint_damping=0.05)
    assert sim.dt == pytest.approx(0.001)
    assert np.allclose(sim.model.dof_armature[sim.mapping.qvel_adr], 0.01)
    assert np.allclose(sim.model.dof_damping[sim.mapping.qvel_adr], 0.05)
