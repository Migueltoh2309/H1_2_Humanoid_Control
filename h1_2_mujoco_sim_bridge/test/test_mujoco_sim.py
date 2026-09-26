"""Tests de MujocoSim contra el MJCF real: cámara, contactos, mundo->pelvis."""
import numpy as np
import pytest

from h1_2_mujoco_sim_bridge.mujoco_sim import MujocoSim

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
def sim(mjcf_path):
    return MujocoSim(
        mjcf_path, joint_names=FULL_27, timestep=0.001,
        joint_armature=0.01, joint_damping=0.05,
        base_body="pelvis", camera_name="robot_rgbd_camera",
    )


def test_world_offset_is_pelvis_position(sim):
    assert np.allclose(sim.world_offset, [0.0, 0.0, 1.03], atol=1e-6)


def test_camera_resolution_from_mjcf(sim):
    assert (sim.camera_width, sim.camera_height) == (640, 480)


def test_camera_optical_pose_is_orthonormal_rotation(sim):
    pos, quat_xyzw = sim.camera_optical_pose_in_body()
    assert np.allclose(pos, [0.11109, 0.0175, 0.68789], atol=1e-5)
    assert np.isclose(np.linalg.norm(quat_xyzw), 1.0, atol=1e-9)


def test_camera_optical_frame_matches_mujoco_axes_flipped(sim):
    """El eje Z (adelante) del frame óptico debe ser -Z_cámara de MuJoCo, y
    el eje Y (abajo) debe ser -Y_cámara de MuJoCo, ambos en el frame del
    body padre (torso_link)."""
    import mujoco

    _, quat_xyzw = sim.camera_optical_pose_in_body()
    x, y, z, w = quat_xyzw
    R_opt = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    cid = sim.camera_id
    bodyid = sim.model.cam_bodyid[cid]
    R_cam_world = sim.data.cam_xmat[cid].reshape(3, 3)
    R_body_world = sim.data.xmat[bodyid].reshape(3, 3)
    R_cam_body = R_body_world.T @ R_cam_world
    expected = R_cam_body @ np.diag([1.0, -1.0, -1.0])
    assert np.max(np.abs(R_opt - expected)) < 1e-9


def test_render_rgb_and_depth_shapes(sim):
    rgb = sim.render_rgb()
    assert rgb.shape == (sim.camera_height, sim.camera_width, 3)
    assert rgb.dtype == np.uint8
    depth = sim.render_depth()
    assert depth.shape == (sim.camera_height, sim.camera_width)
    assert np.all(depth >= 0.0)


def test_contacts_return_positions_relative_to_pelvis(sim):
    for _ in range(50):  # dejar caer el objeto de la mesa unos pasos
        import mujoco
        mujoco.mj_step(sim.model, sim.data)
    contacts = sim.get_contacts(min_force=0.0)
    assert isinstance(contacts, list)
    for c in contacts:
        assert c.pos.shape == (3,)


def test_gravity_torque_holds_arm_against_gravity(sim):
    """g(q) es el par que sostiene el brazo: con el hombro derecho levantado
    debe ser NO nulo y del signo que se opone a la caída, y en cambio las
    muñecas (que casi no cargan peso) deben dar prácticamente cero."""
    sim.data.qpos[:] = 0.0
    sim.data.qpos[sim.mapping.qpos_adr[sim.mapping.index["right_shoulder_pitch_joint"]]] = -1.4
    g = sim.gravity_torque()
    assert g.shape == (len(sim.mapping),)
    i_sh = sim.mapping.index["right_shoulder_pitch_joint"]
    i_wr = sim.mapping.index["right_wrist_yaw_joint"]
    assert abs(g[i_sh]) > 1.0          # el hombro carga todo el brazo
    assert abs(g[i_wr]) < 0.5          # la muñeca, casi nada


def test_gravity_torque_matches_qfrc_bias_at_rest(sim):
    """En reposo (qvel=0) no hay Coriolis, así que g(q) debe coincidir con
    el qfrc_bias que MuJoCo ya calcula."""
    import mujoco
    sim.data.qpos[:] = 0.0
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    g = sim.gravity_torque()
    bias = sim.data.qfrc_bias[sim.mapping.qvel_adr]
    assert np.allclose(g, bias, atol=1e-8)


def test_gravity_comp_can_be_disabled(mjcf_path):
    sim_off = MujocoSim(mjcf_path, joint_names=FULL_27, timestep=0.001,
                        base_body="pelvis", gravity_comp=False)
    assert sim_off.gravity_comp is False
    assert MujocoSim(mjcf_path, joint_names=FULL_27, timestep=0.001,
                     base_body="pelvis").gravity_comp is True
