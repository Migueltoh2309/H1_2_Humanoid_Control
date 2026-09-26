"""Tests de construcción de mensajes (requieren ROS 2 fuenteado: usan
sensor_msgs/visualization_msgs, a diferencia del resto de tests de este
paquete que corren sin ROS)."""
import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy", reason="requiere un entorno ROS 2 fuenteado")

from builtin_interfaces.msg import Time

from h1_2_mujoco_sim_bridge.msg_builders import (camera_info_msg, depth_image_msg,
                                                  rgb_image_msg)
from h1_2_mujoco_sim_bridge.mujoco_sim import Contact
from h1_2_mujoco_sim_bridge.msg_builders import contact_marker_array

STAMP = Time()


def test_rgb_image_msg_shape_and_bytes():
    img = (np.random.rand(4, 8, 3) * 255).astype(np.uint8)
    msg = rgb_image_msg(img, "frame", STAMP)
    assert msg.height == 4 and msg.width == 8
    assert msg.encoding == "rgb8"
    assert msg.step == 8 * 3
    assert len(msg.data) == 4 * 8 * 3


def test_depth_image_msg_32fc1():
    depth = np.full((4, 8), 1.5, dtype=np.float32)
    msg = depth_image_msg(depth, "frame", STAMP)
    assert msg.encoding == "32FC1"
    assert msg.step == 8 * 4
    assert len(msg.data) == 4 * 8 * 4


def test_camera_info_intrinsics_square_pixels():
    info = camera_info_msg(width=640, height=480, fovy_deg=60.0, frame_id="f", stamp=STAMP)
    fx, cx, fy, cy = info.k[0], info.k[2], info.k[4], info.k[5]
    assert fx == pytest.approx(fy)
    assert cx == pytest.approx(320.0)
    assert cy == pytest.approx(240.0)
    # fy = (h/2)/tan(fovy/2)
    import math
    expected_fy = (480 / 2.0) / math.tan(math.radians(60.0) / 2.0)
    assert fy == pytest.approx(expected_fy)


def test_contact_marker_array_scales_with_force():
    contacts = [
        Contact(pos=np.array([0.0, 0.0, 0.0]), normal_force=0.0, body1="a", body2="b"),
        Contact(pos=np.array([0.0, 0.0, 0.0]), normal_force=100.0, body1="a", body2="b"),
    ]
    arr = contact_marker_array(contacts, "pelvis", STAMP)
    assert len(arr.markers) == 2
    assert arr.markers[1].scale.x > arr.markers[0].scale.x


def test_contact_marker_text_carries_body_pair_and_force():
    contacts = [Contact(pos=np.zeros(3), normal_force=12.5, body1="left_wrist_yaw_link", body2="table_front")]
    arr = contact_marker_array(contacts, "pelvis", STAMP)
    body1, body2, force = arr.markers[0].text.split("|")
    assert {body1, body2} == {"left_wrist_yaw_link", "table_front"}
    assert float(force) == pytest.approx(12.5)
