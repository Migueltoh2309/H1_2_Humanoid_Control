"""
msg_builders.py — Construcción de mensajes ROS 2 a partir de arrays de MuJoCo.

Deliberadamente NO usa cv_bridge (en este entorno cv_bridge está roto por un
choque de ABI con numpy: `AttributeError: _ARRAY_API not found`). Construir
sensor_msgs/Image a mano es trivial y evita esa dependencia frágil.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from .mujoco_sim import Contact


def rgb_image_msg(rgb: np.ndarray, frame_id: str, stamp) -> Image:
    h, w, _ = rgb.shape
    msg = Image()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height, msg.width = h, w
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return msg


def depth_image_msg(depth_m: np.ndarray, frame_id: str, stamp) -> Image:
    h, w = depth_m.shape
    msg = Image()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height, msg.width = h, w
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = w * 4
    msg.data = np.ascontiguousarray(depth_m, dtype=np.float32).tobytes()
    return msg


def camera_info_msg(width: int, height: int, fovy_deg: float, frame_id: str, stamp) -> CameraInfo:
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy  # píxeles cuadrados (misma convención de mujoco.Renderer)
    cx, cy = width / 2.0, height / 2.0
    msg = CameraInfo()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height, msg.width = height, width
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def contact_marker_array(contacts: List[Contact], frame_id: str, stamp,
                          max_radius: float = 0.03, force_scale: float = 300.0) -> MarkerArray:
    arr = MarkerArray()
    for i, c in enumerate(contacts):
        m = Marker()
        m.header = Header(frame_id=frame_id, stamp=stamp)
        m.ns = "contacts"
        m.id = i
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        r = min(0.008 + c.normal_force / force_scale, max_radius)
        m.scale.x = m.scale.y = m.scale.z = 2.0 * r
        m.pose.position = Point(x=float(c.pos[0]), y=float(c.pos[1]), z=float(c.pos[2]))
        m.pose.orientation.w = 1.0
        t = min(c.normal_force / force_scale, 1.0)
        m.color = ColorRGBA(r=1.0, g=1.0 - t, b=0.0, a=0.85)
        m.lifetime.sec = 0
        m.lifetime.nanosec = int(0.5e9)
        # RViz ignora `text` en un marker que no es TEXT_VIEW_FACING, pero el
        # campo sigue viajando en el mensaje: un suscriptor programático (p.ej.
        # un detector de colisiones) puede leer aquí qué dos cuerpos chocan y
        # con qué fuerza, sin necesitar un tipo de mensaje nuevo.
        m.text = f"{c.body1}|{c.body2}|{c.normal_force:.4f}"
        arr.markers.append(m)
    return arr
