"""Mensajes ROS 2 de imagen, camera_info y contactos, sin cv_bridge (en este
entorno cv_bridge choca con numpy; mismo motivo que h1_2_mujoco_sim_bridge)."""
import array
import math

import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray


def imagen(arr, encoding, frame_id, stamp):
    h, w = arr.shape[:2]
    bpp = {"rgb8": 3, "mono8": 1, "32FC1": 4}[encoding]
    msg = Image()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height, msg.width = h, w
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = w * bpp
    dtype = np.float32 if encoding == "32FC1" else np.uint8
    # array.array('B') y no bytes: con bytes el setter de rclpy (Humble) valida
    # byte a byte y una imagen 640x480 RGB tarda ~120 ms; asi, <1 ms
    msg.data = array.array("B", np.ascontiguousarray(arr, dtype=dtype).tobytes())
    return msg


def camera_info(width, height, fovy_deg, frame_id, stamp, tx=0.0):
    """Pinhole de mujoco.Renderer (pixeles cuadrados). `tx` = -fx * linea_de_base
    para la segunda camara de un par estereo (convencion de ROS / realsense2)."""
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    msg = CameraInfo()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height, msg.width = height, width
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, tx * fx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def contactos(lista, frame_id, stamp, r_max=0.03, escala=300.0):
    arr = MarkerArray()
    for i, (pos, f, b1, b2) in enumerate(lista):
        m = Marker()
        m.header = Header(frame_id=frame_id, stamp=stamp)
        m.ns, m.id, m.type, m.action = "contacts", i, Marker.SPHERE, Marker.ADD
        r = min(0.008 + f / escala, r_max)
        m.scale.x = m.scale.y = m.scale.z = 2.0 * r
        m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        m.pose.orientation.w = 1.0
        t = min(f / escala, 1.0)
        m.color = ColorRGBA(r=1.0, g=1.0 - t, b=0.0, a=0.85)
        m.lifetime.nanosec = int(0.5e9)
        m.text = f"{b1}|{b2}|{f:.4f}"
        arr.markers.append(m)
    return arr
