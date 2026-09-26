from visualization_msgs.msg import Marker
import numpy as np

def create_sphere_marker(frame="torso_link",
                         ns="marker",
                         marker_id=0,
                         scale=0.05,
                         color=(1.0,0.0,0.0,1.0)):

    marker = Marker()

    marker.header.frame_id = frame
    marker.ns = ns
    marker.id = marker_id

    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale

    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])

    return marker

def set_marker_pose(marker, pose, node):

    # ===== CHECK NaNs =====
    if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
        node.get_logger().warn("Pose inválida (NaN/Inf), no se publica marker")
        return None

    marker.header.stamp = node.get_clock().now().to_msg()

    # ===== POSICIÓN =====
    marker.pose.position.x = float(pose[0])
    marker.pose.position.y = float(pose[1])
    marker.pose.position.z = float(pose[2])

    # ===== NORMALIZAR CUATERNIÓN =====
    q = np.array(pose[3:7])
    norm_q = np.linalg.norm(q)

    if norm_q < 1e-6:
        node.get_logger().warn("Cuaternión inválido (norma ~0)")
        return None

    q = q / norm_q

    marker.pose.orientation.w = float(q[0])
    marker.pose.orientation.x = float(q[1])
    marker.pose.orientation.y = float(q[2])
    marker.pose.orientation.z = float(q[3])

    return marker

# 🔵 Marker de trayectoria
def create_line_marker(frame="torso_link"):
    marker = Marker()
    marker.header.frame_id = frame
    marker.ns = "trajectory"
    marker.id = 1
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    marker.scale.x = 0.01

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    marker.pose.orientation.w = 1.0
    marker.points = []

    return marker