#!/usr/bin/env python3

# ===== ROS =====
import rclpy
from rclpy.node import Node

# ===== LIBS =====
import numpy as np
from sensor_msgs.msg import JointState

# ===== CUSTOM =====
from h1_2_algoritms.fk_functions import *
from h1_2_algoritms.ik_functions import *
from h1_2_algoritms.null_control_functions import *
from h1_2_algoritms.markers import *


class BimanualNode(Node):

    def __init__(self):
        super().__init__('bimanual_node')

        # ===== Publishers =====
        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.marker_pub = self.create_publisher(Marker, 'ee_marker', 10)

        # ===== Joint names =====
        self.jnames = [
            # LEFT ARM
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",

            # RIGHT ARM
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"
        ]

        # ===== Joint configuration =====
        self.q_left = np.zeros(7)
        self.q_right = np.zeros(7)

        # ===== Objetivos EE (pose) =====
        self.targets = {
            "left":  np.array([0.2, 0.3, 0.3, 1, 0, 0, 0]),
            "right": np.array([0.35, -0.3, 0.3, 1, 0, 0, 0])
        }

        # ===== Objetivos del codo =====
        self.elbow_targets = {
            "left":  np.array([0.2,  0.3, 0.25]),
            "right": np.array([0.2, -0.3, 0.25])
        }

        self.k_elbow = 10  # ganancia null-space

        # ===== Estructura =====
        self.arms = {
            "left": {
                "q": self.q_left,
                "fk": fkine_arm_left_unitree,
                "fk_elbow": fk_elbow_left_unitree
            },
            "right": {
                "q": self.q_right,
                "fk": fkine_arm_right_unitree,
                "fk_elbow": fk_elbow_right_unitree
            }
        }

        # ===== JointState =====
        self.jstate = JointState()
        self.jstate.name = self.jnames

        # ===== Markers (UNO POR BRAZO) =====
        self.markers = {
            "left": create_sphere_marker(
                frame="torso_link",
                ns="end_effectors",
                marker_id=0,
                scale=0.05,
                color=(1.0, 0.0, 0.0, 1.0)
            ),
            "right": create_sphere_marker(
                frame="torso_link",
                ns="end_effectors",
                marker_id=1,
                scale=0.05,
                color=(0.0, 0.0, 1.0, 1.0)
            )
        }

        # ===== Control =====
        self.dt = 1.0 / 20.0
        self.timer = self.create_timer(self.dt, self.update)

        self.Kp_c = 2.5
        self.Kp_o = 1.0

        # ===== Trayectoria =====
        self.t = 0.0
        self.omega = 2.0
        self.radius = 0.05

        self.get_logger().info("Nodo bimanual iniciado")

    def compute_pose(self, fk_func, q):
        T = fk_func(q)
        x = TF2xyzquat(T)
        return T, x

    def update(self):

        self.t += self.dt

        for name, arm in self.arms.items():

            q = arm["q"]
            fk = arm["fk"]
            fk_elbow = arm["fk_elbow"]

            xd = self.targets[name].copy()

            # ===== Trayectoria circular =====
            center = xd[0:3]
            xd[0] = center[0] + self.radius * np.cos(self.omega * self.t)
            xd[2] = center[2] + self.radius * np.sin(self.omega * self.t)

            # ===== FK =====
            T = fk(q)
            x = TF2xyzquat(T)

            # ===== NORMALIZACIÓN quaternion =====
            xd[3:] = xd[3:] / (np.linalg.norm(xd[3:]) + 1e-8)
            x[3:]  = x[3:]  / (np.linalg.norm(x[3:])  + 1e-8)

            if np.dot(xd[3:], x[3:]) < 0:
                xd[3:] = -xd[3:]

            # ===== ERRORES =====
            ep = xd[0:3] - x[0:3]
            eo = orientation_error(xd[3:], x[3:])

            # ===== VELOCIDAD =====
            x_dot = np.hstack((self.Kp_c * ep, self.Kp_o * eo))

            # ===== JACOBIANO 6D =====
            J = numerical_jacobian(fk, q, TF2xyzquat)

            # ===== PSEUDOINVERSA DLS =====
            JT = J.T
            lambda2 = (0.03**2)
            J_pinv = JT @ np.linalg.inv(J @ JT + lambda2 * np.eye(6))

            # ===== NULL SPACE =====
            I = np.eye(len(q))
            N = I - J_pinv @ J

            # =====================================
            # 🔹 CODO SOLO EN Y
            # =====================================
            T_elbow = fk_elbow(q)
            x_elbow = T_elbow[0:3, 3]

            y = x_elbow[1]
            y_d = self.elbow_targets[name][1]

            e_elbow = y_d - y

            J_full = numerical_jacobian_position(fk_elbow, q)
            J_elbow = J_full[1, :]  # solo eje Y

            dq0 = self.k_elbow * J_elbow.T * e_elbow

            # ===== CONTROL FINAL =====
            dq = J_pinv @ x_dot + N @ dq0

            # ===== LIMITAR =====
            dq = np.clip(dq, -2.0, 2.0)

            # ===== INTEGRAR =====
            q_new = q + dq * self.dt

            # ===== EVITAR NAN
            #  =====
            if not np.any(np.isnan(q_new)):
                arm["q"][:] = q_new

            print(f"{name} elbow Y error:", abs(e_elbow))

        # ===== 2. Estado global =====
        q = np.concatenate([self.q_left, self.q_right])

        # ===== 3. Publicar joints =====
        self.jstate.header.stamp = self.get_clock().now().to_msg()
        self.jstate.position = q.tolist()
        self.pub.publish(self.jstate)

        # ===== 4. Cinemática + markers =====
        for i, (name, arm) in enumerate(self.arms.items()):

            T, x = self.compute_pose(arm["fk"], arm["q"])

            # Debug
            # print(f"{name}: ", np.round(T, 3))
            print(f"{name} elbow error:", np.linalg.norm(e_elbow))

            marker = set_marker_pose(self.markers[name], x, self)

            if marker is not None:
                self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)

    node = BimanualNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()