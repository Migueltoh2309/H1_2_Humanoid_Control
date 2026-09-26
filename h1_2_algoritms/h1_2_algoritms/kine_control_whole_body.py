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
from h1_2_algoritms.kine_control_functions import *
from h1_2_algoritms.markers import *

class BimanualNode(Node):

    def __init__(self):
        super().__init__('bimanual_node')

        # ===== Publishers =====
        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.marker_pub = self.create_publisher(Marker, 'ee_marker', 10)

        # ===== Joint names =====
        self.  jnames = [
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
        self.q_left = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.q_right = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ===== Objetivos control cinemático =====
        self.targets = {
            "left":  np.array([0.2, 0.3, 0.5, 1, 0, 0, 0]),
            "right": np.array([0.2, -0.3, 0.5, 1, 0, 0, 0])
        }

        # ===== Estructura de brazos =====
        self.arms = {
            "left": {
                "q": self.q_left,
                "fk": fkine_arm_left_unitree
            },
            "right": {
                "q": self.q_right,
                "fk": fkine_arm_right_unitree
            }
        }

        # ===== DLS o Pseudoinversa ===
        self.use_dls = True

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

        # ===== Timer (20 Hz) =====
        self.timer = self.create_timer(1.0 / 20.0, self.update)

        self.get_logger().info("Nodo bimanual iniciado")

        # ===== Control cinemático =====
        self.dt = 1.0 / 20.0   # mismo que el timer
        self.Kp_c = 2.0           # ganancia (puedes tunear)
        self.Kp_o = 2.0           # ganancia (puedes tunear)

        # ===== Trayectoria =====
        self.t = 0.0
        self.omega = 2.0   # velocidad angular
        self.radius = 0.05

    def compute_pose(self, fk_func, q):
        T = fk_func(q)
        x = TF2xyzquat(T)
        return T, x 
        
        # ===== LOOP PRINCIPAL =====
    def update(self):

        self.t += self.dt

        # =========================================
        # 1. CONTROL CINEMÁTICO (por brazo)
        # =========================================
        for name, arm in self.arms.items():

            q = arm["q"]
            fk = arm["fk"]
            xd = self.targets[name].copy()

            # centro del círculo
            center = xd[0:3]

            # círculo en plano XY
            xd[0] = center[0] + self.radius * np.cos(self.omega * self.t)
            xd[1] = center[1] + self.radius * np.sin(self.omega * self.t)
            # xd[2] se queda igual

            # ===== FK =====
            T = fk(q)
            x = TF2xyzquat(T)

            # ===== NORMALIZACIÓN =====
            xd[3:] = xd[3:] / (np.linalg.norm(xd[3:]) + 1e-8)
            x[3:]  = x[3:]  / (np.linalg.norm(x[3:])  + 1e-8)

            if np.dot(xd[3:], x[3:]) < 0:
                xd[3:] = -xd[3:]

            # ===== ERRORES =====
            ep = xd[0:3] - x[0:3]
            eo = orientation_error(xd[3:], x[3:])

            # ===== VELOCIDAD =====
            x_dot = np.hstack((self.Kp_c * ep, self.Kp_o * eo))

            # ===== JACOBIANO =====
            J = numerical_jacobian(fk, q, TF2xyzquat)

            # ===== DLS =====
            JT = J.T
            JJ = J @ JT + 1e-6 * np.eye(6)
            dq = JT @ np.linalg.inv(JJ + (0.05**2)*np.eye(6)) @ x_dot

            # ===== LIMITAR =====
            # dq = np.clip(dq, -1.0, 1.0)

            # ===== INTEGRAR =====
            q_new = q + dq * self.dt

            # ===== EVITAR NaN =====
            if not np.any(np.isnan(q_new)):
                arm["q"][:] = q_new

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
            print(f"{name}: ", np.round(T, 3))

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