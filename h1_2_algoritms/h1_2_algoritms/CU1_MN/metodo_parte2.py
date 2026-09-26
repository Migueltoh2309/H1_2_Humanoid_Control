#!/usr/bin/env python3

# ===== ROS =====
import rclpy
from rclpy.node import Node

# ===== LIBS =====
import numpy as np
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

# ===== CUSTOM =====
from h1_2_algoritms.fk_functions import *
from h1_2_algoritms.markers import *
from h1_2_algoritms.kine_control_functions import *

# ===== DATA =======
from h1_2_algoritms.data_logger import DataLogger

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

        # ===== Configuración =====
        self.q_left = np.zeros(7)
        self.q_right = np.zeros(7)

        # ===== Targets =====
        self.targets = {
            "left":  np.array([0.3,  0.3, 0.4, 1, 0, 0, 0]),
            "right": np.array([0.3, -0.3, 0.4, 1, 0, 0, 0])
        }

        # ===== Brazos =====
        self.arms = {
            "left":  {"q": self.q_left,  "fk": fkine_arm_left_unitree},
            "right": {"q": self.q_right, "fk": fkine_arm_right_unitree}
        }

        # ===== Método IK =====
        self.use_dls = False 

        # ===== JointState =====
        self.jstate = JointState()
        self.jstate.name = self.jnames

        # ===== Markers =====
        self.markers = {
            "left": create_sphere_marker("torso_link", "ee", 0, 0.05, (1,0,0,1)),
            "right": create_sphere_marker("torso_link", "ee", 1, 0.05, (0,0,1,1))
        }

        # ===== Tiempo =====
        self.dt = 1.0 / 20.0
        self.t = 0.0

        # ===== Trayectoria =====
        self.radius = 0.05
        self.omega = 2.0

        # ===== Timer =====
        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info("Nodo bimanual iniciado")

        # ==== DATA ====
        self.logger = DataLogger("Eliminacion_gaussiana_test.csv")

        # Para identificar método
        self.method_name = "DLS" if self.use_dls else "PINV"
        self.max_time = 5.0  # segundos

    # =========================
    # FK + Pose
    # =========================
    def compute_pose(self, fk, q):
        T = fk(q)
        x = TF2xyzquat(T)
        return T, x

    def update(self):

        self.t += self.dt

        # ===== DETENER A LOS 10 SEGUNDOS =====
        if self.t >= self.max_time:
            self.get_logger().info("Tiempo alcanzado, guardando datos...")

            self.logger.save()

            # detener timer y nodo
            self.timer.cancel()
            rclpy.shutdown()
            return

        # ===== Control por brazo =====
        for name, arm in self.arms.items():

            q = arm["q"]
            fk = arm["fk"]

            xd = self.targets[name].copy()

            # Trayectoria
            xd = circular_trajectory(xd, self.t, self.radius, self.omega)

            # IK step
            q_new = ik_dls_gauss(fk, q, xd, lamb=0.05)

            if not np.any(np.isnan(q_new)):
                arm["q"][:] = q_new

            # =====================================
            # 🔥 LOG SOLO PARA BRAZO IZQUIERDO
            # =====================================
            if name == "left":

                x = TF2xyzquat(fk(arm["q"]))

                # Normalización (importante para error)
                xd_n, x_n = normalize_quaternions(xd.copy(), x.copy())

                e = pose_error(xd_n, x_n)

                self.logger.log(
                    self.t,
                    name,
                    self.method_name,
                    xd_n,
                    x_n,
                    e
                )

        # ===== Publicar joints =====
        q_all = np.concatenate([self.q_left, self.q_right])

        self.jstate.header.stamp = self.get_clock().now().to_msg()
        self.jstate.position = q_all.tolist()
        self.pub.publish(self.jstate)

        # ===== Markers =====
        for name, arm in self.arms.items():

            T, x = self.compute_pose(arm["fk"], arm["q"])

            print(f"{name}: ", np.round(T, 3))

            marker = set_marker_pose(self.markers[name], x, self)

            if marker:
                self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = BimanualNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 🔥 Manejo seguro
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()