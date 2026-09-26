#!/usr/bin/env python3

# ===== ROS =====
import rclpy
from rclpy.node import Node

# ===== LIBS =====
import numpy as np
from scipy import sparse
import osqp

from sensor_msgs.msg import JointState

# ===== CUSTOM =====
from h1_2_algoritms.fk_functions import *
from h1_2_algoritms.ik_functions import *
from h1_2_algoritms.null_control_functions import *
from h1_2_algoritms.QP_functions import *
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

        # ===== Objetivos EE =====
        self.targets = {
            "left":  np.array([0.2, 0.3, 0.3, 1, 0, 0, 0]),
            "right": np.array([0.35, -0.3, 0.3, 1, 0, 0, 0])
        }

        # ===== Objetivos del codo =====
        self.elbow_targets = {
            "left":  np.array([0.2,  0.3, 0.25]),
            "right": np.array([0.2, -0.3, 0.25])
        }

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

        # ===== Markers =====
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

        # ===== Tiempo =====
        self.dt = 1.0 / 20.0
        self.timer = self.create_timer(self.dt, self.update)

        # ===== Ganancias de servo cinemático =====
        self.Kp_c = 2.5
        self.Kp_o = 1.0

        # ===== Pesos del QP =====
        self.W_ee = np.diag([
            10.0, 10.0, 10.0,   # posición x, y, z
            2.0,  2.0,  2.0     # orientación
        ])

        self.w_elbow = 3.0       # peso del objetivo secundario del codo
        self.w_reg = 1e-3        # regularización para suavizar dq

        # ===== Ganancia para velocidad deseada del codo =====
        self.k_elbow_qp = 5.0

        # ===== Límites de velocidad articular =====
        self.dq_max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.dq_min = -self.dq_max

        # ===== Límites articulares aproximados =====
        # IMPORTANTE:
        # Reemplaza estos valores por los límites reales del H1-2.
        self.q_min = np.array([-2.6, -1.5, -2.6, -2.0, -2.6, -1.5, -2.6])
        self.q_max = np.array([ 2.6,  1.5,  2.6,  0.0,  2.6,  1.5,  2.6])

        # ===== Trayectoria =====
        self.t = 0.0
        self.omega = 2.0
        self.radius = 0.05

        self.get_logger().info("Nodo bimanual iniciado con QP + OSQP")

    # ============================================================
    # Función auxiliar: obtener pose
    # ============================================================
    def compute_pose(self, fk_func, q):
        T = fk_func(q)
        x = TF2xyzquat(T)
        return T, x

    # ============================================================
    # Loop principal
    # ============================================================
    def update(self):

        self.t += self.dt

        elbow_errors = {}

        for name, arm in self.arms.items():

            q = arm["q"]
            fk = arm["fk"]
            fk_elbow = arm["fk_elbow"]

            xd = self.targets[name].copy()

            # ===== Trayectoria circular =====
            center = xd[0:3].copy()
            xd[0] = center[0] + self.radius * np.cos(self.omega * self.t)
            xd[2] = center[2] + self.radius * np.sin(self.omega * self.t)

            # ===== FK EE =====
            T = fk(q)
            x = TF2xyzquat(T)

            # ===== Normalización quaternion =====
            xd[3:] = xd[3:] / (np.linalg.norm(xd[3:]) + 1e-8)
            x[3:]  = x[3:]  / (np.linalg.norm(x[3:])  + 1e-8)

            if np.dot(xd[3:], x[3:]) < 0:
                xd[3:] = -xd[3:]

            # ===== Errores EE =====
            ep = xd[0:3] - x[0:3]
            eo = orientation_error(xd[3:], x[3:])

            # ===== Velocidad deseada EE =====
            x_dot = np.hstack((
                self.Kp_c * ep,
                self.Kp_o * eo
            ))

            # ===== Jacobiano EE 6D =====
            J = numerical_jacobian(fk, q, TF2xyzquat)

            # =====================================================
            # Objetivo secundario: codo en eje Y
            # =====================================================
            T_elbow = fk_elbow(q)
            x_elbow = T_elbow[0:3, 3]

            y = x_elbow[1]
            y_d = self.elbow_targets[name][1]

            e_elbow = y_d - y
            elbow_errors[name] = abs(e_elbow)

            J_full_elbow = numerical_jacobian_position(fk_elbow, q)
            J_elbow_y = J_full_elbow[1, :]  # solo eje Y

            # Velocidad deseada del codo en Y
            y_dot_elbow = self.k_elbow_qp * e_elbow

            # =====================================================
            # Resolver QP con OSQP
            # =====================================================
            dq = solve_qp_arm(
                J=J,
                x_dot=x_dot,
                J_elbow=J_elbow_y,
                y_dot_elbow=y_dot_elbow,
                q_current=q,
                q_min=self.q_min,
                q_max=self.q_max,
                dq_min=self.dq_min,
                dq_max=self.dq_max,
                dt=self.dt,
                W_ee=self.W_ee,
                w_elbow=self.w_elbow,
                w_reg=self.w_reg,
                logger=self.get_logger()
            )

            # ===== Seguridad extra =====
            dq = np.clip(dq, self.dq_min, self.dq_max)

            # ===== Integración =====
            q_new = q + dq * self.dt

            # ===== Evitar NaN =====
            if not np.any(np.isnan(q_new)):
                arm["q"][:] = q_new
            else:
                self.get_logger().warn(f"NaN detectado en q_new del brazo {name}")

            print(f"{name} elbow Y error:", abs(e_elbow))

        # ===== Estado global =====
        q_global = np.concatenate([self.q_left, self.q_right])

        # ===== Publicar joints =====
        self.jstate.header.stamp = self.get_clock().now().to_msg()
        self.jstate.position = q_global.tolist()
        self.pub.publish(self.jstate)

        # ===== Cinemática + markers =====
        for name, arm in self.arms.items():

            T, x = self.compute_pose(arm["fk"], arm["q"])

            if name in elbow_errors:
                print(f"{name} elbow error:", elbow_errors[name])

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