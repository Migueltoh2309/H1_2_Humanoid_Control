#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import threading
import sys
import termios
import tty

from sensor_msgs.msg import JointState

from h1_2_algoritms.movimiento.fk_functions import *
from h1_2_algoritms.movimiento.ik_functions import *
from h1_2_algoritms.movimiento.null_control_functions import *
from h1_2_algoritms.movimiento.QP_functions import *
from h1_2_algoritms.markers import *


def get_key():
    """Leer una tecla sin bloquear ROS"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return ch


class BimanualTeleop(Node):

    def __init__(self):
        super().__init__('bimanual_teleop')

        # ===== CONFIG =====
        self.step = 0.02
        self.dt = 1.0 / 20.0

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

        # ===== ESTADO =====
        self.q_left = np.zeros(7)
        self.q_right = np.zeros(7)

        # ===== TARGETS = POSICIÓN ACTUAL =====
        T_left = fkine_arm_left_unitree(self.q_left)
        T_right = fkine_arm_right_unitree(self.q_right)

        self.targets = {
            "left": TF2xyzquat(T_left),
            "right": TF2xyzquat(T_right)
        }

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

        # ===== Objetivo secundario QP: mantener el codo donde inicia =====
        self.elbow_targets = {
            name: arm["fk_elbow"](arm["q"])[0:3, 3].copy()
            for name, arm in self.arms.items()
        }

        # ===== MARKERS =====
        self.markers = {
            "left": create_sphere_marker("torso_link","ee",0,0.05,(1,0,0,1)),
            "right": create_sphere_marker("torso_link","ee",1,0.05,(0,0,1,1))
        }

        # ===== MSG =====
        self.jstate = JointState()
        self.jstate.name = self.jnames

        # ===== Ganancias de servo cinemático =====
        self.Kp_c = 2.5
        self.Kp_o = 1.0

        # ===== Pesos del QP =====
        self.W_ee = np.diag([
            10.0, 10.0, 10.0,   # posición x, y, z
            2.0,  2.0,  2.0     # orientación
        ])
        self.w_elbow = 3.0
        self.w_reg = 1e-3
        self.k_elbow_qp = 5.0

        # ===== Límites =====
        self.dq_max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.dq_min = -self.dq_max

        # Reemplazar por límites reales del H1-2 si ya los tienes medidos.
        self.q_min = np.array([-2.6, -1.5, -2.6, -2.0, -2.6, -1.5, -2.6])
        self.q_max = np.array([ 2.6,  1.5,  2.6,  0.0,  2.6,  1.5,  2.6])

        # ===== TIMER =====
        self.timer = self.create_timer(self.dt, self.update)

        # ===== THREAD TECLADO =====
        self.thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.thread.start()

        self.get_logger().info("Teleop QP iniciado")

    # =========================================
    # 🔹 TECLADO
    # =========================================
    def keyboard_loop(self):
        while True:
            key = get_key()

            # LEFT ARM
            if key == 'w': self.targets["left"][0] += self.step
            elif key == 's': self.targets["left"][0] -= self.step
            elif key == 'a': self.targets["left"][1] += self.step
            elif key == 'd': self.targets["left"][1] -= self.step
            elif key == 'q': self.targets["left"][2] += self.step
            elif key == 'e': self.targets["left"][2] -= self.step

            # RIGHT ARM
            elif key == 'i': self.targets["right"][0] += self.step
            elif key == 'k': self.targets["right"][0] -= self.step
            elif key == 'j': self.targets["right"][1] += self.step
            elif key == 'l': self.targets["right"][1] -= self.step
            elif key == 'u': self.targets["right"][2] += self.step
            elif key == 'o': self.targets["right"][2] -= self.step

            elif key == 'x':
                self.get_logger().info("Saliendo...")
                rclpy.shutdown()
                break

    # =========================================
    # 🔹 QP
    # =========================================
    def compute_qp_step(self, name, arm, xd):
        q = arm["q"]
        fk = arm["fk"]
        fk_elbow = arm["fk_elbow"]

        T = fk(q)
        x = TF2xyzquat(T)

        xd = xd.copy()
        xd[3:] = xd[3:] / (np.linalg.norm(xd[3:]) + 1e-8)
        x[3:] = x[3:] / (np.linalg.norm(x[3:]) + 1e-8)

        if np.dot(xd[3:], x[3:]) < 0:
            xd[3:] = -xd[3:]

        ep = xd[0:3] - x[0:3]
        eo = orientation_error(xd[3:], x[3:])

        x_dot = np.hstack((
            self.Kp_c * ep,
            self.Kp_o * eo
        ))

        J = numerical_jacobian(fk, q, TF2xyzquat)

        T_elbow = fk_elbow(q)
        x_elbow = T_elbow[0:3, 3]
        y_dot_elbow = self.k_elbow_qp * (self.elbow_targets[name][1] - x_elbow[1])

        J_full_elbow = numerical_jacobian_position(fk_elbow, q)
        J_elbow_y = J_full_elbow[1, :]

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

        dq = np.clip(dq, self.dq_min, self.dq_max)
        return q + dq * self.dt

    # =========================================
    # 🔹 LOOP
    # =========================================
    def update(self):

        # QP por brazo
        for name, arm in self.arms.items():

            xd = self.targets[name]

            q_new = self.compute_qp_step(name, arm, xd)

            if not np.any(np.isnan(q_new)):
                arm["q"][:] = q_new
            else:
                self.get_logger().warn(f"NaN detectado en q_new del brazo {name}")

        # publicar joints
        q_all = np.concatenate([self.q_left, self.q_right])

        self.jstate.header.stamp = self.get_clock().now().to_msg()
        self.jstate.position = q_all.tolist()
        self.pub.publish(self.jstate)

        # markers
        for name, arm in self.arms.items():
            T = arm["fk"](arm["q"])
            x = TF2xyzquat(T)

            if not np.any(np.isnan(x)):
                marker = set_marker_pose(self.markers[name], x, self)
                self.marker_pub.publish(marker)


def main():
    rclpy.init()
    node = BimanualTeleop()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
