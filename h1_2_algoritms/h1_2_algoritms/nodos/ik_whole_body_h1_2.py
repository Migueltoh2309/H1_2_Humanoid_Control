#!/usr/bin/env python3

# ===== ROS =====
import rclpy
from rclpy.node import Node

# ===== LIBS =====
import numpy as np
from sensor_msgs.msg import JointState

# ===== CUSTOM =====
from h1_2_algoritms.movimiento.fk_functions import *
from h1_2_algoritms.movimiento.ik_functions import *
from h1_2_algoritms.markers import *
from h1_2_algoritms.movimiento import joint_limits as JL

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

        # ===== Objetivos IK =====
        # Cuaternión en formato [w, x, y, z] (el mismo que devuelve
        # TF2xyzquat): [1,0,0,0] es la identidad. Ojo: [0,0,0,1] NO es la
        # identidad sino un giro de 180° en Z.
        self.targets = {
            "left":  np.array([0.4, 0.24, 0.5, 1, 0, 0, 0]),
            "right": np.array([0.4, -0.24, 0.5, 1, 0, 0, 0])
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

        # ===== Límites articulares (del MJCF) =====
        for name, arm in self.arms.items():
            arm["q_min"], arm["q_max"] = JL.get_limits(name)

        # ===== DLS o Pseudoinversa ===
        self.use_dls = True

        # Respetar los rangos mecánicos. Con False se recupera el
        # comportamiento anterior (sin límites), útil para comparar.
        self.respect_limits = True

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

    def compute_pose(self, fk_func, q):
        T = fk_func(q)
        x = TF2xyzquat(T)
        return T, x 
        
        # ===== LOOP PRINCIPAL =====
    def update(self):

        # =========================================
        # 1. IK (por brazo)
        # =========================================
        for name, arm in self.arms.items():

            q = arm["q"]
            fk = arm["fk"]
            xd = self.targets[name]

            # IMPORTANTE: usar q como q0
            if self.respect_limits:
                q_new, _ = ik_dls_step_limited(
                    fk, TF2xyzquat, q, xd, arm["q_min"], arm["q_max"],
                    lamb=0.05)
            elif self.use_dls:
                q_new = ik_dls_step(fk, TF2xyzquat, q, xd, lamb=0.05)
            else:
                q_new = ik_pseudo_step(fk, TF2xyzquat, q, xd, alpha=0.3)

            # evitar NaNs
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