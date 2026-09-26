#!/usr/bin/env python3
"""
ik_whole_body_bridge_demo.py — Copia de ik_whole_body_h1_2.py adaptada para
correr sobre h1_2_mujoco_sim_bridge (simulación con física real en MuJoCo)
en vez de publicar un JointState "de mentira" directamente.

Diferencias respecto al original:
  * Publica el resultado de la IK en /joint_cmd (no en /joint_states: ese
    tópico lo publica el bridge a partir de la física real, no hay que
    pisarlo). El bridge aplica un PD y MuJoCo decide qué pasa de verdad
    (incluidas colisiones).
  * Se suscribe a /contacts (el bridge lo publica con la identidad de los
    cuerpos en `Marker.text`, ver h1_2_mujoco_sim_bridge/msg_builders.py) y
    vigila los contactos que involucran a los brazos. Los contactos ya
    presentes al arrancar (pies en el suelo, mesa, etc.) se toman como línea
    base "normal"; si aparece un contacto NUEVO en un cuerpo de brazo — con
    la mesa, con el propio cuerpo o entre los dos brazos — se considera un
    choque: se cancela el timer de movimiento (no se manda más /joint_cmd,
    el PD del bridge sostiene la última posición) y se publica
    /ik_whole_body_bridge_demo/collision (std_msgs/Bool, True) una vez.
  * Si NINGÚN brazo choca y ambos llegan a su objetivo dentro de tolerancia,
    se loggea éxito y también se detiene el timer (ya no hay nada que
    mandar).

Requiere el bridge corriendo (ver README raíz del workspace):
    ros2 launch h1_2_mujoco_sim_bridge sim_bridge.launch.py

Uso:
    ros2 run h1_2_algoritms ik_whole_body_bridge_demo
    # objetivos que fuerzan a los dos brazos a converger al mismo punto,
    # para ver el camino de cancelación por choque:
    ros2 run h1_2_algoritms ik_whole_body_bridge_demo --ros-args -p collision_test:=true
"""

# ===== ROS =====
import rclpy
from rclpy.node import Node

# ===== LIBS =====
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray

# ===== CUSTOM =====
from h1_2_algoritms.fk_functions import *
from h1_2_algoritms.ik_functions import *
from h1_2_algoritms.markers import *
from h1_2_algoritms import joint_limits as JL

# Nombres de BODY (no de joint) de cada brazo en el MJCF
# (h1_2_description/mjcf/h1_2_scene_qp_reachable.xml, variante sin manos: el
# último eslabón es *_wrist_yaw_link). Se usan para filtrar, de todos los
# contactos que reporta el bridge, cuáles involucran a un brazo.
LEFT_ARM_BODIES = {
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
}
RIGHT_ARM_BODIES = {
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
}
ARM_BODIES = LEFT_ARM_BODIES | RIGHT_ARM_BODIES


class BimanualBridgeDemoNode(Node):

    def __init__(self):
        super().__init__('ik_whole_body_bridge_demo')

        # ===== Parámetros =====
        self.declare_parameter('collision_test', False)
        self.declare_parameter('collision_force_threshold', 0.5)  # N
        self.declare_parameter('tol_pos', 0.01)   # m
        self.declare_parameter('tol_ori', 0.05)   # rad (parte vectorial del error de orientación)
        collision_test = bool(self.get_parameter('collision_test').value)
        self.collision_force_threshold = float(self.get_parameter('collision_force_threshold').value)
        self.tol_pos = float(self.get_parameter('tol_pos').value)
        self.tol_ori = float(self.get_parameter('tol_ori').value)

        # ===== Publishers =====
        # IMPORTANTE: /joint_cmd (comando al bridge), NO /joint_states (eso
        # lo publica el bridge desde la física real de MuJoCo).
        self.pub_cmd = self.create_publisher(JointState, '/joint_cmd', 10)
        self.marker_pub = self.create_publisher(Marker, 'ee_marker', 10)
        self.collision_pub = self.create_publisher(
            Bool, '/ik_whole_body_bridge_demo/collision', 10)

        # ===== Joint names (mismo orden que el JointState de comando) =====
        self.jnames = [
            # LEFT ARM
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            # RIGHT ARM
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]

        # ===== Estado articular interno (referencia cinemática, no la real:
        # la real la decide MuJoCo con el PD del bridge) =====
        self.q_left = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.q_right = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ===== Objetivos IK =====
        if collision_test:
            # Mismo punto para los dos brazos: fuerza que las manos/antebrazos
            # converjan una sobre la otra para poder ver la cancelación por
            # choque (autocolisión brazo-brazo, un par que NO está en la
            # línea base de reposo).
            self.targets = {
                "left":  np.array([0.35, 0.0, 0.55, 1, 0, 0, 0]),
                "right": np.array([0.35, 0.0, 0.55, 1, 0, 0, 0]),
            }
            self.get_logger().warn(
                "collision_test=true: objetivos elegidos a propósito para que "
                "los brazos choquen entre sí y se pueda ver la cancelación.")
        else:
            self.targets = {
                "left":  np.array([0.4, 0.24, 0.5, 1, 0, 0, 0]),
                "right": np.array([0.4, -0.24, 0.5, 1, 0, 0, 0]),
            }

        # ===== Estructura de brazos =====
        self.arms = {
            "left": {"q": self.q_left, "fk": fkine_arm_left_unitree, "converged": False},
            "right": {"q": self.q_right, "fk": fkine_arm_right_unitree, "converged": False},
        }
        for name, arm in self.arms.items():
            arm["q_min"], arm["q_max"] = JL.get_limits(name)

        self.respect_limits = True

        # ===== JointState de comando =====
        self.cmd_msg = JointState()
        self.cmd_msg.name = self.jnames

        # ===== Markers (uno por brazo) =====
        self.markers = {
            "left": create_sphere_marker(frame="torso_link", ns="end_effectors",
                                          marker_id=0, scale=0.05, color=(1.0, 0.0, 0.0, 1.0)),
            "right": create_sphere_marker(frame="torso_link", ns="end_effectors",
                                           marker_id=1, scale=0.05, color=(0.0, 0.0, 1.0, 1.0)),
        }

        # ===== Detección de choques =====
        self._baseline_pairs = None      # set(frozenset({body1, body2})) en reposo
        self._collided = False
        self._done = False
        self.timer = None                # se crea recién al fijar la línea base
        self.create_subscription(MarkerArray, '/contacts', self._on_contacts, 10)

        self.get_logger().info(
            "Nodo bimanual (bridge) iniciado: esperando /contacts para fijar la "
            "línea base de contactos 'normales' antes de empezar a moverse...")

    # ==================================================================== IK
    def compute_pose(self, fk_func, q):
        T = fk_func(q)
        x = TF2xyzquat(T)
        return T, x

    def update(self):
        if self._collided or self._done:
            return

        all_converged = True

        for name, arm in self.arms.items():
            q = arm["q"]
            fk = arm["fk"]
            xd = self.targets[name]

            if self.respect_limits:
                q_new, e = ik_dls_step_limited(
                    fk, TF2xyzquat, q, xd, arm["q_min"], arm["q_max"], lamb=0.05)
            else:
                q_new = ik_dls_step(fk, TF2xyzquat, q, xd, lamb=0.05)
                e = pose_error(xd, TF2xyzquat(fk(q)))

            if not np.any(np.isnan(q_new)):
                arm["q"][:] = q_new

            converged = (np.linalg.norm(e[:3]) < self.tol_pos
                         and np.linalg.norm(e[3:]) < self.tol_ori)
            arm["converged"] = converged
            all_converged = all_converged and converged

        # ===== Publicar /joint_cmd (comando de posición al bridge) =====
        q = np.concatenate([self.q_left, self.q_right])
        self.cmd_msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_msg.position = q.tolist()
        self.pub_cmd.publish(self.cmd_msg)

        # ===== Markers de los end-effectors =====
        for name, arm in self.arms.items():
            _, x = self.compute_pose(arm["fk"], arm["q"])
            marker = set_marker_pose(self.markers[name], x, self)
            if marker is not None:
                self.marker_pub.publish(marker)

        if all_converged and not self._done:
            self._done = True
            self.get_logger().info(
                "Movimiento bimanual completado SIN colisiones: ambos brazos "
                "llegaron a su objetivo. Manteniendo la última posición.")
            self.timer.cancel()

    # =============================================================== CONTACTOS
    def _parse_contacts(self, msg: MarkerArray):
        """{frozenset({body1, body2}): normal_force}. Ignora markers sin el
        `text` esperado (por compatibilidad con otras fuentes de /contacts)."""
        pairs = {}
        for m in msg.markers:
            parts = m.text.split("|")
            if len(parts) != 3:
                continue
            body1, body2, force_str = parts
            try:
                force = float(force_str)
            except ValueError:
                continue
            pairs[frozenset((body1, body2))] = force
        return pairs

    def _on_contacts(self, msg: MarkerArray):
        pairs = self._parse_contacts(msg)

        if self._baseline_pairs is None:
            self._baseline_pairs = set(pairs.keys())
            names = ", ".join(sorted(" <-> ".join(sorted(p)) for p in self._baseline_pairs)) \
                or "(ninguno)"
            self.get_logger().info(
                f"Línea base de contactos fijada ({len(self._baseline_pairs)} "
                f"par(es) ya presentes en reposo, se ignoran): {names}")
            self.timer = self.create_timer(1.0 / 20.0, self.update)
            self.get_logger().info("Movimiento bimanual iniciado.")
            return

        if self._collided or self._done:
            return

        for pair, force in pairs.items():
            if pair in self._baseline_pairs:
                continue
            if force < self.collision_force_threshold:
                continue
            if pair & ARM_BODIES:
                self._trigger_collision(pair, force)
                return

    def _trigger_collision(self, pair, force):
        self._collided = True
        a, b = tuple(pair) if len(pair) == 2 else (next(iter(pair)), next(iter(pair)))
        self.get_logger().error(
            f"¡CHOQUE detectado! {a} <-> {b} (fuerza normal {force:.2f} N). "
            f"Cancelando el movimiento: no se manda más /joint_cmd, el bridge "
            f"sostiene la última posición.")
        msg = Bool()
        msg.data = True
        self.collision_pub.publish(msg)
        if self.timer is not None:
            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)

    node = BimanualBridgeDemoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
