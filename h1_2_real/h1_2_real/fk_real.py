#!/usr/bin/env python3

import math
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

from unitree_hg.msg import LowCmd
from unitree_hg.msg import LowState
from unitree_hg.msg import IMUState
from unitree_hg.msg import MotorState

from h1_2_real.fk_functions import *
from h1_2_real.markers import *


# ============================================================
# CONFIGURACIÓN
# ============================================================

INFO_IMU = False
INFO_MOTOR = False
HIGH_FREQ = True

H1_2_NUM_CMD_MOTOR = 27
H1_2_NUM_STATE_MOTOR = 35


# ============================================================
# MODOS
# ============================================================

class PRorAB:
    PR = 0
    AB = 1


# ============================================================
# ÍNDICES DEL H1-2
# ============================================================

class H12JointIndex:
    # legs
    LEFT_HIP_YAW = 0
    LEFT_HIP_PITCH = 1
    LEFT_HIP_ROLL = 2
    LEFT_KNEE = 3

    LEFT_ANKLE_PITCH = 4
    LEFT_ANKLE_B = 4
    LEFT_ANKLE_ROLL = 5
    LEFT_ANKLE_A = 5

    RIGHT_HIP_YAW = 6
    RIGHT_HIP_PITCH = 7
    RIGHT_HIP_ROLL = 8
    RIGHT_KNEE = 9

    RIGHT_ANKLE_PITCH = 10
    RIGHT_ANKLE_B = 10
    RIGHT_ANKLE_ROLL = 11
    RIGHT_ANKLE_A = 11

    # torso
    WAIST_YAW = 12

    # left arm
    LEFT_SHOULDER_PITCH = 13
    LEFT_SHOULDER_ROLL = 14
    LEFT_SHOULDER_YAW = 15
    LEFT_ELBOW = 16
    LEFT_WRIST_ROLL = 17
    LEFT_WRIST_PITCH = 18
    LEFT_WRIST_YAW = 19

    # right arm
    RIGHT_SHOULDER_PITCH = 20
    RIGHT_SHOULDER_ROLL = 21
    RIGHT_SHOULDER_YAW = 22
    RIGHT_ELBOW = 23
    RIGHT_WRIST_ROLL = 24
    RIGHT_WRIST_PITCH = 25
    RIGHT_WRIST_YAW = 26


LEFT_ARM_INDICES = [
    H12JointIndex.LEFT_SHOULDER_PITCH,
    H12JointIndex.LEFT_SHOULDER_ROLL,
    H12JointIndex.LEFT_SHOULDER_YAW,
    H12JointIndex.LEFT_ELBOW,
    H12JointIndex.LEFT_WRIST_ROLL,
    H12JointIndex.LEFT_WRIST_PITCH,
    H12JointIndex.LEFT_WRIST_YAW,
]

RIGHT_ARM_INDICES = [
    H12JointIndex.RIGHT_SHOULDER_PITCH,
    H12JointIndex.RIGHT_SHOULDER_ROLL,
    H12JointIndex.RIGHT_SHOULDER_YAW,
    H12JointIndex.RIGHT_ELBOW,
    H12JointIndex.RIGHT_WRIST_ROLL,
    H12JointIndex.RIGHT_WRIST_PITCH,
    H12JointIndex.RIGHT_WRIST_YAW,
]


# ============================================================
# CRC
# ============================================================

def get_crc(low_cmd_msg: LowCmd) -> None:
    """
    IMPORTANTE:
    Esta función está vacía porque el CRC real depende de Unitree.

    En C++ el ejemplo original usa:
        get_crc(low_command_);

    Si el robot recibe /lowcmd pero no se mueve, esta es casi
    con seguridad la causa.

    Opciones:
    1. Reimplementar el CRC de motor_crc_hg.h en Python.
    2. Crear un binding Python con pybind11.
    3. Usar un nodo C++ solo para publicar LowCmd.
    """
    pass


# ============================================================
# NODO BIMANUAL REAL
# ============================================================

class BimanualRealNode(Node):

    def __init__(self):
        super().__init__("bimanual_real_node")

        # ====================================================
        # TÓPICOS DE UNITREE
        # ====================================================

        self.lowstate_topic = "/lowstate" if HIGH_FREQ else "/lf/lowstate"
        self.lowcmd_topic = "/lowcmd"

        self.lowstate_sub = self.create_subscription(
            LowState,
            self.lowstate_topic,
            self.lowstate_callback,
            qos_profile_sensor_data
        )

        self.lowcmd_pub = self.create_publisher(
            LowCmd,
            self.lowcmd_topic,
            10
        )

        # ====================================================
        # PUBLISHERS PARA RVIZ
        # ====================================================

        self.joint_pub = self.create_publisher(
            JointState,
            "joint_states",
            10
        )

        self.marker_pub = self.create_publisher(
            Marker,
            "ee_marker",
            10
        )

        # ====================================================
        # JOINT NAMES PARA RVIZ
        # ====================================================

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

        self.jstate = JointState()
        self.jstate.name = self.jnames

        # ====================================================
        # ESTADO REAL DEL ROBOT
        # ====================================================

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(H1_2_NUM_STATE_MOTOR)]

        self.received_lowstate = False
        self.lowstate_counter = 0
        self.control_counter = 0

        self.mode_machine = 0
        self.mode = PRorAB.PR

        self.low_cmd = LowCmd()

        # ====================================================
        # CONFIGURACIÓN ARTICULAR DESEADA
        # ====================================================
        # Estas son tus referencias articulares.
        # Inicialmente se ponen en cero, pero luego las actualizaremos
        # con la posición real del robot para evitar saltos bruscos.

        self.q_left_des = np.zeros(7)
        self.q_right_des = np.zeros(7)

        self.q_left_start = np.zeros(7)
        self.q_right_start = np.zeros(7)

        self.q_left_cmd = np.zeros(7)
        self.q_right_cmd = np.zeros(7)

        self.initialized_motion = False

        # Duración de interpolación hacia q_des
        self.motion_time = 0.0
        self.motion_duration = 2.0

        # Ganancias conservadoras para brazos
        self.arm_kp = 25.0
        self.arm_kd = 1.0

        # ====================================================
        # FK PARA CADA BRAZO
        # ====================================================

        self.arms = {
            "left": {
                "fk": fkine_arm_left_unitree
            },
            "right": {
                "fk": fkine_arm_right_unitree
            }
        }

        # ====================================================
        # MARKERS
        # ====================================================

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

        # ====================================================
        # TIMER
        # ====================================================
        # 500 Hz para publicar LowCmd.
        # Si quieres ir más seguro al inicio, puedes probar 100 Hz:
        # self.control_dt = 0.01

        self.control_dt = 0.002
        self.timer = self.create_timer(
            self.control_dt,
            self.update
        )

        self.get_logger().info("Nodo bimanual real iniciado.")
        self.get_logger().info(f"Subscribed to: {self.lowstate_topic}")
        self.get_logger().info(f"Publishing to: {self.lowcmd_topic}")
        self.get_logger().info("Esperando /lowstate...")

    # ========================================================
    # CALLBACK LOWSTATE
    # ========================================================

    def lowstate_callback(self, msg: LowState):
        self.received_lowstate = True
        self.lowstate_counter += 1

        self.mode_machine = int(msg.mode_machine)
        self.imu = msg.imu_state

        n = min(len(msg.motor_state), H1_2_NUM_STATE_MOTOR)

        for i in range(n):
            self.motor[i] = msg.motor_state[i]

        if not self.initialized_motion:
            self.capture_current_arm_position()
            self.initialized_motion = True

        if self.lowstate_counter % 500 == 0:
            self.get_logger().info(
                f"LowState OK | mode_machine: {self.mode_machine} | "
                f"motor_state size: {len(msg.motor_state)}"
            )

    # ========================================================
    # CAPTURAR POSICIÓN REAL INICIAL
    # ========================================================

    def capture_current_arm_position(self):
        q_left_real, q_right_real = self.get_real_arm_positions()

        self.q_left_start = q_left_real.copy()
        self.q_right_start = q_right_real.copy()

        self.q_left_cmd = q_left_real.copy()
        self.q_right_cmd = q_right_real.copy()

        self.q_left_des = q_left_real.copy()
        self.q_right_des = q_right_real.copy()

        self.get_logger().info("Posición inicial real de brazos capturada.")

    # ========================================================
    # LEER POSICIONES REALES DE BRAZOS
    # ========================================================

    def get_real_arm_positions(self):
        q_left = np.array([
            self.motor[H12JointIndex.LEFT_SHOULDER_PITCH].q,
            self.motor[H12JointIndex.LEFT_SHOULDER_ROLL].q,
            self.motor[H12JointIndex.LEFT_SHOULDER_YAW].q,
            self.motor[H12JointIndex.LEFT_ELBOW].q,
            self.motor[H12JointIndex.LEFT_WRIST_ROLL].q,
            self.motor[H12JointIndex.LEFT_WRIST_PITCH].q,
            self.motor[H12JointIndex.LEFT_WRIST_YAW].q,
        ], dtype=float)

        q_right = np.array([
            self.motor[H12JointIndex.RIGHT_SHOULDER_PITCH].q,
            self.motor[H12JointIndex.RIGHT_SHOULDER_ROLL].q,
            self.motor[H12JointIndex.RIGHT_SHOULDER_YAW].q,
            self.motor[H12JointIndex.RIGHT_ELBOW].q,
            self.motor[H12JointIndex.RIGHT_WRIST_ROLL].q,
            self.motor[H12JointIndex.RIGHT_WRIST_PITCH].q,
            self.motor[H12JointIndex.RIGHT_WRIST_YAW].q,
        ], dtype=float)

        return q_left, q_right

    # ========================================================
    # DEFINIR NUEVA REFERENCIA
    # ========================================================

    def set_desired_arm_positions(self, q_left_des, q_right_des):
        """
        Esta función actualiza el objetivo articular.

        q_left_des:  array de 7 valores
        q_right_des: array de 7 valores
        """

        self.q_left_start = self.q_left_cmd.copy()
        self.q_right_start = self.q_right_cmd.copy()

        self.q_left_des = np.array(q_left_des, dtype=float)
        self.q_right_des = np.array(q_right_des, dtype=float)

        self.motion_time = 0.0

    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================

    def update(self):
        self.control_counter += 1

        if not self.received_lowstate:
            if self.control_counter % 500 == 0:
                self.get_logger().warn("Esperando /lowstate. No se publica /lowcmd todavía.")
            return

        # ====================================================
        # EJEMPLO DE MOVIMIENTO
        # ====================================================
        # Aquí puedes reemplazar por tu IK, QP o trayectoria.
        #
        # Por seguridad, este ejemplo solo mueve un poco las muñecas
        # después de 3 segundos de haber iniciado.

        elapsed = self.control_counter * self.control_dt

        if elapsed > 3.0 and elapsed < 3.0 + self.control_dt:
            q_left_real, q_right_real = self.get_real_arm_positions()

            q_left_goal = q_left_real.copy()
            q_right_goal = q_right_real.copy()

            # Movimiento pequeño en wrist_roll
            q_left_goal[4] += 0.10
            q_right_goal[4] += 0.10

            self.set_desired_arm_positions(q_left_goal, q_right_goal)

            self.get_logger().info("Nueva referencia enviada: wrist_roll +0.10 rad")

        # ====================================================
        # INTERPOLACIÓN SUAVE
        # ====================================================

        self.motion_time += self.control_dt

        s = self.clamp(
            self.motion_time / self.motion_duration,
            0.0,
            1.0
        )

        self.q_left_cmd = (1.0 - s) * self.q_left_start + s * self.q_left_des
        self.q_right_cmd = (1.0 - s) * self.q_right_start + s * self.q_right_des

        # ====================================================
        # PUBLICAR JOINT STATES Y MARKERS PARA RVIZ
        # ====================================================

        self.publish_visualization()

        # ====================================================
        # PUBLICAR LOWCMD PARA ROBOT REAL
        # ====================================================

        self.publish_lowcmd()

    # ========================================================
    # VISUALIZACIÓN
    # ========================================================

    def publish_visualization(self):
        q = np.concatenate([
            self.q_left_cmd,
            self.q_right_cmd
        ])

        self.jstate.header.stamp = self.get_clock().now().to_msg()
        self.jstate.position = q.tolist()
        self.joint_pub.publish(self.jstate)

        # FK brazo izquierdo
        T_left, x_left = self.compute_pose(
            fkine_arm_left_unitree,
            self.q_left_cmd
        )

        marker_left = set_marker_pose(
            self.markers["left"],
            x_left,
            self
        )

        self.marker_pub.publish(marker_left)

        # FK brazo derecho
        T_right, x_right = self.compute_pose(
            fkine_arm_right_unitree,
            self.q_right_cmd
        )

        marker_right = set_marker_pose(
            self.markers["right"],
            x_right,
            self
        )

        self.marker_pub.publish(marker_right)

        if self.control_counter % 500 == 0:
            self.get_logger().info(
                f"Left EE xyz: {np.round(x_left[:3], 3)} | "
                f"Right EE xyz: {np.round(x_right[:3], 3)}"
            )

    def compute_pose(self, fk_func, q):
        T = fk_func(q)
        x = TF2xyzquat(T)
        return T, x

    # ========================================================
    # PUBLICAR LOWCMD
    # ========================================================

    def publish_lowcmd(self):
        self.low_cmd.mode_pr = self.mode
        self.low_cmd.mode_machine = self.mode_machine

        # ----------------------------------------------------
        # 1. Mantener todo el robot en su posición actual
        # ----------------------------------------------------
        # Esto evita mandar ceros a piernas y torso.
        # Solo sobreescribiremos brazos luego.

        for i in range(H1_2_NUM_CMD_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].q = float(self.motor[i].q)
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0

            if i < 13:
                self.low_cmd.motor_cmd[i].kp = 40.0
                self.low_cmd.motor_cmd[i].kd = 1.0
            else:
                self.low_cmd.motor_cmd[i].kp = self.arm_kp
                self.low_cmd.motor_cmd[i].kd = self.arm_kd

        # ----------------------------------------------------
        # 2. Sobreescribir brazo izquierdo
        # ----------------------------------------------------

        for local_i, motor_i in enumerate(LEFT_ARM_INDICES):
            self.low_cmd.motor_cmd[motor_i].mode = 1
            self.low_cmd.motor_cmd[motor_i].q = float(self.q_left_cmd[local_i])
            self.low_cmd.motor_cmd[motor_i].dq = 0.0
            self.low_cmd.motor_cmd[motor_i].kp = self.arm_kp
            self.low_cmd.motor_cmd[motor_i].kd = self.arm_kd
            self.low_cmd.motor_cmd[motor_i].tau = 0.0

        # ----------------------------------------------------
        # 3. Sobreescribir brazo derecho
        # ----------------------------------------------------

        for local_i, motor_i in enumerate(RIGHT_ARM_INDICES):
            self.low_cmd.motor_cmd[motor_i].mode = 1
            self.low_cmd.motor_cmd[motor_i].q = float(self.q_right_cmd[local_i])
            self.low_cmd.motor_cmd[motor_i].dq = 0.0
            self.low_cmd.motor_cmd[motor_i].kp = self.arm_kp
            self.low_cmd.motor_cmd[motor_i].kd = self.arm_kd
            self.low_cmd.motor_cmd[motor_i].tau = 0.0

        # ----------------------------------------------------
        # 4. Calcular CRC
        # ----------------------------------------------------
        # Esta línea es necesaria, pero ahora get_crc está vacío.
        # Si no implementas el CRC real, el robot puede ignorar el comando.

        get_crc(self.low_cmd)

        # ----------------------------------------------------
        # 5. Publicar comando
        # ----------------------------------------------------

        self.lowcmd_pub.publish(self.low_cmd)

        if self.control_counter % 500 == 0:
            self.get_logger().info(
                f"Publishing /lowcmd | "
                f"q_left_cmd[4]: {self.q_left_cmd[4]:.3f} | "
                f"q_right_cmd[4]: {self.q_right_cmd[4]:.3f} | "
                f"mode_machine: {self.mode_machine}"
            )

    # ========================================================
    # UTILIDAD
    # ========================================================

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(value, high))


# ============================================================
# MAIN
# ============================================================

def main(args=None):
    rclpy.init(args=args)

    node = BimanualRealNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Nodo detenido por teclado.")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()