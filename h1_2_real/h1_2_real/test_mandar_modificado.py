#!/usr/bin/env python3

"""
ROS2 low-level command sender for Unitree H1-2 (HG version).
CRC corregido con el layout real del mensaje:

LowCmd:
  uint8 mode_pr
  uint8 mode_machine
    MotorCmd[35] motor_cmd
    uint32[4] reserve
    uint32 crc

    MotorCmd:
    uint8 mode
    float32 q
    float32 dq
    float32 tau      <- ANTES del kp/kd
    float32 kp
    float32 kd
  uint32 reserve
"""

import math
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from unitree_hg.msg import LowCmd
from unitree_hg.msg import LowState
from unitree_hg.msg import IMUState
from unitree_hg.msg import MotorState


# =========================
# CONFIGURATION
# =========================

INFO_IMU   = False
INFO_MOTOR = False
HIGH_FREQ  = True

H1_2_NUM_CMD_MOTOR   = 27
H1_2_NUM_STATE_MOTOR = 35


class PRorAB:
    PR = 0
    AB = 1


class H12JointIndex:
    LEFT_HIP_YAW     = 0
    LEFT_HIP_PITCH   = 1
    LEFT_HIP_ROLL    = 2
    LEFT_KNEE        = 3
    LEFT_ANKLE_PITCH = 4
    LEFT_ANKLE_B     = 4
    LEFT_ANKLE_ROLL  = 5
    LEFT_ANKLE_A     = 5
    RIGHT_HIP_YAW    = 6
    RIGHT_HIP_PITCH  = 7
    RIGHT_HIP_ROLL   = 8
    RIGHT_KNEE       = 9
    RIGHT_ANKLE_PITCH = 10
    RIGHT_ANKLE_B     = 10
    RIGHT_ANKLE_ROLL  = 11
    RIGHT_ANKLE_A     = 11
    WAIST_YAW            = 12
    LEFT_SHOULDER_PITCH  = 13
    LEFT_SHOULDER_ROLL   = 14
    LEFT_SHOULDER_YAW    = 15
    LEFT_ELBOW           = 16
    LEFT_WRIST_ROLL      = 17
    LEFT_WRIST_PITCH     = 18
    LEFT_WRIST_YAW       = 19
    RIGHT_SHOULDER_PITCH = 20
    RIGHT_SHOULDER_ROLL  = 21
    RIGHT_SHOULDER_YAW   = 22
    RIGHT_ELBOW          = 23
    RIGHT_WRIST_ROLL     = 24
    RIGHT_WRIST_PITCH    = 25
    RIGHT_WRIST_YAW      = 26


# =========================
# CRC — layout corregido con los campos reales
# =========================

def _crc32_core(uint32_words: list) -> int:
    """Algoritmo CRC32 propio de Unitree (distinto al CRC32 estandar de Python/zlib)."""
    crc_register = 0xFFFFFFFF
    poly = 0x04C11DB7

    for word in uint32_words:
        xbit = 1 << 31
        data = word & 0xFFFFFFFF
        for _ in range(32):
            if crc_register & 0x80000000:
                crc_register = ((crc_register << 1) ^ poly) & 0xFFFFFFFF
            else:
                crc_register = (crc_register << 1) & 0xFFFFFFFF
            if data & xbit:
                crc_register ^= poly
            xbit >>= 1

    return crc_register


def _serialize_low_cmd(msg: LowCmd) -> bytes:
    """
    Serializa LowCmd con el layout exacto del struct C++.

    MotorCmd (28 bytes cada uno):
      [0]     uint8   mode
      [1-3]   padding (3 bytes)
      [4]     float32 q
      [8]     float32 dq
      [12]    float32 tau     <- tau va ANTES que kp/kd
      [16]    float32 kp
      [20]    float32 kd
      [24]    uint32  reserve
    Total: 28 bytes

    LowCmd (1004 bytes total):
      [0]       uint8  mode_pr
      [1]       uint8  mode_machine
      [2-3]     padding
      [4-983]   MotorCmd[35]      (35 x 28 = 980 bytes)
      [984-999] uint32[4] reserve (4 x 4 = 16 bytes)
      [1000-1003] uint32 crc      <- NO incluido en el calculo

    CRC cubre los primeros 1000 bytes = 250 palabras uint32.
    """
    buf = bytearray()

    # Header (4 bytes con padding)
    buf += struct.pack('<B', int(msg.mode_pr))
    buf += struct.pack('<B', int(msg.mode_machine))
    buf += b'\x00\x00'  # padding

    # MotorCmd[35] — 35 x 28 = 980 bytes
    for i in range(35):
        if i < len(msg.motor_cmd):
            cmd  = msg.motor_cmd[i]
            mode = int(cmd.mode)
            q    = float(cmd.q)
            dq   = float(cmd.dq)
            tau  = float(cmd.tau)   # tau ANTES de kp/kd
            kp   = float(cmd.kp)
            kd   = float(cmd.kd)
        else:
            mode, q, dq, tau, kp, kd = 0, 0.0, 0.0, 0.0, 0.0, 0.0

        buf += struct.pack('<B', mode)
        buf += b'\x00\x00\x00'         # padding
        buf += struct.pack('<f', q)
        buf += struct.pack('<f', dq)
        buf += struct.pack('<f', tau)   # tau primero
        buf += struct.pack('<f', kp)
        buf += struct.pack('<f', kd)
        buf += struct.pack('<I', 0)     # uint32 reserve = 0

    # uint32[4] reserve del LowCmd — 16 bytes
    buf += struct.pack('<4I', 0, 0, 0, 0)

    assert len(buf) == 1000, f"Error de serializacion: {len(buf)} bytes (esperados 1000)"

    return bytes(buf)


def get_crc(msg: LowCmd) -> None:
    """Calcula y asigna msg.crc. Llamar justo antes de publicar."""
    data  = _serialize_low_cmd(msg)
    words = list(struct.unpack_from('<250I', data))  # 1000 / 4 = 250 palabras
    msg.crc = _crc32_core(words)


# =========================
# NODO ROS2
# =========================

class LowLevelCmdSender(Node):

    def __init__(self):
        super().__init__('low_level_cmd_sender')

        self.lowstate_topic = '/lowstate' if HIGH_FREQ else '/lf/lowstate'
        self.lowcmd_topic   = '/lowcmd'

        self.lowstate_subscriber = self.create_subscription(
            LowState,
            self.lowstate_topic,
            self.low_state_callback,
            qos_profile_sensor_data
        )

        self.lowcmd_publisher = self.create_publisher(
            LowCmd,
            self.lowcmd_topic,
            10
        )

        self.low_command = LowCmd()
        self.imu         = IMUState()
        self.motor       = [MotorState() for _ in range(H1_2_NUM_STATE_MOTOR)]

        self.received_lowstate = False
        self.lowstate_counter  = 0
        self.control_counter   = 0

        self.control_dt = 0.002   # 500 Hz
        self.time       = 0.0
        self.duration   = 3.0

        self.mode         = PRorAB.PR
        self.mode_machine = 0

        self.timer = self.create_timer(self.control_dt, self.control)

        self.get_logger().info(f'Suscrito a: {self.lowstate_topic}')
        self.get_logger().info(f'Publicando en: {self.lowcmd_topic}')
        self.get_logger().info('Esperando primer LowState antes de enviar comandos...')

    def low_state_callback(self, data: LowState):
        self.lowstate_counter += 1
        self.received_lowstate = True

        self.mode_machine = int(data.mode_machine)
        self.imu = data.imu_state

        n = min(len(data.motor_state), H1_2_NUM_STATE_MOTOR)
        for i in range(n):
            self.motor[i] = data.motor_state[i]

        if self.lowstate_counter % 200 == 0:
            self.get_logger().info(
                f'LowState OK | count: {self.lowstate_counter} | '
                f'mode_machine: {self.mode_machine} | '
                f'motor_state size: {len(data.motor_state)}'
            )

    def control(self):
        self.control_counter += 1

        if not self.received_lowstate:
            if self.control_counter % 500 == 0:
                self.get_logger().warn('Todavia esperando LowState...')
            return

        self.time += self.control_dt

        self.low_command.mode_pr      = self.mode
        self.low_command.mode_machine = self.mode_machine

        # Comando base para los 27 motores
        for i in range(H1_2_NUM_CMD_MOTOR):
            self.low_command.motor_cmd[i].mode = 1
            self.low_command.motor_cmd[i].tau  = 0.0
            self.low_command.motor_cmd[i].q    = 0.0
            self.low_command.motor_cmd[i].dq   = 0.0
            self.low_command.motor_cmd[i].kp   = 100.0 if i < 13 else 50.0
            self.low_command.motor_cmd[i].kd   = 1.0

        # Stage 1: mover suavemente a postura cero
        if self.time < self.duration:
            ratio = self.clamp(self.time / self.duration, 0.0, 1.0)
            for i in range(H1_2_NUM_CMD_MOTOR):
                self.low_command.motor_cmd[i].q = (1.0 - ratio) * self.motor[i].q

        # Stage 2: movimiento sinusoidal de tobillos y munecas
        else:
            self.mode = PRorAB.PR
            t = self.time - self.duration

            max_pitch = 0.25
            max_roll  = 0.25

            left_pitch_des  =  max_pitch * math.cos(2.0 * math.pi * t)
            left_roll_des   =  max_roll  * math.sin(2.0 * math.pi * t)
            right_pitch_des =  max_pitch * math.cos(2.0 * math.pi * t)
            right_roll_des  = -max_roll  * math.sin(2.0 * math.pi * t)

            kp_pitch, kd_pitch = 80.0, 1.0
            kp_roll,  kd_roll  = 80.0, 1.0

            self._set_motor(H12JointIndex.LEFT_ANKLE_PITCH,  left_pitch_des,  0.0, kp_pitch, kd_pitch, 0.0)
            self._set_motor(H12JointIndex.LEFT_ANKLE_ROLL,   left_roll_des,   0.0, kp_roll,  kd_roll,  0.0)
            self._set_motor(H12JointIndex.RIGHT_ANKLE_PITCH, right_pitch_des, 0.0, kp_pitch, kd_pitch, 0.0)
            self._set_motor(H12JointIndex.RIGHT_ANKLE_ROLL,  right_roll_des,  0.0, kp_roll,  kd_roll,  0.0)

            wrist_roll_des = 0.5 * math.sin(2.0 * math.pi * t)
            self._set_motor(H12JointIndex.LEFT_WRIST_ROLL,  wrist_roll_des, 0.0, 50.0, 1.0, 0.0)
            self._set_motor(H12JointIndex.RIGHT_WRIST_ROLL, wrist_roll_des, 0.0, 50.0, 1.0, 0.0)

        # CRC obligatorio — sin esto el robot descarta el mensaje
        get_crc(self.low_command)

        self.lowcmd_publisher.publish(self.low_command)

        if self.control_counter % 500 == 0:
            self.get_logger().info(
                f'LowCmd | t: {self.time:.3f} s | '
                f'crc: {self.low_command.crc} | '
                f'mode_pr: {self.low_command.mode_pr} | '
                f'mode_machine: {self.low_command.mode_machine}'
            )

    def _set_motor(self, index, q, dq, kp, kd, tau):
        self.low_command.motor_cmd[index].q   = float(q)
        self.low_command.motor_cmd[index].dq  = float(dq)
        self.low_command.motor_cmd[index].kp  = float(kp)
        self.low_command.motor_cmd[index].kd  = float(kd)
        self.low_command.motor_cmd[index].tau = float(tau)

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(value, high))


def main(args=None):
    rclpy.init(args=args)
    node = LowLevelCmdSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Nodo detenido por el usuario.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
