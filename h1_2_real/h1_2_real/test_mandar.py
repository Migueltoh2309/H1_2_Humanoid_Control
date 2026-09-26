#!/usr/bin/env python3

"""
ROS2 low-level command sender for Unitree H1-2 using the same LowState
subscription style that works with the user's sensor/motor reader.

Key points:
- Subscribes to /lowstate or /lf/lowstate with qos_profile_sensor_data.
- Publishes to /lowcmd.
- Waits until the first LowState message is received before sending motion commands.
- Uses 27 command motors for H1-2, but stores up to 35 motor states because LowState may contain more states.
- CRC is still a required Unitree-specific step. This script leaves a placeholder.
"""

import math
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

INFO_IMU = False
INFO_MOTOR = False
HIGH_FREQ = True

H1_2_NUM_CMD_MOTOR = 27
H1_2_NUM_STATE_MOTOR = 35


class PRorAB:
    PR = 0
    AB = 1


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

    # arms
    LEFT_SHOULDER_PITCH = 13
    LEFT_SHOULDER_ROLL = 14
    LEFT_SHOULDER_YAW = 15
    LEFT_ELBOW = 16
    LEFT_WRIST_ROLL = 17
    LEFT_WRIST_PITCH = 18
    LEFT_WRIST_YAW = 19

    RIGHT_SHOULDER_PITCH = 20
    RIGHT_SHOULDER_ROLL = 21
    RIGHT_SHOULDER_YAW = 22
    RIGHT_ELBOW = 23
    RIGHT_WRIST_ROLL = 24
    RIGHT_WRIST_PITCH = 25
    RIGHT_WRIST_YAW = 26


def get_crc(low_cmd_msg: LowCmd) -> None:
    """
    Placeholder for Unitree CRC calculation.

    The original C++ code calls:
        get_crc(low_command_)

    If the robot receives /lowcmd but ignores it, this is probably the missing part.
    Replace this function with the real Unitree CRC implementation or a Python binding.
    """
    pass


class LowLevelCmdSender(Node):

    def __init__(self):
        super().__init__('low_level_cmd_sender')

        # Same topic style as your working reader.
        self.lowstate_topic = '/lowstate' if HIGH_FREQ else '/lf/lowstate'
        self.lowcmd_topic = '/lowcmd'

        # Subscriber with official sensor QoS.
        self.lowstate_subscriber = self.create_subscription(
            LowState,
            self.lowstate_topic,
            self.low_state_callback,
            qos_profile_sensor_data
        )

        # Publisher. Start with depth=10; if needed, this can also be replaced
        # by a custom QoS profile after checking `ros2 topic info /lowcmd -v`.
        self.lowcmd_publisher = self.create_publisher(
            LowCmd,
            self.lowcmd_topic,
            10
        )

        self.low_command = LowCmd()

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(H1_2_NUM_STATE_MOTOR)]

        self.received_lowstate = False
        self.lowstate_counter = 0
        self.control_counter = 0

        self.control_dt = 0.002  # 500 Hz
        self.time = 0.0
        self.duration = 3.0

        self.mode = PRorAB.PR
        self.mode_machine = 0

        self.timer = self.create_timer(self.control_dt, self.control)

        self.get_logger().info(f'Subscribed to topic: {self.lowstate_topic}')
        self.get_logger().info(f'Publishing to topic: {self.lowcmd_topic}')
        self.get_logger().info('Waiting for LowState before sending motion commands...')

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

        if INFO_IMU and self.lowstate_counter % 200 == 0:
            imu = self.imu
            self.get_logger().info(
                f'RPY: {imu.rpy[0]:.3f}, {imu.rpy[1]:.3f}, {imu.rpy[2]:.3f}'
            )

        if INFO_MOTOR and self.lowstate_counter % 200 == 0:
            i = H12JointIndex.LEFT_WRIST_YAW
            self.get_logger().info(
                f'Motor {i} | q: {self.motor[i].q:.4f} | dq: {self.motor[i].dq:.4f}'
            )

    def control(self):
        self.control_counter += 1

        if not self.received_lowstate:
            if self.control_counter % 500 == 0:
                self.get_logger().warn('Still waiting for LowState. Not publishing movement command yet.')
            return

        self.time += self.control_dt

        self.low_command.mode_pr = self.mode
        self.low_command.mode_machine = self.mode_machine

        # Base command for the 27 H1-2 actuated joints.
        for i in range(H1_2_NUM_CMD_MOTOR):
            self.low_command.motor_cmd[i].mode = 1
            self.low_command.motor_cmd[i].tau = 0.0
            self.low_command.motor_cmd[i].q = 0.0
            self.low_command.motor_cmd[i].dq = 0.0
            self.low_command.motor_cmd[i].kp = 100.0 if i < 13 else 50.0
            self.low_command.motor_cmd[i].kd = 1.0

        # Stage 1: move smoothly from current posture to zero posture.
        if self.time < self.duration:
            ratio = self.clamp(self.time / self.duration, 0.0, 1.0)

            for i in range(H1_2_NUM_CMD_MOTOR):
                self.low_command.motor_cmd[i].q = (1.0 - ratio) * self.motor[i].q

        # Stage 2: sinusoidal motion for ankles and wrists.
        else:
            self.mode = PRorAB.PR

            t = self.time - self.duration

            max_pitch = 0.25
            max_roll = 0.25

            left_pitch_des = max_pitch * math.cos(2.0 * math.pi * t)
            left_roll_des = max_roll * math.sin(2.0 * math.pi * t)

            right_pitch_des = max_pitch * math.cos(2.0 * math.pi * t)
            right_roll_des = -max_roll * math.sin(2.0 * math.pi * t)

            kp_pitch = 80.0
            kd_pitch = 1.0
            kp_roll = 80.0
            kd_roll = 1.0

            self.set_motor_cmd(H12JointIndex.LEFT_ANKLE_PITCH, left_pitch_des, 0.0, kp_pitch, kd_pitch, 0.0)
            self.set_motor_cmd(H12JointIndex.LEFT_ANKLE_ROLL, left_roll_des, 0.0, kp_roll, kd_roll, 0.0)
            self.set_motor_cmd(H12JointIndex.RIGHT_ANKLE_PITCH, right_pitch_des, 0.0, kp_pitch, kd_pitch, 0.0)
            self.set_motor_cmd(H12JointIndex.RIGHT_ANKLE_ROLL, right_roll_des, 0.0, kp_roll, kd_roll, 0.0)

            max_wrist_roll_angle = 0.5
            wrist_roll_des = max_wrist_roll_angle * math.sin(2.0 * math.pi * t)

            self.set_motor_cmd(H12JointIndex.LEFT_WRIST_ROLL, wrist_roll_des, 0.0, 50.0, 1.0, 0.0)
            self.set_motor_cmd(H12JointIndex.RIGHT_WRIST_ROLL, wrist_roll_des, 0.0, 50.0, 1.0, 0.0)

        get_crc(self.low_command)
        self.lowcmd_publisher.publish(self.low_command)

        if self.control_counter % 500 == 0:
            self.get_logger().info(
                f'Publishing LowCmd | t: {self.time:.3f} s | '
                f'mode_pr: {self.low_command.mode_pr} | '
                f'mode_machine: {self.low_command.mode_machine}'
            )

    def set_motor_cmd(self, index, q, dq, kp, kd, tau):
        self.low_command.motor_cmd[index].q = float(q)
        self.low_command.motor_cmd[index].dq = float(dq)
        self.low_command.motor_cmd[index].kp = float(kp)
        self.low_command.motor_cmd[index].kd = float(kd)
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
        node.get_logger().info('Node stopped by keyboard interrupt.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()