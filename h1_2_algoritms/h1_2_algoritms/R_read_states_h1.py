#!/usr/bin/env python3

"""
ROS2 subscriber for Unitree H1-2 LowState (HG version)
"""

import rclpy
from rclpy.node import Node

# 🔥 Usar perfil oficial de sensores (evita problemas DDS)
from rclpy.qos import qos_profile_sensor_data

from unitree_hg.msg import IMUState
from unitree_hg.msg import LowState
from unitree_hg.msg import MotorState


# Configuration flags
INFO_IMU = False
INFO_MOTOR = True
HIGH_FREQ = True


class LowStateSuber(Node):

    def __init__(self):
        super().__init__('low_state_suber')

        # 🔥 Topic absoluto (evita problemas de namespace)
        topic_name = "/lowstate" if HIGH_FREQ else "/lf/lowstate"

        # 🔥 Subscription con QoS sensor_data
        self.suber = self.create_subscription(
            LowState,
            topic_name,
            self.topic_callback,
            qos_profile_sensor_data
        )

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(35)]

        self.counter = 0

        self.get_logger().info(f"Subscribed to topic: {topic_name}")
        self.get_logger().info("Waiting for LowState messages...")

    def topic_callback(self, data: LowState):

        self.counter += 1

        if INFO_IMU and self.counter % 200 == 0:
            imu = data.imu_state

            self.get_logger().info(
                f"RPY: {imu.rpy[0]:.3f}, "
                f"{imu.rpy[1]:.3f}, "
                f"{imu.rpy[2]:.3f}"
            )

        if INFO_MOTOR and self.counter % 200 == 0:
            i = 19
            self.motor[i] = data.motor_state[i]

            self.get_logger().info(
                f"Motor {i} | q: {self.motor[i].q:.4f}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = LowStateSuber()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()