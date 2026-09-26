"""Launch del bridge de bajo nivel en modo headless (sin viewer)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('h1_2_mujoco_lowlevel_bridge')
    bridge_yaml = os.path.join(share, 'config', 'bridge.yaml')
    mapping_yaml = os.path.join(share, 'config', 'joint_mapping.yaml')
    limits_yaml = os.path.join(share, 'config', 'motor_limits.yaml')

    verify_crc = DeclareLaunchArgument('verify_crc', default_value='true')

    node = Node(
        package='h1_2_mujoco_lowlevel_bridge',
        executable='lowlevel_bridge',
        name='h1_2_mujoco_lowlevel_bridge',
        output='screen',
        parameters=[
            bridge_yaml,
            {
                'joint_mapping_file': mapping_yaml,
                'motor_limits_file': limits_yaml,
                'use_viewer': False,
                'verify_crc': LaunchConfiguration('verify_crc'),
            },
        ],
    )
    return LaunchDescription([verify_crc, node])
