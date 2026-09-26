"""Launch del bridge de simulación en modo headless (sin viewer nativo ni RViz).

Sigue publicando /joint_states, TF (via robot_state_publisher), cámara y
contactos por si otro proceso (RViz en otra máquina, grabación de bags, un
loop de RL) los necesita.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bridge_share = get_package_share_directory('h1_2_mujoco_sim_bridge')
    description_share = get_package_share_directory('h1_2_description')

    bridge_yaml = os.path.join(bridge_share, 'config', 'sim_bridge.yaml')
    joint_gains_yaml = os.path.join(bridge_share, 'config', 'joint_gains.yaml')
    urdf_path = os.path.join(description_share, 'urdf', 'h1_2_handless_bright.urdf')

    with open(urdf_path, 'r') as f:
        robot_description = {'robot_description': f.read()}

    joint_names_file = DeclareLaunchArgument('joint_names_file', default_value='')

    sim_bridge_node = Node(
        package='h1_2_mujoco_sim_bridge',
        executable='sim_bridge',
        name='h1_2_mujoco_sim_bridge',
        output='screen',
        parameters=[
            bridge_yaml,
            {
                'joint_gains_file': joint_gains_yaml,
                'joint_names_file': LaunchConfiguration('joint_names_file'),
                'use_viewer': False,
            },
        ],
    )

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description],
        output='screen',
    )

    return LaunchDescription([joint_names_file, sim_bridge_node, rsp_node])
