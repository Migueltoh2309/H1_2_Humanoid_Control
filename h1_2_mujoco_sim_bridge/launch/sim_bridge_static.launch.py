"""Sim bridge con la mandarina QUIETA en el centro de la faja (estado por
defecto del MJCF, sin keyframe) — para probar algoritmos de picking sobre
un objeto estático. Ver también sim_bridge_moving.launch.py.
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    base_launch = PathJoinSubstitution([
        FindPackageShare('h1_2_mujoco_sim_bridge'), 'launch', 'sim_bridge.launch.py'
    ])
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={'initial_keyframe': ''}.items(),
        ),
    ])
