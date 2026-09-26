"""Sim bridge con la mandarina EN MOVIMIENTO: arranca en una punta de la
faja y se desplaza sola a velocidad constante (~0.07 m/s) hasta la otra
punta, como si la faja estuviera realmente activa — para probar algoritmos
de picking/tracking sobre un objeto en movimiento. Ver también
sim_bridge_static.launch.py.
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
            launch_arguments={'initial_keyframe': 'mandarina_moving'}.items(),
        ),
    ])
