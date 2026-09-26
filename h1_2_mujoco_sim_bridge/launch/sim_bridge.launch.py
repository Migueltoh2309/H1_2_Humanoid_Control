"""Launch del bridge de simulación MuJoCo + robot_state_publisher + RViz2.

MuJoCo corre la física/colisiones/cámara (nodo sim_bridge) y publica
/joint_states; robot_state_publisher usa esos joint_states + el URDF para
calcular y publicar el TF completo, que RViz2 visualiza. El robot es de base
fija (URDF sin floating_base_joint activo), así que 'pelvis' es la raíz del
árbol de TF — no hace falta ningún nodo de odometría.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bridge_share = get_package_share_directory('h1_2_mujoco_sim_bridge')
    description_share = get_package_share_directory('h1_2_description')

    bridge_yaml = os.path.join(bridge_share, 'config', 'sim_bridge.yaml')
    joint_gains_yaml = os.path.join(bridge_share, 'config', 'joint_gains.yaml')
    default_rviz = os.path.join(bridge_share, 'rviz', 'h1_2_sim.rviz')
    urdf_path = os.path.join(description_share, 'urdf', 'h1_2_handless_bright.urdf')

    with open(urdf_path, 'r') as f:
        robot_description = {'robot_description': f.read()}

    use_viewer = DeclareLaunchArgument('use_viewer', default_value='true')
    use_rviz = DeclareLaunchArgument('use_rviz', default_value='true')
    joint_names_file = DeclareLaunchArgument('joint_names_file', default_value='')
    rviz_config = DeclareLaunchArgument('rviz_config', default_value=default_rviz)
    # '' = mandarina quieta en el centro de la faja (estado por defecto del
    # MJCF). 'mandarina_moving' = arranca en una punta y se mueve sola a lo
    # largo de la faja (ver keyframe en h1_2_scene_surgery_table.xml).
    initial_keyframe = DeclareLaunchArgument('initial_keyframe', default_value='')

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
                'use_viewer': LaunchConfiguration('use_viewer'),
                'initial_keyframe': LaunchConfiguration('initial_keyframe'),
            },
        ],
    )

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription([
        use_viewer, use_rviz, joint_names_file, rviz_config, initial_keyframe,
        sim_bridge_node, rsp_node, rviz_node,
    ])
