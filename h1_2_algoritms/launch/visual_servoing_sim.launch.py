"""Visual servoing + agarre de la mandarina, todo en simulación.

    MuJoCo (escena con manos Inspire)  --/camera/*-->  color_depth_detector_node
          ^            |                                (ajuste de esfera)
          |       /joint_states                               |
     /joint_cmd        v                            /perception/target_position
    (brazos+manos) <-- QP_bimanual_avoidance (scenario:=grasp) <--'

Argumentos:
    belt:=static|moving|fast   fruta quieta (y = -0.03), faja a 0.05 m/s o a
                               0.07 m/s (keyframes de la escena con manos)
    use_viewer:=true|false     viewer nativo de MuJoCo
    use_rviz:=true|false
    grasp_assist:=true|false   weld palma-fruta al cerrar con contacto real
                               (solo simulación; ver VISUAL_SERVOING_PLAN §4)
    localization:=sphere|median+R|median   estimador del centro de la fruta

Ejemplo:
    ros2 launch h1_2_algoritms visual_servoing_sim.launch.py belt:=moving
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

KEYFRAMES = {"static": "", "moving": "mandarina_moving", "fast": "mandarina_moving_fast"}


def _nodes(context):
    bridge_share = get_package_share_directory('h1_2_mujoco_sim_bridge')
    description_share = get_package_share_directory('h1_2_description')
    urdf_path = os.path.join(description_share, 'urdf', 'h1_2_handless_bright.urdf')
    with open(urdf_path, 'r') as f:
        robot_description = {'robot_description': f.read()}

    belt = LaunchConfiguration('belt').perform(context)
    if belt not in KEYFRAMES:
        raise RuntimeError(f"belt:='{belt}' no es uno de {list(KEYFRAMES)}")

    bridge = Node(
        package='h1_2_mujoco_sim_bridge', executable='sim_bridge',
        name='h1_2_mujoco_sim_bridge', output='screen',
        parameters=[
            os.path.join(bridge_share, 'config', 'sim_bridge.yaml'),
            {
                'joint_gains_file': os.path.join(bridge_share, 'config', 'joint_gains.yaml'),
                'model_relpath': 'mjcf/h1_2_scene_surgery_table_hands.xml',
                'initial_keyframe': KEYFRAMES[belt],
                'grasp_assist': LaunchConfiguration('grasp_assist'),
                'use_viewer': LaunchConfiguration('use_viewer'),
                # 15 Hz como en la batería (demos/visual_servoing_grasp.py).
                'camera_frequency': 15.0,
                # Solo para medir el resultado (no lo lee el control).
                'ground_truth_bodies': ['mandarina'],
            },
        ],
    )
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[robot_description], output='screen')
    perception = Node(
        package='h1_2_algoritms', executable='color_depth_detector_node', output='screen',
        parameters=[{'localization': LaunchConfiguration('localization'),
                     'target_frame': 'torso_link',
                     'publish_debug_image': False}],
    )
    servo = Node(
        package='h1_2_algoritms', executable='QP_bimanual_avoidance', output='screen',
        parameters=[{'scenario': 'grasp', 't_home': 2.0}],
    )
    rviz = Node(
        package='rviz2', executable='rviz2', output='screen',
        arguments=['-d', os.path.join(bridge_share, 'rviz', 'h1_2_sim.rviz')],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )
    return [bridge, rsp, perception, servo, rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('belt', default_value='static'),
        DeclareLaunchArgument('use_viewer', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('grasp_assist', default_value='true'),
        DeclareLaunchArgument('localization', default_value='sphere'),
        OpaqueFunction(function=_nodes),
    ])
