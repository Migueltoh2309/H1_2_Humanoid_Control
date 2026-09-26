"""Launch del bridge de simulación con la escena VACÍA: solo el robot.

Igual que sim_bridge.launch.py (MuJoCo corre la física y publica
/joint_states, robot_state_publisher arma el TF, RViz2 visualiza), pero
cargando `h1_2_description/mjcf/h1_2_scene_empty.xml`: el H1-2 sobre el piso,
SIN mesa quirúrgica, SIN faja transportadora y SIN mandarina. El robot, sus
27 actuadores, los sensores de IMU y la cámara RGB-D del torso son los mismos
que en la escena de la mesa, así que un mismo experimento (/joint_cmd) corre
igual en las dos y lo único que cambia es el entorno.

Pensado para caracterizar movimientos "en el aire" — p.ej. levantar el brazo
derecho hacia adelante (`ros2 run h1_2_algoritms fk_right_arm_raise`) — sin
que ningún objeto de la escena meta contactos en el medio.

    ros2 launch h1_2_mujoco_sim_bridge sim_bridge_empty.launch.py
    ros2 launch h1_2_mujoco_sim_bridge sim_bridge_empty.launch.py use_viewer:=false
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
    # Config propia del demo del brazo derecho: igual que h1_2_sim.rviz pero
    # con un display Marker en /ee_marker (esfera roja = comando, verde = real,
    # línea = camino del efector) en lugar de las dos imágenes de la cámara,
    # que en esta escena no publican nada por defecto.
    default_rviz = os.path.join(bridge_share, 'rviz', 'h1_2_right_arm.rviz')
    urdf_path = os.path.join(description_share, 'urdf', 'h1_2_handless_bright.urdf')

    with open(urdf_path, 'r') as f:
        robot_description = {'robot_description': f.read()}

    use_viewer = DeclareLaunchArgument('use_viewer', default_value='true')
    use_rviz = DeclareLaunchArgument('use_rviz', default_value='true')
    joint_names_file = DeclareLaunchArgument('joint_names_file', default_value='')
    rviz_config = DeclareLaunchArgument('rviz_config', default_value=default_rviz)
    # Expuesto por si se quiere reusar este launch con otra escena del
    # paquete h1_2_description (el de la mesa vive en sim_bridge.launch.py).
    model_relpath = DeclareLaunchArgument(
        'model_relpath', default_value='mjcf/h1_2_scene_empty.xml')
    # Por defecto FALSE, al revés que sim_bridge.launch.py: en esta escena no
    # hay nada que mirar (ni mesa, ni faja, ni mandarina) y el render offscreen
    # de la RGB-D es lo más caro del bridge (10-100x un mj_step). Con la cámara
    # encendida en una GPU modesta el hilo de física avanza a tirones: medido
    # en este workspace, ~50% de las muestras de /joint_states repiten el
    # estado anterior (el movimiento termina EXACTAMENTE en la misma pose, pero
    # el registro sale escalonado y la velocidad calculada, llena de picos).
    # Apagada, ese número baja a ~0.3%. Ponerla en true si se necesita la RGB-D.
    publish_camera = DeclareLaunchArgument('publish_camera', default_value='false')

    sim_bridge_node = Node(
        package='h1_2_mujoco_sim_bridge',
        executable='sim_bridge',
        name='h1_2_mujoco_sim_bridge',
        output='screen',
        parameters=[
            bridge_yaml,
            {
                # Va DESPUÉS de bridge_yaml para pisar el model_relpath de la
                # mesa quirúrgica que trae ese archivo.
                'model_relpath': LaunchConfiguration('model_relpath'),
                'joint_gains_file': joint_gains_yaml,
                'joint_names_file': LaunchConfiguration('joint_names_file'),
                'use_viewer': LaunchConfiguration('use_viewer'),
                'publish_camera': LaunchConfiguration('publish_camera'),
                # La escena vacía no tiene el keyframe "mandarina_moving".
                'initial_keyframe': '',
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
        use_viewer, use_rviz, joint_names_file, rviz_config, model_relpath,
        publish_camera, sim_bridge_node, rsp_node, rviz_node,
    ])
