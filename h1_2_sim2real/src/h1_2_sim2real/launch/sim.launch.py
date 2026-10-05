"""El H1-2 simulado (equivale a ENCENDER el robot): MuJoCo + SDK + manos Modbus + D435.

    ros2 launch h1_2_sim2real sim.launch.py                    # con el viewer de MuJoCo
    ros2 launch h1_2_sim2real sim.launch.py use_viewer:=false visor:=true
    ros2 launch h1_2_sim2real sim.launch.py keyframe:=mandarina_moving grasp_assist:=true
    ros2 launch h1_2_sim2real sim.launch.py escena:=suelo            # de pie en el suelo, fisica completa
    ros2 launch h1_2_sim2real sim.launch.py escena:=suelo banda:=true # colgado de la cinta (para lowcmd)

visor:=true abre RViz con /sim/joint_states (verdad de terreno): sirve para ver el
robot cuando se le manda con scripts del SDK sin el puente (p.ej. los de Codigos).
Con el puente, el RViz lo abre puente.launch.py con los /joint_states del agente.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("h1_2_sim2real")
    with open(os.path.join(share, "urdf", "h1_2_manos.urdf")) as f:
        urdf = f.read()
    args = [
        DeclareLaunchArgument("escena", default_value="faja",
                              description="faja (base soldada, mesa+faja) | suelo (de pie en el suelo, base flotante)"),
        DeclareLaunchArgument("banda", default_value="false", description="cinta elastica enganchada (escena suelo)"),
        DeclareLaunchArgument("modelo", default_value="", description="ruta a un MJCF (manda sobre escena)"),
        DeclareLaunchArgument("marcha", default_value="ninguna",
                              description="ninguna | cinematica (base rigida sigue Move) | politica (camina con la politica de unitree_rl_gym)"),
        DeclareLaunchArgument("marcha_semilla", default_value="-1"),
        DeclareLaunchArgument("emisor_ir", default_value="true", description="patron de puntos del emisor IR"),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(share, "rviz", "h1_2_sim2real.rviz")),
        DeclareLaunchArgument("visor", default_value="false"),
        DeclareLaunchArgument("modo_robot", default_value="auto", description="auto | ai | debug"),
        DeclareLaunchArgument("keyframe", default_value="", description="'' | mandarina_moving | mandarina_moving_fast"),
        DeclareLaunchArgument("grasp_assist", default_value="false"),
        DeclareLaunchArgument("camaras", default_value="true"),
        DeclareLaunchArgument("ir", default_value="true"),
        DeclareLaunchArgument("hz_camara", default_value="15.0"),
        DeclareLaunchArgument("hz_lowstate", default_value="500.0"),
        DeclareLaunchArgument("contactos", default_value="true"),
    ]
    sim = Node(package="h1_2_sim2real", executable="sim", name="h1_2_sim", output="screen", parameters=[{
        "escena": LC("escena"),
        "banda": ParameterValue(LC("banda"), value_type=bool),
        "modelo": LC("modelo"),
        "marcha": LC("marcha"),
        "marcha_semilla": ParameterValue(LC("marcha_semilla"), value_type=int),
        "emisor_ir": ParameterValue(LC("emisor_ir"), value_type=bool),
        "use_viewer": ParameterValue(LC("use_viewer"), value_type=bool),
        "modo_robot": LC("modo_robot"),
        "keyframe": LC("keyframe"),
        "grasp_assist": ParameterValue(LC("grasp_assist"), value_type=bool),
        "camaras": ParameterValue(LC("camaras"), value_type=bool),
        "ir": ParameterValue(LC("ir"), value_type=bool),
        "hz_camara": ParameterValue(LC("hz_camara"), value_type=float),
        "hz_lowstate": ParameterValue(LC("hz_lowstate"), value_type=float),
        "contactos": ParameterValue(LC("contactos"), value_type=bool),
    }])
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", output="log",
               parameters=[{"robot_description": urdf}], remappings=[("joint_states", "/sim/joint_states")],
               condition=IfCondition(LC("visor")))
    rviz = Node(package="rviz2", executable="rviz2", output="log",
                arguments=["-d", LC("rviz_config")],
                condition=IfCondition(LC("visor")))
    return LaunchDescription(args + [sim, rsp, rviz])
