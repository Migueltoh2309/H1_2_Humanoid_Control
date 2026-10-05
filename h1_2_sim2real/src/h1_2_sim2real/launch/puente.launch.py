"""Puente ROS 2 -> SDK. EL MISMO para simulador y robot real.

    # simulador (antes: sim.launch.py). Arranca tambien el agente en este PC.
    ros2 launch h1_2_sim2real puente.launch.py objetivo:=sim
    # robot real: el agente corre EN EL ROBOT (scripts/agente_robot.sh) y aqui solo el puente
    ros2 launch h1_2_sim2real puente.launch.py objetivo:=real robot_ip:=192.168.0.143

Despues, cualquier nodo publica /joint_cmd (sensor_msgs/JointState) y lee /joint_states.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def _montar(context):
    share = get_package_share_directory("h1_2_sim2real")
    objetivo = LC("objetivo").perform(context)
    modo = LC("modo").perform(context)
    if objetivo not in ("sim", "real"):
        raise RuntimeError(f"objetivo:={objetivo}: tiene que ser sim o real")
    acciones = []
    if objetivo == "sim":
        ip = "127.0.0.1"
        extra = LC("agente_args").perform(context).split()
        acciones.append(ExecuteProcess(
            cmd=["python3", "-u", os.path.join(share, "robot", "agente_sdk.py"), "--sim", "--modo", modo] + extra,
            name="agente_sdk", output="screen"))
    else:
        ip = LC("robot_ip").perform(context)
        acciones.append(LogInfo(msg=f"ROBOT REAL: el agente tiene que estar corriendo en {ip} "
                                    f"(scripts/agente_robot.sh {modo}). L2+B = parada de emergencia."))
    acciones.append(Node(package="h1_2_sim2real", executable="puente", name="h1_2_puente", output="screen",
                         parameters=[{"agente_ip": ip}]))
    return acciones


def generate_launch_description():
    share = get_package_share_directory("h1_2_sim2real")
    with open(os.path.join(share, "urdf", "h1_2_manos.urdf")) as f:
        urdf = f.read()
    return LaunchDescription([
        DeclareLaunchArgument("objetivo", default_value="sim", description="sim | real"),
        DeclareLaunchArgument("modo", default_value="arm_sdk", description="arm_sdk | lowcmd (solo para el agente local)"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.0.143"),
        DeclareLaunchArgument("agente_args", default_value="", description="argumentos extra del agente (sim)"),
        DeclareLaunchArgument("rviz", default_value="true"),
        OpaqueFunction(function=_montar),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="log",
             parameters=[{"robot_description": urdf}]),
        Node(package="rviz2", executable="rviz2", output="log",
             arguments=["-d", os.path.join(share, "rviz", "h1_2_sim2real.rviz")],
             condition=IfCondition(LC("rviz"))),
    ])
