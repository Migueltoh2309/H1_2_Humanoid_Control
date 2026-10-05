"""Tabla de juntas, limites y ganancias del H1-2 (27 DOF) + manos Inspire.

FICHERO MAESTRO, compartido por los tres lados:
  * el agente SDK (robot/agente_sdk.py), que corre en el robot o en este PC,
  * el simulador MuJoCo (paquete ROS 2 h1_2_sim2real),
  * el puente ROS 2 (puente_ros2.py).
Solo usa la libreria estandar: el agente tiene que poder importarlo en el PC2
del robot sin ROS ni numpy.

Indices DDS = orden del URDF, sin huecos (ver Codigos/pc/simulacion_mujoco_h1_2/
Simulacion_H1_2_Mujoco.md): 0-11 piernas, 12 torso, 13-19 brazo izq, 20-26 brazo
der. El slot 27 del LowCmd es el PESO de rt/arm_sdk.
"""

N = 27
PESO_ARM_SDK = 27
PIERNAS = list(range(12))
BRAZOS = list(range(12, 27))      # torso + 14 de brazos: lo que manda rt/arm_sdk
IZQ = list(range(13, 20))
DER = list(range(20, 27))

NOMBRES = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
INDICE = {n: i for i, n in enumerate(NOMBRES)}

# Limites de posicion (rad). Torso y brazos: los de caja_cuadrado.py (web de
# Unitree ∩ URDF), los que ya se usan en el robot. Piernas: URDF h1_2.urdf.
LIMITES = {
    0: (-0.43, 0.43), 1: (-3.14, 2.5), 2: (-0.43, 3.14), 3: (-0.12, 2.19),
    4: (-0.897, 0.523), 5: (-0.261, 0.261),
    6: (-0.43, 0.43), 7: (-3.14, 2.5), 8: (-3.14, 0.43), 9: (-0.12, 2.19),
    10: (-0.897, 0.523), 11: (-0.261, 0.261),
    12: (-2.35, 1.57), 13: (-3.14, 1.57), 14: (-0.38, 3.40), 15: (-2.66, 2.66),
    16: (-0.95, 1.60), 17: (-2.967, 2.75), 18: (-0.4625, 0.349), 19: (-1.012, 1.012),
    20: (-3.14, 1.57), 21: (-3.40, 0.38), 22: (-2.66, 2.66), 23: (-0.95, 1.60),
    24: (-2.75, 2.967), 25: (-0.4625, 0.349), 26: (-1.012, 1.012),
}
MARGEN = 0.05

# Par maximo (N·m), URDF.
TAU_MAX = [200, 200, 200, 300, 60, 40] * 2 + [200] + [40, 40, 18, 18, 19, 19, 19] * 2

# Ganancias por defecto: las probadas en el robot (caja_cuadrado.py): brazos
# kp 100 / kd 2 por arm_sdk, torso 200 / 5; piernas las del ejemplo oficial
# h1_27dof_example (solo se usan en modo lowcmd, robot colgado).
KP = [100, 100, 100, 200, 80, 80] * 2 + [200] + [100] * 14
KD = [3, 3, 3, 5, 2, 2] * 2 + [5] + [2] * 14


def recortar(i, q):
    lo, hi = LIMITES[i]
    return min(max(q, lo + MARGEN), hi - MARGEN)


# ---------------------------------------------------------------- manos
# Inspire RH56DFTP: 6 DOF por mano, por Modbus TCP (puerto 6000).
# Orden de los registros: [menique, anular, medio, indice, pulgar flexion,
# pulgar rotacion]; 0 = cerrado, 1000 = abierto.
MANO_IP = {"izq": "192.168.124.211", "der": "192.168.124.210"}   # al reves que la doc de Unitree
MANO_IP_SIM = {"izq": "127.0.1.211", "der": "127.0.1.210"}       # el simulador escucha aqui
MANO_PUERTO = 6000
ANGLE_SET, FORCE_SET, SPEED_SET, ANGLE_ACT, FORCE_ACT = 1486, 1498, 1522, 1546, 1582
ERR, STATUS, TEMP = 1606, 1612, 1618

# Junta del URDF/MJCF (sin el prefijo L_/R_) de cada DOF y su recorrido en rad:
# angulo 1000 (abierto) -> 0 rad; angulo 0 (cerrado) -> q_cerrado.
DOF_MANO = [
    ("pinky_proximal_joint", 1.7),
    ("ring_proximal_joint", 1.7),
    ("middle_proximal_joint", 1.7),
    ("index_proximal_joint", 1.7),
    ("thumb_proximal_pitch_joint", 0.6),
    ("thumb_proximal_yaw_joint", 1.3),
]
PREFIJO = {"izq": "L_", "der": "R_"}
LADOS = ("izq", "der")
NOMBRES_MANO = [PREFIJO[l] + j for l in LADOS for j, _ in DOF_MANO]   # 12, izq primero


def angulo_a_rad(dof, a):
    return (1.0 - min(max(a, 0), 1000) / 1000.0) * DOF_MANO[dof][1]


def rad_a_angulo(dof, q):
    return int(round(1000.0 * (1.0 - min(max(q / DOF_MANO[dof][1], 0.0), 1.0))))
