"""Límites articulares de los brazos del Unitree H1-2.

Los valores se toman del MJCF de `h1_2_description` (paquete `h1_2_utec`)
(`mjcf/h1_2_scene_qp_reachable.xml`, campo `jnt_range`), que es el mismo
modelo que usa el bridge de bajo nivel para saturar `q_des`. Se replican
aquí como constantes para que los algoritmos de cinemática no dependan de
MuJoCo en tiempo de ejecución; `limits_from_mjcf()` permite releerlos del
XML para verificar que no se hayan desincronizado.

El orden de las filas es el mismo que el de `fkine_arm_*_unitree`:

    [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
     wrist_roll, wrist_pitch, wrist_yaw]
"""
import numpy as np

# Nombres de las articulaciones en el orden de la cadena cinemática.
ARM_JOINT_NAMES = {
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
}

# Índices de motor Unitree (LowCmd.motor_cmd[i]) de cada cadena, según el
# mapeo que el bridge valida al arrancar.
ARM_MOTOR_IDX = {
    "left": [13, 14, 15, 16, 17, 18, 19],
    "right": [20, 21, 22, 23, 24, 25, 26],
}

# [q_min, q_max] en radianes, tomados de jnt_range del MJCF.
ARM_LIMITS = {
    "left": np.array([
        [-3.14, 1.57],       # left_shoulder_pitch_joint
        [-0.38, 3.40],       # left_shoulder_roll_joint
        [-2.66, 3.01],       # left_shoulder_yaw_joint
        [-0.95, 3.18],       # left_elbow_joint
        [-3.01, 2.75],       # left_wrist_roll_joint
        [-0.4625, 0.4625],   # left_wrist_pitch_joint
        [-1.27, 1.27],       # left_wrist_yaw_joint
    ]),
    "right": np.array([
        [-3.14, 1.57],       # right_shoulder_pitch_joint
        [-3.40, 0.38],       # right_shoulder_roll_joint
        [-3.01, 2.66],       # right_shoulder_yaw_joint
        [-0.95, 3.18],       # right_elbow_joint
        [-2.75, 3.01],       # right_wrist_roll_joint
        [-0.4625, 0.4625],   # right_wrist_pitch_joint
        [-1.27, 1.27],       # right_wrist_yaw_joint
    ]),
}


def get_limits(side):
    """Devuelve (q_min, q_max) como dos vectores de 7 para 'left' o 'right'."""
    lim = ARM_LIMITS[side]
    return lim[:, 0].copy(), lim[:, 1].copy()


def center(side):
    """Punto medio del rango de cada articulación (postura de referencia)."""
    lo, hi = get_limits(side)
    return 0.5 * (lo + hi)


def clamp(q, side):
    """Satura q al rango admisible del brazo indicado."""
    lo, hi = get_limits(side)
    return np.clip(q, lo, hi)


def violations(q, side, tol=1e-9):
    """Exceso sobre el límite de cada articulación (0 si está dentro)."""
    lo, hi = get_limits(side)
    over = np.maximum(np.maximum(lo - q, q - hi), 0.0)
    over[over <= tol] = 0.0
    return over


def within_limits(q, side, tol=1e-9):
    """True si todas las articulaciones están dentro de su rango."""
    lo, hi = get_limits(side)
    return bool(np.all(q >= lo - tol) and np.all(q <= hi + tol))


def limits_from_mjcf(mjcf_path, side):
    """Relee los límites del MJCF. Requiere mujoco; útil para verificar."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    rows = []
    for name in ARM_JOINT_NAMES[side]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"El MJCF no contiene la articulación '{name}'")
        rows.append(model.jnt_range[jid].copy())
    return np.array(rows)
