"""Geometría del agarre de la mandarina con la mano Inspire.

Todo en `torso_link` (el frame de `fkine_arm_*_unitree`, del QP y de la
nube de la cámara). El efector que controla el QP es la MUÑECA
(`*_wrist_yaw_link`); la fruta tiene que quedar en un punto fijo respecto a
ella, delante de la palma. Este módulo traduce "centro de la fruta" ->
"pose objetivo de la muñeca", para el agarre y para el pre-agarre.

Frame de la muñeca (medido en MuJoCo con los dedos abiertos/cerrados, ver
VISUAL_SERVOING_PLAN.md §4):
    x  a lo largo de los dedos (nudillos en x ≈ 0.19, puntas en ≈ 0.24)
    y  normal de la palma: los dedos cierran hacia −y en la mano IZQUIERDA
       y hacia +y en la DERECHA
    z  lado del pulgar (índice en z = +0.03, meñique en −0.025)

Orientación del agarre (barrido de alcanzabilidad con la IK sobre la faja
+ pruebas físicas de cierre/levantamiento, ver VISUAL_SERVOING_PLAN.md §4):
desde ARRIBA, dedos 20° bajo la horizontal y girados 40° hacia el centro
del robot, palma sobre la fruta. El agarre lateral (palma vertical) solo es
alcanzable con los dedos ~65° hacia abajo, y ahí las puntas entran en la
faja. Aproximación con el pulgar ABIERTO (en la preforma opuesta cuelga
bajo la palma y empuja la fruta); se opone al cerrar.
"""
import numpy as np

from h1_2_algoritms.fk_functions import (
    fkine_arm_left_unitree, fkine_arm_right_unitree, TF2xyzquat, rot2quat)
from h1_2_algoritms.ik_functions import ik_solve_limited
from h1_2_algoritms import joint_limits as JL

TORSO_Z = 1.0282          # z de torso_link en el mundo de la escena de la faja
FK = {"left": fkine_arm_left_unitree, "right": fkine_arm_right_unitree}

# Centro de la fruta en el frame de la muñeca para el agarre de potencia
# (y se refleja entre manos). Ajustado en la prueba física.
GRASP_OFFSET = {"left": np.array([0.17, -0.058, 0.0]),
                "right": np.array([0.17, 0.058, 0.0])}
GRASP_MODE = "top"
APPROACH_ELEV = np.deg2rad(-20.0)   # dedos 20° bajo la horizontal
APPROACH_AZ = {"left": np.deg2rad(-40.0), "right": np.deg2rad(40.0)}   # hacia el centro

# Aperturas de la mano (radianes de los 6 actuadores: pulgar yaw, pulgar
# pitch, índice, medio, anular, meñique). "preshape" deja el pulgar ya
# opuesto a los dedos para que al cerrar no barra la fruta.
HAND_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
HAND_PRESHAPE = np.array([0.0, 0.0, 0.1, 0.1, 0.1, 0.1])
HAND_CLOSED = np.array([1.2, 0.5, 1.4, 1.4, 1.4, 1.4])
HAND_ACTUATORS = ["thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
                  "index_proximal_joint", "middle_proximal_joint",
                  "ring_proximal_joint", "pinky_proximal_joint"]


def hand_actuator_names(side):
    pre = "L" if side == "left" else "R"
    return [f"{pre}_{n}" for n in HAND_ACTUATORS]


def grasp_rotation(side, elev=APPROACH_ELEV, az=None, mode=None):
    """R (3x3) de la muñeca en torso_link: x = dirección de los dedos.
    mode "top": palma hacia abajo; "side": palma vertical mirando hacia el
    centro del robot (mano derecha -> +y, izquierda -> -y), pulgar arriba."""
    az = APPROACH_AZ[side] if az is None else az
    mode = GRASP_MODE if mode is None else mode
    x = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
    if mode == "top":
        # normal de palma: la dirección más "hacia abajo" perpendicular a x
        ref = np.array([0.0, 0.0, -1.0])
    else:
        ref = np.array([0.0, 1.0, 0.0]) if side == "right" else np.array([0.0, -1.0, 0.0])
    n = ref - np.dot(ref, x) * x
    n /= np.linalg.norm(n)
    y = -n if side == "left" else n
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def palm_normal(side, R):
    return -R[:, 1] if side == "left" else R[:, 1]


def grasp_wrist_target(side, fruit_torso, R=None):
    """[x y z qw qx qy qz] de la muñeca para que la fruta quede en el punto
    de agarre."""
    R = grasp_rotation(side) if R is None else R
    p = np.asarray(fruit_torso) - R @ GRASP_OFFSET[side]
    return np.hstack((p, rot2quat(R)))


def pregrasp_wrist_target(side, fruit_torso, approach=0.08, R=None):
    """Mismo agarre desplazado `approach` m hacia atrás de la palma (arriba
    de la fruta): desde ahí la mano baja en línea recta a lo largo de la
    normal de la palma, sin barrer la fruta con los dedos."""
    R = grasp_rotation(side) if R is None else R
    xd = grasp_wrist_target(side, fruit_torso, R)
    xd[0:3] -= approach * palm_normal(side, R)
    return xd


def solve_arm_ik(side, xd, q0=None):
    lo, hi = JL.get_limits(side)
    q0 = JL.center(side) if q0 is None else q0
    r = ik_solve_limited(FK[side], TF2xyzquat, q0, xd, lo, hi, restarts=4, max_iters=200)
    return r["q"], bool(r["ok"])


def wrist_to_fruit(side, q):
    """Dónde quedaría el centro de la fruta con la muñeca en q (FK)."""
    T = FK[side](q)
    return T[0:3, 3] + T[0:3, 0:3] @ GRASP_OFFSET[side]
