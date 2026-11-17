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
import os

import numpy as np

from h1_2_algoritms.movimiento.fk_functions import (
    fkine_arm_left_unitree, fkine_arm_right_unitree, TF2xyzquat, rot2quat)
from h1_2_algoritms.movimiento.ik_functions import ik_solve_limited
from h1_2_algoritms.movimiento import joint_limits as JL

# Cuerpo del robot (env H12_CUERPO):
#   "soldado"  la pelvis fija al mundo, piernas rectas (todo hasta 2026-11-15);
#   "pie"      de pie con su equilibrio tras el arranque con el arnés (aprendizaje/cuerpo_pie.py):
#              rodillas dobladas, el torso más abajo (TORSO_Z_PIE, medido con cuerpo_pie.medir()).
CUERPO = os.environ.get("H12_CUERPO", "soldado")
TORSO_Z_PIE = 1.0059     # medido 2026-11-16: de pie tras el arranque, rodillas 0.46, suelas de caja
TORSO_Z = TORSO_Z_PIE if CUERPO == "pie" else 1.0282   # z de torso_link en el mundo de la escena de la faja
# z (torso_link) del centro de una fruta apoyada en la faja: superficie medida
# con mj_ray (0.9645, mesa u_table; con la mesa vieja 0.8245 -> -0.164) + radio.
Z_FRUTA_FAJA = 0.9645 + 0.04 - TORSO_Z
# Cuánto subió la faja respecto de la mesa con la que se ajustaron las poses
# de tarea (depósito, mostrar) y se grabaron las demos de la DMP (fruta a
# z = -0.164). Esas poses se definen relativas a la faja: suben con ella.
# De pie (CUERPO "pie") el torso baja 2.2 cm, pero el depósito (cesto abstracto al costado) y la
# pose de mostrar (frente a la cámara, que va en el torso) no son de la faja: se quedan en el marco
# del torso, con el valor de la pelvis soldada (con el de pie, el depósito sale de los límites).
DZ_MESA = (0.9645 + 0.04 - 1.0282) - (-0.164)
# caja de la superficie útil de la faja (torso_link) en x, z: ±4 cm en z
# Faja (y mesa) acercada al robot FAJA_DX m en -x (env H12_FAJA_DX; montaje).
# Todo lo que se define respecto de la faja se corre con ella: la caja de la
# percepción, la espera, el depósito y la pose de mostrar.
FAJA_DX = float(os.environ.get("H12_FAJA_DX", "0"))
BELT_BOX_XZ = {"x": (0.26 - FAJA_DX, 0.47 - FAJA_DX), "z": (Z_FRUTA_FAJA - 0.04, Z_FRUTA_FAJA + 0.04)}
FK = {"left": fkine_arm_left_unitree, "right": fkine_arm_right_unitree}

# Centro de la fruta en el frame de la muñeca para el agarre de potencia
# (y se refleja entre manos). Ajustado en la prueba física.
GRASP_OFFSET = {"left": np.array([0.17, -0.058, 0.0]),
                "right": np.array([0.17, 0.058, 0.0])}
if os.environ.get("H12_MANOS", "rh56") == "rh56":
    # RH56DFTP oficial: palma hasta x = 0.211 (vieja 0.196) y dedos hasta
    # 0.297 (0.272) en el frame de la muñeca; la cara de la palma sigue en
    # y = 0.025. El agarre se corre lo mismo que la palma.
    GRASP_OFFSET = {"left": np.array([0.185, -0.058, 0.0]),
                    "right": np.array([0.185, 0.058, 0.0])}
GRASP_MODE = "top"
APPROACH_ELEV = np.deg2rad(-20.0)   # dedos 20° bajo la horizontal
APPROACH_AZ = {"left": np.deg2rad(-40.0), "right": np.deg2rad(40.0)}   # hacia el centro

# Aperturas de la mano (radianes de los 6 actuadores: pulgar yaw, pulgar
# pitch, índice, medio, anular, meñique). "preshape" deja el pulgar ya
# opuesto a los dedos para que al cerrar no barra la fruta.
HAND_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
HAND_PRESHAPE = np.array([0.0, 0.0, 0.1, 0.1, 0.1, 0.1])
HAND_CLOSED = np.array([1.2, 0.5, 1.4, 1.4, 1.4, 1.4])
# Manos del modelo (env H12_MANOS):
#   "rh56"    las RH56DFTP OFICIALES del H1-2 del laboratorio (h1_2_utec,
#             h1_2_inspire_description; escena_faja las injerta en la escena):
#             juntas {lado}_thumb_swing / thumb_1 / index_1 ... little_1,
#             q = 0 abierta, máx = cerrada (1.2 / 0.624 / 1.517), mano de
#             0.84 kg montada a x = 0.161 de la muñeca.
#   "vieja"   las de h1_2_description (L_/R_ *_proximal_joint), con las que se
#             hizo todo hasta 2026-11-10.
MANOS = os.environ.get("H12_MANOS", "rh56")
if MANOS == "rh56":
    HAND_ACTUATORS = ["thumb_swing_joint", "thumb_1_joint", "index_1_joint", "middle_1_joint",
                      "ring_1_joint", "little_1_joint"]
    HAND_MAX = np.array([1.2, 0.624, 1.517, 1.517, 1.517, 1.517])
else:
    HAND_ACTUATORS = ["thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
                      "index_proximal_joint", "middle_proximal_joint",
                      "ring_proximal_joint", "pinky_proximal_joint"]
    HAND_MAX = np.array([1.3, 0.6, 1.7, 1.7, 1.7, 1.7])


def hand_actuator_names(side):
    if MANOS == "rh56":
        return [f"{side}_{n}" for n in HAND_ACTUATORS]
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


# Pose de "mostrar" para verificar el agarre con la cámara de la cabeza:
# palma ARRIBA (agarre girado 180° sobre el eje de los dedos), fruta a 26 cm
# sobre la faja y hacia afuera. Con la mano cerrada en la orientación del
# agarre (o en home) los dedos tapan la fruta: medido 0-13% del área visible,
# y la verificación daba falsos negativos con la fruta en la mano. Palma
# arriba: 55-64% visible y 1.3-1.5 mm de error de la esfera, aunque la fruta
# haya quedado ±3 cm corrida en la mano (barrido en docs/LFD_DMP.md §4.2).
SHOW_FRUIT = {"left": np.array([0.36 - FAJA_DX, 0.18, 0.10 + DZ_MESA]),
              "right": np.array([0.36 - FAJA_DX, -0.18, 0.10 + DZ_MESA])}


def show_wrist_target(side, fruit_torso=None):
    """[x y z qw qx qy qz] de la muñeca con la fruta en la pose de mostrar."""
    R = grasp_rotation(side, elev=0.0) @ np.diag([1.0, -1.0, -1.0])    # Rx(180°)
    fruit_torso = SHOW_FRUIT[side] if fruit_torso is None else fruit_torso
    return grasp_wrist_target(side, fruit_torso, R)


def solve_arm_ik(side, xd, q0=None):
    lo, hi = JL.get_limits(side)
    q0 = JL.center(side) if q0 is None else q0
    r = ik_solve_limited(FK[side], TF2xyzquat, q0, xd, lo, hi, restarts=4, max_iters=200)
    return r["q"], bool(r["ok"])


def wrist_to_fruit(side, q):
    """Dónde quedaría el centro de la fruta con la muñeca en q (FK)."""
    T = FK[side](q)
    return T[0:3, 3] + T[0:3, 0:3] @ GRASP_OFFSET[side]
