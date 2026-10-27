"""Cinemática de los brazos para el reto: la de la tesis (`h1_2_algoritms`), sin ROS ni MuJoCo.

Qué se reutiliza de la tesis (importado tal cual, no copiado):
  * `fkine_arm_{left,right}_unitree` y `arm_frames`: cadena DH 7R hasta `*_wrist_yaw_link`
    en torso_link. Verificada contra el MJCF del simulador (h1_2_escena_suelo.xml): error
    < 1e-6 m y 0.000 grados en 200 posturas aleatorias por brazo (tests/test_cinematica.py).
    El reto pide comprobar el marco y el efector del IK, no suponerlos (sec. 3).
  * `collision_model.CAPSULES` y `TORSO_CAPSULE`: cápsulas de los eslabones.
  * `grasp_geometry.grasp_rotation`: orientación de la muñeca "palma abajo" / "de lado".
  * `joint_limits`: límites del URDF (muñeca en cabeceo ±0.4625 rad = ±26.5 grados) y el
    tope de 0.10 rad del alabeo del hombro hacia dentro (malla convexa del torso).

Lo nuevo: la IK usa el jacobiano GEOMÉTRICO de `arm_frames` (columna i = z_{i-1} x
(p - o_{i-1}), z_{i-1}), una FK por iteración en vez de las 8 del jacobiano numérico de
`ik_functions`: el mapa de alcance resuelve decenas de miles de IK. Mismo método que
`ik_dls_step_limited` (mínimos cuadrados amortiguados con límites).

Lados: 'izq' / 'der' (como h1_2_juntas). Marco: torso_link.
"""
import os
import sys

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))


def _ruta_tesis():
    """El código FUENTE de h1_2_algoritms (no la copia de humanoid_ws/install, que puede
    estar desfasada). tareas/ es un enlace al repo: se prueba la ruta del repo y la de
    humanoid_ws/src."""
    for c in (os.path.normpath(os.path.join(_AQUI, "..", "..", "..", "h1_2_algoritms")),
              os.path.expanduser("~/Github/H1_2_Humanoid_Control/h1_2_algoritms"),
              os.path.expanduser("~/humanoid_ws/src/h1_2_algoritms")):
        if os.path.isfile(os.path.join(c, "h1_2_algoritms", "movimiento", "fk_functions.py")):
            return c
    raise ImportError("no encuentro el código de la tesis (h1_2_algoritms)")


_T = _ruta_tesis()
if _T not in sys.path:
    sys.path.insert(0, _T)
for _m in [k for k in sys.modules if k == "h1_2_algoritms" or k.startswith("h1_2_algoritms.")]:
    if not (getattr(sys.modules[_m], "__file__", None) or "").startswith(_T):
        del sys.modules[_m]           # una copia instalada cargada antes: fuera

from h1_2_algoritms.bimanual import collision_model as CM  # noqa: E402
from h1_2_algoritms.movimiento import joint_limits as JL  # noqa: E402
from h1_2_algoritms.movimiento.fk_functions import (  # noqa: E402
    fkine_arm_left_unitree, fkine_arm_right_unitree)
from h1_2_algoritms.vision import grasp_geometry as GG  # noqa: E402

LADOS = ("izq", "der")
INGLES = {"izq": "left", "der": "right"}
FK = {"izq": fkine_arm_left_unitree, "der": fkine_arm_right_unitree}
NOMBRES_JUNTAS = {lado: JL.ARM_JOINT_NAMES[INGLES[lado]] for lado in LADOS}

# Puntos de la mano en el marco de la muñeca (*_wrist_yaw_link): x a lo largo de los
# dedos, y normal de la palma (la palma mira a -y en la izquierda y a +y en la derecha),
# z hacia el pulgar. Medidos sobre las mallas del MJCF con los dedos estirados.
PUNTOS_MANO = {
    # centro del agarre de potencia (tesis, GRASP_OFFSET): delante de la palma
    "palma": {"izq": np.array(GG.GRASP_OFFSET["left"]), "der": np.array(GG.GRASP_OFFSET["right"])},
    # yema del índice estirado (punta de la malla en x = 0.268, z = 0.029)
    "indice": {"izq": np.array([0.262, -0.006, 0.029]), "der": np.array([0.262, 0.006, 0.029])},
    "muneca": {"izq": np.zeros(3), "der": np.zeros(3)},
    # yema del índice algo flexionado (900): el que toca las dianas (nivel 1). Doblado, la mesa
    # lo empuja hacia abrirse y su motor lo nota (FORCE_ACT); estirado del todo apoya en el tope
    "indice_toque": {"izq": np.array([0.264, -0.024, 0.029]), "der": np.array([0.264, 0.024, 0.029])},
}
# Dorso de la mano (marcador ArUco): centro y normal en el marco de la muñeca. La malla de
# la base de la mano llega a |y| = 0.0274 en la zona de la palma.
DORSO = {"izq": (np.array([0.11, 0.029, 0.0]), np.array([0.0, 1.0, 0.0])),
         "der": (np.array([0.11, -0.029, 0.0]), np.array([0.0, -1.0, 0.0]))}


def limites(lado, alabeo_hombro_dentro=0.10):
    """(q_min, q_max) del URDF con el alabeo del hombro hacia dentro limitado (tesis:
    con más de ~0.15 rad la malla convexa del hombro entra en la del torso)."""
    lo, hi = JL.get_limits(INGLES[lado])
    if alabeo_hombro_dentro is not None:
        if lado == "izq":
            lo[1] = max(lo[1], -alabeo_hombro_dentro)
        else:
            hi[1] = min(hi[1], alabeo_hombro_dentro)
    return lo, hi


def fk(lado, q):
    """T (4x4) de la muñeca en torso_link."""
    return FK[lado](np.asarray(q, dtype=float))


def frames(lado, q):
    return CM.arm_frames(INGLES[lado], np.asarray(q, dtype=float))


def punto(lado, q, nombre="palma"):
    T = fk(lado, q)
    return T[:3, 3] + T[:3, :3] @ PUNTOS_MANO[nombre][lado]


def jacobiano(fr, p):
    """Jacobiano geométrico 6x7 del punto p (torso_link) rígido a la muñeca."""
    F = np.asarray(fr[:7])
    Z, Org = F[:, :3, 2], F[:, :3, 3]
    J = np.empty((6, 7))
    J[:3] = np.cross(Z, p - Org).T
    J[3:] = Z.T
    return J


def log_rot(R):
    """Vector de rotación (eje * ángulo) de R."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(ang)
    if s < 1e-6:
        if ang < 1e-6:
            return 0.5 * v
        # ~180 grados: eje de la diagonal
        w = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(w))
        eje = R[:, k] + np.eye(3)[:, k]
        return ang * eje / (np.linalg.norm(eje) + 1e-12)
    return ang / (2.0 * s) * v


def punto_agarre(lado, prof_palma, avance=0.17, lateral=0.0):
    """Punto de agarre (centro del objeto) en el marco de la muñeca: `avance` a lo largo de los
    dedos, `prof_palma` por delante de la palma y `lateral` hacia el pulgar. La tesis usaba
    (0.17, 0.058, 0) para una mandarina de 4 cm de radio; con el pulgar opuesto (rotación 0)
    pulgar y dedos se encuentran en ~(0.21, -0.06, +0.015) (mallas del MJCF)."""
    return np.array([avance, -prof_palma if lado == "izq" else prof_palma, lateral])


def ik(lado, p_des, R_des=None, q0=None, nombre="palma", lim=None, iters=150, lamb=0.02,
       peso_ori=0.3, paso_max=0.25, tol_pos=0.002, tol_ori_deg=2.0, off=None):
    """IK de un punto de la mano (PUNTOS_MANO[nombre], u `off` en el marco de la muñeca) a
    p_des con la muñeca en R_des (None = solo posición). Mínimos cuadrados amortiguados con
    límites (como ik_dls_step_limited de la tesis). Devuelve (q, err_pos m, err_ori grados, ok)."""
    lo, hi = lim if lim is not None else limites(lado)
    q = np.clip(np.asarray(q0 if q0 is not None else postura_inicial(lado), dtype=float), lo, hi)
    off = PUNTOS_MANO[nombre][lado] if off is None else np.asarray(off, dtype=float)
    p_des = np.asarray(p_des, dtype=float)
    W = np.array([1, 1, 1, peso_ori, peso_ori, peso_ori]) if R_des is not None else np.array([1, 1, 1, 0, 0, 0.0])
    ep, eo = np.inf, 0.0
    mejor, sin_mejora = np.inf, 0
    for _ in range(iters):
        fr = frames(lado, q)
        T = fr[7]
        p = T[:3, 3] + T[:3, :3] @ off
        e = np.zeros(6)
        e[:3] = p_des - p
        if R_des is not None:
            e[3:] = log_rot(R_des @ T[:3, :3].T)
        ep, eo = float(np.linalg.norm(e[:3])), float(np.degrees(np.linalg.norm(e[3:])))
        if ep < tol_pos and (R_des is None or eo < tol_ori_deg):
            return q, ep, eo, True
        coste = ep + 0.01 * eo           # 1 cm ~ 1 grado
        if coste < 0.99 * mejor:
            mejor, sin_mejora = coste, 0
        else:
            sin_mejora += 1
            if sin_mejora >= 12:         # atascado (fuera de alcance o contra un límite)
                break
        J = jacobiano(fr, p) * W[:, None]
        ew = e * W
        dq = J.T @ np.linalg.solve(J @ J.T + lamb ** 2 * np.eye(6), ew)
        n = np.linalg.norm(dq)
        if n > paso_max:
            dq *= paso_max / n
        q = np.clip(q + dq, lo, hi)
    return q, ep, eo, (ep < tol_pos and (R_des is None or eo < tol_ori_deg))


def postura_inicial(lado):
    """Semilla de la IK: brazo hacia delante y abajo, codo doblado (mano sobre la mesa)."""
    s = 1.0 if lado == "izq" else -1.0
    return np.array([-0.5, 0.1 * s, 0.0, 0.9, 0.0, 0.0, 0.0])


def rotacion_agarre(lado, modo="arriba", elev_deg=0.0, az_deg=0.0):
    """R de la muñeca en torso_link: x = dirección de los dedos con elevación `elev_deg`
    (negativa = hacia abajo) y azimut `az_deg` (0 = adelante, positivo hacia la izquierda).
    modo 'arriba': palma hacia abajo (la de la tesis). modo 'lado': palma VERTICAL, pulgar
    arriba; con azimut 0 mira a la línea media del robot y el azimut la gira alrededor de la
    vertical. (La 'side' de la tesis proyecta una dirección fija sobre el plano normal a los
    dedos: con elevación y azimut a la vez la palma se inclina y la mano rueda, y entonces el
    pulgar acaba encima del objeto.)"""
    if modo == "arriba":
        return GG.grasp_rotation(INGLES[lado], np.radians(elev_deg), np.radians(az_deg), "top")
    R0 = GG.grasp_rotation(INGLES[lado], np.radians(elev_deg), 0.0, "side")
    c, s = np.cos(np.radians(az_deg)), np.sin(np.radians(az_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R0


def capsulas(lado, q, fr=None):
    """Cápsulas del brazo (tesis) en torso_link: [{name, a, b, r}]."""
    return CM.arm_capsules(INGLES[lado], q, fr if fr is not None else frames(lado, q))


def dist_torso(lado, q, fr=None):
    """Distancia mínima (m, superficie a superficie) de antebrazo y mano al torso."""
    _, ta, tb, tr = CM.TORSO_CAPSULE
    d = np.inf
    for c in capsulas(lado, q, fr):
        if c["name"] in CM.TORSO_PAIRS:
            pa, pb = CM.closest_points_segments(c["a"], c["b"], ta, tb)
            d = min(d, float(np.linalg.norm(pa - pb)) - c["r"] - tr)
    return d


def holgura_plano(lado, q, z_plano, x_min, y_max, fr=None, muestras=6):
    """Holgura vertical (m) de las cápsulas del brazo sobre un plano horizontal z = z_plano
    (torso_link) en la región x >= x_min, |y| <= y_max (la mesa). Negativa = el brazo
    atraviesa la mesa. Fuera de la región, inf."""
    h = np.inf
    for c in capsulas(lado, q, fr):
        for s in np.linspace(0.0, 1.0, muestras):
            p = c["a"] + s * (c["b"] - c["a"])
            if p[0] + c["r"] >= x_min and abs(p[1]) - c["r"] <= y_max:
                h = min(h, float(p[2] - c["r"] - z_plano))
    return h


# puntos de la mano en el marco de la muñeca (palma, dedos estirados, pulgar abierto): para la
# altura de seguridad sobre la mesa (los dedos y el pulgar sobresalen de la cápsula de la mano)
_PUNTOS_MANO_BAJA = [np.array([x, 0.0, z]) for x in (0.03, 0.10, 0.165, 0.268) for z in (-0.04, 0.0, 0.04)] + \
    [np.array([0.17, 0.0, 0.11])]


def mano_baja(lado, q, tablero, margen=0.03):
    """z mínima (torso_link) de la mano sobre el tablero (Rect en x, y), o inf si no está encima."""
    T = fk(lado, q)
    s = 1.0 if lado == "izq" else -1.0
    pts = _PUNTOS_MANO_BAJA + [np.array([0.17, -0.058 * s, 0.0])]
    W = np.array([T[:3, 3] + T[:3, :3] @ p for p in pts])
    sobre = tablero.contiene(W[:, 0], W[:, 1], -margen)
    return float(W[sobre, 2].min()) if sobre.any() else np.inf
