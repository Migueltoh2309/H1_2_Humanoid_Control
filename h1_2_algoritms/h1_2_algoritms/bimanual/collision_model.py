"""Modelo geométrico de colisión de los brazos del H1-2 para el QP bimanual.

Cada eslabón se aproxima con una **cápsula** (segmento + radio), que es la
primitiva estándar para evasión en tiempo real: la distancia entre dos
cápsulas es la distancia entre dos segmentos menos los radios, y su
gradiente respecto a q sale en forma cerrada con el jacobiano del punto más
cercano. Todo se expresa en `torso_link`, que es el mismo frame en el que
trabajan `fkine_arm_*_unitree` (verificado contra MuJoCo: el frame 7 de DH
coincide con `*_wrist_yaw_link` al 1e-4 m) y del que cuelga la cámara RGB-D
del torso, así que brazos, torso y nube de puntos comparten frame sin TF
intermedio.

Cápsulas (extremos fijos en un frame de DH, radios medidos sobre los
vértices de las mallas de colisión del MJCF; ver `CAPSULES`):

    upper  hombro -> codo          frame 3, r = 0.056 m
    fore   codo -> muñeca (pitch)  frame 4, r = 0.060 m
    hand   muñeca -> punta mano    frame 7, r = 0.046 m

más una cápsula fija para el torso (`TORSO_CAPSULE`).

Restricción que se añade al QP (velocity damper, Faverjon & Tournassoud
1987; Kanehiro et al. 2008): para cada par (a, b) con distancia d < d_i,

    d_dot = n^T (J_a dq_a - J_b dq_b) >= -xi * (d - d_s) / (d_i - d_s)

con n el unitario de b hacia a entre los puntos más cercanos. Lejos del
obstáculo (d >= d_i) no hay restricción; al acercarse la velocidad de
aproximación permitida baja linealmente hasta ser 0 en d = d_s, así que la
distancia nunca cae por debajo de d_s (en tiempo continuo). Si ya está por
dentro (d < d_s) el lado derecho es positivo: obliga a alejarse.
"""
import numpy as np

from h1_2_algoritms.movimiento.fk_functions import dh

# =====================================================================
# Cadena DH con frames intermedios
# =====================================================================
# Misma tabla que fkine_arm_{left,right}_unitree (fk_functions.py); aquí
# se devuelven los 8 frames (base + uno por articulación) porque el
# jacobiano de un punto cualquiera del brazo los necesita. La única
# diferencia entre lados es el signo del offset de 15° del hombro y la
# base; test_collision_model.py verifica que el frame 7 sea idéntico al de
# fk_functions.

_TBASE = {
    "left": np.array([
        [0.0, -1.0, 0.0, 0.0],
        [0.25881905, 0.0, 0.96592583, 0.20794643],
        [-0.96592583, 0.0, 0.25881905, 0.43937656],
        [0.0, 0.0, 0.0, 1.0],
    ]),
    "right": np.array([
        [0.0, -1.0, 0.0, 0.0],
        [-0.25881905, 0.0, 0.96592583, -0.20794643],
        [-0.96592583, 0.0, -0.25881905, 0.43937656],
        [0.0, 0.0, 0.0, 1.0],
    ]),
}
_SHOULDER_OFFSET = {"left": np.pi / 2.0 + 0.2618, "right": np.pi / 2.0 - 0.2618}


def arm_frames(side, q):
    """Lista de 8 transformaciones homogéneas en torso_link: T[0] es la base
    y T[i] el frame tras la articulación i. La articulación i gira sobre el
    eje z de T[i-1] y pasa por su origen (convención DH estándar)."""
    rows = [
        (0.0, q[0], 0.0060011, np.pi / 2.0),
        (0.0, q[1] - _SHOULDER_OFFSET[side], 0.0, np.pi / 2.0),
        (-0.3276, q[2] + np.pi / 2.0, 0.006, -np.pi / 2.0),
        (0.0, q[3] + np.pi / 2.0, 0.011, np.pi / 2.0),
        (0.208, q[4] + np.pi, 0.0, np.pi / 2.0),
        (0.0, q[5] + np.pi / 2.0, 0.020, np.pi / 2.0),
        (0.0, q[6], 0.0, 0.0),
    ]
    T = [_TBASE[side]]
    for r in rows:
        T.append(T[-1] @ dh(*r))
    return T


def point_jacobian(frames, k, p):
    """Jacobiano de posición (3x7) de un punto p (en torso_link) rígido al
    eslabón k: columna i = z_{i-1} x (p - o_{i-1}) para i <= k, 0 después."""
    J = np.zeros((3, 7))
    for i in range(1, k + 1):
        z = frames[i - 1][0:3, 2]
        o = frames[i - 1][0:3, 3]
        J[:, i - 1] = np.cross(z, p - o)
    return J


# =====================================================================
# Cápsulas
# =====================================================================
# (nombre, frame k, extremo a local, extremo b local, radio). Los extremos
# están en coordenadas del frame k, así que la cápsula se mueve con las
# articulaciones 1..k. Radios = distancia máxima de los vértices de la malla
# de colisión al segmento (calibrado con MuJoCo, h1_2_scene_surgery_table.xml):
#   upper: shoulder_roll + shoulder_yaw  -> máx 0.0562
#   fore:  elbow + wrist_roll + wrist_pitch -> máx 0.0622 (bulto del codo;
#          0.060 cubre el 99.9%)
#   hand:  wrist_yaw + base de la mano   -> máx 0.0456
CAPSULES = [
    ("upper", 3, np.array([-0.006, -0.3276, 0.0]), np.array([0.0, 0.0, 0.0]), 0.056),
    ("fore", 4, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.208]), 0.060),
    ("hand", 7, np.array([0.03, 0.0, 0.0]), np.array([0.165, 0.0, 0.0]), 0.046),
]

# Dedos de la mano Inspire (escena h1_2_scene_surgery_table_hands.xml):
# desde los nudillos (x ≈ 0.165) hasta las puntas extendidas (x ≈ 0.245)
# del frame 7; radio = semiancho de la mano (±0.043) con margen. Solo se
# usa para el AUTO-FILTRADO de la nube (sin esto la cámara ve los dedos
# como un obstáculo pegado a la mano y el damper congela el brazo). No
# entra en las restricciones: al agarrar, los dedos DEBEN tocar la fruta y
# acercarse a la faja.
FINGER_CAPSULE = ("fingers", 7, np.array([0.165, 0.0, 0.0]), np.array([0.245, 0.0, 0.0]), 0.05)
# Pulgar ABIERTO (preforma de aproximación, yaw = 0): sale de la base
# (x 0.116, z 0.025) hacia el costado, punta en x ≈ 0.165, z ≈ 0.105. Sin
# esta cápsula la cámara lo ve como obstáculo a ~4 cm de la mano y el
# damper empuja la mano hacia atrás (medido: la aproximación no converge).
THUMB_CAPSULE = ("thumb", 7, np.array([0.11, 0.0, 0.02]), np.array([0.17, 0.0, 0.11]), 0.03)

# Torso fijo en torso_link (no depende de q de los brazos). El tramo z
# 0.35-0.45 es más ancho (y = ±0.159) pero ahí están los soportes de los
# hombros, pegados al brazo superior por construcción: se deja fuera a
# propósito para que el brazo superior no "choque" con su propio hombro.
TORSO_CAPSULE = ("torso", np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.30]), 0.11)

# Pares que se vigilan: todas las cápsulas del brazo izquierdo contra las
# del derecho, y las distales (fore, hand) contra el torso. upper-torso se
# omite porque en reposo ya están a pocos cm (el hombro está montado en el
# torso) y ese par no puede separarse moviendo el brazo.
TORSO_PAIRS = ("fore", "hand")


def arm_capsules(side, q, frames=None):
    """Cápsulas del brazo en torso_link: lista de dicts con
    name, k, a, b (extremos en torso_link), r."""
    if frames is None:
        frames = arm_frames(side, q)
    out = []
    for name, k, a_loc, b_loc, r in CAPSULES:
        R = frames[k][0:3, 0:3]
        o = frames[k][0:3, 3]
        out.append({"name": name, "k": k, "a": o + R @ a_loc, "b": o + R @ b_loc, "r": r})
    return out


# =====================================================================
# Distancias
# =====================================================================

def closest_points_segments(p1, q1, p2, q2, eps=1e-12):
    """Puntos más cercanos entre los segmentos [p1,q1] y [p2,q2]
    (Ericson, Real-Time Collision Detection, 5.1.9). Devuelve (c1, c2)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    if a <= eps and e <= eps:
        return p1.copy(), p2.copy()
    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    return p1 + s * d1, p2 + t * d2


def points_to_segment(P, a, b):
    """Distancia de cada fila de P (Nx3) al segmento [a,b] y el punto del
    segmento más cercano a cada una. Vectorizado para la nube de puntos."""
    ab = b - a
    den = ab @ ab
    t = np.zeros(len(P)) if den < 1e-12 else np.clip(((P - a) @ ab) / den, 0.0, 1.0)
    C = a + t[:, None] * ab
    return np.linalg.norm(P - C, axis=1), C


# =====================================================================
# Filas de restricción para el QP
# =====================================================================

def _damper_rhs(d, d_s, d_i, xi):
    return -xi * (d - d_s) / (d_i - d_s)


def self_collision_constraints(frames_l, frames_r, d_s, d_i, xi):
    """Restricciones brazo-brazo y brazo-torso.

    Devuelve (A, lb, info): A es (m x 14) sobre dq = [dq_left, dq_right],
    lb el lado derecho del damper (A dq >= lb) e info una lista con
    (nombre del par, distancia) de TODOS los pares, activos o no, para
    registro/visualización.
    """
    caps = {"left": arm_capsules("left", None, frames_l),
            "right": arm_capsules("right", None, frames_r)}
    frames = {"left": frames_l, "right": frames_r}
    rows, lbs, info = [], [], []

    # ---- brazo izquierdo vs brazo derecho ----
    for ca in caps["left"]:
        for cb in caps["right"]:
            pa, pb = closest_points_segments(ca["a"], ca["b"], cb["a"], cb["b"])
            diff = pa - pb
            dist_axes = np.linalg.norm(diff)
            d = dist_axes - ca["r"] - cb["r"]
            info.append((f"L.{ca['name']}-R.{cb['name']}", d))
            if d >= d_i or dist_axes < 1e-9:
                continue
            n = diff / dist_axes
            Ja = point_jacobian(frames["left"], ca["k"], pa)
            Jb = point_jacobian(frames["right"], cb["k"], pb)
            rows.append(np.hstack((n @ Ja, -(n @ Jb))))
            lbs.append(_damper_rhs(d, d_s, d_i, xi))

    # ---- brazos vs torso (fijo) ----
    _, ta, tb, tr = TORSO_CAPSULE
    for col, side in enumerate(("left", "right")):
        for c in caps[side]:
            if c["name"] not in TORSO_PAIRS:
                continue
            pa, pb = closest_points_segments(c["a"], c["b"], ta, tb)
            diff = pa - pb
            dist_axes = np.linalg.norm(diff)
            d = dist_axes - c["r"] - tr
            info.append((f"{side[0].upper()}.{c['name']}-torso", d))
            if d >= d_i or dist_axes < 1e-9:
                continue
            n = diff / dist_axes
            row = np.zeros(14)
            row[7 * col:7 * col + 7] = n @ point_jacobian(frames[side], c["k"], pa)
            rows.append(row)
            lbs.append(_damper_rhs(d, d_s, d_i, xi))

    A = np.array(rows).reshape(-1, 14)
    return A, np.array(lbs), info


def obstacle_constraints(frames_l, frames_r, points, d_s, d_i, xi, max_per_capsule=1):
    """Restricciones de las cápsulas de ambos brazos contra una nube de
    puntos de obstáculo (Nx3 en torso_link, típicamente de la cámara de
    profundidad). Para cada cápsula se toman los `max_per_capsule` puntos
    más cercanos dentro de d_i (con 1 basta si la nube está voxelizada: el
    damper se reevalúa en cada ciclo con el nuevo punto más cercano).

    Devuelve (A, lb, info) con A (m x 14); info = [(nombre, d_min, punto)].
    """
    rows, lbs, info = [], [], []
    if points is None or len(points) == 0:
        return np.zeros((0, 14)), np.zeros(0), info
    for col, (side, frames) in enumerate((("left", frames_l), ("right", frames_r))):
        for c in arm_capsules(side, None, frames):
            dist_axes, C = points_to_segment(points, c["a"], c["b"])
            d = dist_axes - c["r"]
            order = np.argsort(d)[:max_per_capsule]
            info.append((f"{side[0].upper()}.{c['name']}-env", float(d[order[0]]), points[order[0]]))
            for j in order:
                if d[j] >= d_i or dist_axes[j] < 1e-9:
                    break
                n = (C[j] - points[j]) / dist_axes[j]
                row = np.zeros(14)
                row[7 * col:7 * col + 7] = n @ point_jacobian(frames, c["k"], C[j])
                rows.append(row)
                lbs.append(_damper_rhs(d[j], d_s, d_i, xi))
    return np.array(rows).reshape(-1, 14), np.array(lbs), info


def robot_self_filter(points, frames_l, frames_r, padding):
    """Quita de la nube los puntos que pertenecen al propio robot (la cámara
    del torso VE los brazos: sin esto, cada brazo se tomaría a sí mismo
    como obstáculo y quedaría congelado). Se descarta todo punto a menos de
    r + padding de alguna cápsula de los brazos o del torso; `padding` cubre
    el error del modelo de cápsulas y el desfase entre la imagen y q."""
    if points is None or len(points) == 0:
        return points
    keep = np.ones(len(points), dtype=bool)
    caps = arm_capsules("left", None, frames_l) + arm_capsules("right", None, frames_r)
    for _, k, a_loc, b_loc, r in (FINGER_CAPSULE, THUMB_CAPSULE):
        for frames in (frames_l, frames_r):
            R, o = frames[k][0:3, 0:3], frames[k][0:3, 3]
            caps.append({"a": o + R @ a_loc, "b": o + R @ b_loc, "r": r})
    _, ta, tb, tr = TORSO_CAPSULE
    caps.append({"a": ta, "b": tb, "r": tr})
    for c in caps:
        dist_axes, _ = points_to_segment(points, c["a"], c["b"])
        keep &= dist_axes > c["r"] + padding
    return points[keep]
