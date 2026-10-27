"""Demostraciones para la DMP de alcance: de las trazas del lazo cerrado
(`demos/visual_servoing_grasp.py`, un CSV por ensayo) a trayectorias
relativas a la meta.

Una demostración es el tramo de ALCANCE del brazo que agarra: desde que la
máquina elige el brazo (APPROACH) hasta que empieza a cerrar (CLOSE). La
meta g(t) es la pose de muñeca de agarre sobre la fruta en ese instante
(verdad de terreno de la simulación), así que y(t) = x(t) − g(t) es el
movimiento de la mano VISTO DESDE LA FRUTA, también con la faja en marcha.

Todo se expresa en la convención del brazo DERECHO: la izquierda se refleja
en y (los dos agarres son simétricos, grasp_geometry), y así las demos de
los dos brazos alimentan la misma primitiva.

Estas son demostraciones "robot" (las genera la máquina de estados): sirven
para montar y validar el pipeline antes de tener las humanas
(Multiview_Demo_Capture), que entrarán por la misma interfaz (t, y).
"""
import csv

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from h1_2_algoritms.vision import grasp_geometry as GG

MIRROR = np.array([1.0, -1.0, 1.0])


def to_right(side, v):
    """Lleva un vector de posición/velocidad a la convención del brazo
    derecho (y vuelve: la reflexión es su propia inversa)."""
    return np.asarray(v) * MIRROR if side == "left" else np.asarray(v)


def read_trace(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def reach_segment(rows):
    """(side, t, x_muñeca, fruta) del tramo APPROACH..CLOSE (incluido el
    primer instante de CLOSE), o None si el ensayo no llegó a cerrar."""
    side = next((r["side"] for r in rows if r["side"]), None)
    if side is None or not any(r["phase"] == "CLOSE" for r in rows):
        return None
    i0 = next(i for i, r in enumerate(rows) if r["phase"] in ("APPROACH", "REACH"))
    i1 = next(i for i, r in enumerate(rows) if r["phase"] == "CLOSE")
    seg = rows[i0:i1 + 1]
    k = side[0]
    t = np.array([float(r["t"]) for r in seg])
    x = np.array([[float(r[f"wr_{k}_{a}"]) for a in "xyz"] for r in seg])
    fruit = np.array([[float(r[f"gt_{a}"]) for a in "xyz"] for r in seg])
    return side, t - t[0], x, fruit


def demo_from_trace(path, dt=0.01, smooth=7, goal="moving"):
    """Demostración (t, y, ẏ, ÿ) a `dt`, relativa a la meta y en la
    convención derecha. Las trazas van a 20 Hz y la máquina de estados
    cambia de meta de golpe (APPROACH -> DESCEND): se suaviza con
    Savitzky-Golay antes de derivar, y se interpola con un spline cúbico.

    goal="moving" (por omisión): la fruta en cada instante, la meta móvil
    de GraspServoing. goal="intercept": el agarre sobre la fruta donde
    ESTABA al cerrar (fija durante todo el alcance), para la opción
    `reach_intercept`. Con la fruta quieta son iguales."""
    seg = reach_segment(read_trace(path))
    if seg is None:
        return None
    side, t, x, fruit = seg
    if goal == "intercept":
        fruit = np.repeat(fruit[-1:], len(fruit), axis=0)
    g = np.array([GG.grasp_wrist_target(side, p)[0:3] for p in fruit])
    y = to_right(side, x - g)
    if smooth and len(t) > smooth:
        y = savgol_filter(y, smooth, 3, axis=0)
    sp = CubicSpline(t, y, axis=0)
    tt = np.arange(0.0, t[-1] + 1e-9, dt)
    return tt, sp(tt), sp(tt, 1), sp(tt, 2)
