"""Geometria de la camara sobre el robot (reto, 6.1): pixel <-> punto del suelo.

Marcos:
  robot   origen en el suelo bajo la pelvis; x adelante, y izquierda, z arriba.
          Gira con el rumbo del robot; NO se inclina con el balanceo.
  camara  optico de ROS: x derecha, y abajo, z adelante (hacia donde mira).

Montaje (constante, se MIDE con calibrar_camara.py): centro optico en
(x, y, altura) del robot, inclinado `inclinacion` rad bajo la horizontal y con
`alabeo` rad de giro sobre su eje optico. Columnas de R_montaje = ejes de la
camara expresados en el robot:
  z_c = ( cos(inc), 0, -sin(inc))     mira adelante y abajo
  x_c = ( 0, -1, 0 )                  derecha de la imagen = -y del robot
  y_c = (-sin(inc), 0, -cos(inc))     abajo de la imagen

Balanceo (6.1.4 / 6.4.1): la camara va en la cabeza, rigida con el torso, que
es donde Unitree pone la IMU. Con compensacion, cada fotograma usa
R = R_cuerpo(d_roll, d_pitch) * R_montaje, girando el conjunto alrededor de un
pivote (la pelvis), donde d_* = IMU - IMU de referencia (la de la calibracion).

Un punto del suelo (X, Y, 0) se ve en la imagen con la homografia
  H = K [ r1  r2  -R^T C ]   (r1, r2 = columnas 1 y 2 de R^T)
y la vista cenital (BEV) se obtiene con warpPerspective(ir, H*A, WARP_INVERSE_MAP),
con A la rejilla del suelo (metros <- pixeles de la vista cenital).
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class ModeloCamara:
    fx: float
    fy: float
    cx: float
    cy: float
    ancho: int = 640
    alto: int = 480
    altura: float = 1.703
    inclinacion: float = 0.8859
    alabeo: float = 0.0
    x: float = 0.111
    y: float = 0.0
    pivote_z: float = 1.015          # altura de la pelvis: centro del balanceo

    @classmethod
    def desde_config(cls, cfg, K=None, ancho=640, alto=480):
        c = cfg["camara"]
        fx, fy, cx, cy = K if K is not None else (c["fx"], c["fy"], c["cx"], c["cy"])
        return cls(fx, fy, cx, cy, ancho, alto, c["altura"], c["inclinacion"], c.get("alabeo", 0.0),
                   c.get("x", 0.0), c.get("y", 0.0), c.get("pivote_z", 1.015))

    # ------------------------------------------------------------ extrinsecos
    def R_montaje(self):
        th = self.inclinacion
        R0 = np.array([[0.0, -math.sin(th), math.cos(th)],
                       [-1.0, 0.0, 0.0],
                       [0.0, -math.cos(th), -math.sin(th)]])     # columnas x_c, y_c, z_c
        return R0 @ rot_z(self.alabeo)

    def pose(self, d_roll=0.0, d_pitch=0.0):
        """(R camara->robot, centro optico en el robot) con el cuerpo inclinado."""
        Rc = rot_y(d_pitch) @ rot_x(d_roll)
        piv = np.array([0.0, 0.0, self.pivote_z])
        C0 = np.array([self.x, self.y, self.altura])
        return Rc @ self.R_montaje(), Rc @ (C0 - piv) + piv

    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    # ------------------------------------------------------------ proyecciones
    def pixel_a_suelo(self, u, v, d_roll=0.0, d_pitch=0.0):
        """Puntos del suelo (N x 2) para pixeles (arrays); NaN si el rayo no baja."""
        R, C = self.pose(d_roll, d_pitch)
        u = np.atleast_1d(np.asarray(u, float))
        v = np.atleast_1d(np.asarray(v, float))
        rayos = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], 1) @ R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(rayos[:, 2] < -1e-6, -C[2] / rayos[:, 2], np.nan)
        return C[:2] + rayos[:, :2] * t[:, None]

    def suelo_a_pixel(self, X, Y, d_roll=0.0, d_pitch=0.0):
        H = self.homografia(d_roll, d_pitch)
        p = np.stack([np.asarray(X, float), np.asarray(Y, float), np.ones(np.shape(X))], -1) @ H.T
        return p[..., :2] / p[..., 2:3]

    def homografia(self, d_roll=0.0, d_pitch=0.0):
        """H: (X, Y, 1) del suelo -> pixel homogeneo."""
        R, C = self.pose(d_roll, d_pitch)
        Rt = R.T
        return self.K() @ np.column_stack([Rt[:, 0], Rt[:, 1], -Rt @ C])

    # ------------------------------------------------------------ 6.1: respuestas
    def fov_vertical(self):
        return 2 * math.atan(self.alto / 2 / self.fy)

    def fov_horizontal(self):
        return 2 * math.atan(self.ancho / 2 / self.fx)

    def alcance_suelo(self):
        """(distancia del robot al primer suelo visible, ultima distancia visible)
        en el eje de la imagen. inf si el borde superior mira por encima del horizonte."""
        sup = self.inclinacion - math.atan((self.cy) / self.fy)                # rayo del borde superior
        inf = self.inclinacion + math.atan((self.alto - self.cy) / self.fy)    # borde inferior
        cerca = self.x + self.altura / math.tan(inf) if inf < math.pi / 2 else self.x - self.altura / math.tan(math.pi - inf)
        lejos = self.x + self.altura / math.tan(sup) if sup > 0 else math.inf
        return cerca, lejos


@dataclass
class RejillaBEV:
    """Vista cenital del suelo en el marco del robot. Fila 0 = lo mas lejano,
    columna 0 = lo mas a la izquierda (y maximo)."""
    x0: float
    x1: float
    y0: float
    y1: float
    res: float

    @property
    def forma(self):
        return int(round((self.x1 - self.x0) / self.res)), int(round((self.y1 - self.y0) / self.res))

    def A(self):
        """(X, Y, 1) <- (col, fila, 1)."""
        return np.array([[0.0, -self.res, self.x1], [-self.res, 0.0, self.y1], [0.0, 0.0, 1.0]])

    def a_metros(self, col, fila):
        return self.x1 - np.asarray(fila) * self.res, self.y1 - np.asarray(col) * self.res

    def a_pixel(self, X, Y):
        return (self.y1 - np.asarray(Y)) / self.res, (self.x1 - np.asarray(X)) / self.res

    def fila_de_x(self, X):
        return int(round((self.x1 - X) / self.res))


def matriz_bev(cam: ModeloCamara, rej: RejillaBEV, d_roll=0.0, d_pitch=0.0):
    """M para cv2.warpPerspective(ir, M, (ancho, alto), flags=WARP_INVERSE_MAP | INTER_LINEAR)."""
    return cam.homografia(d_roll, d_pitch) @ rej.A()
