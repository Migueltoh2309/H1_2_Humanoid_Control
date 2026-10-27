"""Geometría de la estación (sección 4) a partir de config/estacion.yaml.

Todo en el marco "suelo" (origen en el suelo bajo la pelvis, x adelante, y izquierda,
z arriba). `a_torso()` pasa al marco de torso_link, que es el de la cinemática de la tesis
y el de la salida de la percepción: es el mismo desplazado `torso_z` hacia arriba (el
robot está de pie y no gira; la inclinación de ~1 grado del torso la absorbe la
calibración cámara-torso, que se mide en torso_link).
"""
from dataclasses import dataclass

import numpy as np

LETRAS = ("A", "B", "C", "D")


@dataclass
class Rect:
    """Rectángulo horizontal: centro (x, y) y medidas (fondo en x, ancho en y)."""
    cx: float
    cy: float
    fondo: float
    ancho: float

    def contiene(self, x, y, margen=0.0):
        x, y = np.asarray(x), np.asarray(y)
        return (np.abs(x - self.cx) <= self.fondo / 2 - margen) & (np.abs(y - self.cy) <= self.ancho / 2 - margen)

    def esquinas(self):
        fx, fy = self.fondo / 2, self.ancho / 2
        return np.array([[self.cx - fx, self.cy + fy], [self.cx + fx, self.cy + fy],
                         [self.cx + fx, self.cy - fy], [self.cx - fx, self.cy - fy]])


class Estacion:
    def __init__(self, cfg, d=None):
        self.cfg = cfg
        r, me = cfg["robot"], cfg["mesa"]
        self.pecho_x = float(r["pecho_x"])
        self.torso_z = float(r["torso_z"])
        self.d = float(me["d"] if d is None else d)
        self.z_mesa = float(me["altura"])
        self.grosor = float(me["grosor"])
        self.x_borde = self.pecho_x + self.d                      # borde cercano del tablero
        self.tablero = Rect(self.x_borde + me["fondo"] / 2, 0.0, me["fondo"], me["ancho"])
        z = cfg["zona_recogida"]
        self.zona = Rect(self.x_borde + z["centro_desde_borde"], 0.0, z["fondo"], z["ancho"])
        b = cfg["bandejas"]
        self.bandeja_lado, self.bandeja_borde = float(b["lado"]), float(b["borde"])
        self.bandejas = {}
        for letra in LETRAS:
            fila, lado = b["posiciones"][letra]
            x = self.x_borde + (b["x_cerca"] if fila == "cerca" else b["x_lejos"])
            y = b["y"] if lado == "izq" else -b["y"]
            self.bandejas[letra] = Rect(x, y, b["lado"], b["lado"])
        a = cfg["aruco"]
        self.aruco_lado = float(a["lado"])
        m_ = float(a["margen_esquina"])
        t = self.tablero
        x0, x1 = t.cx - t.fondo / 2 + m_, t.cx + t.fondo / 2 - m_
        y0, y1 = t.cy + t.ancho / 2 - m_, t.cy - t.ancho / 2 + m_
        # 0 cerca-izq, 1 lejos-izq, 2 lejos-der, 3 cerca-der
        self.aruco_esquinas = dict(zip(a["esquinas"], [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    def lado_bandeja(self, letra):
        """'izq' o 'der': de qué lado del robot está la bandeja."""
        return "izq" if self.bandejas[letra].cy > 0 else "der"

    def a_torso(self, p):
        """Punto(s) del marco suelo al de torso_link."""
        p = np.array(p, dtype=float)
        p[..., 2] -= self.torso_z
        return p

    def a_suelo(self, p):
        p = np.array(p, dtype=float)
        p[..., 2] += self.torso_z
        return p

    def resumen(self):
        return {
            "d": self.d, "x_borde": self.x_borde, "z_mesa": self.z_mesa, "torso_z": self.torso_z,
            "tablero": vars(self.tablero), "zona": vars(self.zona),
            "bandejas": {k: vars(v) for k, v in self.bandejas.items()},
            "aruco_esquinas": {int(k): list(v) for k, v in self.aruco_esquinas.items()},
        }
