"""Interfaces entre los cinco bloques (sec. 5 del reto). Ver INTERFACES.md."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Fotograma:
    """Lo que entrega una fuente (RealSense, ROS 2 del simulador o conjunto de datos)."""
    t: float                         # s, reloj de la fuente (monotonic del PC2 o del PC)
    ir: np.ndarray                   # uint8 HxW, infrarrojo izquierdo
    K: Tuple[float, float, float, float]   # fx, fy, cx, cy del IR
    seq: int = 0
    profundidad: Optional[np.ndarray] = None     # float32 HxW en metros (opcional)
    K_prof: Optional[Tuple[float, float, float, float]] = None


@dataclass
class Imu:
    t: float
    roll: float
    pitch: float
    yaw: float
    gyro: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    motores_ok: bool = True


@dataclass
class MedidaLinea:
    """Salida de Percepcion, en el SUELO y en el marco del robot (x adelante, y izquierda)."""
    t: float
    valida: bool
    desplazamiento: float = 0.0      # m: y de la linea en x = 0 (positivo = linea a la izquierda)
    angulo: float = 0.0              # rad: rumbo de la linea respecto del robot en x = 0
    curvatura: float = 0.0           # 1/m (positivo = gira a la izquierda)
    objetivo: Optional[Tuple[float, float]] = None   # punto de la linea a la distancia de mira
    confianza: float = 0.0           # 0..1
    barra: Optional[float] = None    # m hasta la barra de fin (si se ve)
    esquina: Optional[Tuple[float, int]] = None      # (m hasta la esquina, +1 izquierda / -1 derecha)
    puntos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # (x, y) detectados
    coef: Optional[np.ndarray] = None   # y = c0 + c1 x + c2 x^2
    x_visible: Tuple[float, float] = (0.0, 0.0)
    ms: float = 0.0                  # tiempo de proceso


@dataclass
class EstadoLinea:
    """Salida de Estimacion: la linea respecto del robot, con memoria y dead reckoning."""
    t: float
    valida: bool
    desplazamiento: float = 0.0
    angulo: float = 0.0
    curvatura: float = 0.0
    objetivo: Optional[Tuple[float, float]] = None
    confianza: float = 0.0
    sin_linea_m: float = 0.0         # m recorridos desde la ultima medida buena
    sin_linea_s: float = 0.0
    barra: Optional[float] = None    # m que faltan hasta la barra (estimados tras perderla de vista)
    esquina: Optional[Tuple[float, int]] = None
    recorrido: float = 0.0           # m estimados desde el arranque (sin odometria: vx * t)
    yaw: float = 0.0                 # rumbo de la IMU
    sesgo_yaw: float = 0.0           # rad/s: deriva de rumbo estimada (giroscopo - giro esperado)


@dataclass
class Orden:
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    parar: bool = False              # StopMove en lugar de Move
    motivo: str = ""
