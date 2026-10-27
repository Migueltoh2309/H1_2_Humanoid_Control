"""Tipos que se pasan entre los bloques del sistema (sección 5 del reto). Es la interfaz
escrita ANTES de programar los bloques (INTERFACES.md la explica).

Convenciones: SI (m, rad, s, N; las fuerzas de la mano en gf como sus registros); tiempos
de time.monotonic() del proceso; posiciones y orientaciones en el MARCO DEL TORSO
(torso_link: x adelante, y izquierda, z arriba, origen en la cintura) salvo que el nombre
diga otra cosa; lados 'izq' / 'der'; brazos con 7 juntas en el orden de la tesis
[hombro cabeceo, alabeo, guiñada, codo, muñeca alabeo, cabeceo, guiñada].
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ------------------------------------------------------------------ calibración
@dataclass
class MuestraCalibracion:
    """Una pose quieta del brazo: lo que dicen las juntas y lo que ve la cámara."""
    t: float
    lado: str
    q_medida: np.ndarray            # 7, de rt/lowstate (NO la consigna: el brazo queda ~5 grados corto)
    q_consigna: np.ndarray          # 7, lo que se mandó (para la pregunta 6.1.4)
    marcador_id: int
    esquinas_px: np.ndarray         # 4x2, esquinas del ArUco en la imagen de color
    T_cam_marcador: Optional[np.ndarray] = None   # 4x4 de solvePnP, si se resolvió
    prof_centro: Optional[float] = None           # m, profundidad en el centro del marcador


@dataclass
class Calibracion:
    """Salida del bloque de calibración; se guarda en datos/calibracion/calib_<fecha>.json
    y es la ÚNICA fuente de la pose de la cámara para el resto del sistema."""
    fecha: str
    T_torso_cam: np.ndarray         # 4x4: punto en el marco óptico de color -> torso_link
    T_muneca_marcador: Dict[str, np.ndarray]      # lado -> 4x4 (dónde quedó pegado el ArUco)
    residuo_mm_medio: float
    residuo_mm_max: float
    n_poses: int
    intrinsecos_color: Dict[str, float]           # fx, fy, cx, cy, ancho, alto
    T_color_prof: Optional[np.ndarray] = None     # 4x4: marco de la profundidad nativa -> color
    metodo: str = ""
    notas: str = ""


# ------------------------------------------------------------------ percepción
@dataclass
class Fotograma:
    """Lo que entra a la percepción: igual del simulador, del robot o de un conjunto grabado."""
    t: float
    color: Optional[np.ndarray]     # HxWx3 uint8 RGB (None si no hay)
    prof: np.ndarray                # HxW float32 en m (0 = sin dato), profundidad NATIVA
    K_color: np.ndarray             # 3x3
    K_prof: np.ndarray              # 3x3 (la de IR1: la profundidad nativa está en su marco)
    ir: Optional[np.ndarray] = None  # HxW uint8, IR1 (mismo marco y K que la profundidad)
    q_brazos: Optional[np.ndarray] = None   # 14 (izq + der) medidas en ese instante: autofiltro
    cintura: float = 0.0            # rad, guiñada de la cintura (si no es 0, el torso giró)


@dataclass
class ObjetoDetectado:
    clase: str                      # 'cubo' | 'vaso' | ... (config/objetos.yaml) o 'desconocido'
    centro: np.ndarray              # 3, centro del objeto en torso_link
    guinada: float                  # rad, orientación en planta (módulo su simetría)
    simetria: float                 # rad: pi/2 caja cuadrada, pi rotulador, 0 = redondo (sin guiñada)
    medidas: np.ndarray             # 3, largo, ancho, alto estimados (m)
    confianza: float                # 0..1 (identidad y geometría)
    fraccion_visible: float         # 0..1 de la huella esperada: baja = ocluido por la mano/brazo
    n_puntos: int


@dataclass
class ListaObjetos:
    t: float                        # del fotograma
    objetos: List[ObjetoDetectado]
    plano_mesa: np.ndarray          # 4: n·p + d = 0 en torso_link (n hacia arriba)
    ms: float                       # tiempo de cómputo


# ------------------------------------------------------------------ planificación
@dataclass
class Agarre:
    """Fila de la tabla por objeto (medida en el hito 2: config/agarres.yaml)."""
    tipo: str                       # 'arriba' | 'lado' | 'pinza'
    preforma: List[int]             # 6 ángulos Inspire (0 cerrado .. 1000 abierto)
    cierre: List[int]               # 6 ángulos de consigna CERRADA (la fuerza la limita FORCE_SET)
    force_set: List[int]            # 6, gf por dedo (<= 800)
    altura_agarre: float            # m del centro de agarre sobre la mesa
    elev_deg: float = 0.0           # dedos respecto de la horizontal
    tolerancia_m: float = 0.0       # error de posición que tolera la preforma (6.4.1)
    previo: Optional[List[int]] = None   # paso antes del cierre (lateral: oponer el pulgar)
    suelta: Optional[List[int]] = None   # postura al soltar (si no, la preforma)


@dataclass
class Tramo:
    """Movimiento articular de un brazo entre dos posturas, con perfil de mínima sacudida."""
    lado: str
    q0: np.ndarray                  # 7
    q1: np.ndarray                  # 7
    duracion: float                 # s, >= la mínima medida en 6.3 y >= 8 s
    nombre: str = ""                # 'aproximar', 'descender', 'elevar', 'transportar', ...
    cartesiano: bool = False        # True: q(t) muestreado de una recta cartesiana (descenso final)
    q_ruta: Optional[np.ndarray] = None   # Nx7 puntos intermedios (RRT o recta), si hay


@dataclass
class Plan:
    objeto: ObjetoDetectado
    mano: str
    agarre: Agarre
    bandeja: str
    tramos: List[Tramo]
    ok: bool
    motivo: str = ""                # si no es ok: por qué (fuera de alcance, choque, ...)
    holgura_min_mesa: float = float("nan")   # m, comprobada contra el modelo (sec. 5)


# ------------------------------------------------------------------ brazo y mano
@dataclass
class EstadoBrazos:
    t: float
    q_medida: np.ndarray            # 15: cintura + izq 7 + der 7
    q_consigna: np.ndarray          # 15
    peso_arm_sdk: float             # 0..1, el que se está publicando en motor_cmd[27].q
    error_seguimiento: np.ndarray   # 15, consigna - medida (rad)


@dataclass
class OrdenMano:
    lado: str
    accion: str                     # 'preforma' | 'cerrar' | 'sujetar' | 'soltar' | 'abrir'
    agarre: Optional[Agarre] = None


@dataclass
class EstadoMano:
    t: float
    lado: str
    angulos: np.ndarray             # 6
    fuerzas: np.ndarray             # 6, gf
    status: np.ndarray              # 6 (0 abriendo, 1 cerrando, 2 en posición, 3 parado por fuerza...)
    error: np.ndarray               # 6
    tactil_max: Dict[str, float] = field(default_factory=dict)   # zona -> máximo sobre su reposo
    responde: bool = True


# ------------------------------------------------------------------ supervisor
@dataclass
class Salud:
    """Lo que vigila el supervisor en cada ciclo (sección 10)."""
    t: float
    fsm: int
    rpy: np.ndarray                 # 3, rad (IMU)
    yaw0: float                     # rad, yaw al empezar la tirada
    pies_desplazados: float         # m, máximo de los dos pies respecto de su huella (si se sabe)
    edad_camara: float              # s
    edad_lowstate: float            # s
    manos_responden: bool
    fuerza_max_gf: float
    error_seguimiento_max: float    # rad
