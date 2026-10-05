"""Bloque de ESTIMACION (reto, sec. 5 y 6.4): medidas de percepcion + IMU -> EstadoLinea.

No hay odometria (el GetOdom del PC2 da 0, 0, 0). La pose se estima en un marco
fijo puesto al arrancar:
  * rumbo: yaw de la IMU (como cuadrado.py), relativo al del arranque;
  * avance: la velocidad PEDIDA pasada por un modelo de la marcha (retardo puro +
    primer orden, constantes a medir con escalones, 6.3.1) y por factor_vx
    (velocidad real / pedida, a calibrar con cinta como el --factor de cuadrado.py).

Memoria de la linea: cada medida buena se pasa al marco fijo y se guarda (una
muestra por celda de 2 cm, la mas reciente). En cada ciclo la linea se reconstruye
en el marco del robot con lo recordado: asi la zona ciega bajo el robot (unos
0.4 m) y el hueco de 40 cm del nivel 3 se cubren con lo que la camara vio antes,
y la barra y la esquina se siguen "viendo" cuando ya entraron en la zona ciega.

Balanceo (6.4.1): se compensa en percepcion con el roll/pitch de la IMU de cada
fotograma (sin retardo, a diferencia de un filtro); la memoria, al promediar
medidas de varios pasos, filtra lo que quede.
"""
import math
from collections import deque

import numpy as np

from .tipos import EstadoLinea, MedidaLinea


def envolver(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ModeloMarcha:
    """v_real(t) ~ primer orden de v_pedida(t - retardo)."""

    def __init__(self, retardo=0.3, tau=0.35):
        self.retardo, self.tau = retardo, tau
        self.cola = deque()
        self.v = np.zeros(3)

    def paso(self, t, v_pedida, dt):
        self.cola.append((t, np.asarray(v_pedida, float)))
        while len(self.cola) > 1 and self.cola[1][0] <= t - self.retardo:
            self.cola.popleft()
        v_ret = self.cola[0][1] if self.cola[0][0] <= t - self.retardo else np.zeros(3)
        self.v += (v_ret - self.v) * min(1.0, dt / max(self.tau, 1e-3))
        return self.v


class Estimador:
    def __init__(self, cfg):
        e = cfg["estimacion"]
        self.cfg = cfg
        self.conf_min = e["confianza_min"]
        self.memoria_m = e["memoria_m"]
        self.factor = e.get("factor_vx", 1.0)
        self.factor_w = e.get("factor_vyaw", 1.0)
        self.marcha = ModeloMarcha(e.get("retardo_marcha", 0.3), e.get("tau_marcha", 0.35))
        self.tau_deriva = e.get("tau_deriva", 3.0)
        self.esquina_n = e.get("esquina_confirmaciones", 3)
        self.esquina_min_d = e.get("esquina_distancia_min", 0.45)
        self.sesgo_yaw = 0.0
        self.celda = 0.02
        self.t = None
        self._reiniciar()

    def olvidar_linea(self):
        """Tras el giro de una esquina: la linea recordada queda de lado y estorba."""
        self.mem.clear()
        self._mira_ang = None
        self._esq_cand = None
        self.esquina = None
        self.t_buena, self.rec_buena = self.t, self.recorrido

    def reiniciar(self, yaw_imu=0.0, conservar=None):
        """Marco fijo nuevo en la pose actual (rumbo de la IMU AHORA). `conservar`:
        una MedidaLinea que se vuelve a meter en la memoria."""
        self._reiniciar(yaw_imu)
        if conservar is not None and conservar.valida and len(conservar.puntos):
            self.ultima = conservar
            for (qx, qy) in conservar.puntos:
                self.mem[(int(qx // self.celda), int(qy // self.celda))] = (qx, qy, self.t or 0.0)
            self.t_buena, self.rec_buena = self.t, 0.0

    def _reiniciar(self, yaw_imu=0.0):
        self.x = self.y = 0.0
        self.yaw0 = yaw_imu
        self.psi = 0.0
        self.recorrido = 0.0
        self._mira_ang = None
        self._esq_cand = None
        self._barra_cand = None
        self.mem = {}                 # celda -> (x, y, t)  en el marco fijo
        self.t_buena = None
        self.rec_buena = 0.0
        self.barra = None             # (x, y) en el marco fijo
        self.esquina = None           # (x, y, lado)
        self.ultima = None
        self.n_barra = 0

    # ------------------------------------------------------------ marcos
    def a_fijo(self, P):
        c, s = math.cos(self.psi), math.sin(self.psi)
        return np.column_stack([self.x + P[:, 0] * c - P[:, 1] * s, self.y + P[:, 0] * s + P[:, 1] * c])

    def a_robot(self, Q):
        c, s = math.cos(self.psi), math.sin(self.psi)
        dx, dy = Q[:, 0] - self.x, Q[:, 1] - self.y
        return np.column_stack([dx * c + dy * s, -dx * s + dy * c])

    # ------------------------------------------------------------ ciclo
    def actualizar(self, t, yaw_imu, v_pedida, medida: MedidaLinea = None, gyro_z=None) -> EstadoLinea:
        dt = 0.0 if self.t is None else max(0.0, t - self.t)
        self.t = t
        v = self.marcha.paso(t, v_pedida, dt) * np.array([self.factor, self.factor, self.factor_w])
        # Deriva de rumbo (reto: ~2 grados/s andando, signo distinto entre tiradas): lo que
        # gira el robot (giroscopo) menos lo que deberia girar con la orden (modelo de la
        # marcha). Media exponencial, solo andando: el control la resta de vyaw.
        if gyro_z is not None and dt > 0 and math.hypot(v[0], v[1]) > 0.05:
            a = min(1.0, dt / self.tau_deriva)
            self.sesgo_yaw += (gyro_z - v[2] - self.sesgo_yaw) * a
        self.psi = envolver(yaw_imu - self.yaw0)
        c, s = math.cos(self.psi), math.sin(self.psi)
        self.x += (v[0] * c - v[1] * s) * dt
        self.y += (v[0] * s + v[1] * c) * dt
        self.recorrido += math.hypot(v[0], v[1]) * dt

        if medida is not None and medida.valida and medida.confianza >= self.conf_min and len(medida.puntos):
            self.ultima = medida
            Q = self.a_fijo(medida.puntos)
            for (qx, qy) in Q:
                self.mem[(int(qx // self.celda), int(qy // self.celda))] = (qx, qy, t)
            self.t_buena, self.rec_buena = t, self.recorrido
        # la barra solo con una linea valida detras (con la marcha real, un fotograma sin
        # linea vio una "barra" en la esquina del nivel 4 y el robot paro 2 m antes)
        # (valida = hay linea ajustada detras; NO se pide confianza: al acercarse al final la
        # linea visible se acorta, la confianza baja y la barra no se llegaba a registrar)
        if (medida is not None and medida.barra is not None and medida.valida
                and not self._es_la_esquina(medida.barra)):
            self._recordar_barra(medida)
        if medida is not None and medida.esquina is not None and medida.esquina[0] > self.esquina_min_d:
            self._recordar_esquina(medida)
        self._podar()
        return self._estado(t)

    def _recordar_esquina(self, medida):
        """Esquina confirmada: n_min detecciones que coinciden (+-20 cm), posicion
        promediada y lado por mayoria. Cerca de la zona ciega el tramo transversal
        se ve recortado y el lado puede salir al reves: esas no cuentan."""
        d, lado = medida.esquina
        q = self.a_fijo(np.array([[d, 0.0]]))[0]
        c = self._esq_cand
        if c is None or math.hypot(q[0] - c["x"], q[1] - c["y"]) > 0.20:
            self._esq_cand = {"x": q[0], "y": q[1], "n": 1, "votos": lado}
        else:
            c["x"] = (c["x"] * c["n"] + q[0]) / (c["n"] + 1)
            c["y"] = (c["y"] * c["n"] + q[1]) / (c["n"] + 1)
            c["n"] += 1
            c["votos"] += lado
        c = self._esq_cand
        if c["n"] >= self.esquina_n:
            self.esquina = (c["x"], c["y"], 1 if c["votos"] > 0 else -1)

    def _es_la_esquina(self, d):
        """Una "barra" donde ya hay una esquina CONFIRMADA es la esquina. (Con una candidata
        no basta: al llegar a la barra con el robot algo desplazado, la barra a veces parece
        "de un solo lado", crea una candidata y la barra no se registraba nunca.)"""
        if self.esquina is None:
            return False
        p = self.a_robot(np.array([self.esquina[:2]]))[0]
        return abs(p[0] - d) < 0.3

    def _recordar_barra(self, medida):
        """Barra confirmada: esquina_confirmaciones detecciones coherentes (+-20 cm),
        posicion promediada (la barra se ve varias veces al acercarse)."""
        y = 0.0
        if medida.coef is not None:
            cc = medida.coef
            y = cc[0] + cc[1] * medida.barra + cc[2] * medida.barra ** 2
        q = self.a_fijo(np.array([[medida.barra, y]]))[0]
        c = self._barra_cand
        if c is None or math.hypot(q[0] - c[0], q[1] - c[1]) > 0.20:
            self._barra_cand, self.n_barra = q, 1
        else:
            self._barra_cand = (c * self.n_barra + q) / (self.n_barra + 1)
            self.n_barra = min(self.n_barra + 1, 10)
        if self.n_barra >= self.esquina_n:
            self.barra = self._barra_cand

    def _podar(self):
        if not self.mem:
            return
        Q = np.array([(v[0], v[1]) for v in self.mem.values()])
        P = self.a_robot(Q)
        claves = list(self.mem.keys())
        for k, (px, py) in zip(claves, P):
            if px < -self.memoria_m or math.hypot(px, py) > 4.0:
                del self.mem[k]

    def puntos_robot(self):
        if not self.mem:
            return np.zeros((0, 2))
        Q = np.array([(v[0], v[1]) for v in self.mem.values()])
        P = self.a_robot(Q)
        return P[np.argsort(P[:, 0])]

    # ------------------------------------------------------------ estado
    def _estado(self, t) -> EstadoLinea:
        est = EstadoLinea(t=t, valida=False, recorrido=self.recorrido, yaw=self.psi, sesgo_yaw=self.sesgo_yaw)
        est.sin_linea_m = self.recorrido - self.rec_buena if self.t_buena is not None else math.inf
        est.sin_linea_s = t - self.t_buena if self.t_buena is not None else math.inf
        if self.barra is not None:
            est.barra = float(self.a_robot(self.barra[None])[0, 0])
        if self.esquina is not None:
            est.esquina = (float(self.a_robot(np.array([self.esquina[:2]]))[0, 0]), self.esquina[2])
        P = self.puntos_robot()
        cerca = P[(P[:, 0] > -0.5) & (P[:, 0] < 1.6)] if len(P) else P
        if len(cerca) < 4:
            return est
        w = np.exp(-np.abs(cerca[:, 0] - 0.2) / 0.6)
        grado = 2 if np.ptp(cerca[:, 0]) > 0.6 else 1
        c = np.polyfit(cerca[:, 0], cerca[:, 1], grado, w=np.sqrt(w))[::-1]
        if grado == 1:
            c = np.array([c[0], c[1], 0.0])
        est.valida = True
        est.desplazamiento = float(c[0])
        est.angulo = float(math.atan(c[1]))
        est.curvatura = float(2 * c[2] / (1 + c[1] ** 2) ** 1.5)
        est.confianza = self.ultima.confianza if self.ultima is not None else 0.0
        if est.sin_linea_s > 0.5:
            est.confianza *= math.exp(-(est.sin_linea_s - 0.5))
        self.coef = c
        return est

    def punto_a_distancia(self, L):
        """Punto de mira de pure pursuit: donde la linea recordada corta la
        circunferencia de radio L alrededor del robot. Si hay varios cortes (una S
        que vuelve, el tramo ya recorrido), el mas coherente en angulo con la mira
        anterior (o con el rumbo de la linea). Si la linea es mas corta que L (se
        acaba en la barra o en una esquina), su punto mas lejano por delante.
        None si no hay linea por delante.

        (Una primera version ordenaba los puntos encadenando vecinos: con el ruido de
        la memoria la cadena se cortaba y devolvia un punto a 3 cm del robot; la
        curvatura 2y/(x^2+y^2) se disparaba y el robot sobregiraba hasta 22 cm.)"""
        P = self.puntos_robot()
        P = P[P[:, 0] > -0.05]
        if len(P) == 0:
            return None
        r = np.hypot(P[:, 0], P[:, 1])
        th = np.arctan2(P[:, 1], P[:, 0])
        ref = self._mira_ang if getattr(self, "_mira_ang", None) is not None else 0.0
        dang = np.abs((th - ref + math.pi) % (2 * math.pi) - math.pi)
        anillo = (np.abs(r - L) < 0.06) & (dang < math.radians(75))
        if anillo.any():
            i = np.flatnonzero(anillo)[int(np.argmin(dang[anillo]))]
            cerca = anillo & (np.abs(th - th[i]) < math.radians(6))
            q = P[cerca].mean(0)
        else:
            delante = (r < L) & (dang < math.radians(75))
            if not delante.any():
                return None
            i = np.flatnonzero(delante)[int(np.argmax(r[delante]))]
            q = P[i]
        self._mira_ang = math.atan2(q[1], q[0])
        return float(q[0]), float(q[1])
