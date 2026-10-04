"""Bloque de CONTROL (reto, sec. 5 y 6.3): EstadoLinea -> Orden (vx, vy, vyaw) saturada.

Ley por defecto: pure pursuit sobre la linea recordada. Con el punto de mira
(x_t, y_t) a distancia L, la curvatura que lleva al robot hasta el es
    kappa = 2 y_t / (x_t^2 + y_t^2)       y       vyaw = vx * kappa.
Por que pure pursuit (6.3.3): mira a L ~ 1 m, mas alla de la zona ciega y del
retardo (a 0.3 m/s, 0.3 s de retardo son 9 cm: la mira los absorbe), y no deriva
la senal de error (el balanceo a 1.43 Hz meteria ruido en un termino D).
Alternativa `pd`: vyaw = k_y * desplazamiento + k_psi * angulo.

vx baja con la curvatura y con la confianza (6.3.5); vy opcional (6.3.4).
Topes del reto (sec. 10): vx 0.4, vy 0.2, vyaw 0.5, por `escala` (0.5 en las
primeras tiradas de cada nivel). Limite de aceleracion para suavidad.
"""
import math

from .tipos import EstadoLinea, Orden


def saturar(v, tope):
    return max(-tope, min(tope, v))


class Control:
    def __init__(self, cfg):
        self.c = cfg["control"]
        self.ultima = Orden()

    def mirada(self):
        return self.c["mirada"] + self.c["mirada_por_vx"] * abs(self.ultima.vx)

    def calcular(self, est: EstadoLinea, objetivo, dt) -> Orden:
        c = self.c
        esc = c.get("escala", 1.0)
        vx_max, vy_max, w_max = c["vx_max"] * esc, c["vy_max"] * esc, c["vyaw_max"] * esc
        if not est.valida:
            return self._suavizar(Orden(0.0, 0.0, 0.0, motivo="sin estado"), dt)
        conf = max(0.0, min(1.0, (est.confianza - 0.2) / 0.6))
        if objetivo is not None and math.hypot(*objetivo) < c.get("mira_min", 0.3):
            objetivo = None                  # una mira pegada al robot da curvaturas absurdas: PD
        if c["ley"] == "pd" or objetivo is None:
            kappa = None
            w = c["k_y"] * est.desplazamiento + c["k_psi"] * est.angulo
            curv = abs(est.curvatura)
        else:
            xt, yt = objetivo
            kappa = 2.0 * yt / max(xt * xt + yt * yt, 1e-3)
            curv = abs(kappa)
        vx = vx_max / (1.0 + c["k_curvatura"] * curv)
        vx = max(c["vx_min"] * esc, vx * (0.4 + 0.6 * conf))
        if kappa is not None:
            w = vx * kappa
        if c.get("compensar_deriva", True):
            w -= est.sesgo_yaw               # la deriva medida por el giroscopo, fuera
        vy = c["k_vy"] * est.desplazamiento if c.get("usar_vy") else 0.0
        o = Orden(saturar(vx, vx_max), saturar(vy, vy_max), saturar(w, w_max), motivo=c["ley"])
        return self._suavizar(o, dt)

    def _suavizar(self, o: Orden, dt):
        a = self.ultima
        dvx = self.c["acel_vx"] * max(dt, 1e-3)
        dw = self.c["acel_vyaw"] * max(dt, 1e-3)
        o.vx = a.vx + saturar(o.vx - a.vx, dvx)
        o.vy = a.vy + saturar(o.vy - a.vy, dvx)
        o.vyaw = a.vyaw + saturar(o.vyaw - a.vyaw, dw)
        self.ultima = o
        return o

    def reiniciar(self):
        self.ultima = Orden()
