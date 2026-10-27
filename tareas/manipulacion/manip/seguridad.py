"""Comprobaciones de seguridad de la sección 10 que corren en cada ciclo de un movimiento.
Lanzan brazo.Parada; quien mueve el brazo congela la consigna y hace la salida segura.

Las usan las herramientas del hito 2 (calibrar, curva_pasos) y el supervisor (hito 3).
"""
import math
import time

import numpy as np

from .brazo import Parada


class Vigilante:
    def __init__(self, estado, fsm, brazo, fuente=None, yaw0=None, giro_max_deg=5.0, incl_max_deg=10.0,
                 edad_camara_max=2.0, cada_fsm=2.0, abortar=True):
        self.estado, self.fsm, self.brazo, self.fuente = estado, fsm, brazo, fuente
        self.yaw0 = estado.rpy()[2] if yaw0 is None else yaw0
        self.giro_max, self.incl_max = math.radians(giro_max_deg), math.radians(incl_max_deg)
        self.edad_camara_max, self.cada_fsm = edad_camara_max, cada_fsm
        self.abortar = abortar                 # False: solo registra (curva de pasos en el simulador)
        self.t_fsm = 0.0
        self.err_max = 0.0
        self.giro_obs = 0.0
        self.incl_obs = 0.0
        self.mirar_seguimiento = False
        self.mirar_camara = fuente is not None

    def _parar(self, motivo):
        if self.abortar:
            raise Parada(motivo)

    def __call__(self):
        e = self.estado
        if e.edad() > 0.5:
            raise Parada("rt/lowstate parado")
        r, p, y = e.rpy()
        giro = abs(math.remainder(y - self.yaw0, 2 * math.pi))
        self.giro_obs = max(self.giro_obs, giro)
        self.incl_obs = max(self.incl_obs, abs(r), abs(p))
        if abs(r) > self.incl_max or abs(p) > self.incl_max:
            self._parar(f"inclinación {math.degrees(r):.1f}/{math.degrees(p):.1f} grados")
        if giro > self.giro_max:
            self._parar(f"el cuerpo giró {math.degrees(giro):.1f} grados")
        if not e.motores_ok():
            raise Parada("fallo de motor")
        if self.mirar_seguimiento and self.brazo is not None:
            err = float(np.max(np.abs(self.brazo.error_seguimiento()[1:])))
            self.err_max = max(self.err_max, err)
            if err > self.brazo.cfg["error_seguimiento_max"]:
                raise Parada(f"error de seguimiento {math.degrees(err):.0f} grados (¿choque?)")
        if self.fsm is not None and time.monotonic() - self.t_fsm > self.cada_fsm:
            self.t_fsm = time.monotonic()
            f = self.fsm.leer()
            if f is not None and f != 201:
                raise Parada(f"FSM {f} (no 201)")
        if self.mirar_camara and self.fuente.edad() > self.edad_camara_max:
            raise Parada("la cámara no da imágenes")
