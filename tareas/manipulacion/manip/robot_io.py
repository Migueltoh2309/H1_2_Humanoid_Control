"""Entradas del robot con el SDK de Unitree. EL MISMO CÓDIGO en el robot y en el simulador
de h1_2_sim2real; solo cambian dominio e interfaz DDS:
  robot real:  dominio 0, eth0 (PC2)        simulador: dominio 1, lo

  * Estado: rt/lowstate (juntas, IMU, estado de motores), como caja_cuadrado.Estado.
  * leer_fsm(): GetFsmId (8001) por el RPC "loco". rt/arm_sdk se ignora EN SILENCIO si la FSM
    no es la de marcha (sec. 7): hay que leerla y exigir 201 antes de publicar.
  * confirmar(): la palabra que hay que teclear antes de mover el brazo (sec. 10).
Nada de Move, ReleaseMode ni rt/lowcmd (sec. 10).
"""
import json
import math
import sys
import threading
import time

import numpy as np

N_MOTORES = 27
CINTURA = 12
IZQ = list(range(13, 20))
DER = list(range(20, 27))
BRAZOS = [CINTURA] + IZQ + DER        # lo que manda rt/arm_sdk (15)
PESO_ARM_SDK = 27                     # motor_cmd[27].q = peso de mezcla
INDICES = {"izq": IZQ, "der": DER}


def iniciar_dds(sim: bool, iface: str = "eth0"):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    if sim:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, iface)


class Estado:
    """Último rt/lowstate (hilo del SDK) con su instante de llegada (monotonic)."""

    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        self.msg, self.t, self.n = None, 0.0, 0
        self.cerrojo = threading.Lock()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._al_recibir, 10)

    def _al_recibir(self, m):
        with self.cerrojo:
            self.msg, self.t = m, time.monotonic()
            self.n += 1

    def esperar(self, timeout=5.0):
        t0 = time.monotonic()
        while self.msg is None:
            if time.monotonic() - t0 > timeout:
                return False
            time.sleep(0.05)
        return True

    def ultimo(self):
        with self.cerrojo:
            return self.msg, self.t

    def edad(self):
        return time.monotonic() - self.ultimo()[1]

    def q(self):
        m, _ = self.ultimo()
        return np.array([m.motor_state[i].q for i in range(N_MOTORES)])

    def dq(self):
        m, _ = self.ultimo()
        return np.array([m.motor_state[i].dq for i in range(N_MOTORES)])

    def tau(self):
        m, _ = self.ultimo()
        return np.array([m.motor_state[i].tau_est for i in range(N_MOTORES)])

    def brazo(self, lado):
        return self.q()[INDICES[lado]]

    def rpy(self):
        m, _ = self.ultimo()
        return np.array([float(v) for v in m.imu_state.rpy])

    def gyro(self):
        m, _ = self.ultimo()
        return np.array([float(v) for v in m.imu_state.gyroscope])

    def motores_ok(self):
        m, _ = self.ultimo()
        return all(m.motor_state[i].motorstate == 0 for i in range(N_MOTORES))

    def mode_machine(self):
        return self.ultimo()[0].mode_machine


class LectorFsm:
    """GetFsmId por el RPC del servicio "loco" (el LocoClient de Python no lo trae)."""

    def __init__(self, timeout=3.0):
        from unitree_sdk2py.h1.loco.h1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
        from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient
        self.api = ROBOT_API_ID_LOCO_GET_FSM_ID
        self.cliente = LocoClient()
        self.cliente.SetTimeout(timeout)
        self.cliente.Init()

    def leer(self):
        try:
            code, data = self.cliente._Call(self.api, "{}")
            if code != 0 or not data:
                return None
            return int(json.loads(data)["data"])
        except Exception:
            return None


def confirmar(palabra, sim, log=print):
    """Pide teclear `palabra` (como caja_cuadrado.py) antes de mover nada. Sin terminal solo
    se acepta en el simulador (tiradas automáticas); en el robot, nunca."""
    if not sys.stdin.isatty():
        if sim:
            log(f"(simulador sin terminal: se da por tecleada la palabra {palabra})")
            return True
        log("sin terminal no se puede confirmar: en el robot hay que teclear la palabra")
        return False
    r = input(f"  Escribe {palabra} para mover los brazos (operador con L2+B listo): ").strip()
    return r == palabra


def grados(x):
    return math.degrees(x)
