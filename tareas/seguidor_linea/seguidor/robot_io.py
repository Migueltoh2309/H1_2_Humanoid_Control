"""Entradas/salidas del robot con el SDK de Unitree. EL MISMO CODIGO en el robot
real y en el simulador de h1_2_sim2real: solo cambia dominio e interfaz DDS
  robot real:  dominio 0, eth0     (en el PC2)
  simulador :  dominio 1, lo       (rt/lowstate y el servicio "loco" los da el sim)

  * Imu: rt/lowstate -> roll, pitch, yaw, gyro, motores (como wasd.py y cuadrado.py).
  * Marcha: LocoClient.Move(vx, vy, vyaw) con duration = 1 s (si el programa muere,
    el robot para solo) y StopMove(). En modo SIMULACRO no se llama a nada: las
    ordenes solo se registran (hito 3 del reto).
Solo alto nivel: nada de rt/lowcmd, rt/arm_sdk ni ReleaseMode (sec. 10).
"""
import math
import threading
import time

from .tipos import Imu, Orden


def iniciar_dds(sim: bool, iface: str = "eth0"):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    if sim:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, iface)


class FuenteImu:
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        self.msg, self.t = None, 0.0
        self.cerrojo = threading.Lock()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._al_recibir, 10)

    def _al_recibir(self, m):
        with self.cerrojo:
            self.msg, self.t = m, time.monotonic()

    def esperar(self, timeout=5.0):
        t0 = time.monotonic()
        while self.msg is None:
            if time.monotonic() - t0 > timeout:
                return False
            time.sleep(0.05)
        return True

    def leer(self) -> Imu:
        with self.cerrojo:
            m, t = self.msg, self.t
        if m is None:
            return None
        r, p, y = (float(v) for v in m.imu_state.rpy)
        motores_ok = all(m.motor_state[i].motorstate == 0 for i in range(27))
        return Imu(t, r, p, y, tuple(float(v) for v in m.imu_state.gyroscope), motores_ok)


class Marcha:
    def __init__(self, simulacro: bool, timeout=10.0):
        self.simulacro = simulacro
        self.cliente = None
        self.n_move = 0
        if not simulacro:
            from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient
            self.cliente = LocoClient()
            self.cliente.SetTimeout(timeout)
            self.cliente.Init()

    def aplicar(self, o: Orden):
        if self.simulacro:
            return
        if o.parar:
            self.cliente.StopMove()
        else:
            self.cliente.Move(float(o.vx), float(o.vy), float(o.vyaw))     # duration = 1 s
            self.n_move += 1

    def parar(self):
        if not self.simulacro:
            for _ in range(3):
                try:
                    self.cliente.StopMove()
                    return
                except Exception:
                    time.sleep(0.05)


def yaw_grados(imu: Imu):
    return math.degrees(imu.yaw)
