"""Memoria compartida entre los tres procesos del simulador.

Por que tres procesos: en Python, serializar un LowState_ del SDK cuesta ~0.7 ms
(medido), mas que un mj_step (~0.2 ms), y renderizar las 3 camaras de la D435 son
decenas de ms. En un solo proceso se pisan por el GIL y la fisica no llegaba ni
al 40 % del tiempo real. Asi:
  * fisica   (nodo principal): MuJoCo a 500 Hz, manos Modbus, viewer, /sim/*
  * dds      (proceso_dds):    rt/lowstate a 500 Hz como el robot; rt/lowcmd y
                               rt/arm_sdk recibidos -> aqui
  * camaras  (proceso_camaras): su propio MjData con el qpos de aqui, render y
                               topicos ROS de la D435
CLOCK_MONOTONIC es comun a todos los procesos, asi que los tiempos (t de los
comandos) se comparan directamente.
"""
from multiprocessing import shared_memory

import numpy as np

N = 27


def campos(nq):
    return [
        ("tick", 1), ("q", N), ("dq", N), ("tau", N), ("quat", 4), ("rpy", 3), ("gyro", 3), ("acc", 3),
        ("fuente", 1),                       # 0 interno, 1 interno+arm_sdk, 2 lowcmd, 3 debug sin ordenes
        ("lc_t", 1), ("lc_q", N), ("lc_dq", N), ("lc_kp", N), ("lc_kd", N), ("lc_tau", N), ("lc_mode", N),
        ("arm_t", 1), ("arm_q", N), ("arm_dq", N), ("arm_kp", N), ("arm_kd", N), ("arm_tau", N), ("arm_w", 1),
        ("int_q", N), ("int_kp", N), ("int_kd", N),
        ("qpos_n", 1), ("qpos", nq),
        ("parar", 1),
        ("ls_n", 1),                          # rt/lowstate publicados por el proceso dds (vigilancia)
        # servicio "loco" (LocoClient): lo escribe proceso_loco, lo lee la marcha cinematica
        ("loco_v", 3),                        # vx, vy, vyaw pedidos (SetVelocity)
        ("loco_t_fin", 1),                    # monotonic hasta el que vale (duration de Move)
        ("loco_n", 1),                        # SetVelocity recibidos
        ("fsm", 1),                           # FSM simulada (201 = andar, como con L2+UP en el mando)
    ]


class Compartido:
    def __init__(self, nombre, nq, crear=False):
        tam = sum(n for _, n in campos(nq))
        if crear:
            try:                                # restos de una ejecucion que murio mal
                viejo = shared_memory.SharedMemory(name=nombre)
                viejo.close()
                viejo.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(name=nombre, create=True, size=8 * tam)
        else:
            self.shm = shared_memory.SharedMemory(name=nombre)
        self.creador = crear
        buf = np.ndarray((tam,), dtype=np.float64, buffer=self.shm.buf)
        if crear:
            buf[:] = 0.0
        self.v = {}
        o = 0
        for nombre_c, n in campos(nq):
            self.v[nombre_c] = buf[o:o + n]
            o += n
        if crear:
            self.v["lc_t"][0] = self.v["arm_t"][0] = -1e9

    def cerrar(self):
        self.v = {}
        self.shm.close()
        if self.creador:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
