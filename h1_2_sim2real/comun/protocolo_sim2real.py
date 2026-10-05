"""Datagramas UDP entre el puente ROS 2 (PC) y el agente SDK (robot o simulador).

FICHERO MAESTRO: lo importan robot/agente_sdk.py y el paquete ROS 2. Al robot
se copia con scripts/instalar_en_robot.sh. Si cambias el formato, sube VERSION:
el receptor descarta lo que no reconoce.

Por que UDP y no DDS: el DDS de Unitree vive en la eth0 interna del robot
(192.168.123.x) y el PC llega por la WiFi; es la misma solucion que ya funciona
en Codigos/rviz_h1_2 (estado por UDP 47927). Aqui va en los dos sentidos y con
puertos distintos (47930/47931), para que convivan.

ORDEN (PC -> agente), PUERTO_ORDEN:
    4s 'S2RO' | B version | B orden | H reservado | I seq
    I mascara_cuerpo (bit i = junta i comandada) | H mascara_mano (bit k = dof k) | H reservado
    27f q | 27f dq | 27f tau_ff | 12h angulo_mano (0 cerrado .. 1000 abierto; izq 0-5, der 6-11)

ESTADO (agente -> PC), PUERTO_ESTADO:
    4s 'S2RE' | B version | B modo | B banderas | B mode_machine | I seq | Q t_ns | f peso
    27f q | 27f dq | 27f tau_est | 27f q_objetivo
    4f quat (w x y z) | 3f rpy | 3f gyro | 3f acc
    12h angulo_mano | 12h fuerza_mano
"""
import struct

VERSION = 1
PUERTO_ORDEN = 47930
PUERTO_ESTADO = 47931
N = 27
NM = 12

# ordenes
NORMAL = 0      # aplicar las consignas enmascaradas
SOLTAR = 1      # devolver el control: rampa del peso de arm_sdk a 0 (o mantener, en lowcmd) y salir

# modos del agente
MODO_ARM_SDK = 0
MODO_LOWCMD = 1
NOMBRE_MODO = {MODO_ARM_SDK: "arm_sdk", MODO_LOWCMD: "lowcmd"}

# banderas del estado
B_LOWSTATE = 1 << 0     # llega rt/lowstate
B_ORDENES = 1 << 1      # llegan ordenes del PC (sin watchdog)
B_MANOS = 1 << 2        # las dos manos responden por Modbus
B_SOLTANDO = 1 << 3     # rampa de salida en curso
B_ACTIVO = 1 << 4       # el agente esta mandando (peso > 0 o lowcmd)
B_FALLO = 1 << 5        # motor en fallo o inclinacion: el agente ha soltado

_ORD = struct.Struct("<4sBBHIIHH%df%dh" % (3 * N, NM))
_EST = struct.Struct("<4sBBBBIQf%df13f%dh" % (4 * N, 2 * NM))
TAM_ORDEN = _ORD.size
TAM_ESTADO = _EST.size


def empaquetar_orden(seq, orden, mascara, mascara_mano, q, dq, tau, angulos):
    return _ORD.pack(b"S2RO", VERSION, orden, 0, seq & 0xFFFFFFFF, mascara, mascara_mano, 0,
                     *q, *dq, *tau, *[int(a) for a in angulos])


def desempaquetar_orden(datos):
    if len(datos) != TAM_ORDEN:
        return None
    v = _ORD.unpack(datos)
    if v[0] != b"S2RO" or v[1] != VERSION:
        return None
    b = 8
    return {"orden": v[2], "seq": v[4], "mascara": v[5], "mascara_mano": v[6],
            "q": v[b:b + N], "dq": v[b + N:b + 2 * N], "tau": v[b + 2 * N:b + 3 * N],
            "angulos": v[b + 3 * N:b + 3 * N + NM]}


def empaquetar_estado(seq, t_ns, modo, banderas, mode_machine, peso, q, dq, tau, q_obj,
                      quat, rpy, gyro, acc, ang_mano, fuerza_mano):
    return _EST.pack(b"S2RE", VERSION, modo, banderas, mode_machine & 0xFF, seq & 0xFFFFFFFF,
                     t_ns, peso, *q, *dq, *tau, *q_obj, *quat, *rpy, *gyro, *acc,
                     *[int(a) for a in ang_mano], *[int(max(-32768, min(32767, f))) for f in fuerza_mano])


def desempaquetar_estado(datos):
    if len(datos) != TAM_ESTADO:
        return None
    v = _EST.unpack(datos)
    if v[0] != b"S2RE" or v[1] != VERSION:
        return None
    b = 8
    c = b + 4 * N
    return {"modo": v[2], "banderas": v[3], "mode_machine": v[4], "seq": v[5], "t_ns": v[6],
            "peso": v[7], "q": v[b:b + N], "dq": v[b + N:b + 2 * N], "tau": v[b + 2 * N:b + 3 * N],
            "q_obj": v[b + 3 * N:c], "quat": v[c:c + 4], "rpy": v[c + 4:c + 7],
            "gyro": v[c + 7:c + 10], "acc": v[c + 10:c + 13],
            "ang_mano": v[c + 13:c + 13 + NM], "fuerza_mano": v[c + 13 + NM:c + 13 + 2 * NM]}
