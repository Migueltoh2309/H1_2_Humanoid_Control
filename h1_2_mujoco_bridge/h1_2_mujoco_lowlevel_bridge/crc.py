"""
crc.py — CRC32 de Unitree para unitree_hg/msg/LowCmd.

La lógica de serialización y el algoritmo CRC32 son una réplica EXACTA de la
implementación proporcionada en `test_mandar_modificado.py` (que a su vez
reproduce el layout del struct C++ del SDK). El CRC de Unitree NO es el CRC32
estándar de zlib: es una variante propia bit a bit con polinomio 0x04C11DB7.

Layout serializado (little-endian, 1000 bytes cubiertos por el CRC):

  LowCmd:
    [0]         uint8  mode_pr
    [1]         uint8  mode_machine
    [2-3]       padding (2 bytes)
    [4-983]     MotorCmd[35]        (35 x 28 = 980 bytes)
    [984-999]   uint32[4] reserve   (16 bytes)
    [1000-1003] uint32 crc          <- NO entra en el cálculo

  MotorCmd (28 bytes):
    [0]     uint8   mode
    [1-3]   padding (3 bytes)
    [4]     float32 q
    [8]     float32 dq
    [12]    float32 tau     <- tau ANTES de kp/kd
    [16]    float32 kp
    [20]    float32 kd
    [24]    uint32  reserve

Las funciones son duck-typed: aceptan tanto mensajes reales de `unitree_hg`
como objetos de prueba con los mismos atributos (útil para los tests sin ROS).
"""
from __future__ import annotations

import struct

import numpy as np

NUM_MOTORS_MSG = 35          # tamaño fijo del array motor_cmd en el mensaje
_CRC_POLY = 0x04C11DB7
_SERIALIZED_LEN = 1000       # bytes cubiertos por el CRC (250 palabras uint32)

# Struct precompilado del layout completo (mucho más rápido que packs sueltos):
#   header BBxx (4) + 35 x [B3x f f f f f I] (28) + 4I (16) = 1000 bytes
_LOWCMD_STRUCT = struct.Struct('<BBxx' + 'B3xfffffI' * NUM_MOTORS_MSG + '4I')
assert _LOWCMD_STRUCT.size == _SERIALIZED_LEN

# Tabla de 256 entradas para el CRC (MSB-first, sin reflexión, init 0xFFFFFFFF).
# El bucle bit a bit de Unitree es algebraicamente equivalente al CRC-32/MPEG-2
# procesando los bytes de cada palabra uint32 en orden big-endian; la
# equivalencia se verifica en test_crc.py contra la implementación de
# referencia bit a bit (_crc32_core), copiada literal del script del usuario.
def _build_table():
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ _CRC_POLY) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return tuple(table)


_CRC_TABLE = _build_table()


def _crc32_core(uint32_words) -> int:
    """Implementación de REFERENCIA bit a bit (idéntica a la de
    test_mandar_modificado). Lenta (~1.3 ms/msg); se conserva como oráculo
    para los tests. El camino rápido es _crc32_fast()."""
    crc_register = 0xFFFFFFFF
    poly = _CRC_POLY

    for word in uint32_words:
        xbit = 1 << 31
        data = word & 0xFFFFFFFF
        for _ in range(32):
            if crc_register & 0x80000000:
                crc_register = ((crc_register << 1) ^ poly) & 0xFFFFFFFF
            else:
                crc_register = (crc_register << 1) & 0xFFFFFFFF
            if data & xbit:
                crc_register ^= poly
            xbit >>= 1

    return crc_register


def _crc32_fast(data_le: bytes) -> int:
    """CRC por tabla sobre el buffer serializado (~0.1 ms/msg).
    Las palabras uint32 little-endian se reordenan a big-endian porque el
    algoritmo de Unitree consume cada palabra desde su bit más significativo."""
    be = np.frombuffer(data_le, dtype='<u4').byteswap().tobytes()
    crc = 0xFFFFFFFF
    table = _CRC_TABLE
    for b in be:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ b) & 0xFF]
    return crc


def serialize_low_cmd(msg) -> bytes:
    """Serializa un LowCmd (o duck-type equivalente) al layout binario del SDK."""
    vals = [int(msg.mode_pr) & 0xFF, int(msg.mode_machine) & 0xFF]

    motor_cmd = msg.motor_cmd
    n = len(motor_cmd)
    for i in range(NUM_MOTORS_MSG):
        if i < n:
            cmd = motor_cmd[i]
            vals.extend((int(cmd.mode) & 0xFF, float(cmd.q), float(cmd.dq),
                         float(cmd.tau), float(cmd.kp), float(cmd.kd), 0))
        else:
            vals.extend((0, 0.0, 0.0, 0.0, 0.0, 0.0, 0))

    vals.extend((0, 0, 0, 0))   # uint32[4] reserve
    return _LOWCMD_STRUCT.pack(*vals)


def compute_crc(msg) -> int:
    """Calcula el CRC de Unitree de un LowCmd sin modificar el mensaje."""
    return _crc32_fast(serialize_low_cmd(msg))


def attach_crc(msg) -> None:
    """Calcula y asigna msg.crc (equivalente al get_crc() del script de referencia)."""
    msg.crc = compute_crc(msg)


def verify_crc(msg) -> bool:
    """True si msg.crc coincide con el CRC recalculado del contenido."""
    return (int(msg.crc) & 0xFFFFFFFF) == compute_crc(msg)
