"""Tests del CRC de Unitree (serialización + algoritmo + verificación)."""
import struct

from h1_2_algoritms import unitree_crc as crc_mod


# Mensajes falsos con el layout de unitree_hg/LowCmd (sin depender de ROS)
class FakeMotorCmd:
    def __init__(self):
        self.mode = 0
        self.q = self.dq = self.tau = self.kp = self.kd = 0.0
        self.reserve = 0


class FakeLowCmd:
    def __init__(self):
        self.mode_pr = 0
        self.mode_machine = 0
        self.motor_cmd = [FakeMotorCmd() for _ in range(35)]
        self.reserve = [0, 0, 0, 0]
        self.crc = 0


def _reference_get_crc(msg) -> int:
    """Copia LITERAL de la implementación de test_mandar_modificado.py,
    usada como oráculo para detectar regresiones en crc.py."""
    buf = bytearray()
    buf += struct.pack('<B', int(msg.mode_pr))
    buf += struct.pack('<B', int(msg.mode_machine))
    buf += b'\x00\x00'
    for i in range(35):
        if i < len(msg.motor_cmd):
            cmd = msg.motor_cmd[i]
            mode, q, dq = int(cmd.mode), float(cmd.q), float(cmd.dq)
            tau, kp, kd = float(cmd.tau), float(cmd.kp), float(cmd.kd)
        else:
            mode, q, dq, tau, kp, kd = 0, 0.0, 0.0, 0.0, 0.0, 0.0
        buf += struct.pack('<B', mode)
        buf += b'\x00\x00\x00'
        buf += struct.pack('<f', q)
        buf += struct.pack('<f', dq)
        buf += struct.pack('<f', tau)
        buf += struct.pack('<f', kp)
        buf += struct.pack('<f', kd)
        buf += struct.pack('<I', 0)
    buf += struct.pack('<4I', 0, 0, 0, 0)
    assert len(buf) == 1000
    words = list(struct.unpack_from('<250I', bytes(buf)))
    crc_register = 0xFFFFFFFF
    poly = 0x04C11DB7
    for word in words:
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


def _make_cmd():
    msg = FakeLowCmd()
    msg.mode_pr = 0
    msg.mode_machine = 4
    for i in range(27):
        mc = msg.motor_cmd[i]
        mc.mode = 1
        mc.q = 0.1 * i
        mc.dq = -0.01 * i
        mc.tau = 0.5
        mc.kp = 100.0 if i < 13 else 50.0
        mc.kd = 1.0
    return msg


def test_serialized_length_is_1000():
    assert len(crc_mod.serialize_low_cmd(_make_cmd())) == 1000


def test_crc_matches_reference_implementation():
    msg = _make_cmd()
    assert crc_mod.compute_crc(msg) == _reference_get_crc(msg)


def test_attach_and_verify_roundtrip():
    msg = _make_cmd()
    crc_mod.attach_crc(msg)
    assert crc_mod.verify_crc(msg)


def test_verify_detects_field_tampering():
    msg = _make_cmd()
    crc_mod.attach_crc(msg)
    msg.motor_cmd[5].q += 1e-3       # corromper un campo tras firmar
    assert not crc_mod.verify_crc(msg)


def test_verify_detects_wrong_crc():
    msg = _make_cmd()
    crc_mod.attach_crc(msg)
    msg.crc ^= 0xDEADBEEF
    assert not crc_mod.verify_crc(msg)


def test_mode_and_header_affect_crc():
    msg = _make_cmd()
    c0 = crc_mod.compute_crc(msg)
    msg.mode_pr = 1
    assert crc_mod.compute_crc(msg) != c0


def test_fast_table_crc_equals_bitwise_reference_random():
    """El CRC por tabla debe coincidir con el algoritmo bit a bit de Unitree
    para mensajes con contenido aleatorio (equivalencia algebraica)."""
    import random
    import struct as _st
    rng = random.Random(42)
    for _ in range(5):
        msg = FakeLowCmd()
        msg.mode_pr = rng.randint(0, 1)
        msg.mode_machine = rng.randint(0, 255)
        for mc in msg.motor_cmd:
            mc.mode = rng.randint(0, 1)
            mc.q = rng.uniform(-3, 3)
            mc.dq = rng.uniform(-5, 5)
            mc.tau = rng.uniform(-50, 50)
            mc.kp = rng.uniform(0, 200)
            mc.kd = rng.uniform(0, 5)
        data = crc_mod.serialize_low_cmd(msg)
        words = _st.unpack_from('<250I', data)
        assert crc_mod.compute_crc(msg) == crc_mod._crc32_core(words)
