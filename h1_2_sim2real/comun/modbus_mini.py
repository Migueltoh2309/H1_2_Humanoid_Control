"""Modbus TCP minimo (cliente y servidor) para las manos Inspire.

Solo libreria estandar, para que el agente no dependa de la version de pymodbus
del robot y para que el simulador pueda hacerse pasar por las manos reales:
el servidor escucha en 127.0.1.211:6000 (izq) y 127.0.1.210:6000 (der), y el
mismo cliente (o los scripts de Codigos/manos con --ip) habla con el igual que
con la mano de verdad.

Funciones: 3 (leer holding registers) y 16 (escribir varios registros), las
dos unicas que usan los scripts de Codigos (Mano.regs / Mano.escribir). La 6
(escribir uno) tambien se acepta en el servidor.
"""
import socket
import socketserver
import struct
import threading


class ErrorModbus(Exception):
    pass


class Cliente:
    """Cliente bloqueante, un socket. unidad = 1 como en caja_cuadrado.Mano."""

    def __init__(self, ip, puerto=6000, timeout=1.0, unidad=1):
        self.dir = (ip, puerto)
        self.timeout = timeout
        self.unidad = unidad
        self.s = None
        self.tid = 0
        self.cerrojo = threading.Lock()

    def conectar(self):
        self.cerrar()
        self.s = socket.create_connection(self.dir, timeout=self.timeout)
        self.s.settimeout(self.timeout)

    def cerrar(self):
        if self.s is not None:
            try:
                self.s.close()
            except OSError:
                pass
            self.s = None

    def _recibir(self, n):
        b = b""
        while len(b) < n:
            t = self.s.recv(n - len(b))
            if not t:
                raise ErrorModbus("conexion cerrada")
            b += t
        return b

    def _pedir(self, pdu):
        with self.cerrojo:
            if self.s is None:
                self.conectar()
            self.tid = (self.tid + 1) & 0xFFFF
            try:
                self.s.sendall(struct.pack(">HHHB", self.tid, 0, len(pdu) + 1, self.unidad) + pdu)
                tid, _, n, _ = struct.unpack(">HHHB", self._recibir(7))
                r = self._recibir(n - 1)
            except (OSError, ErrorModbus):
                self.cerrar()
                raise
            if r[0] & 0x80:
                raise ErrorModbus(f"excepcion modbus {r[1]} en la funcion {r[0] & 0x7F}")
            return r

    def leer(self, direccion, n, con_signo=True):
        r = self._pedir(struct.pack(">BHH", 3, direccion, n))
        return list(struct.unpack(">%d%s" % (n, "h" if con_signo else "H"), r[2:2 + 2 * n]))

    def escribir(self, direccion, valores):
        v = [int(x) & 0xFFFF for x in valores]
        self._pedir(struct.pack(">BHHB%dH" % len(v), 16, direccion, len(v), 2 * len(v), *v))


class _Manejador(socketserver.BaseRequestHandler):
    def handle(self):
        mapa = self.server.mapa
        s = self.request
        while True:
            cab = _leer_exacto(s, 7)
            if cab is None:
                return
            tid, proto, n, unidad = struct.unpack(">HHHB", cab)
            pdu = _leer_exacto(s, n - 1)
            if pdu is None:
                return
            f = pdu[0]
            try:
                if f == 3:
                    d, k = struct.unpack(">HH", pdu[1:5])
                    vals = mapa.leer(d, k)
                    resp = struct.pack(">BB%dH" % k, 3, 2 * k, *[v & 0xFFFF for v in vals])
                elif f == 16:
                    d, k, _ = struct.unpack(">HHB", pdu[1:6])
                    mapa.escribir(d, struct.unpack(">%dH" % k, pdu[6:6 + 2 * k]))
                    resp = struct.pack(">BHH", 16, d, k)
                elif f == 6:
                    d, v = struct.unpack(">HH", pdu[1:5])
                    mapa.escribir(d, [v])
                    resp = struct.pack(">BHH", 6, d, v)
                else:
                    resp = struct.pack(">BB", f | 0x80, 1)
            except Exception:
                resp = struct.pack(">BB", f | 0x80, 4)
            s.sendall(struct.pack(">HHHB", tid, proto, len(resp) + 1, unidad) + resp)


def _leer_exacto(s, n):
    b = b""
    while len(b) < n:
        try:
            t = s.recv(n - len(b))
        except OSError:
            return None
        if not t:
            return None
        b += t
    return b


class _Servidor(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def servir(ip, puerto, mapa):
    """Arranca un servidor Modbus TCP en un hilo. `mapa` tiene leer(dir, n) y
    escribir(dir, valores_uint16). Devuelve el servidor (shutdown() para pararlo)."""
    srv = _Servidor((ip, puerto), _Manejador)
    srv.mapa = mapa
    threading.Thread(target=srv.serve_forever, daemon=True, name=f"modbus_{ip}").start()
    return srv
