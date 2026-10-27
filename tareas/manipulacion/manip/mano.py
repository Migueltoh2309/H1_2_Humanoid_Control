"""Bloque MANO: órdenes de alto nivel -> Modbus TCP de la Inspire RH56DFTP (sec. 5).

Patrones de caja_cuadrado.Mano y manos/mano.py (probados en el robot):
  * registros leídos de uno en uno POR GRUPO: leer 1546..1620 de un bloque da basura
    (fuerzas de 11000 gf y error 255, sec. 7);
  * FORCE_SET (gf por dedo) se escribe SIEMPRE antes de cerrar y nunca con el de fábrica
    (2000 = sin límite); máximo 800 gf (sec. 10); se relee para comprobarlo;
  * para sujetar, la consigna se deja CERRADA con la fuerza limitada: fijarla en el ángulo
    actual suelta el objeto al mover el brazo (sec. 7);
  * corte: una fuerza medida > 1200 gf abre la mano (sec. 10);
  * error 5 / status 7: escribir 1 en 1004 y volver a cerrar (sec. 7);
  * velocidad 150 (caja_cuadrado), nunca SAVE.
Orden de los DOF: [meñique, anular, medio, índice, flexión del pulgar, rotación del pulgar];
0 = cerrado, 1000 = abierto. IPs: .211 izquierda, .210 derecha (al revés que la doc).
"""
import os
import sys
import threading
import time

import numpy as np

from .tipos import EstadoMano


def _comun():
    d = os.environ.get("H12_S2R_COMUN")
    cands = [d] if d else []
    aqui = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.normpath(os.path.join(aqui, "..", "..", "..", "h1_2_sim2real", "comun")),
              os.path.expanduser("~/humanoid_ws/src/h1_2_sim2real/comun")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "modbus_mini.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return
    raise ImportError("no encuentro h1_2_sim2real/comun (modbus_mini, h1_2_juntas)")


_comun()
import h1_2_juntas as J  # noqa: E402
import modbus_mini  # noqa: E402

FORCE_SET_MAX = 800
CORTE_GF = 1200
DEDOS = ["menique", "anular", "medio", "indice", "pulgar", "rot_pulgar"]
# zonas táctiles por dedo (inspire_hand_defaut.py, como caja_cuadrado.ZONAS): (dirección, n)
ZONAS = {"menique": [(3000, 9), (3018, 96), (3210, 80)], "anular": [(3370, 9), (3388, 96), (3580, 80)],
         "medio": [(3740, 9), (3758, 96), (3950, 80)], "indice": [(4110, 9), (4128, 96), (4320, 80)],
         "pulgar": [(4480, 9), (4498, 96), (4690, 9), (4708, 96)], "palma": [(4900, 112)]}


class ErrorMano(Exception):
    pass


def _bytes(regs):
    return [int(b) for r in regs for b in (((int(r) & 0xFFFF) >> 8) & 0xFF, int(r) & 0xFF)]


class Mano:
    def __init__(self, lado, sim, velocidad=150, timeout=1.0):
        self.lado = lado
        ip = (J.MANO_IP_SIM if sim else J.MANO_IP)[lado]
        self.c = modbus_mini.Cliente(ip, J.MANO_PUERTO, timeout=timeout)
        self.c.conectar()
        self.cerrojo = threading.Lock()
        self.reposo_tactil = {}
        self.force_set_actual = None
        self._escribir(J.SPEED_SET, [velocidad] * 6)

    # ------------------------------------------------------------ registros
    def _leer(self, d, n, signo=True):
        with self.cerrojo:
            return np.array(self.c.leer(d, n, signo), dtype=float)

    def _escribir(self, d, vals):
        with self.cerrojo:
            self.c.escribir(d, [int(v) & 0xFFFF for v in vals])

    def angulos(self):
        return self._leer(J.ANGLE_ACT, 6)

    def fuerzas(self):
        return self._leer(J.FORCE_ACT, 6)

    def estado(self):
        try:
            a, f = self.angulos(), self.fuerzas()
            st = np.array(_bytes(self._leer(J.STATUS, 3, False)))
            er = np.array(_bytes(self._leer(J.ERR, 3, False)))
            return EstadoMano(time.monotonic(), self.lado, a, f, st, er, responde=True)
        except Exception:
            return EstadoMano(time.monotonic(), self.lado, np.full(6, np.nan), np.full(6, np.nan),
                              np.full(6, -1), np.full(6, -1), responde=False)

    def tactil(self, dedo):
        """Máximo de cada zona del dedo sobre su reposo (medido con tactil_reposo())."""
        out = []
        for k, (d, n) in enumerate(ZONAS[dedo]):
            v = self._leer(d, n, False)
            out.append(float(v.max() - self.reposo_tactil.get((dedo, k), 0.0)))
        return max(out)

    def tactil_reposo(self, dedos=("indice", "pulgar", "palma"), muestras=5):
        """Reposo de los táctiles que se van a usar (leer solo lo que haga falta: el táctil es
        lento, sec. 7)."""
        for dedo in dedos:
            for k, (d, n) in enumerate(ZONAS[dedo]):
                vals = [self._leer(d, n, False).max() for _ in range(muestras)]
                self.reposo_tactil[(dedo, k)] = float(np.mean(vals) + 2 * np.std(vals))

    # ------------------------------------------------------------ órdenes
    def poner_force_set(self, gf):
        gf = [int(min(max(g, 0), FORCE_SET_MAX)) for g in (gf if np.ndim(gf) else [gf] * 6)]
        self._escribir(J.FORCE_SET, gf)
        leido = self._leer(J.FORCE_SET, 6).astype(int).tolist()
        if leido != gf:
            raise ErrorMano(f"FORCE_SET no se escribió: {leido} != {gf}")
        self.force_set_actual = gf
        return gf

    def mover(self, angulos):
        """ANGLE_SET; -1 en un DOF = no tocarlo (como caja_cuadrado)."""
        self._escribir(J.ANGLE_SET, [int(a) if a >= 0 else 0xFFFF for a in angulos])

    def preforma(self, preforma):
        self.mover(preforma)

    def cerrar(self, cierre, force_set):
        """FORCE_SET primero (siempre), luego la consigna CERRADA."""
        self.poner_force_set(force_set)
        self.mover(cierre)

    def abrir(self, rot_pulgar=None):
        """Dedos y flexión del pulgar abiertos; la rotación del pulgar se deja (o se pone)."""
        self.mover([1000, 1000, 1000, 1000, 1000, -1 if rot_pulgar is None else rot_pulgar])

    def borrar_error(self):
        self._escribir(J.BORRAR_ERROR, [1])

    def esperar_quieta(self, timeout=3.0, tol=5.0):
        """Hasta que los ángulos dejan de cambiar (dedos parados: en posición o por fuerza)."""
        t0 = time.monotonic()
        ant = self.angulos()
        while time.monotonic() - t0 < timeout:
            time.sleep(0.15)
            a = self.angulos()
            if np.max(np.abs(a - ant)) < tol:
                return a
            ant = a
        return ant

    def agarre(self, f_min=None, a_vacio=60, dedos=(0, 1, 2, 3, 4), dedos_min=1):
        """¿Hay algo entre los dedos? (pregunta 6.4.3) Un dedo que se para lejos de 0 con
        fuerza tiene algo delante; uno que llega a 0 está vacío. Hace falta el pulgar y al menos
        `dedos_min` dedos más. Devuelve (ok, detalle)."""
        a, f = self.angulos(), self.fuerzas()
        fs = self.force_set_actual or [300] * 6
        tocan = [k for k in dedos if a[k] > a_vacio and f[k] >= (f_min if f_min else 0.5 * fs[k])]
        pulgar = 4 in tocan
        ok = pulgar and len([k for k in tocan if k < 4]) >= dedos_min
        return ok, {"angulos": a.tolist(), "fuerzas": f.tolist(), "tocan": [DEDOS[k] for k in tocan]}

    def vigilar_corte(self):
        f = self.fuerzas()
        if np.nanmax(f) > CORTE_GF:
            self.abrir()
            raise ErrorMano(f"fuerza {np.nanmax(f):.0f} gf > {CORTE_GF}: mano abierta")
        return f

    def cerrar_conexion(self):
        self.c.cerrar()
