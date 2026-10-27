"""Bloque BRAZO: trayectorias articulares -> rt/arm_sdk a frecuencia fija (sec. 5).

Patrones de caja_cuadrado.Brazo (probados en el robot), en un bloque aparte:
  * un hilo publica a `hz` la consigna de cintura + 14 juntas de brazo y el peso en
    motor_cmd[27].q. El peso SOLO lo cambian rampa_entrada / rampa_salida, y solo las llama
    el supervisor (sec. 5);
  * rampa de entrada de 12 s con la consigna siguiendo la postura medida, con tope de 2
    grados (con la postura fija la política dio pasos; copiándola sin tope el codo cayó
    18 grados); rampa de salida >= 5 s que termina en 0.00 de verdad (el ejemplo oficial se
    queda en 0.6, sec. 7);
  * movimientos con perfil de mínima sacudida (v = a = 0 en los extremos), con duración
    >= max(duracion_min, 1.875 · |Δq|max / v_max): con mínima sacudida el pico de velocidad
    es 1.875 · Δq / T, así que así nunca se pasa de v_max (0.5 rad/s, sec. 10);
  * la cintura (índice 12) recibe su postura medida al empezar: no se mueve (6.3.6);
  * consignas recortadas a los límites del URDF con margen;
  * `simulacro`: calcula y registra todo pero NO publica (hito 3).

Las posturas son de 7 juntas en el orden de la tesis por brazo: {'izq': q7, 'der': q7}.
"""
import threading
import time

import numpy as np

from . import cinematica as C
from .robot_io import BRAZOS, CINTURA, INDICES, PESO_ARM_SDK


class Parada(Exception):
    """La lanza un `comprobar` del supervisor para cortar un movimiento."""


def minima_sacudida(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


class Brazo:
    def __init__(self, estado, cfg, simulacro=False, comp=None):
        self.estado, self.cfg, self.simulacro = estado, cfg, simulacro
        # compensación del error de seguimiento (manip/compensacion.py, 6.3.4): self.obj es la
        # consigna DESEADA (la del plan, contra la que se mide el error); lo publicado lleva
        # además el desvío del modelo y del lazo, escalado con el peso (sin salto en la rampa)
        self.comp = comp
        self.hz = float(cfg["hz"])
        self.kp = {j: (cfg["kp_cintura"] if j == CINTURA else cfg["kp"]) for j in BRAZOS}
        self.kd = {j: (cfg["kd_cintura"] if j == CINTURA else cfg["kd"]) for j in BRAZOS}
        assert max(v for j, v in self.kp.items() if j != CINTURA) <= 150.0, "kp del brazo > 150 (sec. 10)"
        q0 = estado.q()
        self.obj = {j: float(q0[j]) for j in BRAZOS}       # consigna actual (15)
        self.peso = 0.0
        self.cerrojo = threading.Lock()
        self.vivo = True
        self.n_pub = 0
        lim = {lado: C.limites(lado, None) for lado in C.LADOS}
        mg = float(cfg["margen_limites"])
        self.lim = {}
        for lado, idx in INDICES.items():
            for k, j in enumerate(idx):
                self.lim[j] = (lim[lado][0][k] + mg, lim[lado][1][k] - mg)
        if not simulacro:
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.utils.crc import CRC
            self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self.pub.Init()
            self.crc = CRC()
            self.mode_machine = estado.mode_machine()
        self.ultima_publicada = dict(self.obj)
        self.hilo = threading.Thread(target=self._bucle, daemon=True, name="arm_sdk")
        self.hilo.start()

    # ------------------------------------------------------------ publicación
    def _bucle(self):
        periodo = 1.0 / self.hz
        t_sig = time.monotonic()
        while self.vivo:
            with self.cerrojo:
                obj, peso = dict(self.obj), self.peso
            obj = self.publicada(obj, peso)
            self.ultima_publicada = obj
            if not self.simulacro:
                from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
                cmd = unitree_hg_msg_dds__LowCmd_()
                cmd.mode_pr = 0
                cmd.mode_machine = self.mode_machine
                for j, q in obj.items():
                    c = cmd.motor_cmd[j]
                    c.mode, c.q, c.dq, c.tau, c.kp, c.kd = 1, q, 0.0, 0.0, self.kp[j], self.kd[j]
                cmd.motor_cmd[PESO_ARM_SDK].q = peso
                cmd.crc = self.crc.Crc(cmd)
                self.pub.Write(cmd)
            self.n_pub += 1
            t_sig += periodo
            time.sleep(max(0.0, t_sig - time.monotonic()))

    def publicada(self, obj, peso):
        """Consigna que sale por rt/arm_sdk: la deseada + el desvío de la compensación."""
        if self.comp is None or not self.comp.activo:
            return obj
        out = dict(obj)
        for lado, idx in INDICES.items():
            q = np.array([obj[j] for j in idx])
            dv = self.comp.desvio(lado, q) * peso
            for j, v in zip(idx, q + dv):
                out[j] = self.recortar(j, float(v))
        return out

    def asentar(self, lado, segundos=0.6, veces=2, comprobar=None):
        """Parada con el brazo quieto: corrige con los encoders (lazo de la compensación).
        Devuelve el error articular máximo (rad) antes de la última corrección."""
        err = 0.0
        for _ in range(veces):
            fin = time.monotonic() + segundos
            while time.monotonic() < fin:
                if comprobar:
                    comprobar()
                time.sleep(1.0 / self.hz)
            q_med = self.estado.brazo(lado)
            if self.comp is None:
                return float(np.max(np.abs(self.consigna(lado) - q_med)))
            err = float(np.max(np.abs(self.comp.corregir(lado, self.consigna(lado), q_med))))
        return err

    # ------------------------------------------------------------ peso (solo el supervisor)
    def rampa_entrada(self, comprobar=None):
        """Peso 0 -> 1 en `rampa_entrada` s; la consigna sigue la postura medida con tope."""
        tope = float(self.cfg["tope_seguir"])
        q_ini = dict(self.obj)
        pasos = max(1, int(self.cfg["rampa_entrada"] * self.hz))
        for k in range(1, pasos + 1):
            q = self.estado.q()
            with self.cerrojo:
                for j in BRAZOS:
                    self.obj[j] = q_ini[j] + float(np.clip(q[j] - q_ini[j], -tope, tope))
                self.peso = k / pasos
            if comprobar:
                comprobar()
            time.sleep(1.0 / self.hz)

    def rampa_salida(self, segundos=None):
        """Peso -> 0.00 en >= rampa_salida s (no se interrumpe: es la salida segura)."""
        segundos = max(float(self.cfg["rampa_salida"]), segundos or 0.0)
        desde = self.peso
        pasos = max(1, int(segundos * self.hz))
        for k in range(1, pasos + 1):
            with self.cerrojo:
                self.peso = desde * (1.0 - k / pasos)
            time.sleep(1.0 / self.hz)
        with self.cerrojo:
            self.peso = 0.0
        time.sleep(5.0 / self.hz)                # unos ciclos más publicando 0.00

    # ------------------------------------------------------------ movimiento
    def consigna(self, lado=None):
        with self.cerrojo:
            if lado is None:
                return dict(self.obj)
            return np.array([self.obj[j] for j in INDICES[lado]])

    def duracion(self, poses):
        """Duración mínima para ir de la consigna actual a `poses` sin pasar de v_max."""
        dq = 0.0
        for lado, q in poses.items():
            dq = max(dq, float(np.max(np.abs(np.asarray(q) - self.consigna(lado)))))
        return max(float(self.cfg["duracion_min"]), 1.875 * dq / float(self.cfg["v_max"]))

    def recortar(self, j, q):
        lo, hi = self.lim[j]
        return float(min(max(q, lo), hi))

    def mover(self, poses, duracion=None, comprobar=None):
        """Mínima sacudida desde la consigna actual hasta `poses` ({'izq': q7, 'der': q7};
        el brazo que no aparece se queda). Bloquea `duracion` s (>= la mínima)."""
        T = max(self.duracion(poses), duracion or 0.0)
        ini = self.consigna()
        fin = dict(ini)
        for lado, q in poses.items():
            for j, v in zip(INDICES[lado], q):
                fin[j] = self.recortar(j, float(v))
        pasos = max(1, int(T * self.hz))
        for k in range(1, pasos + 1):
            s = minima_sacudida(k / pasos)
            with self.cerrojo:
                for j in BRAZOS:
                    self.obj[j] = ini[j] + (fin[j] - ini[j]) * s
            if comprobar:
                comprobar()
            time.sleep(1.0 / self.hz)
        return T

    def seguir_ruta(self, lado, ruta, duracion=None, comprobar=None):
        """Recorre los puntos de paso `ruta` (Nx7, el primero = consigna actual) con un único
        perfil de mínima sacudida sobre la longitud de arco articular: arranca y para en
        reposo y nunca pasa de v_max (la duración se escala con la longitud total)."""
        ruta = np.vstack([self.consigna(lado), np.asarray(ruta, dtype=float)])
        seg = np.linalg.norm(np.diff(ruta, axis=0), axis=1)
        largo = np.concatenate([[0.0], np.cumsum(seg)])
        if largo[-1] < 1e-9:
            return 0.0
        # |dq_j/ds| <= 1 con s la longitud de arco: la velocidad de cada junta <= 1.875 L / T
        T = max(float(self.cfg["duracion_min"]), 1.875 * largo[-1] / float(self.cfg["v_max"]), duracion or 0.0)
        pasos = max(1, int(T * self.hz))
        for k in range(1, pasos + 1):
            s = minima_sacudida(k / pasos) * largo[-1]
            i = min(int(np.searchsorted(largo, s, side="right")) - 1, len(seg) - 1)
            a = (s - largo[i]) / max(seg[i], 1e-12)
            q = ruta[i] + a * (ruta[i + 1] - ruta[i])
            with self.cerrojo:
                for j, v in zip(INDICES[lado], q):
                    self.obj[j] = self.recortar(j, float(v))
            if comprobar:
                comprobar()
            time.sleep(1.0 / self.hz)
        return T

    def error_seguimiento(self):
        """consigna - medida de las 15 juntas (rad)."""
        q = self.estado.q()
        with self.cerrojo:
            return np.array([self.obj[j] - q[j] for j in BRAZOS])

    def parar(self):
        self.vivo = False
        self.hilo.join(timeout=1.0)
