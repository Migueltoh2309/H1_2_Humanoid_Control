#!/usr/bin/env python3
"""Agente SDK del H1-2: recibe consignas del puente ROS 2 por UDP y las manda al
robot con el SDK de Unitree (rt/arm_sdk o rt/lowcmd) y a las manos Inspire por
Modbus TCP. Devuelve el estado (rt/lowstate + manos) por UDP.

EL MISMO PROGRAMA para el simulador y para el robot real; solo cambia donde
corre y por que red habla:

    # simulador (en este PC, dominio DDS 1 sobre lo, manos en 127.0.1.21x)
    python3 agente_sdk.py --sim
    # robot real (en el PC2 del robot, dominio 0 sobre eth0, manos 192.168.124.21x)
    python3 agente_sdk.py --iface eth0 --modo arm_sdk

Modos:
  arm_sdk  (por defecto) torso + brazos (12-26) por rt/arm_sdk con el peso en el
           slot 27. Las piernas las lleva el control de Unitree: robot DE PIE en
           FSM 201 (L2+UP). Al arrancar comprueba que el control interno publica
           rt/lowcmd; si no, arm_sdk no tendria efecto y sale.
  lowcmd   las 27 juntas por rt/lowcmd. Robot COLGADO y en Debug de verdad
           (CheckMode con name ''; ver Codigos/README.md). Las juntas que el PC
           no comanda se sostienen en la postura medida al arrancar.

Seguridad (todo vive aqui, en el lado del robot, para no depender de la WiFi):
  * rampa del peso de arm_sdk 0 -> 1 (12 s, siguiendo la postura medida con un
    tope de 2 grados, como caja_cuadrado.Brazo.rampa_peso) y 1 -> 0 al salir;
  * consignas recortadas a los limites con 0.05 rad de margen y con velocidad
    maxima (--vel-max rad/s), asi un salto en /joint_cmd no da un latigazo;
  * watchdog: sin ordenes --watchdog s -> sostiene la ultima consigna;
    sin ordenes --soltar-tras s -> en arm_sdk suelta (peso a 0);
  * motor en fallo o inclinacion > 10 grados -> suelta (arm_sdk) o amortigua (lowcmd);
  * orden SOLTAR del PC, Ctrl-C o SIGTERM -> rampa de salida y termina.

L2 + B en el mando sigue siendo la parada de emergencia del robot real.
"""
import argparse
import math
import os
import signal
import socket
import sys
import threading
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
for d in (AQUI, os.path.join(AQUI, "..", "comun")):
    if os.path.isfile(os.path.join(d, "protocolo_sim2real.py")):
        sys.path.insert(0, d)

import h1_2_juntas as J                      # noqa: E402
import protocolo_sim2real as P               # noqa: E402
from modbus_mini import Cliente, ErrorModbus  # noqa: E402

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,  # noqa: E402
                                         ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC     # noqa: E402

INCLINACION_ABORTO = 10.0    # grados (prueba_brazos_armsdk.py)
TOPE_SEGUIR = 0.035          # rad: cuanto sigue la rampa de entrada a la postura medida
KD_AMORTIGUAR = 2.0


def log(*a):
    # si la sesion ssh se cae, stdout deja de existir: el agente sigue (y su
    # watchdog suelta) en vez de morir por un BrokenPipe a mitad de movimiento
    try:
        print(time.strftime("%H:%M:%S"), *a, flush=True)
    except (OSError, ValueError):
        pass


class Estado:
    """Ultimo rt/lowstate."""

    def __init__(self):
        self.msg, self.t, self.n = None, 0.0, 0
        self.cerrojo = threading.Lock()

    def al_recibir(self, m):
        with self.cerrojo:
            self.msg, self.t, self.n = m, time.time(), self.n + 1

    def ultimo(self):
        with self.cerrojo:
            return self.msg, self.t


class Manos:
    """Las dos Inspire por Modbus. Hilo propio: escribe ANGLE_SET cuando cambia la
    consigna y lee angulo y fuerza. Si una mano no responde, reintenta cada 2 s
    sin parar el resto."""

    def __init__(self, ips, vel=None, hz=30.0):
        self.cli = {l: Cliente(ips[l], J.MANO_PUERTO, timeout=0.5) for l in J.LADOS}
        self.obj = [-1] * 12            # -1 = no tocar ese dof
        self.nuevo = False
        self.ang = [0] * 12
        self.fuerza = [0] * 12
        self.ok = {l: False for l in J.LADOS}
        self.vel = vel
        self.periodo = 1.0 / hz
        self.cerrojo = threading.Lock()
        self.vivo = True
        self.hilo = threading.Thread(target=self._bucle, daemon=True, name="manos")
        self.hilo.start()

    def mandar(self, mascara, angulos):
        with self.cerrojo:
            for k in range(12):
                if mascara & (1 << k):
                    a = max(0, min(1000, int(angulos[k])))
                    if a != self.obj[k]:
                        self.obj[k] = a
                        self.nuevo = True

    def _bucle(self):
        reintento = {l: 0.0 for l in J.LADOS}
        configurada = {l: False for l in J.LADOS}
        while self.vivo:
            t0 = time.time()
            with self.cerrojo:
                obj, nuevo, self.nuevo = list(self.obj), self.nuevo, False
            for k, lado in enumerate(J.LADOS):
                c = self.cli[lado]
                if not self.ok[lado] and time.time() < reintento[lado]:
                    continue
                try:
                    if not configurada[lado] and self.vel is not None:
                        c.escribir(J.SPEED_SET, [self.vel] * 6)
                    configurada[lado] = True
                    if nuevo and any(a >= 0 for a in obj[6 * k:6 * k + 6]):
                        c.escribir(J.ANGLE_SET, obj[6 * k:6 * k + 6])
                    self.ang[6 * k:6 * k + 6] = c.leer(J.ANGLE_ACT, 6)
                    self.fuerza[6 * k:6 * k + 6] = c.leer(J.FORCE_ACT, 6)
                    if not self.ok[lado]:
                        log(f"mano {lado} ({c.dir[0]}) responde")
                    self.ok[lado] = True
                except (OSError, ErrorModbus) as e:
                    if self.ok[lado] or reintento[lado] == 0.0:
                        log(f"mano {lado} ({c.dir[0]}) no responde: {e}")
                    self.ok[lado] = False
                    reintento[lado] = time.time() + 2.0
                    if nuevo:          # que se reintente la escritura
                        with self.cerrojo:
                            self.nuevo = True
            time.sleep(max(0.0, self.periodo - (time.time() - t0)))

    def parar(self):
        self.vivo = False
        self.hilo.join(timeout=1.0)
        for c in self.cli.values():
            c.cerrar()


class Agente:
    def __init__(self, a):
        self.a = a
        self.modo = P.MODO_ARM_SDK if a.modo == "arm_sdk" else P.MODO_LOWCMD
        self.ctrl = J.BRAZOS if self.modo == P.MODO_ARM_SDK else list(range(J.N))
        self.dt = 1.0 / a.hz

        self.est = Estado()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.est.al_recibir, 10)
        self.pub = ChannelPublisher("rt/arm_sdk" if self.modo == P.MODO_ARM_SDK else "rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.crc = CRC()

        # consignas del PC (lo que pide) y objetivo enviado (con limite de velocidad)
        self.cerrojo = threading.Lock()
        self.pedido_q = [0.0] * J.N
        self.pedido_dq = [0.0] * J.N
        self.pedido_tau = [0.0] * J.N
        self.comandada = [False] * J.N
        self.t_orden = 0.0
        self.n_ordenes = 0
        self.pc = None                   # (ip, puerto) al que se manda el estado
        self.pedir_soltar = False

        self.obj = [0.0] * J.N
        self.q_sostener = [0.0] * J.N
        self.peso = 0.0
        self.soltando = False
        self.amortiguar = False
        self.fallo = None
        self.mode_machine = 0

        self.manos = None if a.sin_manos else Manos(J.MANO_IP_SIM if a.sim else J.MANO_IP, a.vel_mano)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((a.escuchar, a.puerto))
        self.sock.settimeout(0.2)
        self.sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if a.pc:
            self.pc = (a.pc, a.puerto_estado)
        self.vivo = True
        self.seq_tx = 0

    # ------------------------------------------------------------ red
    def _rx(self):
        while self.vivo:
            try:
                datos, origen = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            o = P.desempaquetar_orden(datos)
            if o is None:
                continue
            if self.pc is None or (not self.a.pc and self.pc[0] != origen[0]):
                if self.pc is not None:
                    log(f"AVISO: ordenes desde otra IP {origen[0]} (antes {self.pc[0]}); el estado va ahora alli")
                self.pc = (origen[0], self.a.puerto_estado)
                log(f"PC conectado: {origen[0]} (estado -> UDP {self.a.puerto_estado})")
            if o["orden"] == P.SOLTAR:
                if not self.pedir_soltar:
                    log("orden SOLTAR del PC")
                self.pedir_soltar = True
                continue
            with self.cerrojo:
                m = o["mascara"]
                for i in range(J.N):
                    if m & (1 << i):
                        self.pedido_q[i] = J.recortar(i, o["q"][i])
                        self.pedido_dq[i] = o["dq"][i]
                        self.pedido_tau[i] = max(-self.a.tau_max, min(self.a.tau_max, o["tau"][i]))
                        self.comandada[i] = True
                self.t_orden = time.time()
                self.n_ordenes += 1
            if self.manos is not None and o["mascara_mano"]:
                self.manos.mandar(o["mascara_mano"], o["angulos"])

    def _tx_estado(self, m, banderas):
        if self.pc is None or m is None:
            return
        ms = m.motor_state
        imu = m.imu_state
        manos = self.manos
        datos = P.empaquetar_estado(
            self.seq_tx, time.time_ns(), self.modo, banderas, m.mode_machine, self.peso,
            [ms[i].q for i in range(J.N)], [ms[i].dq for i in range(J.N)],
            [ms[i].tau_est for i in range(J.N)], self.obj,
            list(imu.quaternion), list(imu.rpy), list(imu.gyroscope), list(imu.accelerometer),
            manos.ang if manos else [0] * 12, manos.fuerza if manos else [0] * 12)
        self.seq_tx += 1
        try:
            self.sock_tx.sendto(datos, self.pc)
        except OSError:
            pass

    # ------------------------------------------------------------ DDS
    def _publicar(self, kp_escala=1.0, solo_amortiguar=False):
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_pr = 0
        cmd.mode_machine = self.mode_machine
        ahora = time.time()
        fresco = ahora - self.t_orden < self.a.watchdog
        with self.cerrojo:
            dq = list(self.pedido_dq)
            tau = list(self.pedido_tau)
        for i in self.ctrl:
            c = cmd.motor_cmd[i]
            c.mode = 1
            if solo_amortiguar:
                c.q, c.dq, c.tau, c.kp, c.kd = 0.0, 0.0, 0.0, 0.0, KD_AMORTIGUAR
                continue
            c.q = self.obj[i]
            # dq/tau del PC solo si la orden es fresca y no estamos limitando velocidad
            lim = abs(self.obj[i] - self.pedido_q[i]) > 1e-4
            c.dq = dq[i] if (fresco and self.comandada[i] and not lim) else 0.0
            c.tau = tau[i] if (fresco and self.comandada[i]) else 0.0
            c.kp = J.KP[i] * kp_escala
            c.kd = J.KD[i]
        if self.modo == P.MODO_ARM_SDK:
            cmd.motor_cmd[J.PESO_ARM_SDK].q = self.peso
        cmd.crc = self.crc.Crc(cmd)
        self.pub.Write(cmd)

    def _avanzar_objetivo(self):
        paso = self.a.vel_max * self.dt
        with self.cerrojo:
            for i in self.ctrl:
                dest = self.pedido_q[i] if self.comandada[i] else self.q_sostener[i]
                d = dest - self.obj[i]
                self.obj[i] += max(-paso, min(paso, d))

    def _vigilar(self, m, t_m):
        if time.time() - t_m > 0.5:
            return "rt/lowstate parado"
        fallos = [i for i in range(J.N) if m.motor_state[i].motorstate != 0]
        if fallos:
            return f"motor en fallo {fallos}"
        if self.modo == P.MODO_ARM_SDK:
            r, p = (math.degrees(v) for v in m.imu_state.rpy[:2])
            if abs(r) > INCLINACION_ABORTO or abs(p) > INCLINACION_ABORTO:
                return f"inclinacion roll {r:+.1f} pitch {p:+.1f}"
        return None

    # ------------------------------------------------------------ fases
    def esperar_lowstate(self):
        t0 = time.time()
        while self.est.ultimo()[0] is None:
            if time.time() - t0 > 5.0:
                sys.exit("No llega rt/lowstate (¿interfaz/dominio? ¿simulador arrancado?).")
            time.sleep(0.05)
        m, _ = self.est.ultimo()
        self.mode_machine = m.mode_machine
        fallos = [i for i in range(J.N) if m.motor_state[i].motorstate != 0]
        log(f"rt/lowstate OK · mode_machine={m.mode_machine} · motores en fallo: {fallos or 'ninguno'}")
        if fallos:
            sys.exit("Motores en fallo: no se arranca.")
        q = [m.motor_state[i].q for i in range(J.N)]
        self.q_sostener = list(q)
        self.obj = list(q)

    def comprobar_control_interno(self):
        """arm_sdk solo tiene efecto si el control de Unitree publica rt/lowcmd
        (FSM 201). Mismo control que prueba_brazos_armsdk.py."""
        n = [0]
        espia = ChannelSubscriber("rt/lowcmd", LowCmd_)
        espia.Init(lambda _m: n.__setitem__(0, n[0] + 1), 10)
        time.sleep(1.0)
        espia.Close()
        log(f"control interno en rt/lowcmd: {n[0]} mensajes en 1 s")
        if n[0] < 50:
            sys.exit("El control interno no publica rt/lowcmd (¿FSM 201? ¿modo Debug?): arm_sdk no tendria efecto.")

    def rampa_peso(self, hasta, segundos):
        """Rampa del peso de arm_sdk en `segundos` de reloj (no en numero de ciclos:
        cada publicacion cuesta ~1 ms de CRC + serializacion y el bucle se alarga)."""
        desde = self.peso
        subir = hasta > desde
        q_ini = list(self.obj)
        t0 = time.monotonic()
        n = 0
        while True:
            x = min(1.0, (time.monotonic() - t0) / max(segundos, 1e-3))
            m, t_m = self.est.ultimo()
            if subir:
                razon = self._vigilar(m, t_m)
                if razon:
                    return razon
                for i in self.ctrl:
                    self.obj[i] = q_ini[i] + max(-TOPE_SEGUIR, min(TOPE_SEGUIR, m.motor_state[i].q - q_ini[i]))
                    self.q_sostener[i] = self.obj[i]
            self.peso = desde + (hasta - desde) * x
            self._publicar()
            n += 1
            if n % max(1, int(self.a.hz / self.a.hz_estado)) == 0:
                self._tx_estado(m, P.B_LOWSTATE | P.B_ACTIVO | (P.B_SOLTANDO if not subir else 0))
            if x >= 1.0:
                return None
            time.sleep(self.dt)

    def bucle(self):
        k_tx = max(1, int(round(self.a.hz / self.a.hz_estado)))
        aviso_wd = False
        tick = 0
        t_sig = time.monotonic()
        while self.vivo and not self.pedir_soltar:
            m, t_m = self.est.ultimo()
            razon = self._vigilar(m, t_m)
            if razon:
                self.fallo = razon
                log(f"PARADA: {razon}")
                return
            ahora = time.time()
            sin_ordenes = ahora - self.t_orden if self.n_ordenes else float("inf")
            if self.modo == P.MODO_ARM_SDK and self.n_ordenes and sin_ordenes > self.a.soltar_tras:
                log(f"sin ordenes del PC en {sin_ordenes:.1f} s: suelto arm_sdk")
                return
            if self.n_ordenes and sin_ordenes > self.a.watchdog and not aviso_wd:
                log(f"watchdog: sin ordenes en {self.a.watchdog} s, sostengo la ultima consigna")
                aviso_wd = True
            elif sin_ordenes < self.a.watchdog and aviso_wd:
                log("ordenes de nuevo")
                aviso_wd = False
            self._avanzar_objetivo()
            self._publicar()
            tick += 1
            if tick % k_tx == 0:
                b = P.B_LOWSTATE | P.B_ACTIVO
                if sin_ordenes < self.a.watchdog:
                    b |= P.B_ORDENES
                if self.manos is not None and all(self.manos.ok.values()):
                    b |= P.B_MANOS
                self._tx_estado(m, b)
            t_sig += self.dt
            espera = t_sig - time.monotonic()
            if espera > 0:
                time.sleep(espera)
            else:
                t_sig = time.monotonic()

    def salir(self, rapido):
        if self.modo == P.MODO_ARM_SDK:
            t = 2.0 if rapido else self.a.t_salida
            log(f"arm_sdk: peso {self.peso:.2f} -> 0 en {t:.0f} s")
            self.rampa_peso(0.0, t)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 0.5:           # medio segundo mas con peso 0
                self._publicar()
                time.sleep(self.dt)
        else:
            # lowcmd: el robot esta colgado. Amortiguacion
            # (kp 0, kd 2) 2 s: los brazos bajan despacio en vez de caer.
            log("lowcmd: amortiguacion (kp 0, kd 2) 2 s antes de salir")
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.0:
                self._publicar(solo_amortiguar=True)
                time.sleep(self.dt)
        m, _ = self.est.ultimo()
        self._tx_estado(m, P.B_LOWSTATE | P.B_SOLTANDO | (P.B_FALLO if self.fallo else 0))

    def parar(self):
        self.vivo = False
        if self.manos:
            self.manos.parar()
        self.sock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="contra el simulador: dominio 1 en lo, manos 127.0.1.21x")
    ap.add_argument("--iface", default="eth0", help="interfaz DDS del robot real (eth0)")
    ap.add_argument("--modo", choices=["arm_sdk", "lowcmd"], default="arm_sdk")
    ap.add_argument("--hz", type=float, default=None,
                    help="frecuencia de rt/arm_sdk o rt/lowcmd (por defecto 100 en arm_sdk, como los 50 Hz ya "
                         "probados de caja_cuadrado; 250 en lowcmd)")
    ap.add_argument("--hz-estado", type=float, default=50.0, help="estado al PC por UDP")
    ap.add_argument("--vel-max", type=float, default=1.0, help="rad/s maximos de la consigna enviada")
    ap.add_argument("--tau-max", type=float, default=10.0, help="N·m maximos de par feedforward del PC")
    ap.add_argument("--t-rampa", type=float, default=12.0, help="s de la rampa de entrada de arm_sdk")
    ap.add_argument("--t-salida", type=float, default=5.0, help="s de la rampa de salida de arm_sdk")
    ap.add_argument("--watchdog", type=float, default=0.5, help="s sin ordenes -> sostener")
    ap.add_argument("--soltar-tras", type=float, default=3.0, help="s sin ordenes -> soltar (solo arm_sdk)")
    ap.add_argument("--sin-manos", action="store_true")
    ap.add_argument("--vel-mano", type=int, default=None, help="SPEED_SET de las manos (0-1000); sin tocar si no se da")
    ap.add_argument("--escuchar", default="0.0.0.0")
    ap.add_argument("--puerto", type=int, default=P.PUERTO_ORDEN)
    ap.add_argument("--puerto-estado", type=int, default=P.PUERTO_ESTADO)
    ap.add_argument("--pc", default=None, help="IP del PC para el estado (si no, la de quien mande ordenes)")
    a = ap.parse_args()
    if a.hz is None:
        a.hz = 100.0 if a.modo == "arm_sdk" else 250.0

    if a.sim:
        ChannelFactoryInitialize(1, "lo")
        log("SIMULADOR: dominio DDS 1 sobre lo")
    else:
        ChannelFactoryInitialize(0, a.iface)
        log(f"ROBOT REAL: dominio DDS 0 sobre {a.iface} · modo {a.modo}")
        print("=" * 66)
        print("  ESTO MUEVE EL ROBOT: " + ("torso y brazos (rt/arm_sdk)" if a.modo == "arm_sdk"
                                           else "TODAS las juntas (rt/lowcmd, robot COLGADO)"))
        print("  L2 + B en el mando es la parada de emergencia. Tenlo en la mano.")
        print("=" * 66, flush=True)

    ag = Agente(a)

    def senal(_s, _f):
        ag.pedir_soltar = True
    signal.signal(signal.SIGTERM, senal)
    # colgar el ssh (WiFi caida) NO mata al agente: sigue y el watchdog decide
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    rapido = False
    threading.Thread(target=ag._rx, daemon=True, name="udp_rx").start()
    try:
        ag.esperar_lowstate()
        if ag.modo == P.MODO_ARM_SDK:
            ag.comprobar_control_interno()
            log(f"arm_sdk: peso 0 -> 1 en {a.t_rampa:.0f} s sosteniendo la postura actual")
            razon = ag.rampa_peso(1.0, a.t_rampa)
            if razon:
                ag.fallo = razon
                log(f"PARADA en la rampa: {razon}")
                rapido = True
        if not ag.fallo:
            log(f"listo: escuchando ordenes en UDP {a.puerto} ({len(ag.ctrl)} juntas, "
                f"{a.hz:.0f} Hz, vel max {a.vel_max} rad/s)")
            ag.bucle()
            rapido = ag.fallo is not None
    except KeyboardInterrupt:
        log("Ctrl-C")
        rapido = True
    finally:
        try:
            ag.salir(rapido)
        except KeyboardInterrupt:
            pass
        ag.parar()
        log("agente terminado" + (f" ({ag.fallo})" if ag.fallo else ""))


if __name__ == "__main__":
    main()
