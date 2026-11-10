#!/usr/bin/env python3
"""Reproduce, contra el agente SDK, el flujo de comandos de un episodio de la
faja grabado con `bc_faja.py ... --robot --registro DIR` (docs/LFD_ROBOT.md).

Es el paso 2 hacia el robot: las MISMAS órdenes UDP que mandaría el PC
(q, dq, tau_ff de las 14 juntas de brazo + 12 ángulos Inspire) llegan a
`h1_2_sim2real/robot/agente_sdk.py`, que las pasa a rt/arm_sdk y a Modbus.
En lazo abierto: sin cámara ni faja, el brazo repite los movimientos del
episodio ("en vacío"). Sirve para ver, antes de cerrar el lazo en el robot,
si el robot (o el sim de h1_2_sim2real, que tiene motores PD puros como el
robot) sigue lo que el QP pidió.

Habla con el agente directamente (sin el puente ROS 2): NO correr a la vez
que `puente.launch.py` (los dos usan el puerto de estado 47931).

    # terminal 1: el robot simulado
    ~/humanoid_ws/src/h1_2_sim2real/scripts/sim.sh robot use_viewer:=true
    # terminal 2: el agente contra el sim (con la velocidad que necesita el episodio)
    python3 robot/agente_sdk.py --sim --vel-max 2 --tau-max 15
    # terminal 3
    python3 demos/reproducir_comandos.py results/bc/robot/comandos/comandos_experto_s5000.npz --hasta 20

En el robot real: `--ip 192.168.0.143 --real` (pide Enter), primero con
`--lento 3` (tres veces más despacio) y el agente con `--vel-max` bajo.
"""
import argparse
import os
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from h1_2_algoritms.aprendizaje import comandos_robot as CR   # noqa: E402

P, J = CR.P, CR.J


class Estado:
    """Recibe los datagramas de estado del agente (UDP 47931)."""

    def __init__(self, puerto):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", puerto))
        self.sock.settimeout(0.2)
        self.ultimo = None
        self.log = []          # (t_local, q27, q_obj27, ang12, fuerza12, peso, banderas)
        self.grabar = False
        self.vivo = True
        threading.Thread(target=self._rx, daemon=True).start()

    def _rx(self):
        while self.vivo:
            try:
                d, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            e = P.desempaquetar_estado(d)
            if e is None:
                continue
            self.ultimo = e
            if self.grabar:
                self.log.append((time.monotonic(), e["q"], e["q_obj"], e["ang_mano"], e["fuerza_mano"],
                                 e["peso"], e["banderas"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--ip", default="127.0.0.1", help="agente: 127.0.0.1 (sim) o la IP del robot")
    ap.add_argument("--puerto", type=int, default=P.PUERTO_ORDEN)
    ap.add_argument("--puerto-estado", type=int, default=P.PUERTO_ESTADO)
    ap.add_argument("--desde", type=float, default=0.0, help="s del episodio desde donde reproducir")
    ap.add_argument("--hasta", type=float, default=np.inf)
    ap.add_argument("--lento", type=float, default=1.0, help="factor de cámara lenta (2 = mitad de velocidad)")
    ap.add_argument("--sin-manos", action="store_true")
    ap.add_argument("--sin-tau", action="store_true", help="tau_ff = 0 (sin la gravedad del PC)")
    ap.add_argument("--sin-dq", action="store_true")
    ap.add_argument("--real", action="store_true", help="robot real: pide confirmación")
    ap.add_argument("--soltar", action="store_true", help="al terminar, orden SOLTAR (el agente devuelve el control)")
    ap.add_argument("--salida", default=None, help=".npz con lo medido (por omisión, junto al de entrada)")
    a = ap.parse_args()

    z = np.load(a.npz)
    t = z["pc_t"]
    sel = (t >= a.desde) & (t <= a.hasta)
    if not sel.any():
        sys.exit("ventana vacía")
    t, q, dq, tau, mano = (z[f"pc_{k}"][sel] for k in ("t", "q", "dq", "tau", "mano"))
    idx = [int(i) for i in z["idx_dds_orden"]]
    mascara = sum(1 << i for i in idx)
    mascara_mano = 0 if a.sin_manos else (1 << 12) - 1
    vmax = float(np.max(np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None])) / a.lento if len(t) > 1 else 0.0
    print(f"{len(t)} órdenes, {t[0]:.2f}-{t[-1]:.2f} s del episodio ({(t[-1] - t[0]) * a.lento:.1f} s reales), "
          f"vel. articular pedida máx {vmax:.2f} rad/s, |tau_ff| máx {np.abs(tau).max():.1f} N·m")
    if "meta_vel_max" in z.files:
        print(f"grabado con el agente emulado a --vel-max {float(z['meta_vel_max'])} --tau-max "
              f"{float(z['meta_tau_max'])} (usa al menos eso en el agente)")

    est = Estado(a.puerto_estado)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = [0]

    def mandar(k, orden=P.NORMAL, m=None):
        q27, dq27, tau27 = [0.0] * J.N, [0.0] * J.N, [0.0] * J.N
        for c, i in enumerate(idx):
            q27[i] = float(q[k, c])
            dq27[i] = 0.0 if a.sin_dq else float(dq[k, c]) / a.lento
            tau27[i] = 0.0 if a.sin_tau else float(tau[k, c])
        ang = [int(v) for v in mano[k]]
        tx.sendto(P.empaquetar_orden(seq[0], orden, mascara if m is None else m, mascara_mano if m is None else 0,
                                     q27, dq27, tau27, ang), (a.ip, a.puerto))
        seq[0] += 1

    # 1) presentarse sin mandar juntas (máscara 0): el agente aprende a quién mandar el estado
    t0 = time.monotonic()
    while est.ultimo is None:
        mandar(0, m=0)
        time.sleep(0.05)
        if time.monotonic() - t0 > 3.0:
            sys.exit(f"sin estado del agente en {a.ip}:{a.puerto} (¿agente arrancado? ¿puente corriendo?)")
    e = est.ultimo
    if not e["banderas"] & P.B_ACTIVO:
        sys.exit("el agente no está activo (¿rampa de arm_sdk sin terminar?)")
    d0 = np.abs(np.array([e["q"][i] for i in idx]) - q[0])
    print(f"agente {P.NOMBRE_MODO.get(e['modo'], e['modo'])}, peso {e['peso']:.2f}; "
          f"distancia a la primera pose: máx {d0.max():.2f} rad ({J.NOMBRES[idx[int(np.argmax(d0))]]})")
    if a.real:
        input("ROBOT REAL: L2+B a mano. Enter para llevar los brazos a la primera pose... ")

    # 2) a la primera pose (el agente rampa a su --vel-max)
    t0 = time.monotonic()
    while True:
        mandar(0)
        time.sleep(0.02)
        qm = np.array([est.ultimo["q"][i] for i in idx])
        if np.max(np.abs(qm - q[0])) < 0.05 or time.monotonic() - t0 > 15.0:
            break
    print(f"en la primera pose (error máx {np.max(np.abs(qm - q[0])):.3f} rad, {time.monotonic() - t0:.1f} s)")
    if a.real:
        input("Enter para reproducir el episodio... ")

    # 3) reproducir con los tiempos grabados
    est.grabar = True
    t_ini = time.monotonic()
    for k in range(len(t)):
        objetivo = t_ini + (t[k] - t[0]) * a.lento
        while time.monotonic() < objetivo:
            time.sleep(min(0.002, max(0.0, objetivo - time.monotonic())))
        mandar(k)
    t_fin = time.monotonic()
    while time.monotonic() - t_fin < 1.0:          # sostener 1 s la última pose
        mandar(len(t) - 1)
        time.sleep(0.02)
    est.grabar = False
    if a.soltar:
        for _ in range(5):
            mandar(len(t) - 1, orden=P.SOLTAR)
            time.sleep(0.02)
    est.vivo = False

    # 4) seguimiento: q medida contra la orden (interpolada al instante del estado)
    L = est.log
    if not L:
        sys.exit("no llegó estado durante la reproducción")
    ts = np.array([x[0] for x in L])
    ok = ts <= t_fin
    tep = t[0] + (ts - t_ini) / a.lento        # instante del episodio de cada estado
    qm = np.array([[x[1][i] for i in idx] for x in L])
    qc = np.stack([np.interp(tep, t, q[:, c]) for c in range(len(idx))], axis=1)
    err = np.abs(qm - qc)[ok]
    print(f"\n{len(L)} estados ({len(L) / max(ts[-1] - ts[0], 1e-3):.0f} Hz). Error |q medida - q pedida| [rad]:")
    print(f"{'junta':28s} {'mediana':>8s} {'p99':>8s} {'máx':>8s}")
    for c, i in enumerate(idx):
        print(f"{J.NOMBRES[i]:28s} {np.median(err[:, c]):8.3f} {np.percentile(err[:, c], 99):8.3f} "
              f"{err[:, c].max():8.3f}")
    print(f"{'TODAS':28s} {np.median(err):8.3f} {np.percentile(err, 99):8.3f} {err.max():8.3f}")
    out = a.salida or a.npz[:-4] + "_medido.npz"
    np.savez_compressed(out, t_local=ts - t_ini, t_episodio=tep, q_medida=qm, q_pedida=qc,
                        q_obj_agente=np.array([[x[2][i] for i in idx] for x in L]),
                        mano_angulo=np.array([x[3] for x in L]), mano_fuerza=np.array([x[4] for x in L]),
                        peso=np.array([x[5] for x in L]), lento=a.lento)
    print("medido ->", out)


if __name__ == "__main__":
    main()
