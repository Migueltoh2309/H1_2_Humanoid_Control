#!/usr/bin/env python3
"""Manipulación del H1-2 (reto R40-RT-H1_2-0002). Une los bloques de la sección 5:
percepción -> planificación -> supervisor (único que cambia el peso de arm_sdk) -> brazo
(rt/arm_sdk, con la compensación de 6.3.4) y mano (Modbus) -> registro.

EL MISMO PROGRAMA en el robot y en el simulador; solo cambian la fuente de imagen y el
dominio DDS:
    # simulador (h1_2_sim2real con la estación: scripts/sim_estacion.sh 2)
    python3 manipulacion.py --sim --nivel 2
    # robot real, en el PC2 (cámara por pyrealsense2)
    python3 manipulacion.py --iface eth0 --nivel 2
    # hito 3: simulacro (cámara y planes en vivo; el brazo NO se publica; la mano abre y cierra en el aire)
    python3 manipulacion.py --sim --nivel 2 --simulacro

Seguridad (sec. 10): FSM 201 comprobada, palabra de confirmación, rampas de peso, duración y
velocidad de los movimientos del config, vigilante (paso/giro/inclinación, seguimiento, FSM,
cámara, manos, fuerza) y salida segura. L2 + B en el mando sigue siendo la parada de emergencia.
"""
import argparse
import math
import os
import signal
import sys
import tempfile
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MUJOCO_GL", "egl")
sys.stdout.reconfigure(line_buffering=True)
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "sim"))


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="simulador: DDS dominio 1 en lo, cámara por ROS 2")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--nivel", type=int, default=2, choices=[1, 2, 3, 4])
    ap.add_argument("--solo-observacion", action="store_true", help="hito 3: a la pose de observación y vuelta")
    ap.add_argument("--simular-giro", type=float, default=None, metavar="T",
                    help="hito 3: a los T s se suman 6 grados a la guiñada de la IMU (el supervisor debe abortar)")
    ap.add_argument("--tapar-camara", type=float, default=None, metavar="T",
                    help="hito 3, SOLO simulador: a los T s se tapa la cámara (/sim/tapar_camara)")
    ap.add_argument("--semilla", type=int, default=None, help="(simulador) semilla de la escena, para el registro")
    ap.add_argument("--objeto", default="prisma", help="nivel 4: el que elige el instructor (no redondo)")
    ap.add_argument("--instructor", action="store_true",
                    help="SOLO simulador (nivel 3): mueve otro objeto 5-10 cm durante el primer transporte")
    ap.add_argument("--simulacro", action="store_true",
                    help="hito 3: calcula y registra; no publica arm_sdk (la mano sí, en el aire)")
    ap.add_argument("--calib", default=os.path.join(AQUI, "datos", "calibracion"))
    ap.add_argument("--registro", default=os.path.join(AQUI, "datos", "tiradas"))
    ap.add_argument("--t-max", type=float, default=None)
    a = ap.parse_args()

    from construir_estacion import construir
    from manip import calibracion as CAL
    from manip import robot_io as IO
    from manip.brazo import Brazo, Parada
    from manip.compensacion import Compensador
    from manip.config import _leer, cargar, cargar_objetos
    from manip.estacion import Estacion
    from manip.mano import Mano
    from manip.modelo import Modelo
    from manip.percepcion import Percepcion
    from manip.planificacion import Planificador
    from manip.registro import Registro
    from manip.supervisor import Supervisor

    cfg, cat = cargar(), cargar_objetos()
    leer = lambda n: _leer(os.path.join(AQUI, "config", n))  # noqa: E731
    cfg_brazo, cfg_plan, cfg_sup = leer("brazo.yaml"), leer("planificacion.yaml"), leer("supervisor.yaml")
    agarres, poses = leer("agarres.yaml"), leer("poses.yaml")
    cfg_sup["pivote_cuerpo"] = cfg_brazo["pivote_cuerpo"]
    est = Estacion(cfg)
    calib = CAL.cargar(a.calib)
    log(f"calibración {calib.get('fecha')}: {a.calib}")
    # el modelo de la estación vacía (robot + mesa + bandejas de la config): comprobador de
    # choques del planificador y modelo de la gravedad de la compensación
    sin_desvio = {k: 0.0 for k in ("cabeceo", "guinada", "alabeo", "dx", "dy", "dz")}
    xml, _ = construir(0, 0, cfg, cat, salida=tempfile.mkdtemp(), desvio=sin_desvio)
    plan = Planificador(est, agarres, cat, cfg_plan, Modelo(xml), poses["observacion"])
    comp = Compensador(xml, cfg_brazo["kp"])
    percep = Percepcion(calib, cat, est)

    IO.iniciar_dds(a.sim, a.iface)
    estado = IO.Estado()
    if not estado.esperar(5.0):
        sys.exit("no llega rt/lowstate (¿simulador / robot encendido? ¿interfaz?)")
    fsm = IO.LectorFsm()
    f = fsm.leer()
    log(f"FSM = {f}")
    if f != 201:
        sys.exit("el robot tiene que estar en FSM 201 (rt/arm_sdk se ignora en silencio en otra FSM)")
    if not estado.motores_ok():
        sys.exit("hay motores con fallo")
    if a.sim:
        from manip.fuentes import FuenteRos
        fuente = FuenteRos()
    else:
        from manip.fuentes import FuenteRealsense
        fuente = FuenteRealsense(emisor=True)
    if fuente.fotograma(timeout=10.0) is None:
        sys.exit("no llegan imágenes de la cámara")
    manos = {lado: Mano(lado, a.sim) for lado in ("izq", "der")}
    for lado, m in manos.items():
        e = m.estado()
        if not e.responde:
            sys.exit(f"la mano {lado} no responde")
        log(f"mano {lado}: ángulos {e.angulos.astype(int).tolist()}")

    brazo = Brazo(estado, cfg_brazo, simulacro=a.simulacro, comp=comp)
    reg = Registro(a.registro, a.nivel, estado, brazo,
                   extra={"sim": a.sim, "simulacro": a.simulacro, "calibracion": calib.get("fecha"),
                          "d": est.d, "mesa": est.z_mesa})
    sup = Supervisor(estado, fsm, brazo, manos, fuente, percep, plan, comp, est, cat, cfg_sup, reg, log,
                     simulacro=a.simulacro)
    log(f"registro: {reg.dir}")
    verdad = None
    if a.sim:
        from verdad import Verdad                 # solo para evaluar (herramientas/metricas.py)
        verdad = Verdad(fuente.nodo, list(cat))
        time.sleep(0.5)
        reg.meta["verdad_inicio"] = verdad.foto()
        reg.meta["escena"] = {"nivel": a.nivel, "semilla": a.semilla}
    if a.instructor:
        if not a.sim:
            sys.exit("--instructor solo en el simulador")
        from instructor import Instructor
        destinos = {k: np.array([est.bandejas[o["bandeja"]].cx, est.bandejas[o["bandeja"]].cy]) for k, o in cat.items()}
        ins = Instructor(fuente.nodo, list(cat), est.zona, log, destinos=destinos)
        sup.al_estado = ins.al_estado
    if not IO.confirmar(cfg_sup["palabra"], a.sim, log):
        reg.cerrar({"cancelado": True})
        sys.exit("cancelado")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    t0 = time.monotonic()
    pendientes, motivo = None, ""
    if a.simular_giro is not None:
        sup.giro_simulado = (t0 + a.simular_giro, 6.0)
    tapar_pub = None
    if a.tapar_camara is not None:
        if not a.sim:
            sys.exit("--tapar-camara solo en el simulador (en el robot se tapa con la mano)")
        import threading

        from std_msgs.msg import Bool
        tapar_pub = fuente.nodo.create_publisher(Bool, "/sim/tapar_camara", 2)

        def tapar(v):
            m = Bool()
            m.data = v
            tapar_pub.publish(m)
        threading.Timer(a.tapar_camara, lambda: (log("  [prueba] cámara tapada"), tapar(True))).start()
    try:
        sup.entrada()
        if a.solo_observacion:
            sup.solo_observacion()
            pendientes = []
        elif a.nivel == 1:
            tocadas, n_d = sup.nivel1(a.t_max or cfg_sup["t_max"])
            pendientes = [] if (n_d == 4 and tocadas == 4) else [f"dianas {tocadas}/{n_d}"]
        elif a.nivel == 4:
            ok4 = sup.nivel4(a.objeto, a.t_max or cfg_sup["t_max"])
            pendientes = [] if ok4 else [a.objeto]
        else:
            pendientes = sup.clasificar(a.t_max or cfg_sup["t_max"], cambio_mano=a.nivel >= 3)
        sup.salida()
    except (Parada, KeyboardInterrupt, Exception) as e:    # noqa: B014 (cualquier cosa -> salida segura)
        motivo = f"{type(e).__name__}: {e}"
        sup.salida_segura(motivo)
    finally:
        log(f"peso de arm_sdk = {brazo.peso:.2f}")
        brazo.parar()
        colocados = [o["clase"] for o in reg.objetos if o.get("colocado")]
        metricas = {"s": time.monotonic() - t0, "colocados": colocados, "pendientes": pendientes,
                    "intentos": len(reg.objetos), "giro_max_deg": float(math.degrees(sup.vig.giro_obs)),
                    "incl_max_deg": float(math.degrees(sup.vig.incl_obs)),
                    "error_seguimiento_max_deg": float(math.degrees(sup.vig.err_max)),
                    "salida_segura": motivo}
        if tapar_pub is not None:
            tapar(False)
        if verdad is not None:
            reg.meta["verdad_final"] = verdad.foto()
        d = reg.cerrar(metricas)
        if a.nivel == 1:
            toq = [o for o in reg.objetos if o.get("tocada")]
            log(f"dianas tocadas {len(toq)}/{len(reg.objetos)}  ({metricas['s']:.0f} s)  -> {d}")
        else:
            log(f"colocados {len(colocados)}/{len(cat)}: {colocados}  ({metricas['s']:.0f} s)  -> {d}")
        for m in manos.values():
            m.cerrar_conexion()
        fuente.cerrar()


if __name__ == "__main__":
    main()
