#!/usr/bin/env python3
"""Seguidor de linea del H1-2 (reto R40-RT-H1_2-0001). Une los cinco bloques:
percepcion -> estimacion -> control -> supervisor (unico que llama a Move/StopMove)
-> registro.

EL MISMO PROGRAMA en el robot y en el simulador; solo cambian la fuente de imagen
y el dominio DDS:
    # simulador (h1_2_sim2real con una pista: scripts/sim_pista.sh 1)
    python3 seguidor.py --sim
    # robot real, EN EL PC2, con sudo (la camara) y el python de teleop_venv
    sudo -E PYTHONNOUSERSITE=1 ~/teleop_venv/bin/python seguidor.py --iface eth0
    # hito 3: simulacro (camara en vivo, ordenes calculadas y registradas, NO enviadas)
    python3 seguidor.py --sim --simulacro

Seguridad (sec. 10): palabra de confirmacion antes de andar, topes de velocidad
del config (con --escala 0.5 en las primeras tiradas), vigilante (camara, IMU,
inclinacion > 20 grados, motores, excepcion, Ctrl+C) y StopMove al salir.
L2 + B en el mando sigue siendo la parada de emergencia del robot real.
"""
import argparse
import math
import os
import signal
import sys
import time

# BLAS en un hilo por lo mismo que cv2.setNumThreads(1) (percepcion.py): los ajustes son
# de decenas de puntos y con los nucleos ocupados los hilos de BLAS solo estorban.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.stdout.reconfigure(line_buffering=True)      # que se vea en vivo aunque se redirija
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

from seguidor import config                                   # noqa: E402
from seguidor.control import Control                          # noqa: E402
from seguidor.estimacion import Estimador                     # noqa: E402
from seguidor.percepcion import Percepcion                    # noqa: E402
from seguidor.registro import Registro                        # noqa: E402
from seguidor.robot_io import FuenteImu, Marcha, iniciar_dds  # noqa: E402
from seguidor.supervisor import FIN, PARADA, Supervisor       # noqa: E402
from seguidor.tipos import Orden                              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="simulador: DDS dominio 1 en lo, camara por ROS 2")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--simulacro", action="store_true", help="calcula y registra, pero no manda Move")
    ap.add_argument("--config", default=None)
    ap.add_argument("--config-extra", action="append", default=[],
                    help="yaml superpuesto (p.ej. config/planta_cinematica.yaml); se puede repetir")
    ap.add_argument("--escala", type=float, default=None, help="multiplica los topes de velocidad (0.5 al empezar)")
    ap.add_argument("--nivel", default="?", help="para el registro")
    ap.add_argument("--emisor", choices=["on", "off"], default="on", help="emisor IR de la D435i (robot real)")
    ap.add_argument("--registro", default=os.path.join(AQUI, "datos", "tiradas"))
    ap.add_argument("--sin-confirmacion", action="store_true", help="SOLO simulador: no pide la palabra")
    ap.add_argument("--t-max", type=float, default=120.0)
    ap.add_argument("--rviz", action="store_true", help="SOLO simulador: marcadores en /seguidor/marcadores")
    a = ap.parse_args()
    if a.sin_confirmacion and not a.sim:
        sys.exit("--sin-confirmacion solo vale en el simulador")
    cfg = config.cargar(a.config, a.config_extra)
    if a.escala is not None:
        cfg["control"]["escala"] = a.escala
    s = cfg["supervisor"]

    iniciar_dds(a.sim, a.iface)
    imu = FuenteImu()
    if not imu.esperar(5.0):
        sys.exit("No llega rt/lowstate (¿simulador / robot encendido? ¿interfaz?)")
    if a.sim:
        from seguidor.fuentes import FuenteROS
        fuente = FuenteROS()
    else:
        from seguidor.fuentes import FuenteRealSense
        fuente = FuenteRealSense(emisor=a.emisor == "on")
    def salir(msg):
        """Salida temprana: cerrar la fuente antes (con rclpy vivo, el proceso acaba en
        'terminate called without an active exception')."""
        if hasattr(fuente, "cerrar"):
            fuente.cerrar()
        sys.exit(msg)

    f = fuente.leer(5.0)
    if f is None:
        salir("La camara no da fotogramas")
    print(f"camara OK: {f.ir.shape[1]}x{f.ir.shape[0]}, K = {tuple(round(k, 1) for k in f.K)}")

    marcadores = None
    if a.rviz and a.sim:
        from seguidor.visual_rviz import MarcadoresRviz
        marcadores = MarcadoresRviz(fuente.nodo, os.path.join(AQUI, "sim", "pistas", f"pista_nivel{a.nivel}.json"))
    per, est, ctl = Percepcion(cfg), Estimador(cfg), Control(cfg)
    sup = Supervisor(cfg)
    # roll/pitch de referencia: el de la calibracion de la camara = robot quieto de pie, AHORA
    m = [imu.leer() for _ in range(10) if not time.sleep(0.02)]
    ref_r = sum(x.roll for x in m) / len(m)
    ref_p = sum(x.pitch for x in m) / len(m)
    est.reiniciar(imu.leer().yaw)

    def arrancar():
        est.reiniciar(imu.leer().yaw, conservar=est.ultima)   # rumbo de referencia en el ultimo momento
        ctl.reiniciar()

    def tras_giro():
        est.olvidar_linea()
        ctl.reiniciar()
    sup.arrancar, sup.on_giro = arrancar, tras_giro

    modo = "SIMULACRO (no se manda Move)" if a.simulacro else ("SIMULADOR" if a.sim else "ROBOT REAL")
    c = cfg["control"]
    print(f"modo: {modo} · topes vx {c['vx_max'] * c['escala']:.2f} m/s, vyaw {c['vyaw_max'] * c['escala']:.2f} rad/s")
    if not a.simulacro and not a.sin_confirmacion:
        print("=" * 66 + "\n  ESTO MUEVE EL ROBOT. FSM 201, nadie en la pista, L2 + B a mano.\n" + "=" * 66)
        try:
            palabra = input(f"Escribe {s['palabra']} para empezar: ").strip()
        except (EOFError, KeyboardInterrupt):
            palabra = ""
        if palabra != s["palabra"]:
            salir("No confirmado: no se mueve nada.")
    marcha = Marcha(simulacro=a.simulacro)
    reg = Registro(a.registro, f"nivel{a.nivel}", {"nivel": a.nivel, "modo": modo, "config": cfg,
                                                     "ref_roll": ref_r, "ref_pitch": ref_p})

    parar_ya = {"motivo": None}
    signal.signal(signal.SIGINT, lambda *_: parar_ya.__setitem__("motivo", "Ctrl+C"))
    signal.signal(signal.SIGTERM, lambda *_: parar_ya.__setitem__("motivo", "SIGTERM"))

    periodo = 1.0 / s["hz"]
    t0 = t_ant = time.monotonic()
    t_cam = f.t
    aplicada = Orden()
    medida = None
    excepcion = None
    try:
        while True:
            t = time.monotonic()
            dt = t - t_ant
            t_ant = t
            if parar_ya["motivo"] or t - t0 > a.t_max:
                excepcion = parar_ya["motivo"] or "tiempo maximo"
            nueva = fuente.leer(0.0)
            u = imu.leer()
            medida_buena = False
            med_ciclo = None
            if nueva is not None:
                t_cam = nueva.t
                medida = per.procesar(nueva, u.roll - ref_r, u.pitch - ref_p, mira=ctl.mirada())
                med_ciclo = medida
                medida_buena = medida.valida and medida.confianza >= cfg["estimacion"]["confianza_min"]
            elif medida is not None and t - medida.t < 0.15:
                medida_buena = medida.valida and medida.confianza >= cfg["estimacion"]["confianza_min"]
            e = est.actualizar(t, u.yaw, (aplicada.vx, aplicada.vy, aplicada.vyaw), med_ciclo, gyro_z=u.gyro[2])
            objetivo = est.punto_a_distancia(ctl.mirada())
            o_ctl = ctl.calcular(e, objetivo, dt)
            salud = {"t_camara": t_cam, "t_imu": u.t, "roll": u.roll, "pitch": u.pitch,
                     "motores_ok": u.motores_ok, "excepcion": excepcion}
            o = sup.paso(t, e, medida_buena, o_ctl, salud)
            marcha.aplicar(o)
            aplicada = Orden() if o.parar else o
            if o.parar:
                ctl.reiniciar()
            gt = fuente.pose_verdad() if hasattr(fuente, "pose_verdad") else None
            reg.fila(t=t - t0, estado=sup.estado, motivo=o.motivo,
                     med_valida=med_ciclo.valida if med_ciclo else None,
                     med_desp=med_ciclo.desplazamiento if med_ciclo else None,
                     med_ang=med_ciclo.angulo if med_ciclo else None,
                     med_curv=med_ciclo.curvatura if med_ciclo else None,
                     med_conf=med_ciclo.confianza if med_ciclo else None,
                     med_barra=med_ciclo.barra if med_ciclo else None,
                     med_esquina=med_ciclo.esquina if med_ciclo else None,
                     med_ms=med_ciclo.ms if med_ciclo else None,
                     est_valida=e.valida, est_desp=e.desplazamiento, est_ang=e.angulo, est_curv=e.curvatura,
                     est_conf=e.confianza, est_barra=e.barra, est_esquina=e.esquina,
                     sin_linea_m=e.sin_linea_m if math.isfinite(e.sin_linea_m) else None,
                     recorrido=e.recorrido, yaw=e.yaw, roll=u.roll, pitch=u.pitch,
                     obj_x=objetivo[0] if objetivo else None, obj_y=objetivo[1] if objetivo else None,
                     vx=o.vx, vy=o.vy, vyaw=o.vyaw, parar=o.parar, edad_camara=t - t_cam,
                     gt_x=gt[0] if gt else None, gt_y=gt[1] if gt else None, gt_yaw=gt[2] if gt else None)
            if marcadores is not None:
                marcadores.publicar(med_ciclo or medida, est.puntos_robot(), objetivo, sup.estado, o, e.confianza)
            if sup.estado in (FIN, PARADA):
                if marcadores is not None:
                    marcadores.publicar(medida, est.puntos_robot(), None, sup.estado, o, e.confianza)
                break
            time.sleep(max(0.0, periodo - (time.monotonic() - t)))
    except Exception as ex:                                   # vigilante: excepcion no controlada
        excepcion = repr(ex)
        print(f"EXCEPCION: {ex}")
        raise
    finally:
        marcha.parar()
        motivo = sup.motivo if sup.estado in (FIN, PARADA) else (excepcion or "salida")
        r = reg.cerrar(sup.eventos, f"{sup.estado}: {motivo}", cfg["estimacion"]["confianza_min"])
        print(f"\nFIN: {sup.estado} ({motivo}). Registro: {reg.base}.csv / .json")
        print(f"  recorrido estimado {r['recorrido_estimado_m']} m en {r['tiempo_recorrido_s']} s · error lateral "
              f"medio {r['error_lateral_medio_m']} m, max {r['error_lateral_max_m']} m · confianza sobre umbral "
              f"{r['pct_confianza_sobre_umbral']} % · percepcion {r['percepcion_ms_media']} ms")
        if hasattr(fuente, "cerrar"):
            fuente.cerrar()


if __name__ == "__main__":
    main()
