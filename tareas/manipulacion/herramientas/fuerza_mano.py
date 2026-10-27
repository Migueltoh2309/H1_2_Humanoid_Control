#!/usr/bin/env python3
"""Hito 2 — fuerza real de la mano frente a FORCE_SET (pregunta 6.4.2), con la báscula.

Robot: la mano con la palma hacia abajo sobre la báscula de cocina (el operador la sostiene o
el brazo está en una pose fija), un dedo (el índice por defecto) encima del plato. Para cada
FORCE_SET de la lista: se escribe FORCE_SET (y se relee), el dedo se cierra hasta 0 (se para
sobre el plato), se espera, se lee FORCE_ACT y el operador teclea lo que marca la báscula
(gramos = gf). Se abre y se repite. Sale la tabla FORCE_SET -> FORCE_ACT -> báscula y cuánto
se pasa el firmware.

    python3 herramientas/fuerza_mano.py --lado izq --dedo indice --lista 200 300 400 500 600 800

Simulador (--sim): sin ROS ni robot. MuJoCo directo con la escena de la estación: un objeto
fijo delante de la palma hace de báscula y su lectura es la fuerza de contacto que calcula
MuJoCo sobre ese dedo. En el simulador el par del dedo es exactamente FORCE_SET por
construcción; lo que mide la prueba es cuánto de esa fuerza llega al objeto por la geometría
del contacto.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sim"))
DEDOS = ["menique", "anular", "medio", "indice", "pulgar"]
DATOS = os.path.join(RAIZ, "datos", "fuerza_mano")


def robot(a):
    from manip.mano import Mano
    k = DEDOS.index(a.dedo)
    mano = Mano(a.lado, sim=False)
    filas = []
    try:
        for fs in a.lista:
            mano.abrir()
            time.sleep(1.0)
            input(f"  Dedo {a.dedo} sobre el plato de la báscula, báscula a cero. "
                  f"Enter para cerrar con FORCE_SET {fs}: ")
            mano.poner_force_set([fs] * 6)
            cmd = [-1] * 6
            cmd[k] = 0
            mano.mover(cmd)
            time.sleep(2.0)
            f_act = float(np.median([mano.fuerzas()[k] for _ in range(5)]))
            a_act = float(mano.angulos()[k])
            try:
                bascula = float(input(f"    FORCE_ACT {f_act:.0f} gf, ángulo {a_act:.0f}. ¿Qué marca la báscula (g)? "))
            except ValueError:
                bascula = float("nan")
            filas.append({"force_set": fs, "force_act": f_act, "bascula_g": bascula, "angulo": a_act})
    finally:
        mano.abrir()
        mano.cerrar_conexion()
    return filas


def simulado(a):
    """Objeto fijo delante de la palma; fuerza de contacto de MuJoCo sobre la falange del dedo."""
    import mujoco
    from construir_estacion import S2R, construir
    sys.path.insert(0, S2R)
    import tempfile

    from h1_2_sim2real.comun import J
    from h1_2_sim2real.robot_simulado import RobotSimulado
    from manip.config import cargar, cargar_objetos
    xml, _ = construir(2, 0, cargar(), cargar_objetos(), salida=tempfile.mkdtemp())
    k = DEDOS.index(a.dedo)
    pre = "L_" if a.lado == "izq" else "R_"
    eslabones = {"menique": "pinky", "anular": "ring", "medio": "middle", "indice": "index", "pulgar": "thumb"}
    filas = []
    for fs in a.lista:
        sim = RobotSimulado(xml, log=lambda s: None)
        m, d = sim.model, sim.data
        mano = sim.manos[a.lado]
        w = m.body(("left" if a.lado == "izq" else "right") + "_wrist_yaw_link").id
        off = np.array([0.17, -0.058 if a.lado == "izq" else 0.058, 0.0])
        cuerpos = {b for b in range(m.nbody) if (m.body(b).name or "").startswith(pre + eslabones[a.dedo])}
        obj = m.body("pelota").id
        mano.escribir(J.FORCE_SET, [fs] * 6)
        mano.escribir(J.SPEED_SET, [300] * 6)
        cmd = [1000] * 6
        cmd[k] = 0
        mano.escribir(J.ANGLE_SET, cmd)
        z = np.zeros(J.N)
        c = {"lc_t": np.array([-1e9]), "lc_q": z, "lc_dq": z, "lc_kp": z, "lc_kd": z, "lc_tau": z, "lc_mode": z,
             "arm_t": np.array([-1e9]), "arm_q": z, "arm_dq": z, "arm_kp": z, "arm_kd": z, "arm_tau": z,
             "arm_w": np.zeros(1), "loco_v": np.zeros(3), "loco_t_fin": np.zeros(1), "fsm": np.array([201.0])}
        t, f6, fuerzas = 0.0, np.zeros(6), []
        for n in range(int(2.0 / sim.dt)):
            sim.poner_cuerpo("pelota", d.xpos[w] + d.xmat[w].reshape(3, 3) @ off, [1, 0, 0, 0])
            sim.paso(t, c)
            t += sim.dt
            if n * sim.dt > 1.5:
                tot = 0.0
                for i in range(d.ncon):
                    cc = d.contact[i]
                    b1, b2 = m.geom_bodyid[cc.geom1], m.geom_bodyid[cc.geom2]
                    if (b1 == obj and b2 in cuerpos) or (b2 == obj and b1 in cuerpos):
                        mujoco.mj_contactForce(m, d, i, f6)
                        tot += abs(f6[0])
                fuerzas.append(tot)
        f_act = float(mano.leer(J.FORCE_ACT, 6)[k])
        filas.append({"force_set": fs, "force_act": f_act, "bascula_g": float(np.mean(fuerzas) * 101.97),
                      "angulo": float(mano.leer(J.ANGLE_ACT, 6)[k])})
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--lado", default="izq", choices=["izq", "der"])
    ap.add_argument("--dedo", default="indice", choices=DEDOS)
    ap.add_argument("--lista", type=int, nargs="+", default=[200, 300, 400, 500, 600, 800])
    a = ap.parse_args()
    if max(a.lista) > 800:
        sys.exit("FORCE_SET <= 800 gf (sec. 10)")
    filas = simulado(a) if a.sim else robot(a)
    print(f"\n{'FORCE_SET':>9} {'FORCE_ACT':>9} {'báscula':>8} {'exceso':>7}")
    for f in filas:
        print(f"{f['force_set']:9d} {f['force_act']:9.0f} {f['bascula_g']:8.0f} "
              f"{100 * (f['bascula_g'] / f['force_set'] - 1):6.0f}%")
    os.makedirs(DATOS, exist_ok=True)
    nombre = f"fuerza_{'sim' if a.sim else 'robot'}_{a.lado}_{a.dedo}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    ruta = os.path.join(DATOS, nombre)
    json.dump({"lado": a.lado, "dedo": a.dedo, "sim": a.sim, "filas": filas}, open(ruta, "w"), indent=1)
    print("guardado", ruta)


if __name__ == "__main__":
    main()
