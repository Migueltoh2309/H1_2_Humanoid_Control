#!/usr/bin/env python3
"""Hito 2 — calibración cámara-torso con el ArUco del dorso (6.1.3, 6.1.4) y balanceo de la
cámara con el robot quieto (6.1.5). Mismo programa en el simulador y en el robot.

Secuencia (robot de pie en FSM 201, nadie al alcance de los brazos):
  1. Comprueba FSM 201 (GetFsmId), rt/lowstate, motores y la cámara. Pide la palabra CALIBRAR.
  2. Rampa de entrada de 12 s; los dos brazos a la pose de OBSERVACIÓN (config/poses.yaml).
  3. --balanceo S: S segundos mirando los ArUco del tablero con el robot quieto (6.1.5).
  4. Por cada mano, las poses de calibración (config/poses.yaml): mínima sacudida (>= 8 s,
     <= 0.5 rad/s), 1.5 s de reposo, varios fotogramas de IR1 promediando las juntas MEDIDAS
     (y guardando las consignas: 6.1.4). Vuelta a observación.
  5. Vuelta a la postura del principio y rampa de salida (>= 5 s hasta 0.00).
  6. Resuelve AX = YB (manip/calibracion.py), valida con los ArUco del tablero y guarda
     datos/calibracion/calib_<fecha>.json (+ ultima.json) y las muestras.
Vigila todo el rato: FSM, roll/pitch > 10 grados, giro > 5 grados, rt/lowstate parado, error de
seguimiento (choque) y cámara sin imágenes -> se congela el brazo y rampa de salida (las poses
de calibración están a >= 10 cm sobre la mesa: ya es la altura de seguridad).

    python3 herramientas/calibrar.py --sim                   # simulador (dominio 1, lo)
    python3 herramientas/calibrar.py --iface eth0 --realsense  # robot (PC2)
    python3 herramientas/calibrar.py --resolver datos/calibracion/muestras_<fecha>   # rehacer el ajuste
En el simulador conviene el emisor IR apagado (sim_estacion.sh 0 emisor_ir:=false), como se
apaga el proyector de la D435 real para calibrar (--realsense apaga el emisor).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
from manip import calibracion as CAL  # noqa: E402
from manip import cinematica as C  # noqa: E402
from manip.config import _leer, cargar  # noqa: E402
from manip.estacion import Estacion  # noqa: E402

DATOS = os.path.join(RAIZ, "datos", "calibracion")


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def capturar(fuente, estado, brazo, lado, cfg_cal, det, ids, n=6):
    """n fotogramas de IR1 con el brazo quieto: esquinas promediadas y juntas medidas medias."""
    qs, qc, esq, prof = [], [], {}, []
    t = time.monotonic()
    for _ in range(n):
        f = fuente.fotograma(timeout=2.0, despues_de=t)
        if f is None:
            continue
        t = f.t
        qs.append(estado.brazo(lado))
        qc.append(brazo.consigna(lado))
        for i, dd in CAL.detectar(f.ir, f.K_prof, cfg_cal["lado"], det, ids).items():
            esq.setdefault(i, []).append(dd["esquinas"])
        prof.append(f)
    return qs, qc, esq, prof


def principal(a):
    from manip import robot_io as IO
    from manip.brazo import Brazo, Parada
    from manip.seguridad import Vigilante
    cfg = cargar()
    cfg_brazo = _leer(os.path.join(RAIZ, "config", "brazo.yaml"))
    poses = _leer(os.path.join(RAIZ, "config", "poses.yaml"))
    ac = cfg["aruco"]
    IO.iniciar_dds(a.sim, a.iface)
    estado = IO.Estado()
    if not estado.esperar():
        sys.exit("no llega rt/lowstate")
    fsm = IO.LectorFsm()
    f = fsm.leer()
    log(f"FSM = {f}")
    if f != 201:
        sys.exit("el robot tiene que estar en FSM 201 (rt/arm_sdk se ignora en silencio en otra FSM)")
    if not estado.motores_ok():
        sys.exit("hay motores con fallo")
    if a.realsense:
        from manip.fuentes import FuenteRealsense
        fuente = FuenteRealsense(emisor=False)
    else:
        from manip.fuentes import FuenteRos
        fuente = FuenteRos(con_color=False)
    fr = fuente.fotograma(timeout=10.0)
    if fr is None:
        sys.exit("no llegan imágenes de la cámara (IR1 + profundidad)")
    K = fr.K_prof
    det = CAL.detector(ac["diccionario"])
    ids_mano = {lado: ac["mano"][lado] for lado in C.LADOS}
    ids_tablero = set(ac["esquinas"])
    if not IO.confirmar("CALIBRAR", a.sim, log):
        sys.exit("cancelado")

    fecha = time.strftime("%Y%m%d_%H%M%S")
    carpeta = os.path.join(DATOS, f"muestras_{fecha}")
    os.makedirs(carpeta, exist_ok=True)
    q_ini = {lado: estado.brazo(lado) for lado in C.LADOS}
    yaw0 = estado.rpy()[2]
    brazo = Brazo(estado, cfg_brazo)
    vig = Vigilante(estado, fsm, brazo, fuente, yaw0)
    vig.mirar_camara = True
    obs = {lado: np.array(poses["observacion"][lado]) for lado in C.LADOS}
    muestras, balanceo = [], []
    try:
        log("rampa de entrada (12 s)")
        brazo.rampa_entrada(comprobar=vig)
        vig.mirar_seguimiento = True
        log(f"a la pose de observación ({brazo.duracion(obs):.1f} s)")
        brazo.mover(obs, comprobar=vig)
        time.sleep(1.0)
        if a.balanceo > 0:
            log(f"balanceo de la cámara: {a.balanceo:.0f} s mirando los ArUco del tablero (6.1.5)")
            t0 = time.monotonic()
            t = t0
            while time.monotonic() - t0 < a.balanceo:
                vig()
                fb = fuente.fotograma(timeout=2.0, despues_de=t)
                if fb is None:
                    continue
                t = fb.t
                d = CAL.detectar(fb.ir, K, ac["lado"], det, ids_tablero)
                if d:
                    c3 = {str(i): CAL.centro_3d(v["esquinas"], fb.prof, K) for i, v in d.items()}
                    balanceo.append({"t": fb.t - t0, "rpy": estado.rpy().tolist(),
                                     **{k: v.tolist() for k, v in c3.items() if v is not None}})
        for lado in (a.lados or C.LADOS):
            lista = poses["calibracion"][lado]
            for k, q in enumerate(lista):
                q = np.array(q)
                T = brazo.mover({lado: q}, comprobar=vig)
                fin = time.monotonic() + 1.5
                while time.monotonic() < fin:
                    vig()
                    time.sleep(0.05)
                ids = set(ids_mano.values()) | ids_tablero
                qs, qc, esq, frames = capturar(fuente, estado, brazo, lado, ac, det, ids)
                i = ids_mano[lado]
                if i not in esq or len(esq[i]) < 2:
                    log(f"  {lado} pose {k + 1}/{len(lista)} ({T:.1f} s): NO se ve el marcador {i}")
                    continue
                m = {"lado": lado, "pose": k, "q_medida": np.mean(qs, axis=0), "q_consigna": np.mean(qc, axis=0),
                     "esquinas": np.mean(esq[i], axis=0), "n_fotogramas": len(esq[i]),
                     "tablero": {int(j): np.mean(v, axis=0).tolist() for j, v in esq.items() if j in ids_tablero}}
                c3 = CAL.centro_3d(m["esquinas"], frames[-1].prof, K)
                m["centro_3d"] = None if c3 is None else c3
                m["tablero_3d"] = {}
                for j in m["tablero"]:
                    c3 = CAL.centro_3d(np.array(m["tablero"][j]), frames[-1].prof, K)
                    if c3 is not None:
                        m["tablero_3d"][j] = c3.tolist()
                d1 = CAL.detectar(frames[-1].ir, K, ac["lado"], det, {i})
                m["T_cam_marcador"] = d1[i]["T_cam_marcador"] if i in d1 else None
                err = np.degrees(np.max(np.abs(m["q_consigna"] - m["q_medida"])))
                log(f"  {lado} pose {k + 1}/{len(lista)} ({T:.1f} s): marcador visto {len(esq[i])} veces; "
                    f"consigna - medida máx {err:.1f} grados")
                np.savez_compressed(os.path.join(carpeta, f"{lado}_{k:02d}.npz"), ir=frames[-1].ir, K=K,
                                    q_medida=m["q_medida"], q_consigna=m["q_consigna"], esquinas=m["esquinas"])
                muestras.append(m)
            brazo.mover(obs, comprobar=vig)
        log("vuelta a la postura del principio")
        brazo.mover(q_ini, comprobar=vig)
    except (Parada, KeyboardInterrupt) as e:
        log(f"PARADA: {e} -> brazo congelado, rampa de salida")
    finally:
        vig.mirar_seguimiento = False
        brazo.rampa_salida()
        log(f"peso de arm_sdk = {brazo.peso:.2f}")
        brazo.parar()
    with open(os.path.join(carpeta, "muestras.json"), "w") as fh:
        json.dump({"fecha": fecha, "K": K.tolist(), "lado_marcador": ac["lado"], "balanceo": balanceo,
                   "error_seguimiento_max_deg": math.degrees(vig.err_max),
                   "muestras": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
                                for m in muestras]}, fh, indent=1)
    fuente.cerrar()           # sin esto, el hilo de rclpy aborta al salir
    log(f"{len(muestras)} muestras en {carpeta}")
    return carpeta


def resolver_carpeta(carpeta, cfg, verdad=None):
    d = json.load(open(os.path.join(carpeta, "muestras.json")))
    K = np.array(d["K"])
    lado_m = d["lado_marcador"]
    mu = []
    for m in d["muestras"]:
        m = dict(m)
        for k in ("q_medida", "q_consigna", "esquinas", "T_cam_marcador", "centro_3d"):
            if m.get(k) is not None:
                m[k] = np.array(m[k])
        mu.append(m)
    mu = [m for m in mu if m.get("T_cam_marcador") is not None]
    if len(mu) < 6:
        sys.exit(f"solo {len(mu)} muestras válidas: hacen falta al menos 6")
    X0 = CAL.camara_nominal(cfg)
    Y0 = {lado: CAL.marcador_nominal(lado) for lado in C.LADOS}
    res = {}
    for usar in ("medida", "consigna"):
        X, Ys, inf = CAL.resolver(mu, K, lado_m, X0, Y0, usar=usar)
        res[usar] = (X, Ys, inf)
        log(f"ajuste con las juntas {usar.upper()}S: reproyección {inf['reproy_px_rms']:.2f} px rms "
            f"(máx {inf['reproy_px_max']:.2f}); centro del marcador {inf['mm_medio']:.1f} mm medio, "
            f"{inf['mm_max']:.1f} máx ({inf['mm_lateral_medio']:.1f} / {inf['mm_lateral_max']:.1f} mm "
            f"en el plano de la imagen)")
    X, Ys, inf = res["medida"]
    Xc = res["consigna"][0]
    dif = {"mm": float(1000 * np.linalg.norm(X[:3, 3] - Xc[:3, 3])), "grados": CAL.ang_deg(X[:3, :3], Xc[:3, :3])}
    log(f"6.1.4: la cámara calibrada con consignas difiere {dif['mm']:.1f} mm y {dif['grados']:.2f} grados")
    X0d = {"mm": float(1000 * np.linalg.norm(X[:3, 3] - X0[:3, 3])), "grados": CAL.ang_deg(X[:3, :3], X0[:3, :3])}
    log(f"respecto de la pose nominal del URDF: {X0d['mm']:.1f} mm y {X0d['grados']:.2f} grados")
    # validación con los ArUco del tablero (posición conocida: geometría de la estación)
    est = Estacion(cfg)
    pos_t = {int(i): est.a_torso([x, y, est.z_mesa]) for i, (x, y) in est.aruco_esquinas.items()}
    errs = {}
    # centro del marcador con la profundidad del píxel (no la de solvePnP); la placa del
    # marcador está 1.2 mm por encima del tablero
    for m in mu:
        for j, c in (m.get("tablero_3d") or {}).items():
            p = (X @ np.append(np.array(c), 1.0))[:3]
            esperado = pos_t[int(j)] + np.array([0.0, 0.0, 0.0012])
            errs.setdefault(int(j), []).append(1000 * np.linalg.norm(p - esperado))
    val = {j: float(np.median(v)) for j, v in errs.items()}
    if val:
        log("validación con los ArUco del tablero (mm, mediana): " +
            ", ".join(f"{j}: {v:.1f}" for j, v in sorted(val.items())))
    extra = {"muestras": carpeta, "consignas_frente_a_medidas": dif, "desde_nominal": X0d,
             "validacion_tablero_mm": val, "informe_consignas": res["consigna"][2]}
    if verdad is not None:
        Xv = verdad
        ver = {"mm": float(1000 * np.linalg.norm(X[:3, 3] - Xv[:3, 3])), "grados": CAL.ang_deg(X[:3, :3], Xv[:3, :3])}
        log(f"SOLO SIMULADOR, frente a la pose verdadera de la cámara: {ver['mm']:.1f} mm y {ver['grados']:.2f} grados")
        extra["verdad_simulador"] = ver
    d_bal = d.get("balanceo") or []
    if d_bal:
        P = {}
        for b in d_bal:
            for k, c in b.items():
                if k.isdigit():
                    P.setdefault(k, []).append(np.array(c))
        sd = {k: float(1000 * np.std(np.array(v), axis=0).max()) for k, v in P.items() if len(v) > 3}
        log(f"6.1.5 balanceo con el robot quieto: desviación típica máx de la posición de cada ArUco (mm): {sd}")
        extra["balanceo_mm"] = sd
        # la validación de arriba usa las poses de calibración, con el brazo extendido y el
        # torso más inclinado; esta es la de la pose de observación, la que usa la percepción
        vo = {int(k): float(1000 * np.linalg.norm((X @ np.append(np.mean(v, axis=0), 1.0))[:3]
                                                  - (pos_t[int(k)] + np.array([0.0, 0.0, 0.0012]))))
              for k, v in P.items() if int(k) in pos_t}
        log("validación con los ArUco del tablero en la pose de observación (mm): " +
            ", ".join(f"{j}: {v:.1f}" for j, v in sorted(vo.items())))
        extra["validacion_observacion_mm"] = vo
    ruta = CAL.guardar(DATOS, X, Ys, inf, K, extra)
    log(f"calibración guardada: {ruta}")
    return ruta


def verdad_ir_sim(xml):
    """Pose VERDADERA del marco óptico de IR1 en torso_link, del MJCF de la escena. Solo para
    evaluar en el simulador: el sistema nunca la lee."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    c = m.camera("robot_ir_left")
    q = c.quat
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    R = R.reshape(3, 3)
    return CAL.T_de(np.column_stack([R[:, 0], -R[:, 1], -R[:, 2]]), c.pos.copy())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--realsense", action="store_true", help="cámara por pyrealsense2 (PC2) en vez de ROS")
    ap.add_argument("--balanceo", type=float, default=15.0, help="s mirando el tablero quieto (6.1.5); 0 = no")
    ap.add_argument("--lados", nargs="+", choices=C.LADOS)
    ap.add_argument("--resolver", help="carpeta de muestras: solo rehacer el ajuste")
    ap.add_argument("--escena", help="(simulador) MJCF de la escena, para comparar con la cámara verdadera")
    a = ap.parse_args()
    cfg = cargar()
    carpeta = a.resolver or principal(a)
    verdad = verdad_ir_sim(a.escena) if a.escena else None
    resolver_carpeta(carpeta, cfg, verdad)


if __name__ == "__main__":
    main()
