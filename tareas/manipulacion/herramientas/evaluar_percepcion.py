#!/usr/bin/env python3
"""Percepción probada en el PC sobre datos grabados (entregable de F3, preguntas 6.2.x).

1. Graba (o reutiliza) un conjunto de datos: por cada escena de la estación (niveles 2-4,
   semillas 0-2), un fotograma de la D435 (profundidad nativa + IR1, color) con los brazos en
   la pose de observación, rendido con el MISMO MJCF que el simulador (EGL, GPU), opcionalmente
   con el ruido de la D435 (modelo de la tesis: fruit_localization.realsense_depth_noise).
   Se guarda como .npz con fuentes.grabar(): la percepción lo lee con FuenteCarpeta, sin saber
   que no es la cámara en vivo.
2. Pasa la percepción con la CALIBRACIÓN del hito 2 (datos/calibracion/ultima.json; nunca la
   pose verdadera de la cámara) y compara con la verdad de terreno de la escena llevada a
   torso_link: error de posición (mm), de guiñada (módulo la simetría del objeto) e identidad.
3. Imágenes con los objetos detectados y reconocidos (datos/percepcion/*.png) y tabla.

    python3 herramientas/evaluar_percepcion.py                # sin ruido y con ruido D435
    python3 herramientas/evaluar_percepcion.py --ruido 1 --tapar   # además, con una mano encima
"""
import argparse
import json
import math
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
import cv2  # noqa: E402
import mujoco  # noqa: E402

from manip import calibracion as CAL  # noqa: E402
from manip import cinematica as C  # noqa: E402
from manip.config import _leer, cargar, cargar_objetos  # noqa: E402
from manip.estacion import Estacion  # noqa: E402
from manip.fuentes import FuenteCarpeta, grabar  # noqa: E402
from manip.percepcion import Percepcion  # noqa: E402
from manip.tipos import Fotograma  # noqa: E402

DATOS = os.path.join(RAIZ, "datos")


def render(xml, q_obs, ruido, rng, tapar=None, n=1):
    """Fotogramas de la D435 de la escena (IR1 + profundidad nativa + color) y la verdad.
    Con ruido, `n` fotogramas con ruido independiente (el robot quieto en OBSERVAR)."""
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("de_pie").id)
    q = {lado: np.array(q_obs[lado], float) for lado in C.LADOS}
    if tapar is not None:
        q[tapar[0]] = tapar[1]
    for lado in C.LADOS:
        d.qpos[[m.jnt_qposadr[m.joint(n).id] for n in C.NOMBRES_JUNTAS[lado]]] = q[lado]
    mujoco.mj_forward(m, d)
    out = {}
    for nombre in ("robot_ir_left", "robot_rgbd_camera"):
        cid = m.camera(nombre).id
        w, h = (int(v) for v in m.cam_resolution[cid])
        r = mujoco.Renderer(m, height=h, width=w)
        r.update_scene(d, camera=cid)
        rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(d, camera=cid)
        z = r.render()
        r.close()
        f = h / 2 / math.tan(math.radians(m.cam_fovy[cid]) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        out[nombre] = (rgb, z, K)
    rgb_ir0, z0, K = out["robot_ir_left"]
    tor = m.body("torso_link").id
    Rt, pt = d.xmat[tor].reshape(3, 3), d.xpos[tor].copy()
    qb = np.concatenate([q["izq"], q["der"]])
    fs = []
    for k in range(n if ruido else 1):
        z, rgb_ir = z0, rgb_ir0
        if ruido:
            from h1_2_algoritms.vision.fruit_localization import image_noise, realsense_depth_noise
            z = realsense_depth_noise(z0, rng, K[0, 0])
            rgb_ir = image_noise(rgb_ir0, rng)
        ir = cv2.cvtColor(rgb_ir, cv2.COLOR_RGB2GRAY)
        fs.append(Fotograma(t=0.033 * k, color=out["robot_rgbd_camera"][0], prof=z.astype(np.float32),
                            K_color=out["robot_rgbd_camera"][2], K_prof=K, ir=ir, q_brazos=qb))
    return fs, (Rt, pt)


def verdad_torso(geo, Rt, pt):
    out = {}
    for k, o in geo["objetos"].items():
        p = Rt.T @ (np.array([o["x"], o["y"], o["z"]]) - pt)
        out[k] = (p, o["guinada"])
    return out


def dibujar(f, lista, T_torso_cam, ruta, verdad=None):
    img = cv2.cvtColor(f.ir, cv2.COLOR_GRAY2BGR)
    K = f.K_prof
    Ti = CAL.inv(T_torso_cam)

    def px(p):
        c = Ti[:3, :3] @ p + Ti[:3, 3]
        return int(K[0, 0] * c[0] / c[2] + K[0, 2]), int(K[1, 1] * c[1] / c[2] + K[1, 2])
    if verdad:
        for k, (p, _) in verdad.items():
            cv2.drawMarker(img, px(p), (0, 200, 0), cv2.MARKER_CROSS, 14, 2)
    for o in lista.objetos:
        u, v = px(o.centro)
        col = (0, 0, 255) if o.clase == "desconocido" else (255, 120, 0)
        cv2.circle(img, (u, v), 5, col, 2)
        if o.simetria > 0:
            dx = 0.04 * np.array([math.cos(o.guinada), math.sin(o.guinada), 0])
            cv2.line(img, px(o.centro - dx), px(o.centro + dx), col, 2)
        cv2.putText(img, f"{o.clase} {o.confianza:.2f}", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(img, f"{lista.ms:.0f} ms", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(ruta, img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ruido", type=int, nargs="+", default=[0, 1], help="0 sin ruido, 1 con ruido D435")
    ap.add_argument("--tapar", action="store_true", help="además: una mano bajando sobre el primer objeto")
    ap.add_argument("--fotogramas", type=int, default=8, help="fotogramas fusionados (con ruido)")
    ap.add_argument("--calib", default=os.path.join(DATOS, "calibracion"))
    ap.add_argument("--niveles", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--semillas", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    cfg, obj = cargar(), cargar_objetos()
    est = Estacion(cfg)
    poses = _leer(os.path.join(RAIZ, "config", "poses.yaml"))
    try:
        calib = CAL.cargar(a.calib)
    except OSError:
        sys.exit("no hay calibración: ./scripts/hito2_sim.sh calibrar")
    per = Percepcion(calib, obj, est)
    rng = np.random.default_rng(0)
    filas = []
    os.makedirs(os.path.join(DATOS, "percepcion"), exist_ok=True)
    casos = [("", None)]
    if a.tapar:
        casos.append(("_tapada", "tapar"))
    for ruido in a.ruido:
        for n in a.niveles:
            for s in a.semillas:
                base = os.path.join(RAIZ, "sim", "estaciones", f"estacion_n{n}_s{s}")
                geo = json.load(open(base + ".json"))
                for suf, modo in casos:
                    tapar = None
                    if modo == "tapar":
                        # mano de ese lado a 8 cm sobre el primer objeto alcanzable, palma abajo
                        for k, o in geo["objetos"].items():
                            lado = o["lado"]
                            p = est.a_torso([o["x"], o["y"], est.z_mesa + o["alto"] + 0.08])
                            R = C.rotacion_agarre(lado, "arriba", 0, -30 if lado == "izq" else 30)
                            q, *_, ok = C.ik(lado, p, R)
                            if ok:
                                tapar = (lado, q, k)
                                break
                    nombre = f"n{n}_s{s}_r{ruido}{suf}"
                    carpeta = os.path.join(DATOS, "dataset", nombre)
                    os.makedirs(carpeta, exist_ok=True)
                    fs, (Rt, pt) = render(base + ".xml", poses["observacion"], ruido, rng,
                                          None if tapar is None else tapar[:2], n=a.fotogramas)
                    for k_, f in enumerate(fs):
                        grabar(os.path.join(carpeta, f"f{k_:03d}.npz"), f)
                    fuente = FuenteCarpeta(carpeta)                 # la percepción lee lo grabado
                    fs = [fuente.fotograma() for _ in range(len(fuente))]
                    lista = per(fs)
                    f = per.fusionar(fs)
                    ver = verdad_torso(geo, Rt, pt)
                    dibujar(f, lista, calib["T_torso_cam"], os.path.join(DATOS, "percepcion", nombre + ".png"), ver)
                    vistos = {o.clase: o for o in lista.objetos}
                    for k, (p, g) in ver.items():
                        o = vistos.get(k)
                        fila = {"escena": nombre, "objeto": k, "visto": o is not None,
                                "tapado": tapar is not None and tapar[2] == k}
                        if o is not None:
                            fila["err_xy_mm"] = float(1000 * np.linalg.norm(o.centro[:2] - p[:2]))
                            fila["err_z_mm"] = float(1000 * (o.centro[2] - p[2]))
                            fila["err_gui_deg"] = float(math.degrees(abs(math.remainder(o.guinada - g, o.simetria)))) \
                                if o.simetria > 0 else None
                            fila["confianza"] = o.confianza
                            fila["visible"] = o.fraccion_visible
                        filas.append(fila)
                    fila_ms = lista.ms
                    extra = [o.clase for o in lista.objetos if o.clase not in ver]
                    print(f"{nombre:16s} {len(lista.objetos)} objetos, {fila_ms:5.1f} ms" +
                          (f"  sobran: {extra}" if extra else ""))
    json.dump(filas, open(os.path.join(DATOS, "percepcion", "resultados.json"), "w"), indent=1)
    print(f"\n{'ruido':>5} {'objeto':10s} {'vistos':>7} {'xy medio':>9} {'xy máx':>7} {'z medio':>8} "
          f"{'guiñada':>8} {'conf':>5}")
    for ruido in a.ruido:
        for k in obj:
            fs = [x for x in filas if x["objeto"] == k and f"_r{ruido}" in x["escena"] and not x["tapado"]]
            vis = [x for x in fs if x["visto"]]
            if not fs:
                continue
            xy = [x["err_xy_mm"] for x in vis]
            z = [abs(x["err_z_mm"]) for x in vis]
            g = [x["err_gui_deg"] for x in vis if x["err_gui_deg"] is not None]
            print(f"{ruido:5d} {k:10s} {len(vis):3d}/{len(fs):<3d} {np.mean(xy) if xy else float('nan'):8.1f} "
                  f"{np.max(xy) if xy else float('nan'):7.1f} {np.mean(z) if z else float('nan'):8.1f} "
                  f"{(np.mean(g) if g else float('nan')):7.1f}° "
                  f"{np.mean([x['confianza'] for x in vis]) if vis else 0:5.2f}")
    if a.tapar:
        tp = [x for x in filas if x["tapado"]]
        print(f"\ncon la mano a 8 cm encima: vistos {sum(x['visto'] for x in tp)}/{len(tp)}, "
              f"fracción visible media {np.mean([x['visible'] for x in tp if x['visto']] or [0]):.2f}, "
              f"error xy medio {np.mean([x['err_xy_mm'] for x in tp if x['visto']] or [float('nan')]):.1f} mm")


if __name__ == "__main__":
    main()
