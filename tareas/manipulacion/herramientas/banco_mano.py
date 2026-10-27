#!/usr/bin/env python3
"""Banco de la MANO suelta: la Inspire sola, llevada por un cuerpo mocap, sobre la mesa de la
estación y con un objeto delante (pregunta 6.4.1). Sirve para elegir la geometría de cada
agarre (orientación, punto de agarre, preforma, FORCE_SET) en segundos y medir cuántos
centímetros de error de posición tolera, antes de pasarla al brazo (planificador + IK).

La mano es la de la escena del robot (mallas, juntas, acoples y actuadores del MJCF) con la
misma ManoSim de h1_2_sim2real: FORCE_SET limita el par, STATUS, FORCE_ACT. La muñeca la
mueve un mocap (rígido: como un brazo perfecto), con la misma secuencia que la tirada:
preforma en el preagarre -> bajar en recta -> FORCE_SET y cerrar -> subir -> sostener.

La orientación y el punto de agarre se definen como en el planificador (marco de torso_link):
`rotacion_agarre(lado, modo, elev, az)` y `punto_agarre(lado, prof_palma, avance, lateral)`.

    python3 herramientas/banco_mano.py --objeto pelota                        # tabla de agarres.yaml
    python3 herramientas/banco_mano.py --objeto vaso --fijar tipo=lado avance=0.2 prof_palma=0.06
    python3 herramientas/banco_mano.py --objeto cubo --error 0.01 0.02        # tolerancia (6.4.1)
    python3 herramientas/banco_mano.py --objeto pelota --barrer prof_palma 0.05 0.06 0.07
    python3 herramientas/banco_mano.py --objeto cubo --foto                   # datos/banco_mano/*.png
"""
import argparse
import itertools
import math
import os
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sim"))
from construir_estacion import S2R, construir, cuerpo_objeto  # noqa: E402

sys.path.insert(0, S2R)
import mujoco  # noqa: E402
from h1_2_sim2real.comun import J  # noqa: E402
from h1_2_sim2real.robot_simulado import ManoSim  # noqa: E402

from manip import cinematica as C  # noqa: E402
from manip.config import _leer, cargar, cargar_objetos  # noqa: E402
from manip.estacion import Estacion  # noqa: E402

DATOS = os.path.join(RAIZ, "datos", "banco_mano")
CUERPO_MUNECA = {"izq": "left_wrist_yaw_link", "der": "right_wrist_yaw_link"}
# constante de tiempo de la soldadura mocap-mano (s): cuánto cede la mano al empujar. Se ajusta
# para que ceda como el brazo con kp 100 (comprobado contra probar_agarre.py)
RIGIDEZ_SOLREF = 0.05


def escena(lado, nombre_obj, obj, p_obj, gui, cfg, cat):
    """MJCF: la estación vacía, el objeto en p_obj (mundo) y la mano `lado` sobre un mocap."""
    xml, _ = construir(0, 0, cfg, cat, salida=tempfile.mkdtemp())
    arbol = ET.parse(xml)
    r = arbol.getroot()
    wb = r.find("worldbody")
    pelvis = next(b for b in wb.findall("body") if b.get("name") == "pelvis")
    muneca = next(b for b in pelvis.iter("body") if b.get("name") == CUERPO_MUNECA[lado])
    wb.remove(pelvis)
    # mocap = la consigna; la mano va en un cuerpo libre unido a él por una soldadura blanda
    # (el brazo con kp 100 cede cuando los dedos empujan) y con su peso compensado (como el
    # brazo con la compensación de gravedad)
    ET.SubElement(wb, "body", {"name": "base_mano", "mocap": "true", "pos": "0 0 1.5"})
    libre = ET.SubElement(wb, "body", {"name": "mano_libre", "pos": "0 0 1.5"})
    ET.SubElement(libre, "freejoint", {"name": "mano_libre"})
    muneca.set("pos", "0 0 0")
    muneca.attrib.pop("quat", None)
    for j in muneca.findall("joint"):              # la guiñada de la muñeca: la lleva el mocap
        muneca.remove(j)
    libre.append(muneca)
    for b in libre.iter("body"):
        b.set("gravcomp", "1")
    # fuera lo que nombra juntas y cuerpos que ya no están
    m_juntas = {j.get("name") for j in r.iter("joint")}
    m_cuerpos = {b.get("name") for b in r.iter("body")}
    for sec in ("actuator", "sensor", "equality", "contact", "keyframe", "tendon"):
        for s in r.findall(sec):
            for e in list(s):
                refs = [e.get(k) for k in ("joint", "joint1", "joint2", "body", "body1", "body2", "site",
                                           "objname") if e.get(k)]
                malo = e.tag == "key" or any(x not in m_juntas and x not in m_cuerpos for x in refs)
                if e.tag in ("fixed", "spatial"):
                    malo = any(j.get("joint") not in m_juntas for j in e.iter("joint"))
                if malo:
                    s.remove(e)
    sitios = {s.get("name") for s in r.iter("site")}
    for s in r.findall("sensor"):
        for e in list(s):
            if e.get("site") and e.get("site") not in sitios:
                s.remove(e)
    cuerpo_objeto(wb, nombre_obj, obj, p_obj[0], p_obj[1], gui, p_obj[2])
    eq = r.find("equality")
    if eq is None:
        eq = ET.SubElement(r, "equality")
    ET.SubElement(eq, "weld", {"body1": "base_mano", "body2": "mano_libre", "solref": f"{RIGIDEZ_SOLREF} 1"})
    return mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))


class _Sim:
    """Lo que ManoSim necesita del simulador."""

    def __init__(self, m, d):
        self.model, self.data, self.cerrojo = m, d, threading.Lock()


def probar(lado, nombre, ag, cat, cfg, err_xy=(0.0, 0.0), gui=0.0, x=0.36, y=None, foto=False, lift=0.10):
    """Un agarre. Devuelve dict con levantado, sujeto, dedos, fuerzas."""
    est = Estacion(cfg)
    obj = cat[nombre]
    y = (0.10 if lado == "izq" else -0.10) if y is None else y
    zm = est.z_mesa
    m = escena(lado, nombre, obj, (x, y, zm), gui, cfg, cat)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mano = ManoSim(_Sim(m, d), lado)
    mano.escribir(J.SPEED_SET, [300] * 6)
    # pose de la muñeca: como el planificador, en torso_link; el mundo es torso + (0, 0, torso_z)
    R = C.rotacion_agarre(lado, ag.get("tipo", "arriba") if ag.get("tipo") != "pinza" else "arriba",
                          ag.get("elev_deg", 0.0), ag.get("az_deg", 0.0) + math.degrees(gui) * 0)
    off = C.punto_agarre(lado, ag["prof_palma"], ag.get("avance", 0.17), ag.get("lateral", 0.0))
    p_ag = np.array([x + err_xy[0], y + err_xy[1], zm + ag["altura_agarre"]])
    p_mun = p_ag - R @ off
    sube = np.array([0, 0, ag.get("preagarre", 0.06)])
    if ag.get("tipo") == "lado":       # el lateral entra de lado, a lo largo de la normal de la palma
        n = -R[:, 1] if lado == "izq" else R[:, 1]
        sube = -n * ag.get("preagarre", 0.06) + np.array([0, 0, 0.01])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    bid = m.body("base_mano").mocapid[0]
    oid = m.body(nombre).id

    def ir(p, T):
        p0 = d.mocap_pos[bid].copy()
        n = max(1, int(T / m.opt.timestep))
        for k in range(1, n + 1):
            s = k / n
            s = s * s * s * (10 - 15 * s + 6 * s * s)
            d.mocap_pos[bid] = p0 + (p - p0) * s
            mano.paso(m.opt.timestep)
            mujoco.mj_step(m, d)

    def esperar(T):
        for _ in range(int(T / m.opt.timestep)):
            mano.paso(m.opt.timestep)
            mujoco.mj_step(m, d)

    d.mocap_pos[bid] = p_mun + sube
    d.mocap_quat[bid] = q
    ja = m.jnt_qposadr[m.joint("mano_libre").id]
    d.qpos[ja:ja + 3], d.qpos[ja + 3:ja + 7] = p_mun + sube, q
    mujoco.mj_forward(m, d)
    mano.escribir(J.ANGLE_SET, ag["preforma"])
    esperar(1.0)
    z0 = d.xpos[oid][2]
    p_ini = d.xpos[oid].copy()
    ir(p_mun, 1.5)
    esperar(0.3)
    empujado = float(np.linalg.norm(d.xpos[oid][:2] - p_ini[:2]))
    mano.escribir(J.FORCE_SET, ag["force_set"])
    if ag.get("previo"):                          # lateral: primero oponer el pulgar
        mano.escribir(J.ANGLE_SET, ag["previo"])
        esperar(2.0)
    mano.escribir(J.ANGLE_SET, ag["cierre"])
    esperar(1.5)
    a = np.array(mano.leer(J.ANGLE_ACT, 6), float)
    f = np.array(mano.leer(J.FORCE_ACT, 6), float)
    fotos = [_foto(m, d, d.xpos[oid])] if foto else []
    ir(p_mun + np.array([0, 0, lift]), 2.0)
    esperar(1.0)
    # transporte: un vaivén lateral de 15 cm como el de ir a la bandeja
    ir(p_mun + np.array([0, 0.15 if lado == "izq" else -0.15, lift]), 2.0)
    esperar(1.0)
    levantado = float(d.xpos[oid][2] - z0)
    if foto:
        fotos.append(_foto(m, d, d.xpos[oid]))
        import cv2
        os.makedirs(DATOS, exist_ok=True)
        cv2.imwrite(os.path.join(DATOS, f"{nombre}_{lado}.png"), np.hstack(fotos)[..., ::-1])
    Ro = d.xmat[oid].reshape(3, 3)
    return {"levantado_mm": 1000 * levantado, "ok": levantado > 0.8 * lift,
            "empujado_mm": 1000 * empujado, "angulos": a.astype(int).tolist(), "fuerzas": f.astype(int).tolist(),
            "inclinado_deg": math.degrees(math.acos(min(1.0, abs(Ro[2, 2]))))}


def _foto(m, d, mira):
    """Dos vistas: desde delante (lado de las puntas) y desde el robot (lado del pulgar)."""
    r = mujoco.Renderer(m, 240, 320)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = mira
    imgs = []
    for az in (200, 340):
        cam.distance, cam.azimuth, cam.elevation = 0.35, az, -15
        r.update_scene(d, cam)
        imgs.append(r.render().copy())
    r.close()
    return np.hstack(imgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objeto", default="cubo")
    ap.add_argument("--lado", default="izq", choices=["izq", "der"])
    ap.add_argument("--fijar", nargs="*", default=[], help="clave=valor del agarre")
    ap.add_argument("--barrer", nargs="+", default=None, help="clave v1 v2 ... (barrido de un parámetro)")
    ap.add_argument("--error", type=float, nargs="*", default=[0.0], help="error de posición (m), 8 direcciones")
    ap.add_argument("--guinadas", type=float, nargs="*", default=[0.0], help="guiñada del objeto (grados)")
    ap.add_argument("--foto", action="store_true")
    ap.add_argument("--rigidez", type=float, default=None, help="solref de la soldadura mocap-mano (s)")
    a = ap.parse_args()
    if a.rigidez is not None:
        global RIGIDEZ_SOLREF
        RIGIDEZ_SOLREF = a.rigidez
    import yaml
    cfg, cat = cargar(), cargar_objetos()
    ag = dict(_leer(os.path.join(RAIZ, "config", "agarres.yaml"))[a.objeto])
    for kv in a.fijar:
        k, v = kv.split("=", 1)
        ag[k] = yaml.safe_load(v)
    barrido = [(None, None)]
    if a.barrer:
        barrido = [(a.barrer[0], yaml.safe_load(v)) for v in a.barrer[1:]]
    for clave, valor in barrido:
        g = dict(ag)
        if clave:
            g[clave] = valor
        filas = []
        for e, gd in itertools.product(a.error, a.guinadas):
            dirs = [(0.0, 0.0)] if e == 0 else [(e * math.cos(t), e * math.sin(t))
                                                for t in np.arange(8) * math.pi / 4]
            for dx, dy in dirs:
                r = probar(a.lado, a.objeto, g, cat, cfg, (dx, dy), math.radians(gd), foto=a.foto and e == 0)
                filas.append((e, gd, dx, dy, r))
        n_ok = sum(r["ok"] for *_, r in filas)
        tit = f"{clave}={valor}" if clave else "tabla"
        print(f"{a.objeto} {a.lado} {tit}: {n_ok}/{len(filas)} sujetos", flush=True)
        for e, gd, dx, dy, r in filas:
            print(f"   err {1000 * e:3.0f} mm ({1000 * dx:+4.0f},{1000 * dy:+4.0f}) gui {gd:3.0f}: "
                  f"{'OK ' if r['ok'] else 'MAL'} sube {r['levantado_mm']:5.0f} mm, "
                  f"empujado {r['empujado_mm']:4.0f} mm, "
                  f"incl {r['inclinado_deg']:3.0f}, ángulos {r['angulos'][:5]}, fuerzas {r['fuerzas'][:5]}", flush=True)


if __name__ == "__main__":
    main()
