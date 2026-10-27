#!/usr/bin/env python3
"""Genera la estación del reto (R40-RT-H1_2-0002, sección 4) para MuJoCo:
sim/estaciones/estacion_n{N}_s{S}.xml + .json (la verdad de terreno con la que se evalúa).

Robot: la escena del suelo de h1_2_sim2real (h1_2_escena_suelo.xml: base flotante, manos
Inspire, D435 en la cabeza), de pie en el origen mirando a +x. Lo que se añade:
  * sala con paredes, suelo gris oscuro y luz cenital difusa (config/estacion.yaml: ambiente);
  * mesa con tablero blanco mate de 80x50 cm (placa blanca sobre canto gris) a la altura de
    la config (92 cm; el reto dice 86; --mesa para probar otra), con el borde cercano a la
    distancia d del pecho (la elige el hito 1);
  * zona de recogida de 40x25 cm marcada con cinta azul y 4 bandejas de 12x12 cm con borde
    de 2 cm, con su centro a 30 cm de la línea media (A y B a la izquierda);
  * marcadores ArUco de 4 cm (DICT_4X4_50) en las 4 esquinas del tablero y en el dorso de
    cada mano, hechos con cajitas negras sobre una placa blanca (sin texturas);
  * nivel 1: cuatro dianas de 2 cm con una cruz; niveles 2-4: los cuatro objetos de
    config/objetos.yaml (cuerpos sueltos con su masa y fricción), colocados al azar dentro de
    la zona según la regla del nivel; nivel 4: además, la cruz del centro del tablero;
    nivel 0: tablero vacío (calibración);
  * cámaras con los intrínsecos del reto (color fx 604.1, fy 603.6; IR 378.8) y la pose de
    la D435 DESVIADA de la del URDF (--desvio-camara): la pose real de la cámara no se
    conoce y la tiene que encontrar la calibración del hito 2. La desviación queda en el
    JSON bajo "verdad_oculta": el sistema NO la lee, solo la evaluación.

Reglas de colocación (sec. 4): dentro de la zona de recogida, sin solaparse. Nivel 2: cada
objeto del lado de su bandeja. Nivel 3 (y 4): al menos dos en el lado contrario a su
bandeja. Nivel 4: además, lejos de la cruz del centro del tablero.

Uso:  python3 sim/construir_estacion.py --nivel 3 --semilla 1
      python3 sim/construir_estacion.py --todos            # niveles 0..4, semillas 0..2
      python3 sim/construir_estacion.py --nivel 0 --sin-desvio
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
from manip import cinematica as C  # noqa: E402
from manip.cinematica import DORSO  # noqa: E402
from manip.config import _leer, cargar, cargar_objetos  # noqa: E402
from manip.estacion import LETRAS, Estacion  # noqa: E402

SALIDA = os.path.join(AQUI, "estaciones")
# desviación por defecto de la cámara respecto del URDF (grados y metros, en el marco de la
# cámara de MuJoCo: x a la derecha de la imagen, y arriba, -z hacia delante)
DESVIO = {"cabeceo": -1.8, "guinada": 1.2, "alabeo": 0.7, "dx": 0.006, "dy": -0.004, "dz": 0.009}
CAMS = ("robot_rgbd_camera", "robot_ir_left", "robot_ir_right")


def _raiz_s2r():
    """h1_2_sim2real/src/h1_2_sim2real (tareas/ puede ser un enlace al repo de GitHub)."""
    candidatos = []
    if os.environ.get("H12_S2R_COMUN"):
        candidatos.append(os.path.join(os.path.dirname(os.environ["H12_S2R_COMUN"]), "src", "h1_2_sim2real"))
    candidatos += [os.path.normpath(os.path.join(AQUI, "..", "..", "..", "h1_2_sim2real", "src", "h1_2_sim2real")),
                   os.path.expanduser("~/humanoid_ws/src/h1_2_sim2real/src/h1_2_sim2real")]
    for c in candidatos:
        if os.path.isfile(os.path.join(c, "mjcf", "h1_2_escena_suelo.xml")):
            return c
    raise SystemExit("no encuentro h1_2_sim2real: source scripts/entorno.sh")


S2R = _raiz_s2r()
ESCENA_ROBOT = os.path.join(S2R, "mjcf", "h1_2_escena_suelo.xml")
MALLAS = os.path.join(S2R, "meshes")


def f(v):
    return " ".join(f"{x:.6g}" for x in np.ravel(v))


def rot(eje, ang):
    c, s = math.cos(ang), math.sin(ang)
    if eje == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if eje == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_de_R(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def caja(padre, nombre, pos, medio, material, xyaxes=None, colision=False, **extra):
    a = {"name": nombre, "type": "box", "pos": f(pos), "size": f(medio), "material": material}
    if xyaxes is not None:
        a["xyaxes"] = f(xyaxes)
    if not colision:
        a.update({"contype": "0", "conaffinity": "0", "density": "0", "group": "2"})
    a.update(extra)
    return ET.SubElement(padre, "geom", a)


# ------------------------------------------------------------------ ArUco
def celdas_aruco(ident, diccionario="DICT_4X4_50"):
    """Matriz 6x6 (borde incluido) del marcador: True = celda negra."""
    import cv2
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, diccionario))
    img = cv2.aruco.generateImageMarker(dic, int(ident), 6, borderBits=1)
    return img < 128


def aruco(padre, prefijo, ident, centro, ex, ey, lado, diccionario):
    """Marcador en el plano (ex, ey) con normal ex x ey, centro `centro` en el marco del
    padre. Ejes como OpenCV: x a la derecha del marcador, y arriba, z hacia fuera. Placa
    blanca con una celda de margen y, encima, las celdas negras fusionadas por filas."""
    ex, ey = np.asarray(ex, float), np.asarray(ey, float)
    ez = np.cross(ex, ey)
    cel = lado / 6.0
    xy = np.concatenate([ex, ey])
    caja(padre, f"{prefijo}_placa", centro + 0.0005 * ez, [lado / 2 + cel, lado / 2 + cel, 0.0005],
         "aruco_blanco", xy)
    negras = celdas_aruco(ident, diccionario)
    k = 0
    for r in range(6):
        c = 0
        while c < 6:
            if not negras[r, c]:
                c += 1
                continue
            c1 = c
            while c1 + 1 < 6 and negras[r, c1 + 1]:
                c1 += 1
            u = ((c + c1) / 2 - 2.5) * cel
            v = (2.5 - r) * cel
            caja(padre, f"{prefijo}_n{k}", centro + u * ex + v * ey + 0.0012 * ez,
                 [(c1 - c + 1) * cel / 2, cel / 2, 0.0002], "aruco_negro", xy)
            k += 1
            c = c1 + 1


# ------------------------------------------------------------------ objetos
def radio_huella(o):
    if o["forma"] == "caja":
        return math.hypot(o["medidas"][0], o["medidas"][1]) / 2
    if o["forma"] in ("cilindro", "esfera"):
        return o["radio"]
    return o["largo"] / 2           # cilindro_tumbado


def alto(o):
    return {"caja": lambda: o["medidas"][2], "cilindro": lambda: o["alto"], "esfera": lambda: 2 * o["radio"],
            "cilindro_tumbado": lambda: 2 * o["radio"]}[o["forma"]]()


def colocar(est, objetos, nivel, rng, evitar=None, intentos=20000, holgura=0.02):
    """(x, y, guiñada) de cada objeto según la regla del nivel. Nivel 2: del lado de su
    bandeja; 3 y 4: al menos dos en el lado contrario. `evitar` = (x, y, radio) libre.
    `holgura`: espacio libre mínimo entre las huellas de dos objetos (sitio para la mano)."""
    nombres = list(objetos)
    lado_bandeja = {n: (1 if est.bandejas[objetos[n]["bandeja"]].cy > 0 else -1) for n in nombres}
    z = est.zona
    for _ in range(intentos):
        if nivel == 2:
            lado = dict(lado_bandeja)
        else:
            contrarios = set(rng.choice(nombres, size=int(rng.integers(2, len(nombres) + 1)), replace=False))
            lado = {n: (-lado_bandeja[n] if n in contrarios else lado_bandeja[n]) for n in nombres}
        poses, ok = {}, True
        for n in nombres:
            r = radio_huella(objetos[n])
            x = rng.uniform(z.cx - z.fondo / 2 + r, z.cx + z.fondo / 2 - r)
            ymin, ymax = 0.01, z.ancho / 2 - r
            if ymax <= ymin:
                ok = False
                break
            y = lado[n] * rng.uniform(ymin, ymax)
            if evitar is not None and math.hypot(x - evitar[0], y - evitar[1]) < evitar[2] + r:
                ok = False
                break
            for m, (xm, ym, _) in poses.items():
                if math.hypot(x - xm, y - ym) < r + radio_huella(objetos[m]) + holgura:
                    ok = False
                    break
            if not ok:
                break
            poses[n] = (x, y, float(rng.uniform(-math.pi, math.pi)))
        if ok:
            return poses
    raise RuntimeError("no caben los objetos en la zona de recogida con esa regla")


def cuerpo_objeto(wb, nombre, o, x, y, gui, z_mesa):
    h = alto(o)
    q = quat_de_R(rot("z", gui) @ (rot("y", math.pi / 2) if o["forma"] == "cilindro_tumbado" else np.eye(3)))
    b = ET.SubElement(wb, "body", {"name": nombre, "pos": f([x, y, z_mesa + h / 2 + 0.0005]), "quat": f(q)})
    ET.SubElement(b, "freejoint", {"name": f"{nombre}_libre"})
    g = {"name": f"{nombre}_geom", "mass": f"{o['masa']:.4g}", "material": f"obj_{nombre}",
         "friction": f(o["friccion"]), "condim": "6" if o["forma"] in ("esfera", "cilindro_tumbado") else "4",
         "solref": f"{o.get('solref', 0.004)} 1", "priority": "1"}
    if "solimp" in o:
        g["solimp"] = f(o["solimp"])
    if o["forma"] == "caja":
        g.update(type="box", size=f(np.array(o["medidas"]) / 2))
    elif o["forma"] == "esfera":
        g.update(type="sphere", size=f(o["radio"]))
    elif o["forma"] == "cilindro":
        g.update(type="cylinder", size=f([o["radio"], o["alto"] / 2]))
    else:
        g.update(type="cylinder", size=f([o["radio"], o["largo"] / 2]))
    ET.SubElement(b, "geom", g)
    return {"x": x, "y": y, "z": z_mesa + h / 2, "guinada": gui, "alto": h, "quat": list(q)}


# ------------------------------------------------------------------ escena
def camara_desviada(cam_rgbd, desvio):
    """Pose real de la cámara de color en torso_link (pos, R con columnas x, y, z de la
    cámara de MuJoCo) aplicando la desviación a la nominal."""
    pos0 = np.array([float(v) for v in cam_rgbd.get("pos").split()])
    xa = np.array([float(v) for v in cam_rgbd.get("xyaxes").split()])
    x, y = xa[:3] / np.linalg.norm(xa[:3]), xa[3:] / np.linalg.norm(xa[3:])
    R0 = np.column_stack([x, y, np.cross(x, y)])
    Rd = rot("x", math.radians(desvio["cabeceo"])) @ rot("y", math.radians(desvio["guinada"])) \
        @ rot("z", math.radians(desvio["alabeo"]))
    R = R0 @ Rd
    pos = pos0 + R0 @ np.array([desvio["dx"], desvio["dy"], desvio["dz"]])
    return pos0, R0, pos, R


def pose_optica(pos, R):
    """Marco óptico ROS (x derecha, y abajo, z adelante) a partir del de MuJoCo."""
    Ro = np.column_stack([R[:, 0], -R[:, 1], -R[:, 2]])
    return {"pos": [float(v) for v in pos], "R": Ro.tolist()}


def brazos_keyframe(arbol, cfg):
    """Brazos del keyframe de pie en la pose de observación (config/poses.yaml). Con la mesa a
    92 cm, en la postura de pie de la escena del suelo las puntas de los dedos quedan a ~1 cm
    del tablero y, con la gravedad, se apoyan en él. En el robot es el protocolo: los brazos
    pasan a la pose de observación antes de acercar la mesa (o de poner los calzos)."""
    import mujoco
    ruta = os.path.join(os.path.dirname(AQUI), "config", "poses.yaml")
    if not os.path.isfile(ruta):
        return
    obs = _leer(ruta)["observacion"]
    r = arbol.getroot()
    m = mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))
    for key in r.iter("key"):
        qpos = [float(v) for v in key.get("qpos").split()]
        for lado in C.LADOS:
            for n, v in zip(C.NOMBRES_JUNTAS[lado], obs[lado]):
                qpos[int(m.jnt_qposadr[m.joint(n).id])] = float(v)
        key.set("qpos", f(qpos))          # el control interno del simulador sostiene este qpos


def construir(nivel, semilla, cfg, objetos, desvio=DESVIO, nombre=None, salida=SALIDA):
    est = Estacion(cfg)
    rng = np.random.default_rng(1000 * nivel + semilla)
    arbol = ET.parse(ESCENA_ROBOT)
    r = arbol.getroot()
    r.set("model", f"h1_2_estacion_n{nivel}_s{semilla}")
    r.find("compiler").set("meshdir", MALLAS + "/")

    # ---- cámaras: intrínsecos del reto y pose desviada (rígida para las tres)
    cc, ci = cfg["camara"]["color"], cfg["camara"]["ir"]
    cams = {c.get("name"): c for c in r.iter("camera") if c.get("name") in CAMS}
    pos0, R0, pos, R = camara_desviada(cams["robot_rgbd_camera"], desvio)
    for nombre_cam, c in cams.items():
        p_c = np.array([float(v) for v in c.get("pos").split()])
        off = R0.T @ (p_c - pos0)              # desplazamiento en el marco de la cámara
        c.set("pos", f(pos + R @ off))
        c.set("xyaxes", f(np.concatenate([R[:, 0], R[:, 1]])))
        fy, alto_px, ancho_px = (cc["fy"], cc["alto"], cc["ancho"]) if nombre_cam == "robot_rgbd_camera" \
            else (ci["fy"], ci["alto"], ci["ancho"])
        c.set("fovy", f"{math.degrees(2 * math.atan(alto_px / 2 / fy)):.4f}")
        c.set("resolution", f"{ancho_px} {alto_px}")

    # ---- contacto para agarrar (documentación de MuJoCo, "Contact"): cono de fricción
    # elíptico e impratio alto. Con el piramidal e impratio 1 (por defecto) lo agarrado
    # resbala entre los dedos aunque la fuerza normal sobre de sobra.
    opt = r.find("option")
    if opt is None:
        opt = ET.SubElement(r, "option")
    opt.set("cone", "elliptic")
    opt.set("impratio", "10")

    # ---- fuera el suelo y las luces de la escena del suelo
    for wb in r.findall("worldbody"):
        for c in list(wb):
            if c.tag == "light" or (c.tag == "geom" and c.get("name") == "floor"):
                wb.remove(c)
    for a in r.findall("asset"):
        for c in list(a):
            if c.get("name") == "groundplane":
                a.remove(c)
    asset = ET.SubElement(r, "asset")
    mat = lambda n, rgba: ET.SubElement(asset, "material", {"name": n, "rgba": rgba, "specular": "0.02",  # noqa: E731
                                                            "shininess": "0", "reflectance": "0"})
    amb = cfg["ambiente"]
    rgb = lambda k: f"{amb[k][0]} {amb[k][1]} {amb[k][2]} 1"  # noqa: E731
    mat("suelo_gris", rgb("suelo"))
    mat("paredes", rgb("paredes"))
    mat("tablero", rgb("tablero"))
    mat("patas", rgb("canto"))
    mat("cinta_zona", rgb("cinta_zona"))
    mat("bandeja", "0.30 0.32 0.34 1")
    mat("aruco_blanco", "0.97 0.97 0.97 1")
    mat("aruco_negro", "0.03 0.03 0.03 1")
    mat("diana", "1.0 0.45 0.05 1")
    mat("negro", "0.05 0.05 0.05 1")
    for n, o in objetos.items():
        mat(f"obj_{n}", f"{o['color'][0]} {o['color'][1]} {o['color'][2]} 1")
    vis = r.find("visual")
    for c in list(vis):
        if c.tag == "headlight":
            vis.remove(c)
    la = f([amb["luz_ambiente"]] * 3)
    ET.SubElement(vis, "headlight", {"ambient": la, "diffuse": "0 0 0", "specular": "0 0 0"})
    ET.SubElement(vis, "quality", {"shadowsize": "8192"})
    ET.SubElement(vis, "map", {"shadowscale": "0.9", "znear": "0.01"})
    stat = r.find("statistic")
    if stat is None:
        stat = ET.SubElement(r, "statistic")
    stat.set("center", f"{est.tablero.cx / 2:.2f} 0 1.0")
    stat.set("extent", "1.6")

    # ---- ArUco en el dorso de las manos (cuerpo *_wrist_yaw_link)
    ac = cfg["aruco"]
    marcadores = []
    for lado, cuerpo in (("izq", "left_wrist_yaw_link"), ("der", "right_wrist_yaw_link")):
        b = next(x for x in r.iter("body") if x.get("name") == cuerpo)
        centro, normal = DORSO[lado]
        ey = np.array([1.0, 0.0, 0.0])                    # arriba del marcador = hacia los dedos
        ex = np.cross(ey, normal)
        aruco(b, f"aruco_{lado}", ac["mano"][lado], centro, ex, ey, ac["lado"], ac["diccionario"])
        marcadores.append({"id": ac["mano"][lado], "cuerpo": cuerpo, "centro": centro.tolist(),
                           "R": np.column_stack([ex, ey, normal]).tolist()})

    # ---- mesa, zona, bandejas y marcadores del tablero (en el mundo)
    wb = ET.SubElement(r, "worldbody")
    ET.SubElement(wb, "geom", {"name": "suelo", "type": "plane", "size": "0 0 0.05", "material": "suelo_gris"})
    t, zm, gr = est.tablero, est.z_mesa, est.grosor
    sala = amb["sala"]
    xd, xa, ly, hz = sala["delante"], sala["detras"], sala["lado"], sala["alto"]
    for nombre_p, pos_p, medio_p in (("pared_delante", [xd, 0, hz / 2], [0.02, ly, hz / 2]),
                                     ("pared_detras", [xa, 0, hz / 2], [0.02, ly, hz / 2]),
                                     ("pared_izq", [(xd + xa) / 2, ly, hz / 2], [(xd - xa) / 2, 0.02, hz / 2]),
                                     ("pared_der", [(xd + xa) / 2, -ly, hz / 2], [(xd - xa) / 2, 0.02, hz / 2])):
        caja(wb, nombre_p, pos_p, medio_p, "paredes")
    ET.SubElement(wb, "light", {"name": "techo", "directional": "true", "pos": f([t.cx, 0.3, hz]),
                                "dir": "-0.05 -0.1 -1", "diffuse": f([amb["luz_cenital"]] * 3),
                                "specular": "0 0 0", "castshadow": "true"})
    for k, sy in enumerate((1, -1)):
        ET.SubElement(wb, "light", {"name": f"foco_{k}", "pos": f([t.cx, sy * 0.9, hz - 0.3]),
                                    "dir": f([0, -sy * 0.5, -1]), "cutoff": "70", "exponent": "1",
                                    "diffuse": f([amb["focos"]] * 3), "specular": "0 0 0", "castshadow": "false"})
    # tablero: placa blanca de 3 mm (la superficie de la mesa) sobre un canto gris, para que
    # el borde se distinga en el color; la colisión es la caja entera
    placa = 0.003
    caja(wb, "tablero", [t.cx, t.cy, zm - gr / 2], [t.fondo / 2, t.ancho / 2, gr / 2], "patas",
         colision=True, friction="0.8 0.005 0.0001")
    # losa de colisión invisible bajo el tablero (solo simulador): con 3 cm, un cilindro que penetra
    # mucho (el vaso apretado contra la pared de una bandeja) sale por debajo y cae al suelo (2 de 50
    # tiradas); con 12 cm el contacto siempre lo empuja hacia arriba
    losa = 0.12
    caja(wb, "tablero_losa", [t.cx, t.cy, zm - gr - losa / 2], [t.fondo / 2 - 0.05, t.ancho / 2 - 0.05, losa / 2],
         "patas", colision=True, rgba="0 0 0 0", group="3")
    caja(wb, "tablero_placa", [t.cx, t.cy, zm - placa / 2 + 0.0001], [t.fondo / 2, t.ancho / 2, placa / 2],
         "tablero")
    for k, (px, py) in enumerate(t.esquinas()):
        sx, sy = -np.sign(px - t.cx), -np.sign(py - t.cy)
        caja(wb, f"pata_{k}", [px + 0.04 * sx, py + 0.04 * sy, (zm - gr) / 2], [0.02, 0.02, (zm - gr) / 2],
             "patas", colision=True)
    zr = est.zona
    a_cinta = 0.006
    for k, (px, py, lx, ly) in enumerate((
            (zr.cx - zr.fondo / 2, zr.cy, a_cinta, zr.ancho), (zr.cx + zr.fondo / 2, zr.cy, a_cinta, zr.ancho),
            (zr.cx, zr.cy + zr.ancho / 2, zr.fondo, a_cinta), (zr.cx, zr.cy - zr.ancho / 2, zr.fondo, a_cinta))):
        caja(wb, f"zona_{k}", [px, py, zm + 0.0003], [lx / 2, ly / 2, 0.0003], "cinta_zona")
    lb, hb, ep = est.bandeja_lado, est.bandeja_borde, cfg["bandejas"]["espesor_pared"]
    for letra in LETRAS:
        b = est.bandejas[letra]
        cb = ET.SubElement(wb, "body", {"name": f"bandeja_{letra}", "pos": f([b.cx, b.cy, zm])})
        caja(cb, f"bandeja_{letra}_fondo", [0, 0, 0.002], [lb / 2, lb / 2, 0.002], "bandeja", colision=True)
        for k, (px, py, lx, ly) in enumerate(((lb / 2 - ep / 2, 0, ep, lb), (-lb / 2 + ep / 2, 0, ep, lb),
                                              (0, lb / 2 - ep / 2, lb, ep), (0, -lb / 2 + ep / 2, lb, ep))):
            caja(cb, f"bandeja_{letra}_pared{k}", [px, py, hb / 2], [lx / 2, ly / 2, hb / 2], "bandeja",
                 colision=True)
    ex_t, ey_t = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])   # leíbles desde el robot
    for ident, (px, py) in est.aruco_esquinas.items():
        aruco(wb, f"aruco_t{ident}", ident, np.array([px, py, zm]), ex_t, ey_t, ac["lado"], ac["diccionario"])
        marcadores.append({"id": int(ident), "cuerpo": "world", "centro": [px, py, zm],
                           "R": np.column_stack([ex_t, ey_t, np.cross(ex_t, ey_t)]).tolist()})

    # ---- lo propio de cada nivel
    geo = {"nivel": nivel, "semilla": semilla, "estacion": est.resumen(), "objetos": {}, "dianas": [],
           "cruz": None, "aruco": marcadores}
    if nivel == 1:
        dcfg = cfg["dianas"]
        dianas = []
        for _ in range(10000):
            x = rng.uniform(zr.cx - zr.fondo / 2 + 0.02, zr.cx + zr.fondo / 2 - 0.02)
            y = rng.uniform(zr.cy - zr.ancho / 2 + 0.02, zr.cy + zr.ancho / 2 - 0.02)
            if all(math.hypot(x - a, y - b) >= dcfg["separacion_min"] for a, b in dianas):
                dianas.append((x, y))
            if len(dianas) == 4:
                break
        for k, (x, y) in enumerate(dianas):
            ET.SubElement(wb, "geom", {"name": f"diana_{k}", "type": "cylinder", "pos": f([x, y, zm + 0.0003]),
                                       "size": f([dcfg["diametro"] / 2, 0.0003]), "material": "diana",
                                       "contype": "0", "conaffinity": "0", "group": "2"})
            for j, (lx, ly) in enumerate(((0.016, 0.0015), (0.0015, 0.016))):
                caja(wb, f"diana_{k}_cruz{j}", [x, y, zm + 0.0007], [lx / 2, ly / 2, 0.0001], "negro")
        geo["dianas"] = [[x, y, zm] for x, y in dianas]
    if nivel >= 2:
        # nivel 4: la cruz va en el CENTRO DEL TABLERO (sec. 4), que cae dentro de la zona
        evitar = (est.tablero.cx, est.tablero.cy, cfg["cruz_n4"]["largo"] / 2 + 0.02) if nivel == 4 else None
        # la holgura de la config si caben; si no (nivel 4: la cruz del centro libre), un cm menos
        holg = float(cfg["zona_recogida"].get("holgura_objetos", 0.02))
        while True:
            try:
                poses = colocar(est, objetos, nivel, rng, evitar=evitar, holgura=holg)
                break
            except RuntimeError:
                if holg <= 0.02:
                    raise
                holg = round(holg - 0.01, 3)
        geo["holgura_objetos"] = holg
        for n, (x, y, gui) in poses.items():
            p = cuerpo_objeto(wb, n, objetos[n], x, y, gui, zm)
            o = objetos[n]
            p.update({"forma": o["forma"], "bandeja": o["bandeja"], "masa": o["masa"],
                      "lado_bandeja": est.lado_bandeja(o["bandeja"]), "lado": "izq" if y > 0 else "der",
                      "medidas": {k: o[k] for k in ("medidas", "radio", "alto", "largo") if k in o}})
            geo["objetos"][n] = p
    if nivel == 4:
        c = cfg["cruz_n4"]
        for j, (lx, ly) in enumerate(((c["largo"], c["ancho"]), (c["ancho"], c["largo"]))):
            caja(wb, f"cruz_{j}", [t.cx, t.cy, zm + 0.0003], [lx / 2, ly / 2, 0.0003], "negro")
        geo["cruz"] = [t.cx, t.cy, zm]

    # el keyframe de pie solo trae el qpos del robot: MuJoCo rellenaría con CEROS el de los
    # objetos (todos en el origen, solapados). Sus freejoint van después del robot, en orden.
    for key in r.iter("key"):
        extra = []
        for o in geo["objetos"].values():
            extra += [o["x"], o["y"], o["z"] + 0.0005] + list(o["quat"])
        if extra:
            key.set("qpos", key.get("qpos") + " " + f(extra))
    geo["camara_nominal"] = pose_optica(pos0, R0)
    geo["verdad_oculta"] = {"camara_torso": pose_optica(pos, R), "desvio": desvio,
                            "nota": "pose REAL de la cámara en torso_link; el sistema no la lee"}
    ET.indent(arbol, space="  ")
    brazos_keyframe(arbol, cfg)
    r.insert(0, ET.Comment(f" GENERADO por tareas/manipulacion/sim/construir_estacion.py (nivel {nivel}, "
                           f"semilla {semilla}, d = {est.d:.3f} m). No editar a mano. "))
    os.makedirs(salida, exist_ok=True)
    base = os.path.join(salida, nombre or f"estacion_n{nivel}_s{semilla}")
    arbol.write(base + ".xml", encoding="unicode", xml_declaration=True)
    with open(base + ".json", "w") as fh:
        json.dump(geo, fh, indent=1)
    return base + ".xml", geo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nivel", type=int, default=3, help="0 vacía (calibración), 1 dianas, 2-4 objetos")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--todos", action="store_true", help="niveles 0..4 con semillas 0..2")
    ap.add_argument("--d", type=float, default=None, help="distancia borde-pecho (por defecto la de la config)")
    ap.add_argument("--mesa", type=float, default=None, help="altura del tablero (m; por defecto la de la config)")
    ap.add_argument("--sin-desvio", action="store_true", help="cámara exactamente en la pose del URDF")
    a = ap.parse_args()
    cfg, objetos = cargar(), cargar_objetos()
    if a.d is not None:
        cfg["mesa"]["d"] = a.d
    if a.mesa is not None:
        cfg["mesa"]["altura"] = a.mesa
    desvio = {k: 0.0 for k in DESVIO} if a.sin_desvio else DESVIO
    trabajos = [(n, s) for n in range(5) for s in range(3)] if a.todos else [(a.nivel, a.semilla)]
    for n, s in trabajos:
        xml, geo = construir(n, s, cfg, objetos, desvio)
        extra = ", ".join(f"{k} {v['lado'][0]}->{v['bandeja']}" for k, v in geo["objetos"].items())
        print(f"nivel {n} semilla {s}: {xml}  {extra}")


if __name__ == "__main__":
    main()
