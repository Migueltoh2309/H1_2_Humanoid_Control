#!/usr/bin/env python3
"""Genera las pistas del reto (R40-RT-H1_2-0001, seccion 4) para MuJoCo:
sim/pistas/pista_nivel{1..4}.xml + pista_nivel{1..4}.json (geometria exacta, la
verdad de terreno con la que se evalua: linea central cada 1 cm, huecos, barra,
sombra).

Robot: la escena del suelo de h1_2_sim2real (h1_2_escena_suelo.xml, manos Inspire,
D435 en la cabeza), de pie en el origen mirando a +x. Lo que se cambia:
  * suelo blanco mate y cinta negra mate de 5 cm (cajas de 1 mm, sin colision);
  * cuadro de inicio de 60x60 cm alrededor del robot y barra de fin de 60 cm;
  * camaras con los intrinsecos MEDIDOS del reto (sec. 3), 640x480:
      color fx=fy~603.6 -> fovy 43.38 grados; IR fx=fy=378.8 -> fovy 64.72 grados
    (MuJoCo pone el punto principal en el centro: 320, 240; el real esta en
    321.5, 236.0, a 4 px: se ignora);
  * inclinacion de la camara: --inclinacion (grados bajo la horizontal). Por
    defecto la del URDF de Unitree (camera_joint, pitch 0.886 rad = 50.8). Unitree no
    la documenta para el robot real: el reto pide MEDIRLA (herramientas/calibrar_camara.py);
  * luz cenital que proyecta sombras + ambiente, y en el nivel 3 una placa por
    encima de la camara que deja 1 m de pista en sombra (la camara no la ve).

Niveles (el robot sale del cuadro hacia +x):
  1  recta de 4 m desalineada 10 grados respecto del robot, barra de fin
  2  recta 2 m, curva 90 grados de radio 2 m, recta 2 m, barra
  3  S de radios 1.2 m, interrupcion de 40 cm y 1 m en sombra, barra
  4  recta 2 m, esquina de 90 grados en angulo recto, recta 2 m, barra

Uso:  python3 sim/construir_pistas.py [--inclinacion 50.8] [--niveles 1 2 3 4]
"""
import argparse
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "pistas")
S2R = os.path.normpath(os.path.join(AQUI, "..", "..", "..", "h1_2_sim2real", "src", "h1_2_sim2real"))
ESCENA_ROBOT = os.path.join(S2R, "mjcf", "h1_2_escena_suelo.xml")
MALLAS = os.path.join(S2R, "meshes")

ANCHO = 0.05            # cinta de 5 cm
ESPESOR = 0.001
LADO_INICIO = 0.60
LARGO_BARRA = 0.60
FY_COLOR, FY_IR, ALTO_IMG, ANCHO_IMG = 603.6, 378.8, 480, 640
PASO = 0.01             # muestreo de la linea central (verdad de terreno)


# ------------------------------------------------------------------ trazados
class Trazado:
    """Linea central como sucesion de rectas y arcos desde una pose inicial."""

    def __init__(self, x=0.0, y=0.0, rumbo=0.0):
        self.x, self.y, self.th = x, y, rumbo
        self.puntos = [(x, y, rumbo, 0.0)]
        self.s = 0.0

    def recta(self, largo):
        n = max(1, int(round(largo / PASO)))
        for _ in range(n):
            self.x += PASO * math.cos(self.th)
            self.y += PASO * math.sin(self.th)
            self.s += PASO
            self.puntos.append((self.x, self.y, self.th, self.s))
        return self

    def arco(self, radio, angulo):
        """angulo > 0 gira a la izquierda."""
        n = max(1, int(round(abs(angulo) * radio / PASO)))
        dth = angulo / n
        for _ in range(n):
            # cuerda de un arco elemental (exacta en el limite)
            self.x += radio * abs(dth) * math.cos(self.th + dth / 2)
            self.y += radio * abs(dth) * math.sin(self.th + dth / 2)
            self.th += dth
            self.s += radio * abs(dth)
            self.puntos.append((self.x, self.y, self.th, self.s))
        return self

    def esquina(self, angulo):
        """Giro instantaneo (angulo recto del nivel 4)."""
        self.th += angulo
        self.puntos.append((self.x, self.y, self.th, self.s))
        return self


def nivel(n):
    """(trazado, huecos [(s0, s1)], sombra (cx, cy, largo, ancho, rumbo) o None)."""
    if n == 1:
        return Trazado(rumbo=math.radians(10.0)).recta(4.0), [], None
    if n == 2:
        return Trazado().recta(2.0).arco(2.0, math.pi / 2).recta(2.0), [], None
    if n == 3:
        t = Trazado().recta(1.0).arco(1.2, math.pi / 2).arco(1.2, -math.pi / 2).recta(1.5)
        s_s = 1.0 + 1.2 * math.pi / 2           # arranque de la segunda curva
        huecos = [(s_s + 0.60, s_s + 1.00)]     # 40 cm sin cinta en la segunda curva
        # 1 m de sombra centrado en la primera curva
        s_c = 1.0 + 1.2 * math.pi / 4
        x, y, th, _ = min(t.puntos, key=lambda p: abs(p[3] - s_c))
        return t, huecos, (x, y, 1.0, 1.4, th)
    if n == 4:
        return Trazado().recta(2.0).esquina(math.pi / 2).recta(2.0), [], None
    raise ValueError(n)


# ------------------------------------------------------------------ MJCF
def caja(padre, nombre, cx, cy, largo, ancho, rumbo, material="cinta"):
    ET.SubElement(padre, "geom", {
        "name": nombre, "type": "box", "material": material, "contype": "0", "conaffinity": "0",
        "pos": f"{cx:.5f} {cy:.5f} {ESPESOR / 2:.5f}",
        "size": f"{largo / 2:.5f} {ancho / 2:.5f} {ESPESOR / 2:.5f}",
        "euler": f"0 0 {rumbo:.6f}", "group": "0"})


def _intervalos_sin_huecos(s0, s1, huecos):
    """[s0, s1] menos los huecos."""
    trozos = [(s0, s1)]
    for a, b in huecos:
        nuevos = []
        for u, v in trozos:
            if b <= u or a >= v:
                nuevos.append((u, v))
                continue
            if a > u:
                nuevos.append((u, a))
            if b < v:
                nuevos.append((b, v))
        trozos = nuevos
    return [(u, v) for u, v in trozos if v - u > 1e-4]


def cinta_de_trazado(padre, puntos, huecos, prefijo):
    """Cajas de cinta: cada tramo recto en una sola caja, las curvas en trozos de
    2 cm (solapados 4 mm para que no se vean juntas). Los huecos se recortan por
    longitud de arco. En una esquina, los tramos se alargan media cinta para que
    el angulo quede lleno."""
    n = len(puntos)
    esquinas = {i for i in range(n - 1) if abs(puntos[i][3] - puntos[i + 1][3]) < 1e-9}
    piezas = []          # (i, j, es_recta)
    i = 0
    while i < n - 1:
        if i in esquinas:
            i += 1
            continue
        th = puntos[i + 1][2]
        j = i + 1
        while j < n - 1 and j not in esquinas and abs(puntos[j + 1][2] - th) < 1e-9:
            j += 1
        if j - i >= 3:
            piezas.append((i, j, True))
        else:
            j = min(i + 2, n - 1)
            if j in esquinas or any(k in esquinas for k in range(i, j)):
                j = i + 1
            piezas.append((i, j, False))
        i = j
    k = 0
    for i, j, recta in piezas:
        x0, y0, _, s0 = puntos[i]
        x1, y1, _, s1 = puntos[j]
        largo = math.hypot(x1 - x0, y1 - y0)
        if largo < 1e-6:
            continue
        ux, uy = (x1 - x0) / largo, (y1 - y0) / largo
        for u, v in _intervalos_sin_huecos(s0, s1, huecos):
            a = (u - s0) / (s1 - s0) * largo
            b = (v - s0) / (s1 - s0) * largo
            ext0 = ANCHO / 2 if (recta and i - 1 in esquinas and u == s0) else 0.0
            ext1 = ANCHO / 2 if (recta and j in esquinas and v == s1) else 0.0
            solape = 0.0 if recta else 0.004
            a, b = a - ext0 - solape / 2, b + ext1 + solape / 2
            cx, cy = x0 + ux * (a + b) / 2, y0 + uy * (a + b) / 2
            caja(padre, f"{prefijo}_{k}", cx, cy, b - a, ANCHO, math.atan2(uy, ux))
            k += 1
    return k


def construir(n, inclinacion_deg, emisor_info=None):
    t, huecos, sombra = nivel(n)
    arbol = ET.parse(ESCENA_ROBOT)
    r = arbol.getroot()
    r.set("model", f"h1_2_pista_nivel{n}")
    r.find("compiler").set("meshdir", MALLAS + "/")
    # camaras de la cabeza: intrinsecos del reto e inclinacion
    th = math.radians(inclinacion_deg)
    xy = f"0 -1 0 {math.sin(th):.9f} 0 {math.cos(th):.9f}"
    for cam in r.iter("camera"):
        nombre = cam.get("name")
        if nombre in ("robot_rgbd_camera", "robot_ir_left", "robot_ir_right"):
            fy = FY_COLOR if nombre == "robot_rgbd_camera" else FY_IR
            cam.set("fovy", f"{math.degrees(2 * math.atan(ALTO_IMG / 2 / fy)):.4f}")
            cam.set("resolution", f"{ANCHO_IMG} {ALTO_IMG}")
            cam.set("xyaxes", xy)
    # fuera el suelo azul y las luces de la escena del suelo
    for wb in r.findall("worldbody"):
        for c in list(wb):
            if c.tag == "light" or (c.tag == "geom" and c.get("name") == "floor"):
                wb.remove(c)
    for a in r.findall("asset"):
        for c in list(a):
            if c.get("name") == "groundplane":
                a.remove(c)
    asset = ET.SubElement(r, "asset")
    ET.SubElement(asset, "material", {"name": "suelo_blanco", "rgba": "0.88 0.88 0.86 1", "specular": "0.05",
                                      "shininess": "0", "reflectance": "0"})
    ET.SubElement(asset, "material", {"name": "cinta", "rgba": "0.05 0.05 0.05 1", "specular": "0",
                                      "shininess": "0", "reflectance": "0"})
    ET.SubElement(asset, "material", {"name": "placa", "rgba": "0.3 0.3 0.3 1"})
    vis = r.find("visual")
    for c in list(vis):
        if c.tag == "headlight":
            vis.remove(c)
    # el headlight va pegado a la camara que renderiza: solo ambiente, sin "flash"
    ET.SubElement(vis, "headlight", {"ambient": "0.32 0.32 0.32", "diffuse": "0 0 0", "specular": "0 0 0"})
    ET.SubElement(vis, "quality", {"shadowsize": "8192"})
    ET.SubElement(vis, "map", {"shadowscale": "0.9", "znear": "0.01"})
    # la sombra de una luz direccional cubre lo que abarca el modelo: que abarque la pista
    xs = [p[0] for p in t.puntos]
    ys = [p[1] for p in t.puntos]
    stat = r.find("statistic")
    if stat is None:
        stat = ET.SubElement(r, "statistic")
    stat.set("center", f"{(min(xs) + max(xs)) / 2:.2f} {(min(ys) + max(ys)) / 2:.2f} 0.8")
    stat.set("extent", f"{max(max(xs) - min(xs), max(ys) - min(ys), 3.0) / 2 + 1.5:.2f}")

    wb = ET.SubElement(r, "worldbody")
    ET.SubElement(wb, "geom", {"name": "suelo", "type": "plane", "size": "0 0 0.05", "material": "suelo_blanco"})
    ET.SubElement(wb, "light", {"name": "cenital", "directional": "true", "pos": "0 0 6", "dir": "0 0 -1",
                                "diffuse": "0.62 0.62 0.62", "specular": "0 0 0", "castshadow": "true"})
    n_trozos = cinta_de_trazado(wb, t.puntos, huecos, "cinta")
    # cuadro de inicio (60x60 alrededor del robot) y barra de fin
    h = LADO_INICIO / 2
    for k, (cx, cy, largo, rumbo) in enumerate(((h, 0, LADO_INICIO + ANCHO, math.pi / 2), (-h, 0, LADO_INICIO + ANCHO, math.pi / 2),
                                                (0, h, LADO_INICIO, 0.0), (0, -h, LADO_INICIO, 0.0))):
        caja(wb, f"inicio_{k}", cx, cy, largo, ANCHO, rumbo)
    xf, yf, thf, sf = t.puntos[-1]
    caja(wb, "barra_fin", xf, yf, LARGO_BARRA, ANCHO, thf + math.pi / 2)
    if sombra is not None:
        cx, cy, largo, ancho, rumbo = sombra
        # placa a 2.4 m: por encima de la camara (1.70 m, mira hacia abajo) -> no se ve
        ET.SubElement(wb, "geom", {"name": "placa_sombra", "type": "box", "material": "placa", "contype": "0",
                                   "conaffinity": "0", "pos": f"{cx:.4f} {cy:.4f} 2.4",
                                   "size": f"{largo / 2:.4f} {ancho / 2:.4f} 0.005", "euler": f"0 0 {rumbo:.6f}"})
    ET.indent(arbol, space="  ")
    r.insert(0, ET.Comment(f" GENERADO por tareas/seguidor_linea/sim/construir_pistas.py (nivel {n}). No editar a mano. "))
    xml = os.path.join(SALIDA, f"pista_nivel{n}.xml")
    arbol.write(xml, encoding="unicode", xml_declaration=True)

    geo = {
        "nivel": n, "ancho_cinta": ANCHO, "paso": PASO, "inclinacion_camara_deg": inclinacion_deg,
        "linea": [[round(x, 5), round(y, 5), round(th_, 6), round(s, 4)] for x, y, th_, s in t.puntos],
        "huecos": huecos, "largo_total": t.s,
        "barra": {"x": xf, "y": yf, "rumbo_linea": thf, "largo": LARGO_BARRA},
        "inicio": {"lado": LADO_INICIO, "x": 0.0, "y": 0.0},
        "sombra": None if sombra is None else dict(zip(("x", "y", "largo", "ancho", "rumbo"), sombra)),
        "robot_inicial": {"x": 0.0, "y": 0.0, "yaw": 0.0},
    }
    with open(os.path.join(SALIDA, f"pista_nivel{n}.json"), "w") as f:
        json.dump(geo, f, indent=1)
    return xml, n_trozos, t.s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inclinacion", type=float, default=math.degrees(0.8859291283123216),
                    help="grados de la camara bajo la horizontal (por defecto la del URDF de Unitree)")
    ap.add_argument("--niveles", type=int, nargs="+", default=[1, 2, 3, 4])
    a = ap.parse_args()
    os.makedirs(SALIDA, exist_ok=True)
    for n in a.niveles:
        xml, k, largo = construir(n, a.inclinacion)
        print(f"nivel {n}: {xml}  ({k} trozos de cinta, linea de {largo:.2f} m)")
    print(f"camara: {a.inclinacion:.1f} grados bajo la horizontal; IR fovy "
          f"{math.degrees(2 * math.atan(ALTO_IMG / 2 / FY_IR)):.2f}, color fovy "
          f"{math.degrees(2 * math.atan(ALTO_IMG / 2 / FY_COLOR)):.2f} (640x480)")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
