#!/usr/bin/env python3
"""Genera mjcf/h1_2_scene_surgery_table_hands.xml a partir de
mjcf/h1_2_scene_surgery_table.xml (la escena de la faja, base fija) y de
h1_2_description/mjcf/h1_2_scene.xml (el modelo completo, que trae las manos
Inspire).

Uso (desde h1_2_scenes/):
    python3 scripts/build_scene_hands.py [--description-dir RUTA]

--description-dir es la carpeta de h1_2_description (con mjcf/ y meshes/).
Por defecto se busca en ../h1_2_utec/h1_2_description y, si no está, en el
share instalado (con el workspace cargado).

Se genera con un script, en vez de editar a mano, para que la escena de la
faja siga siendo la única fuente de verdad de mesa/faja/cámara/postura: si
se corrige algo allá, basta con volver a correr esto.

Qué agrega respecto a la escena de la faja:

1. **Dedos de las manos Inspire** (12 joints por mano, copiados de
   h1_2_scene.xml). La mano real tiene 6 actuadores (pulgar yaw, pulgar
   pitch, índice, medio, anular, meñique); las falanges intermedias/distales
   van acopladas por mecanismo. Ese acople sale de las etiquetas <mimic> del
   URDF (h1_2.urdf) y aquí se modela con <equality joint> lineales:
       thumb_intermediate = 1.6 * thumb_proximal_pitch
       thumb_distal       = 2.4 * thumb_proximal_pitch
       X_intermediate     = 1.0 * X_proximal   (índice, medio, anular, meñique)
   Actuadores de posición (servo interno), como el control de posición de
   la mano real; `forcerange` acotado al `actuatorfrcrange` del modelo.
   Los dedos NO chocan entre sí (contype/conaffinity) pero sí con la fruta,
   la faja y la mesa.

2. **Mandarina libre (freejoint)**, no un slide: para poder levantarla. La
   faja la arrastra un "carro" plano de 1 mm (slide en Y + actuador de
   velocidad, igual que el motor virtual de la escena original) sobre el
   que la fruta se apoya y del que se levanta al agarrarla. Visualmente es
   del color de la faja: en RGB y profundidad es indistinguible de ella
   (1 mm).

3. **Par estéreo IR de la RealSense D435i** (`robot_ir_left`,
   `robot_ir_right`): misma orientación que `robot_rgbd_camera`, línea de
   base de 50 mm, el izquierdo 15 mm hacia -x_cámara respecto al RGB (en la
   D435 el origen de profundidad es el imager IR izquierdo y el RGB está a
   ~15 mm de él). 848x480, fovy 58° (FOV IR de la D435: 87°x58°).

Uso:
    python3 scripts/build_scene_hands.py
"""
import copy
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
MJCF = os.path.join(PKG, "mjcf")
SCENE_MESHES = os.path.join(PKG, "meshes")
SRC_SCENE = os.path.join(MJCF, "h1_2_scene_surgery_table.xml")
OUT = os.path.join(MJCF, "h1_2_scene_surgery_table_hands.xml")


def default_description_dir():
    """h1_2_description al lado de este paquete en el workspace o, si no, su
    share instalado (necesita el workspace con `source install/setup.bash`)."""
    candidates = [os.path.join(PKG, "..", "h1_2_utec", "h1_2_description")]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(get_package_share_directory("h1_2_description"))
    except Exception:
        pass
    for c in candidates:
        if os.path.isfile(os.path.join(c, "mjcf", "h1_2_scene.xml")):
            return os.path.normpath(c)
    return None


# Carpeta de h1_2_description (con mjcf/ y meshes/); se fija en main()
DESCRIPTION_DIR = None


def load_model(path):
    """Compila una escena de este paquete con las mismas rutas absolutas que
    genera el CMakeLists al instalar (mallas del robot en h1_2_description,
    faja y mesa en h1_2_scenes/meshes)."""
    import mujoco
    xml = open(path, encoding="utf-8").read()
    xml = xml.replace('meshdir="../meshes/"',
                      'meshdir="%s/"' % os.path.join(DESCRIPTION_DIR, "meshes"))
    for mesh in os.listdir(SCENE_MESHES):
        xml = xml.replace('file="%s"' % mesh, 'file="%s"' % os.path.join(SCENE_MESHES, mesh))
    return mujoco.MjModel.from_xml_string(xml)


FINGERS = ["index", "middle", "ring", "pinky"]
# (joint actuado, kp, forcerange [N·m]). Con el actuatorfrcrange del modelo
# (1 N·m, ~17-20 N en la punta con ~5-6 cm de palanca) el cierre contra una
# esfera RÍGIDA la lanzaba (medido). Se acota a ~6 N por dedo y ~12 N el
# pulgar: por debajo de la fuerza máxima de la hoja de datos (RH56DFX: 10 N
# en la punta de cada dedo, 15 N el pulgar; la RH56DFTP del H1-2 declara
# hasta 30 N). La mano real permite fijar umbrales de fuerza, así que un
# cierre suave es una configuración realista, no una limitación del modelo.
HAND_ACTUATORS = [("thumb_proximal_yaw_joint", 2.0, 0.6), ("thumb_proximal_pitch_joint", 2.0, 0.6)] + \
                 [(f"{f}_proximal_joint", 2.0, 0.35) for f in FINGERS]
COUPLINGS = [("thumb_intermediate_joint", "thumb_proximal_pitch_joint", 1.6),
             ("thumb_distal_joint", "thumb_proximal_pitch_joint", 2.4)] + \
            [(f"{f}_intermediate_joint", f"{f}_proximal_joint", 1.0) for f in FINGERS]

# Cámara RGB del MJCF: pos en torso_link y eje x de la imagen.
RGB_POS = (0.11109, 0.01750, 0.68789)
RGB_XYAXES = "0 -1 0 0.774503060 0 0.632570162"
CAM_X = (0.0, -1.0, 0.0)          # derecha de la imagen, en torso_link
IR_LEFT_OFFSET = -0.015           # [m] sobre CAM_X, respecto al RGB
IR_BASELINE = 0.050               # [m]

# Mandarina / carro, en mundo (misma posición que la escena original).
FRUIT_POS = (0.34, -0.03, 0.8655)   # 0.824 superficie + 0.0015 carro + 0.04 radio
CARRIER_POS = (0.34, -0.03, 0.8248)
CARRIER_HALF = (0.07, 0.07, 0.0008)
BELT_Y0 = -0.50          # y inicial de la fruta en los keyframes de faja
BELT_SPEED = 0.05        # m/s, valor de referencia del checkpoint S6


def parse(path):
    return ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


def main():
    global DESCRIPTION_DIR
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--description-dir", default=None,
                    help="carpeta de h1_2_description con mjcf/ y meshes/ (def.: se busca)")
    DESCRIPTION_DIR = ap.parse_args().description_dir or default_description_dir()
    if DESCRIPTION_DIR is None:
        raise SystemExit("No encuentro h1_2_description/mjcf/h1_2_scene.xml; indica --description-dir")
    DESCRIPTION_DIR = os.path.abspath(DESCRIPTION_DIR)
    SRC_HANDS = os.path.join(DESCRIPTION_DIR, "mjcf", "h1_2_scene.xml")
    scene = parse(SRC_SCENE)
    hands = parse(SRC_HANDS)
    sroot, hroot = scene.getroot(), hands.getroot()

    # ---- assets: mallas de los dedos que falten ----
    s_asset = sroot.find("asset")
    have = {m.get("name") for m in sroot.iter("mesh")}
    for m in hroot.find("asset").iter("mesh"):
        if m.get("name") not in have and m.get("name", "").startswith("link"):
            s_asset.append(copy.deepcopy(m))

    # ---- dedos ----
    for side, pre in (("left", "L"), ("right", "R")):
        dst = find_body(sroot, f"{side}_wrist_yaw_link")
        src = find_body(hroot, f"{side}_wrist_yaw_link")
        for child in src.findall("body"):
            fb = copy.deepcopy(child)
            for j in fb.iter("joint"):
                # armature/damping: sin esto las falanges (gramos) hacen
                # chattering con dt = 2 ms, igual que las muñecas.
                j.set("armature", "0.0005")
                j.set("damping", "0.02")
            for g in fb.iter("geom"):
                if g.get("contype") == "0":
                    continue          # geom visual
                g.set("contype", "2")
                g.set("conaffinity", "1")
                g.set("friction", "1.5 0.01 0.001")
                g.set("condim", "4")
            dst.append(fb)
        # Sitio en el centro de la palma: 'grasp_site' de la escena original
        # queda en la punta; este es el punto que debe coincidir con el
        # centro de la fruta en un agarre de potencia.
        site = ET.SubElement(dst, "site")
        site.set("name", f"{side}_palm_site")
        site.set("pos", "0.14 0 0")
        site.set("size", "0.01")
        site.set("rgba", "1 1 0 0.6")
        site.set("group", "4")

    # ---- cámaras IR ----
    # El framebuffer offscreen por defecto es 640x480: las IR (848) no caben.
    vis = sroot.find("visual")
    glob = vis.find("global")
    glob.set("offwidth", "1280")
    glob.set("offheight", "800")
    torso = find_body(sroot, "torso_link")
    rgb = next(c for c in torso.iter("camera") if c.get("name") == "robot_rgbd_camera")
    idx = list(torso).index(rgb)
    for k, (name, off) in enumerate((("robot_ir_left", IR_LEFT_OFFSET),
                                     ("robot_ir_right", IR_LEFT_OFFSET + IR_BASELINE))):
        pos = [RGB_POS[i] + off * CAM_X[i] for i in range(3)]
        cam = ET.Element("camera", {
            "name": name, "mode": "fixed", "pos": " ".join(f"{v:.5f}" for v in pos),
            "xyaxes": RGB_XYAXES, "fovy": "58", "resolution": "848 480"})
        torso.insert(idx + 1 + k, cam)

    # ---- mandarina libre + carro de la faja ----
    # La escena tiene dos <worldbody> (robot / entorno): se usa el que
    # contiene la mandarina.
    world = next(w for w in sroot.findall("worldbody")
                 if any(b.get("name") == "mandarina" for b in w.findall("body")))
    world.remove(find_body(world, "mandarina"))
    carrier = ET.SubElement(world, "body", {"name": "belt_carrier",
                                            "pos": " ".join(map(str, CARRIER_POS))})
    ET.SubElement(carrier, "joint", {"name": "belt_carrier_joint", "type": "slide",
                                     "axis": "0 1 0", "range": "-0.52 0.53", "damping": "0.05"})
    ET.SubElement(carrier, "geom", {"name": "belt_carrier_geom", "type": "box",
                                    "size": " ".join(map(str, CARRIER_HALF)), "mass": "0.2",
                                    "rgba": "0.95 0.95 0.93 1", "friction": "1.2 0.01 0.001",
                                    "contype": "4", "conaffinity": "4"})
    # Colisión de la faja: MuJoCo choca contra la envolvente CONVEXA de la
    # malla, que por el bloque motor de una punta queda ~2.5 cm por encima
    # de la superficie plana real (medido: la fruta en reposo "penetraba"
    # 25 mm y salía expulsada; los dedos chocaban con aire). Se deja la
    # malla solo como visual y se agrega una caja invisible (grupo 3, no se
    # renderiza en RGB ni en profundidad) con la cara superior en la
    # superficie real, medida con mj_ray: z = 0.824-0.825 en
    # x ∈ [0.28, 0.49], |y| < 0.55.
    for g in world.iter("geom"):
        if g.get("name") == "conveyor_belt_geom":
            g.set("contype", "0")
            g.set("conaffinity", "0")
    ET.SubElement(world, "geom", {"name": "conveyor_belt_surface", "type": "box",
                                  "pos": "0.385 0 0.8145", "size": "0.11 0.58 0.01",
                                  "group": "3", "rgba": "0 1 0 0.3",
                                  "friction": "1.0 0.005 0.0001"})
    fruit = ET.SubElement(world, "body", {"name": "mandarina",
                                          "pos": " ".join(map(str, FRUIT_POS))})
    ET.SubElement(fruit, "freejoint", {"name": "mandarina_free"})
    ET.SubElement(fruit, "site", {"name": "mandarina_target", "pos": "0 0 0",
                                  "size": "0.01", "rgba": "1 0 0 1", "group": "4"})
    # condim 6 + fricción de rodadura: sin ella una esfera sobre un plano
    # que acelera rueda en vez de ser arrastrada.
    ET.SubElement(fruit, "geom", {"name": "mandarina_geom", "type": "sphere", "size": "0.04",
                                  "mass": "0.10", "rgba": "1.0 0.45 0.05 1",
                                  "friction": "1.2 0.02 0.002", "condim": "6",
                                  "contype": "5", "conaffinity": "5"})

    # ---- actuadores ----
    for act in sroot.findall("actuator"):
        for v in list(act):
            if v.get("name") == "conveyor_motor":
                act.remove(v)
    act = sroot.findall("actuator")[-1]
    ET.SubElement(act, "velocity", {"name": "conveyor_motor", "joint": "belt_carrier_joint",
                                    "kv": "20", "ctrlrange": "-0.15 0.15", "forcerange": "-5 5"})
    for side, pre in (("left", "L"), ("right", "R")):
        for jn, kp, fr in HAND_ACTUATORS:
            j = f"{pre}_{jn}"
            # Rango de control = rango del joint en el modelo.
            ctrlrange = ("-0.1 1.3" if "yaw" in jn else
                         "-0.1 0.6" if "thumb_proximal_pitch" in jn else "-0.1 1.7")
            ET.SubElement(act, "position", {"name": j, "joint": j, "kp": str(kp),
                                            "forcerange": f"-{fr} {fr}", "ctrllimited": "true",
                                            "ctrlrange": ctrlrange})

    # ---- acoples (mimic del URDF) ----
    eq = sroot.find("equality")
    if eq is None:
        eq = ET.SubElement(sroot, "equality")
    for pre in ("L", "R"):
        for j1, j2, k in COUPLINGS:
            ET.SubElement(eq, "joint", {"joint1": f"{pre}_{j1}", "joint2": f"{pre}_{j2}",
                                        "polycoef": f"0 {k} 0 0 0", "solref": "0.005 1"})

    # ---- asistencia de agarre (desactivada por defecto) ----
    # Weld mano-fruta que el controlador ACTIVA solo cuando la mano ya
    # cerró con contacto real de pulgar + dedos (ver
    # demos/visual_servoing_grasp.py): emula la fricción/deformación de
    # una fruta real que un contacto rígido esfera-malla no captura. Los
    # resultados se reportan con y sin asistencia.
    for side in ("left", "right"):
        ET.SubElement(eq, "weld", {"name": f"{side}_grasp_assist", "body1": f"{side}_wrist_yaw_link",
                                   "body2": "mandarina", "active": "false", "solref": "0.02 1"})

    # ---- contacto: la base de la mano no choca con sus falanges ----
    contact = sroot.find("contact")
    for pre, side in (("L", "left"), ("R", "right")):
        for b in ("thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal",
                  *[f"{f}_proximal" for f in FINGERS]):
            ET.SubElement(contact, "exclude", {"body1": f"{side}_wrist_yaw_link", "body2": f"{pre}_{b}"})

    # ---- keyframes: el layout de qpos cambió (dedos + freejoint) ----
    for kf in sroot.findall("keyframe"):
        sroot.remove(kf)

    root_comment = ET.Comment(
        " GENERADO por scripts/build_scene_hands.py a partir de h1_2_scene_surgery_table.xml "
        "+ manos de h1_2_scene.xml. NO editar a mano: editar la fuente y regenerar. ")
    sroot.insert(0, root_comment)
    ET.indent(scene, space="  ")
    scene.write(OUT, encoding="UTF-8", xml_declaration=True)

    # Validación: que compile y que los acoples/actuadores existan.
    m = load_model(OUT)

    # ---- keyframes de la faja en movimiento (el layout de qpos se conoce
    # recién compilado: dedos + carro + freejoint) ----
    # "mandarina_moving": la fruta arranca en y = BELT_Y0 y la faja la lleva
    # a BELT_SPEED hacia +y (mismo nombre que en la escena sin manos, así
    # sim_bridge_moving.launch.py / initial_keyframe sirven igual).
    # "mandarina_moving_fast": lo mismo a 0.07 m/s (tope del manuscrito).
    kf_root = ET.SubElement(sroot, "keyframe")
    for name, v in (("mandarina_moving", BELT_SPEED), ("mandarina_moving_fast", 0.07)):
        qpos = m.qpos0.copy()
        qvel = np.zeros(m.nv)
        ctrl = np.zeros(m.nu)
        jc = m.jnt_qposadr[m.joint("belt_carrier_joint").id]
        jf = m.jnt_qposadr[m.joint("mandarina_free").id]
        vc = m.jnt_dofadr[m.joint("belt_carrier_joint").id]
        vf = m.jnt_dofadr[m.joint("mandarina_free").id]
        qpos[jc] = BELT_Y0 - CARRIER_POS[1]
        qpos[jf:jf + 3] = (FRUIT_POS[0], BELT_Y0, FRUIT_POS[2])
        qvel[vc] = v
        qvel[vf + 1] = v
        ctrl[m.actuator("conveyor_motor").id] = v
        ET.SubElement(kf_root, "key", {
            "name": name,
            "qpos": " ".join(f"{x:.6g}" for x in qpos),
            "qvel": " ".join(f"{x:.6g}" for x in qvel),
            "ctrl": " ".join(f"{x:.6g}" for x in ctrl)})
    ET.indent(scene, space="  ")
    scene.write(OUT, encoding="UTF-8", xml_declaration=True)
    m = load_model(OUT)
    print(f"OK {OUT}\n  nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq} ncam={m.ncam} nkey={m.nkey}")


if __name__ == "__main__":
    main()
