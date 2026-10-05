#!/usr/bin/env python3
"""Genera mjcf/h1_2_escena_suelo.xml: el H1-2 SOLO, de pie en el suelo, con la
fisica completa (base flotante, gravedad, contactos pies-suelo y entre eslabones,
piernas activas), manos Inspire y la D435 (RGB, depth, 2 IR).

Fuente: mjcf/h1_2_escena_faja_manos.xml (copia de la escena con manos de
h1_2_scenes), que ya trae lo bueno del robot: dedos acoplados como la mano real
(equality), actuadores de posicion de las manos, camaras de la D435 y la pelvis
calibrada a la suela. Se genera con un script, en vez de editar a mano, para que
esa escena siga siendo la unica fuente de verdad del robot.

Cambios respecto a la fuente:
  * pelvis con <freejoint name="floating_base_joint"> (base flotante, como el
    h1_2.xml de Unitree): el robot se sostiene sobre sus pies o se cae;
  * fuera mesa, faja, carro, mandarina, el actuador de la faja y los weld de la
    asistencia de agarre; quedan suelo y luces;
  * IMU completa en el sitio `imu` de torso_link (donde la pone Unitree):
    framequat `imu_quat` + el giroscopo y acelerometro que ya habia;
  * sensores de verdad de terreno de la pelvis (`pelvis_pos`, `pelvis_linvel`);
  * pies: la colision pasa de la malla (casco convexo) a una CAJA con las medidas
    de la suela sacadas de los vertices de la malla. Con el casco, los puntos de
    contacto con el suelo saltaban entre 4 y 5 y cada salto metia un impulso: la IMU
    veia picos de +-3 m/s2 (a ~23 Hz) con el robot quieto. Con la caja: 4 esquinas
    estables. La malla sigue como visual;
  * keyframe `de_pie`: rodillas algo flexionadas (cadera -0.16, rodilla 0.36,
    tobillo -0.20: suma 0, el pie queda plano con el tronco vertical) y la altura
    de la pelvis CALCULADA con los vertices de la malla de los pies para que la
    suela quede exactamente en z = 0 (no a ojo).

Uso:  python3 scripts/construir_escena_suelo.py      (desde la raiz de h1_2_sim2real)
"""
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
MJCF = os.path.normpath(os.path.join(AQUI, "..", "src", "h1_2_sim2real", "mjcf"))
FUENTE = os.path.join(MJCF, "h1_2_escena_faja_manos.xml")
SALIDA = os.path.join(MJCF, "h1_2_escena_suelo.xml")

FUERA_CUERPOS = {"surgery_table", "conveyor_belt", "belt_carrier", "mandarina"}
FUERA_GEOMS = {"conveyor_belt_surface"}
FUERA_MALLAS = {"surgery_table", "conveyor_belt"}
FUERA_ACTUADORES = {"conveyor_motor"}

# Postura de pie (rad). Brazos colgando, codos un poco flexionados para que las
# manos no rocen los muslos.
POSTURA = {
    "left_hip_pitch_joint": -0.16, "left_knee_joint": 0.36, "left_ankle_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.16, "right_knee_joint": 0.36, "right_ankle_pitch_joint": -0.20,
    "left_shoulder_roll_joint": 0.12, "right_shoulder_roll_joint": -0.12,
    "left_elbow_joint": 0.30, "right_elbow_joint": 0.30,
}
PIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def quitar(padre, pred):
    for c in list(padre):
        if pred(c):
            padre.remove(c)


def suela_en_cuerpo(fuente_xml):
    """{cuerpo_pie: (centro, medias)} de la caja que envuelve la suela, en el marco
    del cuerpo, a partir de los vertices de la malla de colision."""
    m = mujoco.MjModel.from_xml_path(fuente_xml)
    out = {}
    for pie in PIES:
        b = m.body(pie).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] == 0:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            vb = m.geom_pos[g] + v @ R.reshape(3, 3).T
            zmin = vb[:, 2].min()
            suela = vb[vb[:, 2] < zmin + 0.01]          # el centimetro de abajo: la planta
            lo, hi = suela.min(0), suela.max(0)
            alto = 0.02
            centro = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, zmin + alto / 2])
            out[pie] = (centro, np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, alto / 2]))
    return out


def construir_xml():
    arbol = ET.parse(FUENTE)
    r = arbol.getroot()
    for wb in r.findall("worldbody"):
        quitar(wb, lambda c: (c.tag == "body" and c.get("name") in FUERA_CUERPOS)
               or (c.tag == "geom" and c.get("name") in FUERA_GEOMS))
    for a in r.findall("asset"):
        quitar(a, lambda c: c.tag == "mesh" and c.get("name") in FUERA_MALLAS)
    for a in r.findall("actuator"):
        quitar(a, lambda c: c.get("name") in FUERA_ACTUADORES)
    for e in r.findall("equality"):
        quitar(e, lambda c: c.tag == "weld")
    for k in r.findall("keyframe"):
        r.remove(k)

    for pie, (c, h) in suela_en_cuerpo(FUENTE).items():
        cuerpo = r.find(f".//body[@name='{pie}']")
        for geom in cuerpo.findall("geom"):
            if geom.get("type") == "mesh" and geom.get("contype") != "0":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "1")
                geom.set("density", "0")
        cuerpo.append(ET.Element("geom", {
            "name": pie.replace("_ankle_roll_link", "_suela"), "type": "box",
            "pos": " ".join(f"{x:.5f}" for x in c), "size": " ".join(f"{x:.5f}" for x in h),
            "rgba": "0.2 0.2 0.2 0.0", "group": "3", "friction": "1.0 0.02 0.01"}))

    pelvis = r.find("worldbody").find("body[@name='pelvis']")
    libre = ET.Element("freejoint", {"name": "floating_base_joint"})
    pelvis.insert(1, libre)                     # despues del <inertial>

    sensor = r.find("sensor")
    sensor.insert(0, ET.Element("framequat", {"name": "imu_quat", "objtype": "site", "objname": "imu"}))
    sensor.append(ET.Element("framepos", {"name": "pelvis_pos", "objtype": "body", "objname": "pelvis"}))
    sensor.append(ET.Element("framelinvel", {"name": "pelvis_linvel", "objtype": "body", "objname": "pelvis"}))

    # los comentarios del XML fuente se pierden con ElementTree: cabecera nueva
    r.insert(0, ET.Comment(" GENERADO por scripts/construir_escena_suelo.py a partir de "
                           "h1_2_escena_faja_manos.xml. NO editar a mano: editar el script y regenerar. "))
    r.set("model", "h1_2_suelo")
    return arbol


def altura_pelvis(xml):
    """z de la pelvis que deja el punto mas bajo de las suelas en z = 0 con la POSTURA."""
    tmp = SALIDA + ".tmp.xml"                   # junto a la fuente: el meshdir es relativo
    with open(tmp, "w") as f:
        f.write(xml)
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
    d = mujoco.MjData(m)
    for j, q in POSTURA.items():
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = q
    mujoco.mj_kinematics(m, d)
    z_min = np.inf
    for g in range(m.ngeom):
        cuerpo = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        if cuerpo not in PIES or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        esquinas = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[g]
        z = (d.geom_xpos[g] + esquinas @ d.geom_xmat[g].reshape(3, 3).T)[:, 2]
        z_min = min(z_min, float(z.min()))
    return float(d.qpos[2]) - z_min, m


def main():
    arbol = construir_xml()
    xml = ET.tostring(arbol.getroot(), encoding="unicode")
    z, m = altura_pelvis(xml)
    qpos = np.zeros(m.nq)
    qpos[0:3] = [0.0, 0.0, z]
    qpos[3] = 1.0
    for j, q in POSTURA.items():
        qpos[m.jnt_qposadr[m.joint(j).id]] = q
    ctrl = np.zeros(m.nu)                       # motores a 0; manos abiertas (posicion 0)
    kf = ET.SubElement(arbol.getroot(), "keyframe")
    ET.SubElement(kf, "key", {"name": "de_pie", "qpos": " ".join(f"{x:.6g}" for x in qpos),
                              "ctrl": " ".join(f"{x:.6g}" for x in ctrl)})
    arbol.getroot().find("worldbody").find("body[@name='pelvis']").set("pos", f"0 0 {z:.4f}")
    ET.indent(arbol, space="  ")
    arbol.write(SALIDA, encoding="unicode", xml_declaration=True)
    print(f"{SALIDA}\n  pelvis z = {z:.4f} m (suela a z=0 en la postura de pie) · nq={m.nq} nv={m.nv} nu={m.nu}")


if __name__ == "__main__":
    main()
