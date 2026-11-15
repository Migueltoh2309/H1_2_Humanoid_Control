#!/usr/bin/env python3
"""Manos Inspire RH56DFTP OFICIALES (las del H1-2 del laboratorio) para las
escenas de MuJoCo: genera `meshes/inspire_rh56dftp/` a partir del paquete
`h1_2_inspire_description` de h1_2_utec
(github.com/smorales2405/h1_2_utec, ros_h1_2_ws/src/h1_2_inspire_description).

    python3 scripts/build_manos_rh56.py --src RUTA/h1_2_inspire_description

Escribe:
  * las mallas de las dos manos (colisión + visuales por color) y
    `wrist_yaw_link.STL` con el anillo adaptador. MuJoCo nombra las mallas
    por su archivo y las dos palmas se llaman `base_link.STL`: aquí pasan a
    `{lado}_hand_base_link.STL`;
  * `manos_rh56.xml`: fragmento MJCF con los <mesh> y los dos subárboles
    `{lado}_hand_base_link` (montaje en `{lado}_wrist_yaw_link` del URDF:
    x = 0.161, con anillo de 10 mm y brida de 16 mm), más el ACOPLE de las
    falanges (<equality joint>). El acople sale de las curvas medidas sobre
    fotos de la mano real (`config/hand_coupling.yaml`, "software
    coupling"), no de los <mimic> lineales: cada junta acoplada es un
    polinomio de grado 4 de la actuada (lo que admite `polycoef`).

Las mallas se SIMPLIFICAN (fast_simplification, a <= --max-tri triángulos
cada una; el original tiene ~1 M en total): con las originales cada proceso
de simulación pasaba de ~0.2 a ~1.9 GB al renderizar la cámara (render por
software) y 6 procesos en paralelo agotaban la memoria. La colisión usa la
envolvente convexa de cada malla, que casi no cambia.

Las escenas no cambian: `h1_2_algoritms/aprendizaje/escena_faja.py` injerta
el fragmento al construir la escena de la faja (escena_faja.MANOS).
Convención de la mano oficial: q = 0 mano ABIERTA (ANGLE_SET 1000), el
límite superior es CERRADA (ANGLE_SET 0) en las 12 juntas.
"""
import argparse
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "meshes", "inspire_rh56dftp"))
URDF = "urdf/h1_2_with_RH56DFTP_hands_software_coupling.urdf"
SIDES = ("left", "right")


def _yaml_curves(path):
    """s y curvas de config/hand_coupling.yaml (sin depender de PyYAML)."""
    txt = open(path, encoding="utf-8").read()
    s = np.array([float(v) for v in re.search(r"^s: \[(.*)\]", txt, re.M).group(1).split(",")])
    curves = {k: np.array([float(v) for v in vals.split(",")])
              for k, vals in re.findall(r"^  (\w+): \[(.*)\]", txt, re.M)}
    return s, curves


# junta acoplada -> (junta actuada, curva acoplada, curva actuada)
_ACOPLE = {**{f"{f}_2": (f"{f}_1", "finger_2", "finger_1") for f in ("index", "middle", "ring", "little")},
           "thumb_2": ("thumb_1", "thumb_2", "thumb_1"),
           "thumb_3": ("thumb_1", "thumb_3", "thumb_1")}


def coupling_polys(yaml_path):
    """{acoplada: (actuada, [c0..c4])} con q_acoplada = sum c_k q_actuada^k
    (c0 = 0: las dos curvas parten de la mano abierta)."""
    s, C = _yaml_curves(yaml_path)
    out = {}
    for dep, (act, cd, ca) in _ACOPLE.items():
        x, y = C[ca], C[cd]
        A = np.stack([x ** k for k in range(1, 5)], axis=1)
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        err = np.max(np.abs(A @ c - y))
        out[dep] = (act, [0.0, *c], err)
    return out


def _leer_stl(path):
    """STL binario -> (vértices únicos, caras)."""
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
    rec = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
    v = rec["v"].reshape(-1, 3).astype(np.float64)
    uv, inv = np.unique(np.round(v, 7), axis=0, return_inverse=True)
    return uv, inv.reshape(-1, 3).astype(np.int64)


def _escribir_stl(path, v, f):
    tri = v[f].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(f), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as fh:
        fh.write(b"build_manos_rh56 (simplificada)".ljust(80, b" "))
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())


def simplificar(path, max_tri):
    v, f = _leer_stl(path)
    if len(f) <= max_tri:
        return len(f), len(f)
    import fast_simplification
    v2, f2 = fast_simplification.simplify(v, f, target_reduction=1.0 - max_tri / len(f))
    _escribir_stl(path, v2, f2)
    return len(f), len(f2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="carpeta h1_2_inspire_description")
    ap.add_argument("--max-tri", type=int, default=3000, help="triángulos por malla (0 = sin simplificar)")
    args = ap.parse_args()
    src = os.path.abspath(args.src)
    import mujoco

    os.makedirs(OUT_DIR, exist_ok=True)
    # ---- mallas ----
    ren = {}
    for side in SIDES:
        d_in = os.path.join(src, "meshes", f"hand_{side}")
        for f in os.listdir(d_in):
            if f.endswith(".STL"):
                dst = f"{side}_hand_base_link.STL" if f == "base_link.STL" else f
                shutil.copy(os.path.join(d_in, f), os.path.join(OUT_DIR, dst))
                ren[os.path.join(d_in, f)] = dst
        for f in os.listdir(os.path.join(d_in, "visual")):
            shutil.copy(os.path.join(d_in, "visual", f), os.path.join(OUT_DIR, f))
            ren[os.path.join(d_in, "visual", f)] = f
    shutil.copy(os.path.join(src, "meshes", "h1_2", "wrist_yaw_link.STL"),
                os.path.join(OUT_DIR, "wrist_yaw_link.STL"))
    shutil.copy(os.path.join(src, "config", "hand_coupling.yaml"), os.path.join(OUT_DIR, "hand_coupling.yaml"))
    if args.max_tri:
        antes = despues = 0
        for f in sorted(os.listdir(OUT_DIR)):
            if f.endswith(".STL"):
                a, b = simplificar(os.path.join(OUT_DIR, f), args.max_tri)
                antes, despues = antes + a, despues + b
        print(f"mallas simplificadas: {antes} -> {despues} triángulos")

    # ---- URDF -> MJCF (MuJoCo) con las mallas ya renombradas ----
    u = open(os.path.join(src, URDF), encoding="utf-8").read()
    u = u.replace("package://h1_2_inspire_description/", src + "/")
    for a, b in ren.items():
        u = u.replace(f'filename="{a}"', f'filename="{os.path.join(OUT_DIR, b)}"')
    u = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler discardvisual="false" fusestatic="false" '
               r'strippath="false"/></mujoco>', u, count=1)
    tmp = tempfile.mkdtemp()
    p_urdf, p_xml = os.path.join(tmp, "h.urdf"), os.path.join(tmp, "h.xml")
    open(p_urdf, "w", encoding="utf-8").write(u)
    mujoco.mj_saveLastXML(p_xml, mujoco.MjModel.from_xml_path(p_urdf))
    full = ET.parse(p_xml).getroot()

    frag = ET.Element("mujoco", {"model": "manos_rh56"})
    frag.append(ET.Comment(" GENERADO por h1_2_scenes/scripts/build_manos_rh56.py desde "
                           "h1_2_inspire_description (h1_2_utec). NO editar a mano. "))
    asset, used = ET.SubElement(frag, "asset"), set()
    bodies = ET.SubElement(frag, "worldbody")
    for side in SIDES:
        hb = next(b for b in full.iter("body") if b.get("name") == f"{side}_hand_base_link")
        for g in hb.iter("geom"):
            used.add(g.get("mesh"))
        bodies.append(hb)
    for msh in full.find("asset").iter("mesh"):
        if msh.get("name") in used:
            ET.SubElement(asset, "mesh", {"name": f"rh56_{msh.get('name')}",
                                          "file": os.path.basename(msh.get("file"))})
    for side in SIDES:
        for g in next(b for b in bodies if b.get("name") == f"{side}_hand_base_link").iter("geom"):
            g.set("mesh", f"rh56_{g.get('mesh')}")
    ET.SubElement(asset, "mesh", {"name": "rh56_wrist_yaw_link", "file": "wrist_yaw_link.STL"})

    eq = ET.SubElement(frag, "equality")
    polys = coupling_polys(os.path.join(OUT_DIR, "hand_coupling.yaml"))
    for side in SIDES:
        for dep, (act, c, err) in polys.items():
            ET.SubElement(eq, "joint", {"joint1": f"{side}_{dep}_joint", "joint2": f"{side}_{act}_joint",
                                        "polycoef": " ".join(f"{v:.6g}" for v in c)})
    ET.indent(frag, space="  ")
    out = os.path.join(OUT_DIR, "manos_rh56.xml")
    ET.ElementTree(frag).write(out, encoding="UTF-8", xml_declaration=True)
    shutil.rmtree(tmp)
    print("OK", out)
    for dep, (act, c, err) in polys.items():
        print(f"  {dep:9s} <- {act:8s} error máx del polinomio {err * 1000:.1f} mrad")


if __name__ == "__main__":
    main()
