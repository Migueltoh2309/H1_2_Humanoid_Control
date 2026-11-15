"""Escena de la faja con VARIAS frutas, para la política de alto nivel
(fase B del LfD, docs/LFD_BC.md).

Se genera en tiempo de carga a partir de `h1_2_scene_surgery_table_hands.xml`
(la escena de una fruta, que a su vez genera build_scene_hands.py), sin
tocar ese generador:

  * se quita la mandarina única (y sus welds de asistencia) y se agregan
    `n` frutas iguales `fruta_i`, estacionadas fuera de la vista hasta que
    "llegan" a la faja;
  * un weld de asistencia por brazo y por fruta (`{side}_assist_i`), con la
    misma regla que en demos/visual_servoing_grasp.py (`_try_weld`);
  * la faja es CINEMÁTICA: cada paso, una fruta apoyada en la faja (y no
    sujeta) recibe la velocidad de la faja. El carro de la escena original
    lleva una sola fruta y tiene ±0.5 m de recorrido; con varias frutas en
    fila no sirve.

El depósito es una abstracción: la habilidad lleva la mano a la pose de
depósito al costado y abre; al soltar, la fruta se cuenta como depositada
y se estaciona (no se modela el canasto).
"""
import copy
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from h1_2_algoritms.vision import grasp_geometry as GG

BELT_TOP = 0.9645            # z de la superficie de la faja (mundo), medido con mj_ray (mesa u_table; con la vieja: 0.8245)
FRUIT_R = 0.04
BELT_X = (0.28, 0.49)
BELT_Y = (-0.55, 0.55)
# Desplazamiento de la faja hacia el robot (m, en -x). 0 = la escena tal
# cual. Lo fija el script antes de build_xml (estudio de montaje: con la
# faja donde está, el lado lejano queda fuera del alcance de los brazos).
BELT_DX = GG.FAJA_DX
# la mesa va con la faja (la faja está apoyada en ella)
BELT_BODIES = ("conveyor_belt", "belt_carrier", "conveyor_belt_surface", "u_table")
SPAWN_Y = -0.52              # la faja va hacia +y: entra por la derecha del robot
EXIT_Y = 0.52


def build_xml(src, n_fruits):
    """Ruta a un MJCF temporal con `n_fruits` frutas."""
    tree = ET.parse(src)
    root = tree.getroot()
    comp = root.find("compiler")
    if not os.path.isabs(comp.get("meshdir", "")):
        # El MJCF temporal vive en otro directorio: rutas absolutas.
        meshdir = os.path.normpath(os.path.join(os.path.dirname(src), comp.get("meshdir", "")))
        comp.set("meshdir", meshdir + "/")
    for el in root.iter():
        if el.tag in ("body", "geom") and el.get("name") in BELT_BODIES and BELT_DX:
            p = [float(c) for c in el.get("pos").split()]
            p[0] -= BELT_DX
            el.set("pos", " ".join(map(str, p)))
    for kf in root.findall("keyframe"):
        root.remove(kf)
    if GG.MANOS == "rh56":
        manos_rh56(root, src)
    world = next(w for w in root.findall("worldbody")
                 if any(b.get("name") == "mandarina" for b in w.findall("body")))
    world.remove(next(b for b in world.findall("body") if b.get("name") == "mandarina"))
    eq = root.find("equality")
    for w in list(eq):
        if w.get("body2") == "mandarina":
            eq.remove(w)
    for i in range(n_fruits):
        b = ET.SubElement(world, "body", {"name": f"fruta_{i}", "pos": " ".join(map(str, park_pos(i)))})
        ET.SubElement(b, "freejoint", {"name": f"fruta_{i}_free"})
        ET.SubElement(b, "geom", {"name": f"fruta_{i}_geom", "type": "sphere", "size": str(FRUIT_R),
                                  "mass": "0.10", "rgba": "1.0 0.45 0.05 1",
                                  "friction": "1.2 0.02 0.002", "condim": "6",
                                  "contype": "5", "conaffinity": "5"})
        for side in ("left", "right"):
            ET.SubElement(eq, "weld", {"name": f"{side}_assist_{i}", "body1": f"{side}_wrist_yaw_link",
                                       "body2": f"fruta_{i}", "active": "false", "solref": "0.02 1"})
    fd, path = tempfile.mkstemp(prefix="h1_2_faja_multi_", suffix=".xml")
    os.close(fd)
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return path


# Manos RH56DFTP oficiales (GG.MANOS == "rh56"): fragmento generado por
# h1_2_scenes/scripts/build_manos_rh56.py. Actuadores de posición como los de
# la mano vieja: kp 2 y fuerza acotada a ~6 N por dedo / ~12 N el pulgar
# (build_scene_hands.HAND_ACTUATORS: con el actuatorfrcrange de 1 N·m el
# cierre contra una esfera rígida la lanzaba).
RH56_ACT = [("thumb_swing_joint", 0.6), ("thumb_1_joint", 0.6)] + \
           [(f"{f}_1_joint", 0.35) for f in ("index", "middle", "ring", "little")]


def manos_rh56(root, src):
    """Cambia en el árbol MJCF `root` (la escena de la faja con las manos de
    h1_2_description) las manos por las RH56DFTP oficiales."""
    d = os.path.normpath(os.path.join(os.path.dirname(src), "..", "meshes", "inspire_rh56dftp"))
    frag = ET.parse(os.path.join(d, "manos_rh56.xml")).getroot()
    asset = root.find("asset")
    for msh in frag.find("asset"):
        msh = copy.deepcopy(msh)
        msh.set("file", os.path.join(d, msh.get("file")))
        asset.append(msh)
    vieja = ("L_", "R_")
    for side in ("left", "right"):
        w = next(b for b in root.iter("body") if b.get("name") == f"{side}_wrist_yaw_link")
        for b in [b for b in w.findall("body") if b.get("name", "").startswith(vieja)]:
            w.remove(b)
        for g in [g for g in w.findall("geom") if (g.get("mesh") or "").endswith("hand_base_link")]:
            w.remove(g)
        for g in w.findall("geom"):            # muñeca con el anillo adaptador de 10 mm
            if g.get("mesh") == "wrist_yaw_link":
                g.set("mesh", "rh56_wrist_yaw_link")
        hb = copy.deepcopy(next(b for b in frag.find("worldbody") if b.get("name") == f"{side}_hand_base_link"))
        for j in hb.iter("joint"):
            j.set("armature", "0.0005")        # falanges de gramos con dt = 2 ms (como la mano vieja)
            j.set("damping", "0.02")
        for g in hb.iter("geom"):
            if g.get("contype") == "0":
                continue                       # visual (por color)
            g.attrib.pop("rgba", None)
            g.set("group", "3")                # colisión: no se dibuja (la visual va encima)
            g.set("contype", "2")
            g.set("conaffinity", "1")
            g.set("friction", "1.5 0.01 0.001")
            g.set("condim", "4")
        w.append(hb)
    for act in root.findall("actuator"):
        for a in [a for a in act if (a.get("joint") or "").startswith(vieja)]:
            act.remove(a)
    act = root.findall("actuator")[-1]
    for side in ("left", "right"):
        for jn, fr in RH56_ACT:
            j = next(x for x in frag.iter("joint") if x.get("name") == f"{side}_{jn}")
            ET.SubElement(act, "position", {"name": f"{side}_{jn}", "joint": f"{side}_{jn}", "kp": "2.0",
                                            "forcerange": f"-{fr} {fr}", "ctrllimited": "true",
                                            "ctrlrange": j.get("range")})
    eq = root.find("equality")
    for e in [e for e in eq.findall("joint") if (e.get("joint1") or "").startswith(vieja)]:
        eq.remove(e)
    for e in frag.find("equality"):
        e = copy.deepcopy(e)
        e.set("solref", "0.005 1")
        eq.append(e)
    contact = root.find("contact")
    for e in [e for e in contact.findall("exclude") if (e.get("body2") or "").startswith(vieja)]:
        contact.remove(e)
    for side in ("left", "right"):
        hb = next(b for b in root.iter("body") if b.get("name") == f"{side}_hand_base_link")
        for b in hb.iter("body"):
            if b is not hb:
                ET.SubElement(contact, "exclude", {"body1": f"{side}_hand_base_link", "body2": b.get("name")})
                ET.SubElement(contact, "exclude", {"body1": f"{side}_wrist_yaw_link", "body2": b.get("name")})


def park_pos(i):
    """Estacionamiento: detrás del robot, en el piso, fuera de la cámara."""
    return [-1.0 - 0.12 * (i % 10), 1.5 + 0.12 * (i // 10), FRUIT_R]


class FajaMulti:
    """Estado de las frutas sobre la faja (verdad de terreno) y la faja
    cinemática. Se usa con un MujocoSim ya construido sobre build_xml()."""

    def __init__(self, m, d, n_fruits, v_belt):
        self.m, self.d, self.n, self.v = m, d, n_fruits, float(v_belt)
        self.qadr = [m.jnt_qposadr[m.joint(f"fruta_{i}_free").id] for i in range(n_fruits)]
        self.vadr = [m.jnt_dofadr[m.joint(f"fruta_{i}_free").id] for i in range(n_fruits)]
        self.bid = [m.body(f"fruta_{i}").id for i in range(n_fruits)]
        self.eid = {(s, i): m.equality(f"{s}_assist_{i}").id for s in ("left", "right") for i in range(n_fruits)}
        self.wrist = {s: m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")}
        # "Sensor" de prensión: cuerpos de cada mano (dedos + palma) y del robot.
        root = m.body("pelvis").id
        self.robot_bodies = {b for b in range(m.nbody) if m.body_rootid[b] == root}
        # (la muñeca y todo lo que cuelga de ella: sirve con las dos manos)
        self.hand_bodies = {s: {b for b in self.robot_bodies if self._under(b, self.wrist[s])}
                            for s in ("left", "right")}
        self._wrench = np.zeros(6)
        # estado: "parked" | "belt" | "held:<side>" | "deposited" | "lost" | "dropped"
        self.state = ["parked"] * n_fruits
        self.t_spawn = [None] * n_fruits
        self.held = {"left": None, "right": None}

    def _under(self, b, top):
        while b > 0:
            if b == top:
                return True
            b = self.m.body_parentid[b]
        return False

    # ------------------------------------------------------------ frutas
    def spawn(self, i, x, t):
        a = self.qadr[i]
        self.d.qpos[a:a + 3] = [x, SPAWN_Y, BELT_TOP + FRUIT_R + 0.001]
        self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.d.qvel[self.vadr[i]:self.vadr[i] + 6] = [0, self.v, 0, 0, 0, 0]
        self.state[i] = "belt"
        self.t_spawn[i] = t

    def park(self, i, state):
        for s in ("left", "right"):
            self.d.eq_active[self.eid[(s, i)]] = 0
            if self.held[s] == i:
                self.held[s] = None
        a = self.qadr[i]
        self.d.qpos[a:a + 3] = park_pos(i)
        self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.d.qvel[self.vadr[i]:self.vadr[i] + 6] = 0
        self.state[i] = state

    def pos_torso(self, i):
        return self.d.xpos[self.bid[i]] - [0, 0, GG.TORSO_Z]

    def on_belt(self, i):
        p = self.d.xpos[self.bid[i]]
        return (BELT_X[0] - BELT_DX - 0.02 <= p[0] <= BELT_X[1] - BELT_DX + 0.02 and BELT_Y[0] <= p[1] <= BELT_Y[1]
                and p[2] < BELT_TOP + FRUIT_R + 0.01)

    def visible_truth(self):
        """{i: posición en torso_link} de las frutas sobre la faja (verdad
        de terreno, solo para métricas y para el modo de percepción `gt`)."""
        return {i: self.pos_torso(i) for i in range(self.n) if self.state[i] == "belt"}

    # ------------------------------------------------------------ física
    def before_step(self):
        """Faja cinemática: las frutas apoyadas en ella van a v (sin rodar)."""
        for i in range(self.n):
            if self.state[i] == "belt" and self.on_belt(i):
                va = self.vadr[i]
                self.d.qvel[va:va + 2] = [0.0, self.v]
                self.d.qvel[va + 3:va + 6] = 0.0

    def after_step(self, t):
        """Contabilidad: frutas que salen por la punta de la faja o caen."""
        events = []
        for i in range(self.n):
            p = self.d.xpos[self.bid[i]]
            if self.state[i] == "belt":
                if p[1] > EXIT_Y:
                    self.park(i, "lost")
                    events.append(("lost", i))
                elif p[2] < BELT_TOP - 0.10:
                    self.park(i, "dropped")
                    events.append(("dropped", i))
            elif self.state[i].startswith("held") and p[2] < BELT_TOP - 0.10:
                self.park(i, "dropped")
                events.append(("dropped", i))
        return events

    def grip_force(self, side):
        """Fuerza normal (N) de los contactos de la mano `side` contra lo que
        no es el robot: el análogo en el sim de FORCE_ACT de la Inspire
        (como ManoSim en h1_2_sim2real). Con la fruta levantada de la faja,
        > 0 solo si los dedos la aprietan."""
        m, d = self.m, self.d
        hb, f = self.hand_bodies[side], 0.0
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in hb and b2 not in self.robot_bodies) or (b2 in hb and b1 not in self.robot_bodies):
                mujoco.mj_contactForce(m, d, k, self._wrench)
                f += abs(self._wrench[0])
        return f

    # ------------------------------------------------------------ agarre
    def try_weld(self, side, expected_torso, tol=0.02):
        """Asistencia de agarre (misma regla que `_try_weld` del demo de una
        fruta): ≥ 2 grupos de dedos/palma en contacto con una fruta que está
        a < tol del punto de agarre esperado. Devuelve el índice o None."""
        m, d = self.m, self.d
        groups = {}
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for fb, other in ((b1, b2), (b2, b1)):
                if fb not in self.bid:
                    continue
                name = m.body(other).name
                for g in ("thumb", "index", "middle", "ring", "pinky", "little", "wrist_yaw", "hand_base"):
                    if g in name and name.startswith(("L_", "R_", side)):
                        groups.setdefault(self.bid.index(fb), set()).add(g)
        for i, gs in groups.items():
            if len(gs) < 2 or self.state[i] != "belt":
                continue
            if np.linalg.norm(self.pos_torso(i) - expected_torso) > tol:
                continue
            eid, b1, fb = self.eid[(side, i)], self.wrist[side], self.bid[i]
            R1 = d.xmat[b1].reshape(3, 3)
            q1i, rq = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(q1i, d.xquat[b1])
            mujoco.mju_mulQuat(rq, q1i, d.xquat[fb])
            m.eq_data[eid][0:3] = 0.0
            m.eq_data[eid][3:6] = R1.T @ (d.xpos[fb] - d.xpos[b1])
            m.eq_data[eid][6:10] = rq
            m.eq_data[eid][10] = 1.0
            d.eq_active[eid] = 1
            self.state[i] = f"held:{side}"
            self.held[side] = i
            return i
        return None

    def release(self, side, deposited):
        """Suelta lo que tenga la mano `side`: depositada (cuenta) o no."""
        i = self.held[side]
        if i is None:
            return None
        if deposited:
            self.park(i, "deposited")
        else:
            self.d.eq_active[self.eid[(side, i)]] = 0
            self.held[side] = None
            self.state[i] = "belt"     # cae de vuelta (a la faja o al piso)
        return i
