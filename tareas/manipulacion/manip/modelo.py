"""El modelo MuJoCo de la estación como comprobador (sin simular): choques del brazo con la
mesa, las bandejas, el cuerpo y el otro brazo, con las mallas de colisión reales (como
caja_geometria.py de Codigos, que usa mj_collision), y lo que la cámara ve de la mesa con
los brazos en una postura (render de segmentación por EGL, en la GPU).

Lo usan la pose de observación y de calibración (herramientas/poses.py) y el planificador:
"planificación comprobada sin robot contra el modelo" (sec. 5). El modelo es el MISMO MJCF que
simula el robot, pero aquí solo se pone qpos y se llama a mj_forward.
"""
import math

import mujoco
import numpy as np

from . import cinematica as C

PREFIJO_BRAZO = {"izq": ("left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist", "L_"),
                 "der": ("right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist", "R_")}
ACOPLES = {"thumb_proximal_pitch_joint": (("thumb_intermediate_joint", 1.6), ("thumb_distal_joint", 2.4)),
           "index_proximal_joint": (("index_intermediate_joint", 1.0),),
           "middle_proximal_joint": (("middle_intermediate_joint", 1.0),),
           "ring_proximal_joint": (("ring_intermediate_joint", 1.0),),
           "pinky_proximal_joint": (("pinky_intermediate_joint", 1.0),)}
CUERPO = ("pelvis", "torso_link", "hip", "knee", "ankle", "logo", "head")


class Modelo:
    def __init__(self, xml):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "de_pie")
        if k >= 0:
            mujoco.mj_resetDataKeyframe(m, self.d, k)
        self.qadr = {lado: [int(m.jnt_qposadr[m.joint(n).id]) for n in C.NOMBRES_JUNTAS[lado]] for lado in C.LADOS}
        self.cintura = int(m.jnt_qposadr[m.joint("torso_joint").id])
        nombre_cuerpo = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        colisiona = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        robot = self._subarbol(m.body("pelvis").id)

        def de(pred):
            return [g for g in colisiona if pred(nombre_cuerpo[m.geom_bodyid[g]], m.geom_bodyid[g])]
        self.g_brazo = {lado: de(lambda n, b, p=PREFIJO_BRAZO[lado]: any(n.startswith(x) for x in p))
                        for lado in C.LADOS}
        # distales (codo en adelante): el brazo superior va montado en el torso y está siempre a
        # ~1 cm de él (como TORSO_PAIRS de la tesis, el par hombro-torso no se vigila)
        self.g_distal = {lado: de(lambda n, b, p=PREFIJO_BRAZO[lado][2:]: any(n.startswith(x) for x in p))
                         for lado in C.LADOS}
        self.g_cuerpo = de(lambda n, b: b in robot and any(x in n for x in CUERPO))
        # entorno: lo que no es el robot (mesa, patas, bandejas, objetos), sin el suelo
        self.g_entorno = [g for g in de(lambda n, b: b not in robot) if (m.geom(g).name or "") != "suelo"]
        self.g_mesa = [g for g in self.g_entorno if (m.geom(g).name or "").startswith(("tablero", "bandeja"))]
        self._ft = np.zeros(6)
        # vértices (marco del geom) de las mallas de colisión de los brazos: distancia exacta
        # a las cajas de la mesa y las bandejas
        self.vert = {}
        for g in self.g_brazo["izq"] + self.g_brazo["der"]:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                k = m.geom_dataid[g]
                V = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(float)
                # solo los de la envolvente convexa (la que usa MuJoCo para chocar): de miles a cientos
                try:
                    from scipy.spatial import ConvexHull
                    V = V[ConvexHull(V).vertices]
                except Exception:
                    pass
                self.vert[g] = V
        # margen de contacto en los brazos: mj_collision informa de los pares a menos de MARGEN
        self.margen = 0.04
        for g in self.g_brazo["izq"] + self.g_brazo["der"]:
            m.geom_margin[g] = self.margen
        self.lado_de = {g: lado for lado in C.LADOS for g in self.g_brazo[lado]}
        self.clase = {}
        for g in self.g_mesa:
            self.clase[g] = "mesa"
        for g in self.g_entorno:
            self.clase.setdefault(g, "entorno")
        for g in self.g_cuerpo:
            self.clase[g] = "cuerpo"

    def _subarbol(self, raiz):
        m = self.m
        dentro = {raiz}
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == raiz:
                    dentro.add(b)
                    break
                p = m.body_parentid[p]
        return dentro

    def poner(self, q_izq=None, q_der=None, cintura=0.0):
        d = self.d
        for lado, q in (("izq", q_izq), ("der", q_der)):
            if q is not None:
                d.qpos[self.qadr[lado]] = q
        d.qpos[self.cintura] = cintura
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        mujoco.mj_camlight(self.m, d)      # las cámaras no las mueve mj_kinematics
        mujoco.mj_collision(self.m, d)     # contactos con margen (holguras con el cuerpo)

    def _dist(self, gs1, gs2, dmax):
        best = dmax
        for g1 in gs1:
            for g2 in gs2:
                dd = mujoco.mj_geomDistance(self.m, self.d, g1, g2, dmax, self._ft)
                if dd < best:
                    best = dd
        return best

    def poner_mano(self, lado, angulos):
        """Dedos en la postura Inspire `angulos` (6, 0 cerrado .. 1000 abierto) con los acoples
        del MJCF (falange intermedia = proximal; pulgar: intermedia 1.6 y distal 2.4 veces la
        flexión). Hay que llamarlo antes de poner(): solo escribe qpos."""
        from .mano import J
        m, d = self.m, self.d
        pre = J.PREFIJO[lado]
        for k, (junta, _) in enumerate(J.DOF_MANO):
            if angulos[k] is None or angulos[k] < 0:
                continue
            q = J.angulo_a_rad(k, angulos[k])
            d.qpos[m.jnt_qposadr[m.joint(pre + junta).id]] = q
            for hija, f in ACOPLES.get(junta, ()):
                d.qpos[m.jnt_qposadr[m.joint(pre + hija).id]] = f * q

    def _dist_cajas(self, lado, cajas):
        """Distancia con signo de los vértices de las mallas del brazo a unas cajas (para mallas
        convexas contra una caja, lo más cercano es un vértice salvo arista contra arista, ~mm)."""
        m, d = self.m, self.d
        gs = [g for g in self.g_brazo[lado] if g in self.vert]
        Vw = np.vstack([self.vert[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g in gs])
        best = self.margen
        lo, hi = Vw.min(axis=0) - best, Vw.max(axis=0) + best
        for gc in cajas:
            if m.geom_type[gc] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            c, R, h = d.geom_xpos[gc], d.geom_xmat[gc].reshape(3, 3), m.geom_size[gc]
            ext = np.abs(R) @ h                          # semiejes de la caja alineada con el mundo
            if np.any(c + ext < lo) or np.any(c - ext > hi):
                continue                                 # descarte rápido: no se solapan las cajas
            q = np.abs((Vw - c) @ R) - h
            dist = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
            best = min(best, float(dist.min()))
        return best

    def holguras(self, lado, dmax=None, con_otro_brazo=True):
        """Distancia mínima (m, con signo: < 0 = penetración) del brazo `lado` a la mesa y las
        bandejas (vértices de las mallas contra las cajas: exacta), y al resto del entorno, al
        cuerpo (desde el codo: el brazo superior va montado en el torso) y al otro brazo
        (contactos de mj_collision con margen de 4 cm). Saturada en 4 cm. Llamar antes a poner().

        No se usa mj_geomDistance: en MuJoCo 3.9, entre una malla y una caja devuelve a veces un
        0.0 FALSO (medido: dedo a 2 cm de la mesa -> 0.0 con distmax 0.1, y de vez en cuando
        también con 0.04)."""
        out = {"mesa": self._dist_cajas(lado, self.g_mesa), "entorno": self.margen, "cuerpo": self.margen}
        if con_otro_brazo:
            out["otro_brazo"] = self.margen
        distal = set(self.g_distal[lado])
        mios = set(self.g_brazo[lado])
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if a not in mios:
                    continue
                if b in self.lado_de and self.lado_de[b] != lado:
                    k = "otro_brazo" if con_otro_brazo else None
                else:
                    k = self.clase.get(b)
                    if k == "cuerpo" and a not in distal:
                        k = None
                    if k == "mesa":
                        k = None             # la mesa va por vértices (exacta)
                if k is not None:
                    out[k] = min(out[k], float(c.dist))
        return out

    def punto_cuerpo(self, nombre):
        b = self.m.body(nombre).id
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()


class Vista:
    """Lo que ve una cámara del modelo con los brazos en una postura: render de segmentación
    (EGL). Sirve para saber qué parte del tablero tapan los brazos."""

    def __init__(self, modelo, camara="robot_ir_left"):
        self.mod = modelo
        m = modelo.m
        self.cam = m.camera(camara).id
        w, h = (int(v) for v in m.cam_resolution[self.cam])
        self.r = mujoco.Renderer(m, height=h, width=w)
        self.r.enable_segmentation_rendering()
        # lo que tiene que quedar a la vista: el tablero (la placa blanca de encima, si la hay)
        # y las bandejas (para comprobar que el objeto quedó dentro)
        self.mesa = {g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(("tablero", "bandeja"))}
        self.tablero = m.geom("tablero").id
        self.brazos = set(modelo.g_brazo["izq"]) | set(modelo.g_brazo["der"])
        # geoms visuales de los brazos también (las mallas visuales no colisionan)
        nombres = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        pref = PREFIJO_BRAZO["izq"] + PREFIJO_BRAZO["der"]
        self.brazos |= {g for g in range(m.ngeom) if nombres[m.geom_bodyid[g]].startswith(pref)}
        self.mesa_libre = None

    def segmentar(self):
        self.r.update_scene(self.mod.d, camera=self.cam)
        seg = self.r.render()
        return seg[..., 0]              # id del geom (-1 = fondo)

    def tablero_tapado(self, q_lejos=None):
        """Fracción de los píxeles del tablero y las bandejas que tapan los brazos en la postura actual.
        La referencia (tablero sin brazos) se toma una vez con los brazos en `q_lejos`."""
        mod = self.mod
        actual = mod.d.qpos.copy()
        if self.mesa_libre is None:
            q = q_lejos if q_lejos is not None else {"izq": None, "der": None}
            mod.poner(q.get("izq"), q.get("der"))
            self.mesa_libre = np.isin(self.segmentar(), list(self.mesa))
            mod.d.qpos[:] = actual
            mod.poner()
        seg = self.segmentar()
        tapa = np.isin(seg, list(self.brazos)) & self.mesa_libre
        return float(tapa.sum()) / max(1, int(self.mesa_libre.sum())), seg

    def cerrar(self):
        self.r.close()


def angulo_rot(R1, R2):
    return math.degrees(math.acos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))
