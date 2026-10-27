"""Bloque de PLANIFICACIÓN: objeto + estado de los brazos -> Plan de tramos articulares
(sec. 5), comprobado SIN ROBOT contra el modelo.

Algoritmos de la tesis (h1_2_algoritms.bimanual.bimanual_planner, capa 3), reutilizados tal
cual: ArmCollisionChecker (cápsulas del brazo contra el otro brazo congelado, el torso y una
nube de obstáculos, con márgenes), rrt_connect (RRT-Connect en 7 GDL), shortcut y densify.
Lo que pone el reto:
  * la postura objetivo la da la IK del reto (manip/cinematica.py) con la ORIENTACIÓN de
    agarre (palma abajo, guiñada alineada con el objeto módulo su simetría), probando los
    azimuts y elevaciones de config/planificacion.yaml y quedándose con la más cercana a la
    postura actual que pase todas las comprobaciones;
  * tránsitos (observación -> preagarre -> presuelta -> observación) con RRT: la mesa, las
    paredes de las bandejas y los demás objetos (de la percepción) son la nube de obstáculos;
    el margen de 6 cm de la tesis mantiene el brazo por encima de la mesa;
  * descenso final y retirada: rectas cartesianas (IK punto a punto con la orientación fija);
  * suelta: el objeto centrado en su bandeja con el fondo a 8 mm del fondo; el rotulador
    (14 cm) solo cabe en la diagonal de la bandeja (11.2 cm de interior): a 45 grados;
  * comprobación final contra el MJCF de la estación VACÍA (mallas de colisión, como
    caja_geometria.py): holgura con mesa y bandejas, cuerpo y otro brazo en todos los puntos,
    y altura de seguridad de la mano sobre la mesa en todo salvo el descenso final.
Todo en torso_link. La verdad de terreno no se usa: los objetos llegan de la percepción.
"""
import math

import numpy as np

from . import cinematica as C
from .tipos import Agarre, Plan, Tramo

BP = None   # módulo de la tesis (se importa al crear el planificador)


def _bp():
    global BP
    if BP is None:
        from h1_2_algoritms.bimanual import bimanual_planner
        BP = bimanual_planner
    return BP


def agarre_de(d):
    return Agarre(tipo=d["tipo"], preforma=list(d["preforma"]), cierre=list(d["cierre"]),
                  force_set=list(d["force_set"]), altura_agarre=float(d["altura_agarre"]),
                  elev_deg=float(d.get("elev_deg", 0.0)), tolerancia_m=d.get("tolerancia_m") or 0.0,
                  previo=list(d["previo"]) if d.get("previo") else None,
                  suelta=list(d["suelta"]) if d.get("suelta") else None)


class Planificador:
    def __init__(self, est, agarres, catalogo, cfg, modelo, q_obs):
        self.est, self.cfg, self.mod = est, cfg, modelo
        self.agarres = {k: agarre_de(v) for k, v in agarres.items()}
        self.simetria = {k: math.radians(v.get("simetria_deg", 0)) for k, v in agarres.items()}
        self.prof_palma = {k: float(v.get("prof_palma", 0.058)) for k, v in agarres.items()}
        self.avance = {k: float(v.get("avance", 0.17)) for k, v in agarres.items()}
        self.lateral = {k: float(v.get("lateral", 0.0)) for k, v in agarres.items()}
        # elevaciones de los dedos permitidas por objeto (con los dedos hacia abajo, en una caja
        # apoyan en la cara de arriba en vez de rodearla: medido en probar_agarre.py)
        self.elevaciones = {k: v.get("elevaciones") for k, v in agarres.items()}
        # orientación de la mano: 'arriba' (palma abajo; también la pinza) o 'lado' (palma
        # vertical mirando a la línea media del robot, pulgar arriba)
        self.modo = {k: ("lado" if v.get("tipo") == "lado" else "arriba") for k, v in agarres.items()}
        # guiñada de los dedos respecto de la del objeto (rotulador: 90 = cruzando su eje)
        # altura de suelta sobre el fondo por objeto (rotulador: desde encima del borde)
        self.suelta_sobre = {k: v.get("suelta_sobre_fondo") for k, v in agarres.items()}
        # desalineación de guiñada que tolera el agarre de cada objeto (si no, la de la config)
        self.tol_guinada = {k: v.get("tolerancia_guinada") for k, v in agarres.items()}
        self.suelta_off = {k: v.get("suelta_off") for k, v in agarres.items()}
        # retroceso hacia la muñeca antes de subir, en las sueltas sobre la mesa (el pulgar deja de
        # tocar la cara del prisma); a la pelota, en cambio, los dedos la arrastran de vuelta
        self.retirada_atras = {k: bool(v.get("retirada_atras", True)) for k, v in agarres.items()}
        # giro que el objeto da dentro de la mano entre el agarre y la suelta (grados, por mano)
        self.deriva_giro = {k: v.get("deriva_giro") or {} for k, v in agarres.items()}
        # dedos (además del pulgar) que deben seguir tocando con el objeto ya elevado: con menos,
        # el agarre es débil y se cae en el transporte (la pelota sujeta solo por índice y medio)
        self.dedos_elevado = {k: int(v.get("dedos_elevado", 1)) for k, v in agarres.items()}
        # duración mínima del transporte cargado (s): más lento, menos inercia que saque el objeto
        # de un agarre de tres dedos (la pelota en el cruce de la mano izquierda al centro)
        self.duracion_transporte = {k: v.get("duracion_transporte") for k, v in agarres.items()}
        self.guinada_rel = {k: math.radians(float(v.get("guinada_rel_deg", 0.0))) for k, v in agarres.items()}
        # el objeto se suelta con sus caras paralelas a las paredes de la bandeja (paso en grados):
        # el cubo girado 45 ocupa su diagonal (8.5 cm) y, con 1-2 cm de error, queda apoyado en
        # la pared, inclinado 21 grados (nivel 3, 2026-10-15)
        self.alinear_bandeja = {k: v.get("alinear_bandeja_deg") for k, v in agarres.items()}
        self.catalogo = catalogo
        self.q_obs = {k: np.asarray(v, dtype=float) for k, v in q_obs.items()}
        self.z_mesa = est.z_mesa - est.torso_z
        self.lim = {lado: C.limites(lado) for lado in C.LADOS}
        self.plano = None          # plano de la mesa medido (ListaObjetos.plano)
        _bp()

    # ------------------------------------------------------------ obstáculos
    def nube(self, otros=()):
        """Mesa (rejilla de 2.5 cm), paredes de las bandejas y otros objetos, en torso_link."""
        t, e = self.est.tablero, self.est
        xs = np.arange(t.cx - t.fondo / 2, t.cx + t.fondo / 2 + 1e-9, 0.025)
        ys = np.arange(t.cy - t.ancho / 2, t.cy + t.ancho / 2 + 1e-9, 0.025)
        X, Y = np.meshgrid(xs, ys)
        P = [np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, self.z_mesa)])]
        for b in e.bandejas.values():
            for s in np.linspace(-0.5, 0.5, 6):
                for (dx, dy) in ((s, -0.5), (s, 0.5), (-0.5, s), (0.5, s)):
                    P.append(np.array([[b.cx + dx * b.fondo, b.cy + dy * b.ancho, self.z_mesa + e.bandeja_borde]]))
        for o in otros:
            r = max(o.medidas[0], o.medidas[1]) / 2
            for ang in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                for zz in (0.3, 1.0):
                    P.append(np.array([[o.centro[0] + r * math.cos(ang), o.centro[1] + r * math.sin(ang),
                                        self.z_mesa + zz * o.medidas[2]]]))
        return np.vstack(P)

    # ------------------------------------------------------------ comprobaciones
    def mano_baja(self, lado, q):
        """z mínima (torso_link) de la mano (palma, dedos estirados, pulgar) sobre la mesa."""
        return C.mano_baja(lado, q, self.est.tablero)

    def holguras(self, lado, q, q_otro):
        qi, qd = (q, q_otro) if lado == "izq" else (q_otro, q)
        self.mod.poner(qi, qd)
        return self.mod.holguras(lado)

    def postura_valida(self, lado, q, q_otro, final=False, altura=None):
        """`altura`: la mínima del punto más bajo de la mano sobre la mesa en este tramo (por
        defecto la de seguridad; más si hay objetos altos o lleva uno colgando)."""
        h = self.holguras(lado, q, q_otro)
        c = self.cfg
        if h["mesa"] < c["margen_mesa"] or h["cuerpo"] < c["margen_cuerpo"] or h["otro_brazo"] < c["margen_otro_brazo"]:
            return False, h
        alt = c["altura_seguridad"] if altura is None else altura
        if not final and self.mano_baja(lado, q) < self.z_mesa + alt - 1e-4:
            return False, h
        return True, h

    # ------------------------------------------------------------ IK de agarre y suelta
    def _candidatas(self, lado, p, gui_obj, sim, elevs, q_semilla, q_otro, extra_az=(), azs=None, off=None,
                    mano=None, modo="arriba", clase=None, altura=None):
        """Posturas que llegan a p con la palma abajo y una guiñada compatible con el objeto,
        con su preagarre y su recta de descenso ya comprobados contra el modelo; ordenadas por
        cercanía a q_semilla. El preagarre sube lo justo para que el punto más bajo de la mano
        quede a la altura de seguridad (más alto acerca el antebrazo al pecho)."""
        if azs is None:
            azs = list(self.cfg["azimuts"]) + list(extra_az)
            if sim > 0:
                # la guiñada de los dedos se alinea con una cara del objeto (gui + k·sim), con la
                # desalineación que tolera el agarre (config: tolerancia_guinada)
                base = math.degrees(gui_obj)
                tol_g = self.tol_guinada.get(clase) or self.cfg.get("tolerancia_guinada", [0, -10, 10, -20, 20])
                azs = []
                for dg in tol_g:
                    for k in range(int(round(2 * math.pi / sim))):
                        a = round(math.degrees(math.remainder(math.radians(base + dg + k * math.degrees(sim)),
                                                              2 * math.pi)), 1)
                        if -95 <= a <= 95 and a not in azs:
                            azs.append(a)
        if modo == "lado":
            # palma vertical: con los dedos hacia un lado (|az| -> 90) la palma ya no puede mirar
            # a la línea media (grasp_rotation degenera)
            azs = [a for a in azs if abs(a) <= self.cfg.get("az_max_lado", 60)]
        out = []
        tol = self.cfg.get("tol_ik", 0.004)
        razones = self.razones = {}
        if mano is not None:
            self.mod.poner_mano(lado, mano)

        def no(r):
            razones[r] = razones.get(r, 0) + 1
        for elev in elevs:
            for az in azs:
                R = C.rotacion_agarre(lado, modo, elev, az)
                q, ep, eo, ok = C.ik(lado, p, R, q0=q_semilla, lim=self.lim[lado], tol_pos=tol, off=off)
                if not ok:
                    q, ep, eo, ok = C.ik(lado, p, R, lim=self.lim[lado], tol_pos=tol, off=off)
                if not ok:
                    no("sin IK")
                    continue
                v, h = self.postura_valida(lado, q, q_otro, final=True)
                if not v:
                    no("agarre choca (" + min(h, key=h.get) + ")")
                    continue
                alt = self.cfg["altura_seguridad"] if altura is None else altura
                subir = max(self.cfg["preagarre_min"], self.z_mesa + alt + 0.005 - self.mano_baja(lado, q))
                p_pre = p + np.array([0, 0, subir])
                qp, *_, okp = C.ik(lado, p_pre, R, q0=q, lim=self.lim[lado], tol_pos=tol, off=off)
                if not okp:
                    no("preagarre sin IK")
                    continue
                v, h = self.postura_valida(lado, qp, q_otro, altura=altura)
                if not v:
                    no("preagarre choca o bajo (" + min(h, key=h.get) + ")")
                    continue
                baja = self.linea(lado, qp, p_pre, p, R, "descender", off=off)
                if baja is None:
                    no("recta sin IK")
                    continue
                okb, _, mot = self.comprobar([baja], q_otro)
                if not okb:
                    no("recta choca: " + mot.split(":")[1].split(",")[0].strip())
                    continue
                out.append({"q": baja.q1, "q_pre": qp, "p_pre": p_pre, "az": az, "elev": elev, "R": R,
                            "baja": baja, "coste": float(np.abs(q - q_semilla).sum())})
        return sorted(out, key=lambda c: c["coste"])

    def z_en(self, x, y):
        """Altura de la mesa en (x, y) del torso: del plano medido si lo hay, si no de la config."""
        if self.plano is None or not np.all(np.isfinite(self.plano)):
            return self.z_mesa
        n, d = self.plano[:3], self.plano[3]
        return float(-(d + n[0] * x + n[1] * y) / n[2])

    def _alto(self, clase):
        o = self.catalogo[clase]
        return {"caja": lambda: o["medidas"][2], "cilindro": lambda: o["alto"],
                "esfera": lambda: 2 * o["radio"], "cilindro_tumbado": lambda: 2 * o["radio"]}[o["forma"]]()

    def off(self, lado, clase):
        return C.punto_agarre(lado, self.prof_palma[clase], self.avance[clase], self.lateral[clase])

    def off_suelta(self, lado, clase):
        """Dónde queda DE VERDAD el objeto en la mano al soltarlo (marco de la muñeca): al cerrar,
        la reacción de los dedos sube la mano (el brazo cede) y al elevar el objeto resbala unos
        cm (prisma: 4 cm más lejos de la palma). Medido (tabla `suelta_off`, x y z de la mano
        izquierda); si no hay medida, el punto de agarre."""
        so = self.suelta_off.get(clase)
        if so is None:
            return self.off(lado, clase)
        return np.array([so[0], so[1] if lado == "izq" else -so[1], so[2]])

    def agarre_ik(self, lado, obj, q_semilla, q_otro, altura=None):
        ag = self.agarres[obj.clase]
        p = obj.centro.copy()
        # la mesa donde está el objeto, MEDIDA (plano de la percepción): el torso no está
        # horizontal (~0.5 grados hacia atrás en el simulador) y la config supone que sí
        z_mesa_aqui = obj.centro[2] - self._alto(obj.clase) / 2
        p[2] = z_mesa_aqui + ag.altura_agarre
        sim = self.simetria.get(obj.clase, 0.0)
        elevs = self.elevaciones.get(obj.clase) or \
            [ag.elev_deg] + [e for e in self.cfg["elevaciones"] if e != ag.elev_deg]
        gui = obj.guinada + self.guinada_rel.get(obj.clase, 0.0)
        return self._candidatas(lado, p, gui, sim, elevs, q_semilla, q_otro, off=self.off(lado, obj.clase),
                                mano=ag.preforma, modo=self.modo.get(obj.clase, "arriba"), clase=obj.clase,
                                altura=altura), p

    def suelta_ik(self, lado, bandeja, obj, desfase, q_semilla, q_otro, azs=None):
        """Objeto centrado en la bandeja, o en un punto de la mesa si `bandeja` es (x, y).
        `desfase` = guiñada del objeto - azimut de la mano."""
        ag = self.agarres[obj.clase]
        if isinstance(bandeja, str):
            b = self.est.bandejas[bandeja]
            x, y, fondo = b.cx, b.cy, 0.004
        else:
            x, y, fondo = bandeja[0], bandeja[1], 0.0
        sobre = self.suelta_sobre.get(obj.clase)
        sobre = self.cfg["suelta_sobre_fondo"] if sobre is None else sobre
        p = np.array([x, y, self.z_en(x, y) + fondo + sobre + ag.altura_agarre])
        elevs = [ag.elev_deg] + [e for e in self.cfg["elevaciones"] if e != ag.elev_deg]
        if azs is not None:
            pass                                  # impuestos (nivel 4: girado lo pedido)
        elif self.catalogo[obj.clase]["forma"] == "cilindro_tumbado" and isinstance(bandeja, str):
            # solo cabe en la diagonal: guiñada del objeto 45 + k·90 -> azimut de la mano
            azs = [math.degrees(math.remainder(math.radians(45 + 90 * k) - desfase, 2 * math.pi)) for k in range(4)]
            azs = [a for a in azs if -95 <= a <= 95]
        elif self.alinear_bandeja.get(obj.clase) and isinstance(bandeja, str):
            paso = float(self.alinear_bandeja[obj.clase])
            azs = sorted({round(math.degrees(math.remainder(math.radians(paso * k + dt) - desfase, 2 * math.pi)), 1)
                          for k in range(int(round(360 / paso))) for dt in (0.0, -8.0, 8.0)})
            azs = [a for a in azs if -95 <= a <= 95]
            ok = self._candidatas(lado, p, 0.0, 0.0, elevs, q_semilla, q_otro, azs=azs,
                                  off=self.off_suelta(lado, obj.clase),
                                  mano=[500 if a < 1000 else 1000 for a in ag.cierre],
                                  modo=self.modo.get(obj.clase, "arriba"))
            if ok:
                return ok, p
            azs = None                            # alineado no hay postura: como antes
        # al soltar, la mano aún sujeta: dedos medio cerrados (el objeto los para)
        return self._candidatas(lado, p, 0.0, 0.0, elevs, q_semilla, q_otro, azs=azs,
                                off=self.off_suelta(lado, obj.clase),
                                mano=[500 if a < 1000 else 1000 for a in ag.cierre],
                                modo=self.modo.get(obj.clase, "arriba")), p

    def rehacer_descenso(self, plan, p_nuevo, q_pre):
        """Descenso y elevación del `plan` hacia el punto de agarre corregido `p_nuevo` (marco
        del torso ahora), desde la postura de preagarre q_pre (la consigna actual). La
        orientación de la mano no se toca (la inclinación es de 1-2 grados). Devuelve
        (descender, elevar) o None si la recta no tiene IK."""
        lado, clase = plan.mano, plan.objeto.clase
        off = self.off(lado, clase)
        t = {tr.nombre: tr for tr in plan.tramos}
        T = C.fk(lado, t["descender"].q_ruta[-1])
        R = T[:3, :3]
        Tp = C.fk(lado, q_pre)
        p_pre = Tp[:3, :3] @ off + Tp[:3, 3]
        baja = self.linea(lado, q_pre, p_pre, p_nuevo, R, "descender", off=off)
        if baja is None:
            return None
        Te = C.fk(lado, t["elevar"].q_ruta[-1])             # hasta donde subía el plan (transporte)
        p_up = Te[:3, :3] @ off + Te[:3, 3]
        p_up = np.array([p_nuevo[0], p_nuevo[1], max(p_up[2], p_pre[2])])
        sube = self.linea(lado, baja.q1, p_nuevo, p_up, R, "elevar", off=off)
        if sube is None:
            return None
        return baja, sube

    def punto_suelta(self, plan):
        """Dónde suelta el plan el objeto (marco del torso del plan): FK del final de la bajada."""
        lado, clase = plan.mano, plan.objeto.clase
        T = C.fk(lado, {t.nombre: t for t in plan.tramos}["bajar"].q_ruta[-1])
        return T[:3, :3] @ self.off_suelta(lado, clase) + T[:3, 3]

    def rehacer_suelta(self, plan, p_nuevo, q_actual):
        """Bajada y retirada hacia el punto de suelta corregido (la inclinación del cuerpo con el
        brazo extendido, igual que antes de agarrar). Devuelve (bajar, retirar) o None."""
        lado, clase = plan.mano, plan.objeto.clase
        off = self.off_suelta(lado, clase)
        t = {tr.nombre: tr for tr in plan.tramos}
        R = C.fk(lado, t["bajar"].q_ruta[-1])[:3, :3]
        T0 = C.fk(lado, q_actual)
        p0 = T0[:3, :3] @ off + T0[:3, 3]
        baja = self.linea(lado, q_actual, p0, p_nuevo, R, "bajar", off=off)
        if baja is None:
            return None
        Tr = C.fk(lado, t["retirar"].q_ruta[-1])
        p_ret = Tr[:3, :3] @ off + Tr[:3, 3]
        sube = self.retirada(lado, baja.q1, p_nuevo, p_ret[2], R, off, atras=not isinstance(plan.bandeja, str) and self.retirada_atras.get(clase, True))
        return None if sube is None else (baja, sube)

    def retirada(self, lado, q, p, z_fin, R, off, atras=False, d_atras=0.03):
        """Tras soltar: subir en vertical; sobre la mesa (nivel 4, cambio de mano), antes 3 cm hacia
        atrás en horizontal (hacia la muñeca): el pulgar, que hacía de poste, deja de tocar la cara
        del objeto (si sube tocándola lo arrastra y lo gira 7-12 grados)."""
        tramos = []
        if atras:
            x = R[:, 0].copy()
            x[2] = 0.0
            x /= max(np.linalg.norm(x), 1e-9)
            p_a = p - d_atras * x
            t_a = self.linea(lado, q, p, p_a, R, "retirar", off=off)
            if t_a is not None:
                tramos.append(t_a)
                q, p = t_a.q1, p_a
        t_s = self.linea(lado, q, p, np.array([p[0], p[1], z_fin]), R, "retirar", off=off)
        if t_s is None:
            return None
        tramos.append(t_s)
        ruta = np.vstack([tr.q_ruta for tr in tramos])
        return Tramo(lado, tramos[0].q0, ruta[-1], 0.0, "retirar", cartesiano=True, q_ruta=ruta)

    # ------------------------------------------------------------ tramos
    def linea(self, lado, q0, p0, p1, R, nombre, off=None):
        """Recta cartesiana del punto de agarre de p0 a p1 con la orientación R fija."""
        n = max(2, int(math.ceil(np.linalg.norm(p1 - p0) / self.cfg["paso_cartesiano"])))
        q, ruta = q0.copy(), []
        for k in range(1, n + 1):
            q, ep, eo, ok = C.ik(lado, p0 + (p1 - p0) * k / n, R, q0=q, lim=self.lim[lado],
                                 tol_pos=self.cfg.get("tol_ik", 0.004), off=off)
            if not ok:
                return None
            if ruta and np.max(np.abs(q - ruta[-1])) > 0.15:
                return None                       # salto: la IK cambió de rama
            ruta.append(q.copy())
        return Tramo(lado, q0, ruta[-1], 0.0, nombre, cartesiano=True, q_ruta=np.array(ruta))

    def transito(self, lado, q0, q1, q_otro, obstaculos, nombre, rng, altura=None):
        bp = _bp()
        plan = self
        z_min = self.z_mesa + (self.cfg["altura_seguridad"] if altura is None else altura) - 1e-4

        class Comprobador(bp.ArmCollisionChecker):
            """El de la tesis (cápsulas contra el otro brazo, el torso y la nube) + la altura de
            seguridad de la MANO entera (dedos y pulgar sobresalen de la cápsula de la mano)."""

            def valid(self, q):
                return super().valid(q) and plan.mano_baja(lado, q) >= z_min

        chk = Comprobador(C.INGLES[lado], q_otro, obstaculos, self.lim[lado][0], self.lim[lado][1],
                          self.cfg["margen_rrt_brazos"], self.cfg["margen_rrt_entorno"])
        chk.relax_for_start(q0)
        if not chk.valid(q1):
            return None                     # el objetivo no vale para el RRT: no gastar su tiempo
        if chk.edge_valid(q0, q1):
            ruta = [q0, q1]
        else:
            ruta = bp.rrt_connect(chk, q0, q1, time_limit=self.cfg["tiempo_rrt"], rng=rng)
            if ruta is None:
                return None
            ruta = bp.shortcut(chk, ruta, rng=rng)
        ruta = bp.densify(ruta, max_step=0.1)
        return Tramo(lado, q0, q1, 0.0, nombre, q_ruta=np.array(ruta[1:]))

    def manos_tramo(self, ag):
        """Forma de la mano en cada tramo del plan: preforma al acercarse y bajar; cerrada a
        medias (el objeto la para) mientras sujeta; al soltar vuelve a la PREFORMA (abrir del todo
mete el pulgar abierto 10 cm hacia la mesa y la bandeja). Al ELEVAR, la de la
        preforma: con un objeto bajo los dedos que lo rodean llegan al tablero (con la fuerza
        limitada) y la recta solo aleja la mano de la mesa."""
        sujeta = [500 if a < 1000 else 1000 for a in ag.cierre]
        return {"aproximar": ag.preforma, "descender": ag.preforma, "elevar": ag.preforma, "transportar": sujeta,
                "bajar": sujeta, "retirar": ag.preforma, "volver": ag.preforma}

    def comprobar(self, tramos, q_otro, manos=None, alturas=None):
        """Todos los puntos de todos los tramos contra el modelo, con la forma de la mano de
        cada tramo (`manos`: nombre -> 6 ángulos; si no, la que tenga el modelo).
        (ok, holgura mín mesa, motivo)."""
        hmin, motivo = np.inf, ""
        for t in tramos:
            if manos and t.nombre in manos:
                self.mod.poner_mano(t.lado, manos[t.nombre])
            ruta = np.vstack([t.q0, t.q_ruta]) if t.q_ruta is not None else np.vstack([t.q0, t.q1])
            fina = _bp().densify(list(ruta), max_step=0.05)
            for q in fina:
                v, h = self.postura_valida(t.lado, q, q_otro, final=t.cartesiano,
                                           altura=(alturas or {}).get(t.nombre))
                hmin = min(hmin, h["mesa"])
                if not v:
                    motivo = f"{t.nombre}: holguras {', '.join(f'{k} {v_ * 100:.1f} cm' for k, v_ in h.items())}"
                    if not t.cartesiano:
                        motivo += f", mano a {100 * (self.mano_baja(t.lado, q) - self.z_mesa):.1f} cm de la mesa"
                    return False, hmin, motivo
        return True, hmin, motivo

    # ------------------------------------------------------------ plan completo
    def planificar(self, obj, q_actual, bandeja=None, mano=None, otros=(), semilla=0, plano=None, giro=None,
                   tol_giro_deg=5.0):
        """Plan para coger `obj` (ObjetoDetectado) y dejarlo en `bandeja` (por defecto la suya)
        con `mano` (por defecto la del lado del objeto). `plano`: el de la mesa medido.
        `giro` (rad, nivel 4): el objeto tiene que quedar girado eso respecto de como está (módulo
        su simetría): la mano suelta con su azimut de agarre + giro."""
        rng = np.random.default_rng(semilla)
        if plano is not None:
            self.plano = np.asarray(plano, dtype=float)
        cat = self.catalogo.get(obj.clase)
        if cat is None:
            return Plan(obj, "", None, "", [], False, "objeto desconocido")
        bandeja = bandeja or cat["bandeja"]
        lado = mano or ("izq" if obj.centro[1] > 0 else "der")
        otro = "der" if lado == "izq" else "izq"
        q0, q_otro = np.asarray(q_actual[lado], float), np.asarray(q_actual[otro], float)
        ag = self.agarres[obj.clase]
        if isinstance(bandeja, str) and self.est.lado_bandeja(bandeja) != lado:
            return Plan(obj, lado, ag, bandeja, [], False,
                        f"la bandeja {bandeja} está al otro lado: hace falta la entrega de mano a mano")
        # alturas mínimas de la mano (su punto más bajo sobre la mesa): vacía, por encima del
        # objeto más alto que quede en la mesa; cargada, además lo que el objeto cuelga por debajo
        # de la mano (si no, el objeto que lleva barre a los demás: nivel 3, 2026-10-13)
        seg, margen_obj = self.cfg["altura_seguridad"], self.cfg.get("margen_objetos", 0.03)
        altos = [self._alto(o.clase) if o.clase in self.catalogo else float(o.medidas[2]) for o in otros]
        alto_otros = max(altos) if altos else 0.0
        alt_vacia = max(seg, alto_otros + margen_obj)
        cands, p_ag = self.agarre_ik(lado, obj, self.q_obs[lado], q_otro, altura=alt_vacia)
        if not cands:
            return Plan(obj, lado, ag, bandeja, [], False, "agarre imposible: " +
                        ", ".join(f"{k} x{v}" for k, v in sorted(self.razones.items(), key=lambda x: -x[1])))
        obst = self.nube(otros)
        fallos = {}

        def fallo(r):
            fallos[r] = fallos.get(r, 0) + 1
        for c in cands[:6]:
            desfase = math.remainder(obj.guinada - math.radians(c["az"]), 2 * math.pi)
            azs_su = None
            if giro is not None:
                # guiñada final = az_suelta + (guiñada - az_agarre) = guiñada + giro (mód. simetría)
                paso = math.degrees(obj.simetria) if obj.simetria > 0 else 360.0
                # el objeto CONSERVA su desfase con la mano al cerrar (medido con la verdad del
                # simulador: no se alinea con los dedos; deriva +5-11 grados al elevar): la mano
                # gira exactamente `giro` entre el agarre y la suelta
                base = c["az"] + math.degrees(giro) - float(self.deriva_giro.get(obj.clase, {}).get(lado, 0.0))
                azs_su = sorted({round(math.degrees(math.remainder(math.radians(base + k * paso + dt), 2 * math.pi)), 1)
                                 for k in range(int(round(360 / paso))) for dt in (0.0, -tol_giro_deg, tol_giro_deg)},
                                key=lambda a: abs(math.remainder(math.radians(a - base), math.radians(paso))))
                azs_su = [a for a in azs_su if -95 <= a <= 95]
            sueltas, p_su = self.suelta_ik(lado, bandeja, obj, desfase, c["q_pre"], q_otro, azs=azs_su)
            if not sueltas:
                fallo("suelta imposible: " + ", ".join(f"{k} x{v}" for k, v in self.razones.items()))
            off = self.off(lado, obj.clase)
            colgado = max(0.0, self.mano_baja(lado, c["q"]) - (p_ag[2] - ag.altura_agarre))
            alt_carga = max(seg, colgado + max(alto_otros, self.est.bandeja_borde) + margen_obj)

            def hasta(q, p, alt, p_min):
                """p subido lo justo para que la mano (en la postura q) quede a `alt` sobre la mesa."""
                return p + np.array([0, 0, max(p_min[2] - p[2], self.z_mesa + alt + 0.005 - self.mano_baja(lado, q))])
            p_up_g = hasta(c["q"], p_ag, alt_carga, c["p_pre"])
            for s in sueltas[:3]:
                t1 = self.transito(lado, q0, c["q_pre"], q_otro, obst, "aproximar", rng, altura=alt_vacia)
                t2 = c["baja"]
                t3 = self.linea(lado, c["q"], p_ag, p_up_g, c["R"], "elevar", off=off)
                off_s = self.off_suelta(lado, obj.clase)
                p_up_s = hasta(s["q"], p_su, alt_carga, s["p_pre"])
                q_up_s, _, _, ok_s = C.ik(lado, p_up_s, s["R"], q0=s["q_pre"], lim=self.lim[lado],
                                          tol_pos=self.cfg.get("tol_ik", 0.004), off=off_s)
                t5 = self.linea(lado, q_up_s, p_up_s, p_su, s["R"], "bajar", off=off_s) if ok_s else None
                t4 = self.transito(lado, t3.q1, q_up_s, q_otro, obst, "transportar", rng, altura=alt_carga) \
                    if (t3 is not None and ok_s) else None
                t6 = self.retirada(lado, s["q"], p_su, hasta(s["q"], p_su, alt_vacia, s["p_pre"])[2], s["R"], off_s,
                                   atras=not isinstance(bandeja, str) and self.retirada_atras.get(obj.clase, True))
                t7 = self.transito(lado, t6.q1, self.q_obs[lado], q_otro, obst, "volver", rng, altura=alt_vacia) \
                    if t6 is not None else None
                tramos = [t1, t2, t3, t4, t5, t6, t7]
                if any(t is None for t in tramos):
                    nombres = ("aproximar", "descender", "elevar", "transportar", "bajar", "retirar", "volver")
                    fallo("sin camino: " + ", ".join(n for n, t in zip(nombres, tramos) if t is None))
                    continue
                ok, hmin, motivo = self.comprobar(tramos, q_otro, self.manos_tramo(ag),
                                                  {"aproximar": alt_vacia, "transportar": alt_carga,
                                                   "volver": alt_vacia})
                if not ok:
                    fallo(motivo)
                if ok:
                    p = Plan(obj, lado, ag, bandeja, tramos, True, "", holgura_min_mesa=hmin)
                    p.az_agarre, p.az_suelta = c["az"], s["az"]
                    p.alturas = {"vacia": alt_vacia, "carga": alt_carga, "colgado": colgado}
                    return p
        return Plan(obj, lado, ag, bandeja, [], False, "ningún plan pasa la comprobación: " +
                    "; ".join(f"{k} (x{v})" for k, v in sorted(fallos.items(), key=lambda x: -x[1])[:3]))

    def tocar(self, lado, p, q_actual, q_otro, preforma, elevs=(-45.0, -30.0, -60.0), sobre=0.008, sondeo=0.02,
              semilla=0):
        """Nivel 1: la yema del índice (PUNTOS_MANO['indice_toque']) a `sobre` m sobre el punto p de la
        mesa, con la palma abajo y los dedos inclinados `elevs`; luego una recta de `sondeo` m
        hacia abajo (se para por contacto) y la retirada. Devuelve (tramos, motivo)."""
        rng = np.random.default_rng(semilla)
        off = C.PUNTOS_MANO["indice_toque"][lado]
        objetivo = np.array([p[0], p[1], self.z_en(p[0], p[1]) + sobre])
        azs = [a for a in self.cfg["azimuts"]]
        cands = self._candidatas(lado, objetivo, 0.0, 0.0, list(elevs), self.q_obs[lado], q_otro, azs=azs, off=off,
                                 mano=preforma)
        if not cands:
            return None, "sin postura: " + ", ".join(f"{k} x{v}" for k, v in self.razones.items())
        obst = self.nube(())
        for c in cands[:4]:
            t1 = self.transito(lado, np.asarray(q_actual, float), c["q_pre"], q_otro, obst, "aproximar", rng)
            t2 = c["baja"]
            t3 = self.linea(lado, c["q"], objetivo, objetivo - np.array([0, 0, sondeo]), c["R"], "sondear", off=off)
            t4 = self.linea(lado, c["q"], objetivo, c["p_pre"], c["R"], "retirar", off=off)
            if None in (t1, t3, t4):
                continue
            manos = {n: preforma for n in ("aproximar", "descender", "retirar")}
            ok, _, motivo = self.comprobar([t1, t2, t4], q_otro, manos)
            if ok:
                return {"aproximar": t1, "descender": t2, "sondear": t3, "retirar": t4, "az": c["az"],
                        "elev": c["elev"], "objetivo": objetivo}, ""
        return None, "ningún toque pasa la comprobación"

    def punto_comun(self, obj, objetos, q_actual, separacion=0.10, receptora=None, hacia=0.03):
        """Sitio de la zona donde las DOS manos pueden coger `obj`, lejos de los demás objetos,
        para cambiar de mano por la mesa (nivel 3) en vez de en el aire. Primero un filtro
        rápido (IK de las dos manos con alguna guiñada), luego la comprobación completa.
        Con `receptora`, preferido `hacia` m dentro de su lado: lo que el objeto se mueva al
        soltarlo (la pelota rueda 3-6 cm) no lo devuelve al lado de la que lo entrega.
        Devuelve (x, y) o None."""
        from dataclasses import replace
        z = self.est.zona
        otros = [o for o in objetos if o is not obj]
        cands = [(x, y) for x in np.arange(z.cx - z.fondo / 2 + 0.03, z.cx + z.fondo / 2 - 0.03, 0.02)
                 for y in np.arange(-0.12, 0.121, 0.03)
                 if all(np.hypot(o.centro[0] - x, o.centro[1] - y) >= separacion for o in otros)]
        y_pref = 0.0 if receptora is None else (hacia if receptora == "izq" else -hacia)
        cands.sort(key=lambda c: abs(c[1] - y_pref) + 0.5 * abs(c[0] - 0.32))   # centro y cerca del pecho
        ag = self.agarres[obj.clase]
        for x, y in cands:
            p = np.array([x, y, self.z_en(x, y) + ag.altura_agarre])
            rapido = all(any(C.ik(lado, p, C.rotacion_agarre(lado, "arriba", 0, az), off=self.off(lado, obj.clase),
                                  tol_pos=0.004)[3] for az in range(-75, 76, 25)) for lado in C.LADOS)
            if not rapido:
                continue
            fant = replace(obj, centro=np.array([x, y, obj.centro[2]]))
            otro = {"izq": "der", "der": "izq"}
            if all(self.agarre_ik(lado, fant, self.q_obs[lado], np.asarray(q_actual[otro[lado]], float))[0]
                   for lado in C.LADOS):
                return float(x), float(y)
        return None
