"""Bloque de PERCEPCIÓN: Fotograma -> ListaObjetos en el marco del torso (sec. 5, 6.2).

Sobre la PROFUNDIDAD NATIVA (marco de IR1): con d = 10 cm el color solo ve la mitad lejana de
la zona y la profundidad nativa la ve entera (hito 1). Los cuatro objetos se distinguen por
su geometría; el color no hace falta (6.2.1, 6.2.2).

  1. Profundidad -> puntos -> torso_link con la calibración (única fuente de la pose de la
     cámara).
  2. Plano de la mesa: RANSAC con la normal cerca de la vertical y refinado por mínimos
     cuadrados (covarianza 3x3: nada de SVD sobre la nube entera, que en el seguidor costó
     5 GB). Lo que sobresale de 7 mm a 25 cm es candidato: separa los objetos de la mesa sin
     depender de la luz (pista de 6.2.1); las sombras del brazo no tienen altura (6.2.4).
  3. Autofiltro: fuera lo que está cerca de las cápsulas de los brazos (modelo de colisión de
     la tesis, con las juntas MEDIDAS del instante del fotograma; 6.2.4). Fuera las paredes
     de las bandejas y lo que cae fuera del tablero (geometría de la estación).
  4. Mapa de alturas en la mesa (celdas de 4 mm, la altura máxima de cada una) y
     componentes conexas: un objeto por componente.
  5. Identidad: altura, largo, ancho y "llenado" del rectángulo mínimo (1 caja, pi/4
     redondo), contra lo esperado por config/objetos.yaml; asignación húngara (cada objeto
     una vez) y confianza por la distancia al esperado y al segundo mejor.
  6. Centro y orientación en planta según la forma (6.2.3): caja -> rectángulo mínimo de la
     cara de arriba (guiñada módulo 90); vaso -> círculo de la cara de arriba; pelota ->
     ajuste de esfera de radio conocido (tesis: locate_sphere, 2.4 mm frente a 33 mm de la
     mediana); rotulador -> eje largo del rectángulo mínimo (módulo 180). La z del centro
     sale del plano + la mitad de la altura conocida.
  7. Fracción visible: huella vista / huella esperada (baja = tapado por la mano, 6.2.5).
"""
import math
import time

import cv2
import numpy as np

from . import cinematica as C
from .tipos import ListaObjetos, ObjetoDetectado

cv2.setNumThreads(1)
CM = C.CM
CELDA = 0.004


# llenado medido de una caja rasterizada en celdas de 4 mm (cubo 6 cm y prisma 4x4: 0.83-0.86);
# un círculo da 0.75-0.76 (pi/4). Con sigma 0.04 separan caja de cilindro: el prisma de 4x4x10
# y el vaso de 6x9.5 se parecen en todo lo demás
LLENADO_CAJA = 0.85
# fracción visible de la huella a partir de la cual la confianza no se rebaja (las escenas
# limpias dan 0.86-1.0; con la mano encima, 0.2-0.4)
VISIBLE_PLENO = 0.7


def _esperado(o):
    """(alto, largo, ancho, llenado, simetría rad, cúpula) esperados de un objeto del catálogo.
    llenado: área de la huella / rectángulo mínimo (0.85 una caja rasterizada en celdas de 4 mm
    y girada; pi/4 un círculo). cúpula: percentil 97 - percentil 70 de la altura de los puntos
    vistos desde arriba: 0 en una cara plana; en una esfera (área proyectada uniforme en el
    disco, el 30 % de arriba es rho^2 <= 0.3) 2R - R(1 + sqrt(0.7)) = 0.163 R; en un cilindro
    tumbado 0.046 R."""
    f = o["forma"]
    if f == "caja":
        a, b = sorted(o["medidas"][:2], reverse=True)
        return o["medidas"][2], a, b, LLENADO_CAJA, math.pi / 2 if abs(a - b) < 0.01 else math.pi, 0.0
    if f == "cilindro":
        return o["alto"], 2 * o["radio"], 2 * o["radio"], math.pi / 4, 0.0, 0.0
    if f == "esfera":
        return 2 * o["radio"], 2 * o["radio"], 2 * o["radio"], math.pi / 4, 0.0, 0.163 * o["radio"]
    return 2 * o["radio"], o["largo"], 2 * o["radio"], 0.9, math.pi, 0.046 * o["radio"]   # tumbado


def mascara_robot(P, q14, padding=0.025):
    """True = punto del propio robot (cápsulas de brazos, dedos, pulgar y torso de la tesis,
    como collision_model.robot_self_filter pero devolviendo la máscara)."""
    fl, fr = CM.arm_frames("left", q14[:7]), CM.arm_frames("right", q14[7:])
    caps = CM.arm_capsules("left", None, fl) + CM.arm_capsules("right", None, fr)
    for _, k, a, b, r in (CM.FINGER_CAPSULE, CM.THUMB_CAPSULE):
        for frs in (fl, fr):
            R, o = frs[k][:3, :3], frs[k][:3, 3]
            caps.append({"a": o + R @ a, "b": o + R @ b, "r": r})
    _, ta, tb, tr = CM.TORSO_CAPSULE
    caps.append({"a": ta, "b": tb, "r": tr})
    fuera = np.zeros(len(P), dtype=bool)
    for c in caps:
        d, _ = CM.points_to_segment(P, c["a"], c["b"])
        fuera |= d <= c["r"] + padding
    return fuera


def ajustar_plano(P, z0, rng, iters=80, tol=0.006):
    """RANSAC de un plano casi horizontal cerca de z0. Devuelve (n, d) con n hacia arriba."""
    cand = P[np.abs(P[:, 2] - z0) < 0.08]
    if len(cand) < 200:
        return None
    if len(cand) > 20000:
        cand = cand[rng.choice(len(cand), 20000, replace=False)]
    mejor, n_mejor = None, 0
    for _ in range(iters):
        a, b, c = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n[2] < 0:
            n = -n
        if n[2] < 0.97:
            continue
        d = -n @ a
        k = int(np.sum(np.abs(cand @ n + d) < tol))
        if k > n_mejor:
            mejor, n_mejor = (n, d), k
    if mejor is None:
        return None
    n, d = mejor
    inl = cand[np.abs(cand @ n + d) < tol]
    c = inl.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((inl - c).T))
    n = v[:, 0]
    if n[2] < 0:
        n = -n
    return n, float(-n @ c)


def ajustar_circulo(xy):
    """Círculo algebraico (Kåsa): (cx, cy, r)."""
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    s, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = s[0] / 2, s[1] / 2
    return cx, cy, math.sqrt(max(s[2] + cx ** 2 + cy ** 2, 0.0))


class Percepcion:
    def __init__(self, calib, objetos, estacion=None, padding_robot=0.025, h_min=0.010, area_min_cm2=4.0,
                 alto_min=0.015, semilla=0):
        self.T = np.asarray(calib["T_torso_cam"], dtype=float)
        self.objetos = objetos
        self.est = estacion
        self.padding = padding_robot
        self.h_min = h_min
        self.alto_min = alto_min        # el objeto más bajo (rotulador tumbado) mide 3.2 cm
        self.area_min = area_min_cm2 * 1e-4
        self.semilla = semilla          # RANSAC reproducible: mismo fotograma -> mismo resultado
        self.esp = {k: _esperado(o) for k, o in objetos.items()}
        self.z_mesa0 = (estacion.z_mesa - estacion.torso_z) if estacion is not None else -0.15
        self._uv = None
        # clases que el supervisor ya ha colocado: lo que esté dentro de SU bandeja es ese objeto y
        # no entra en el reparto (el cubo inclinado en su bandeja se parece al prisma y el reparto
        # alternativo cubo<->prisma casi no empeora: la confianza de los que quedan caía a 0.28)
        self.colocados = set()

    # -------------------------------------------------------------- nube
    def nube(self, prof, K, paso=2):
        h, w = prof.shape
        if self._uv is None or self._uv[0].shape != (h // paso, w // paso) or self._uv[2] != paso:
            v, u = np.mgrid[0:h:paso, 0:w:paso]
            self._uv = (u, v, paso)
        u, v, _ = self._uv
        z = prof[v, u]
        ok = np.isfinite(z) & (z > 0.2) & (z < 1.8)
        u, v, z = u[ok], v[ok], z[ok]
        Pc = np.column_stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        P = Pc @ self.T[:3, :3].T + self.T[:3, 3]
        return P, u, v

    def _fuera_estacion(self, P):
        """Paredes de las bandejas (bandas de 1 cm) y lo que cae fuera del tablero."""
        e = self.est
        if e is None:
            return np.zeros(len(P), dtype=bool)
        x, y = P[:, 0], P[:, 1]
        t = e.tablero
        fuera = ~t.contiene(x, y, 0.01)
        for b in e.bandejas.values():
            dentro = b.contiene(x, y, -0.008)
            interior = b.contiene(x, y, 0.012)
            fuera |= dentro & ~interior
        return fuera

    # -------------------------------------------------------------- principal
    @staticmethod
    def fusionar(fotogramas):
        """Varios fotogramas con el robot quieto -> uno: mediana por píxel de la profundidad
        (sin contar los huecos). Baja el ruido de la D435 ~sqrt(N) y rellena huecos."""
        if not isinstance(fotogramas, (list, tuple)):
            return fotogramas
        if len(fotogramas) == 1:
            return fotogramas[0]
        Z = np.stack([np.where(np.isfinite(f.prof) & (f.prof > 0), f.prof, np.nan) for f in fotogramas])
        with np.errstate(all="ignore"):
            z = np.nanmedian(Z, axis=0)
        f = fotogramas[-1]
        from dataclasses import replace
        return replace(f, prof=np.nan_to_num(z, nan=0.0).astype(np.float32))

    def __call__(self, f):
        t0 = time.perf_counter()
        f = self.fusionar(f)
        P, u, v = self.nube(f.prof, f.K_prof)
        roi = (P[:, 0] > 0.10) & (P[:, 0] < 0.90) & (np.abs(P[:, 1]) < 0.55) & \
              (P[:, 2] > self.z_mesa0 - 0.15) & (P[:, 2] < self.z_mesa0 + 0.30)
        P, u, v = P[roi], u[roi], v[roi]
        plano = ajustar_plano(P, self.z_mesa0, np.random.default_rng(self.semilla))
        if plano is None:
            return ListaObjetos(f.t, [], np.full(4, np.nan), 1000 * (time.perf_counter() - t0))
        n, d = plano
        h = P @ n + d
        cand = (h > self.h_min) & (h < 0.25)
        if f.q_brazos is not None:
            cand[cand] &= ~mascara_robot(P[cand], np.asarray(f.q_brazos), self.padding)
        cand[cand] &= ~self._fuera_estacion(P[cand])
        Pc, hc, uc, vc = P[cand], h[cand], u[cand], v[cand]
        objetos = self._agrupar(Pc, hc, uc, vc, n, d, f)
        return ListaObjetos(f.t, objetos, np.append(n, d), 1000 * (time.perf_counter() - t0))

    def _agrupar(self, P, h, u, v, n, d, f):
        if len(P) < 20:
            return []
        x0, y0 = P[:, 0].min() - 0.02, P[:, 1].min() - 0.02
        ix = ((P[:, 0] - x0) / CELDA).astype(int)
        iy = ((P[:, 1] - y0) / CELDA).astype(int)
        H = np.zeros((ix.max() + 3, iy.max() + 3), np.float32)
        np.maximum.at(H, (ix, iy), h.astype(np.float32))
        ocup = (H > 0).astype(np.uint8)
        ocup = cv2.morphologyEx(ocup, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        nlab, lab = cv2.connectedComponents(ocup, connectivity=8)
        grupos = []
        for k in range(1, nlab):
            celdas = np.argwhere(lab == k)
            if len(celdas) * CELDA ** 2 < self.area_min:
                continue
            sel = lab[ix, iy] == k
            if sel.sum() < 15:
                continue
            grupos.append(self._medir(P[sel], h[sel], u[sel], v[sel], celdas, x0, y0, n, d,
                                      H[celdas[:, 0], celdas[:, 1]]))
        # el ruido de la D435 (~4 mm a 1 m, correlado) hace manchas de 1 cm: un objeto es más alto
        grupos = [g for g in grupos if g is not None and g["alto"] >= self.alto_min]
        return self._identificar(grupos, f, n, d)

    def _medir(self, P, h, u, v, celdas, x0, y0, n, d, h_celdas=None):
        xy = np.column_stack([x0 + (celdas[:, 0] + 0.5) * CELDA, y0 + (celdas[:, 1] + 0.5) * CELDA])
        (cx, cy), (a, b), ang = cv2.minAreaRect(xy.astype(np.float32))
        largo, ancho = max(a, b) + CELDA, min(a, b) + CELDA
        area = len(celdas) * CELDA ** 2
        alto = float(np.percentile(h, 97))
        # cúpula sobre el mapa de alturas (lo más alto de cada celda = lo visto desde arriba):
        # con los puntos crudos, la cara lateral de un objeto alto (prisma de 10 cm) baja el
        # percentil 70 y parece una cúpula
        hc = h_celdas[h_celdas > 0] if h_celdas is not None and (h_celdas > 0).sum() >= 5 else h
        cupula = float(np.percentile(hc, 97) - np.percentile(hc, 70))
        return {"cupula": cupula, "P": P, "h": h, "u": u, "v": v, "xy": xy, "rect": ((cx, cy), (a, b), ang),
                "largo": largo,
                "ancho": ancho, "alto": alto, "llenado": area / max(largo * ancho, 1e-6), "area": area}

    def _identificar(self, grupos, f, n, d):
        from scipy.optimize import linear_sum_assignment
        nombres = list(self.objetos)
        sig = np.array([0.010, 0.015, 0.012, 0.08, 0.002])
        coste = np.zeros((len(grupos), len(nombres)))
        for i, g in enumerate(grupos):
            fg = np.array([g["alto"], g["largo"], g["ancho"], g["llenado"], g["cupula"]])
            for j, nm in enumerate(nombres):
                e = np.array(self.esp[nm][:4] + (self.esp[nm][5],))
                r = (fg - e) / sig
                # tapado (huella menor que la esperada) penaliza menos que sobrar
                r[1:3] = np.where(r[1:3] < 0, 0.5 * r[1:3], r[1:3])
                coste[i, j] = float(r @ r)
        # pelota frente a caja de tamaño parecido: residuo del ajuste de esfera de radio conocido
        # (~ruido en una pelota; las aristas y caras planas de un cubo lo suben a > 5 mm)
        esferas = [j for j, nm in enumerate(nombres) if self.objetos[nm]["forma"] == "esfera"]
        cajas = [j for j, nm in enumerate(nombres) if self.objetos[nm]["forma"] == "caja"]
        for i, g in enumerate(grupos):
            for je in esferas:
                if coste[i, je] > 40:
                    continue
                c = self._esfera(g, self.objetos[nombres[je]]["radio"], f)
                if c is None:
                    continue
                res = float(np.median(np.abs(np.linalg.norm(g["P"] - c, axis=1) - self.objetos[nombres[je]]["radio"])))
                g["res_esfera"] = res
                coste[i, je] += (max(res - 0.002, 0.0) / 0.0015) ** 2
                for jc in cajas:
                    coste[i, jc] += (max(0.004 - res, 0.0) / 0.0015) ** 2
        # lo ya colocado, dentro de su bandeja: fijo (uno por clase), fuera del reparto
        fijos = {}
        if self.est is not None:
            for c in self.colocados:
                if c not in nombres or c in fijos.values():
                    continue
                b = self.est.bandejas[self.objetos[c]["bandeja"]]
                for i, g in enumerate(grupos):
                    if i not in fijos and b.contiene(g["rect"][0][0], g["rect"][0][1], margen=-0.01):
                        fijos[i] = c
                        break
        libres = [j for j, nm in enumerate(nombres) if nm not in fijos.values()]
        # lo que no se parece a ninguna clase (la mano, una sombra con altura) no entra en el
        # reparto: si entrase, el óptimo global podría quitarle su etiqueta a un objeto real
        validos = [i for i in range(len(grupos)) if i not in fijos and libres and coste[i, libres].min() <= 60]
        coste_v = coste[np.ix_(validos, libres)] if validos else np.zeros((0, len(libres)))
        fv, cv = linear_sum_assignment(coste_v) if len(validos) else ([], [])
        cols = np.array(cv, dtype=int)
        filas = np.array([validos[k] for k in fv], dtype=int)
        asign = dict(zip(filas, [libres[k] for k in cols]))
        total = float(coste_v[fv, cols].sum()) if len(validos) else 0.0
        out = []
        for i, g in enumerate(grupos):
            if i in fijos:
                out.append(self._objeto(g, fijos[i], 1.0, n, d, f))
                continue
            j = asign.get(i)
            if j is None or coste[i, j] > 60:
                out.append(self._objeto(g, "desconocido", 0.0, n, d, f))
                continue
            # margen de la ASIGNACIÓN: cuánto empeora el coste total óptimo si este grupo no
            # fuera el objeto j (cada objeto aparece una vez: que el grupo se parezca a otro
            # objeto que ya encaja mejor en otro grupo no es una duda)
            c2 = coste_v.copy()
            c2[validos.index(i), libres.index(j)] = 1e6
            f2, k2 = linear_sum_assignment(c2)
            margen = float(c2[f2, k2].sum()) - total if len(libres) > 1 else 10.0
            conf = float(math.exp(-0.5 * coste[i, j] / 4.0) * (1 - math.exp(-max(margen, 0.0) / 4.0)))
            o = self._objeto(g, nombres[j], conf, n, d, f)
            # lo que no se ve no se mide: un objeto tapado (la mano encima) puede ser el único
            # candidato a su clase y tener margen, pero su centro sale de un trozo
            o.confianza = conf * min(1.0, o.fraccion_visible / VISIBLE_PLENO)
            out.append(o)
        return out

    def _z_mesa(self, x, y, n, d):
        return -(d + n[0] * x + n[1] * y) / n[2]

    def _objeto(self, g, clase, conf, n, d, f):
        P, h = g["P"], g["h"]
        (cx, cy), (a, b), ang = g["rect"]
        gui = math.radians(ang) + (0.0 if a >= b else math.pi / 2)     # dirección del lado largo
        o = self.objetos.get(clase)
        alto_obj = self.esp[clase][0] if o else g["alto"]
        sim = self.esp[clase][4] if o else 0.0
        forma = o["forma"] if o else None
        if forma == "caja":
            top = h > g["alto"] - 0.012
            if top.sum() >= 10:
                (cx, cy), (a2, b2), ang2 = cv2.minAreaRect(P[top, :2].astype(np.float32))
                gui = math.radians(ang2) + (0.0 if a2 >= b2 else math.pi / 2)    # el lado LARGO
        elif forma == "cilindro":
            top = h > g["alto"] - 0.010
            if top.sum() >= 10:
                cx, cy, _ = ajustar_circulo(P[top, :2])
        elif forma == "esfera":
            c = self._esfera(g, o["radio"], f)
            if c is not None:
                cx, cy = c[0], c[1]
        z = self._z_mesa(cx, cy, n, d) + alto_obj / 2
        if sim > 0:
            gui = math.remainder(gui, sim)
        else:
            gui = 0.0
        esp = self.esp.get(clase)
        huella = (esp[1] * esp[2] * (esp[3] if esp[3] < 1 else 1.0)) if esp else g["area"]
        return ObjetoDetectado(clase=clase, centro=np.array([cx, cy, z]), guinada=float(gui), simetria=float(sim),
                               medidas=np.array([g["largo"], g["ancho"], g["alto"]]), confianza=conf,
                               fraccion_visible=float(min(1.0, g["area"] / max(huella, 1e-6))),
                               n_puntos=int(len(P)))

    def _esfera(self, g, radio, f):
        """Ajuste de esfera de radio conocido (tesis) sobre los píxeles del grupo."""
        from h1_2_algoritms.vision.fruit_localization import Intrinsics, locate_sphere
        K = f.K_prof
        mask = np.zeros(f.prof.shape, np.uint8)
        mask[g["v"], g["u"]] = 1
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        prof = np.nan_to_num(f.prof, nan=0.0)
        c = locate_sphere(mask, prof, Intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2]), radio, erode_px=1)
        if c is None or not np.all(np.isfinite(c)):
            return None
        return self.T[:3, :3] @ c + self.T[:3, 3]


def cambios(antes, ahora, tol_pos=0.02, tol_gui=math.radians(10)):
    """Objetos que se movieron entre dos listas (nivel 3, 6.2.6): {clase: (dpos m, dguiñada rad)}."""
    a = {o.clase: o for o in antes.objetos if o.clase != "desconocido"}
    out = {}
    for o in ahora.objetos:
        p = a.get(o.clase)
        if p is None:
            continue
        dp = float(np.linalg.norm(o.centro[:2] - p.centro[:2]))
        dg = abs(math.remainder(o.guinada - p.guinada, o.simetria)) if o.simetria > 0 else 0.0
        if dp > tol_pos or dg > tol_gui:
            out[o.clase] = (dp, dg)
    return out
