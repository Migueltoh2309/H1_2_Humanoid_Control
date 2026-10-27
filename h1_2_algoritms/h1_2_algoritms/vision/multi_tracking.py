"""Percepción de VARIAS frutas sobre la faja (fase B del LfD).

    RGB-D -> un blob naranja por fruta -> ajuste de esfera por blob
          -> MultiFruitTracker: una pista (FruitTracker, Kalman de velocidad
             constante) por fruta, asociación por vecino más cercano

Las pistas son los "objetos" del estado simbólico que usa la política de
alto nivel (aprendizaje/alto_nivel.py): posición, velocidad e identidad
estable mientras la fruta viaja por la faja.
"""
import numpy as np

from h1_2_algoritms.vision.fruit_localization import segment_orange_all, locate_sphere, MANDARINA_RADIUS
from h1_2_algoritms.vision.visual_servoing import FruitTracker

# Superficie útil de la faja en torso_link: la MISMA x, z que
# GraspServoing.belt_box (una pista fuera de esa caja haría que el agarre
# se aborte al instante), con todo el largo de la faja en y.
BELT_BOX = {"x": (0.26, 0.47), "y": (-0.60, 0.60), "z": (-0.20, -0.12)}


def detect_fruits(rgb, depth, K, R, p, min_visible=0.35):
    """Centros (torso_link) de todas las frutas visibles. Como en el demo de
    una fruta, se descarta un blob con < 35% del área esperada (con la mano
    encima, la silueta parcial sesga la esfera)."""
    out = []
    for mask in segment_orange_all(rgb):
        c = locate_sphere(mask, depth, K)
        if c is None or c[2] <= 0:
            continue
        area_exp = np.pi * (K.fx * MANDARINA_RADIUS / c[2]) ** 2
        if mask.sum() / 255 < min_visible * area_exp:
            continue
        out.append(R @ c + p)
    return out


def on_belt(p, box=BELT_BOX):
    return all(box[a][0] <= p[k] <= box[a][1] for k, a in enumerate("xyz"))


class Track:
    def __init__(self, tid, kf):
        self.id = tid
        self.kf = kf
        self.t_last = kf.t
        self.hist = [(kf.t, kf.x[0:3].copy())]     # medidas aceptadas (t, z)


class MultiFruitTracker:
    """Pistas de frutas sobre la faja.

    * Asociación: pares (pista, detección) por distancia creciente, con
      compuerta `gate` + 3σ de la predicción (como FruitTracker).
    * Pista nueva: solo con una detección SOBRE la faja (una fruta en la
      mano no abre pistas). Arranca con la velocidad media de las pistas
      maduras: la faja lleva a todas igual.
    * Baja: sin medidas por `max_age` s (el brazo puede tapar una fruta
      ~4 s), o predicha fuera de la faja.
    * Velocidad COMPARTIDA (`shared_velocity`, por omisión): la faja lleva
      a todas las frutas a la misma velocidad constante, así que se estima
      una sola, por mínimos cuadrados sobre la historia de todas las pistas
      (`belt_velocity`), y la posición de cada pista es el promedio de sus
      medidas del último `pos_window` s llevadas al instante pedido con esa
      velocidad. El Kalman por pista (σ_acc chico) se dejaba arrastrar por
      las oclusiones parciales (la fruta pasa bajo un brazo: la esfera se
      corre atrás y luego adelante) y subestimaba la velocidad ~10%
      (medido: −5 mm/s de media, −10 a −13 en los agarres fallidos, 15-30 mm
      de atraso al cerrar tras un alcance de 3 s). El Kalman queda para la
      asociación.
    """

    def __init__(self, gate=0.05, max_age=5.0, min_updates_mature=8, shared_velocity=True,
                 vel_window=8.0, pos_window=1.0, min_span=1.0):
        self.gate = gate
        self.max_age = max_age
        self.min_mature = min_updates_mature
        self.shared_velocity = shared_velocity
        self.vel_window = vel_window
        self.pos_window = pos_window
        self.min_span = min_span
        self.tracks = {}
        self._next = 0

    def remove(self, tid):
        self.tracks.pop(tid, None)

    def belt_velocity(self):
        """Velocidad común de las frutas: pendiente de mínimos cuadrados con
        una ordenada por pista, sobre las medidas del último `vel_window` s
        de las pistas que cubren ≥ `min_span` s. Sin datos suficientes, la
        media de los Kalman maduros (o None)."""
        num, den = np.zeros(3), 0.0
        for tr in self.tracks.values():
            h = [(t, z) for t, z in tr.hist if t >= tr.hist[-1][0] - self.vel_window] if tr.hist else []
            if len(h) < 3 or h[-1][0] - h[0][0] < self.min_span:
                continue
            T = np.array([t for t, _ in h])
            Z = np.array([z for _, z in h])
            dT = T - T.mean()
            num += dT @ (Z - Z.mean(axis=0))
            den += dT @ dT
        if den > 0:
            return num / den
        v = [tr.kf.x[3:6] for tr in self.tracks.values() if tr.kf.n_updates >= self.min_mature]
        return np.mean(v, axis=0) if v else None

    def update(self, dets, t, exclude=(), exclude_radius=0.08, occluders=(), occlusion_radius=0.10):
        """dets: centros en torso_link medidos en el instante de captura t.
        exclude: puntos (p.ej. el punto de agarre de una mano que está
        cerrando o levantando) cerca de los cuales una detección no abre
        pista nueva: es la fruta que esa mano ya tomó.
        occluders: puntos de agarre de las manos que están alcanzando. Una
        detección a < occlusion_radius se asocia a su pista pero NO la
        corrige: la mano tapa parte de la fruta y la esfera ajustada a lo
        visible se corre hacia atrás (medido: la estimación iba 16 mm
        atrasada al cerrar en los agarres fallidos del brazo izquierdo). Con
        la faja a velocidad constante, la predicción vale más que esa
        medida."""
        dets = [np.asarray(z) for z in dets]
        pairs = []
        for tid, tr in self.tracks.items():
            p, _ = tr.kf.predict(t)
            sig = np.sqrt(np.max(np.diag(tr.kf.P[0:3, 0:3])))
            for j, z in enumerate(dets):
                dist = np.linalg.norm(z - p)
                if dist < self.gate + 3 * sig:
                    pairs.append((dist, tid, j))
        used_t, used_d = set(), set()
        for dist, tid, j in sorted(pairs):
            if tid in used_t or j in used_d:
                continue
            if any(np.linalg.norm(dets[j] - o) < occlusion_radius for o in occluders):
                self.tracks[tid].t_last = t
                used_t.add(tid)
                used_d.add(j)
                continue
            tr = self.tracks[tid]
            if tr.kf.update(dets[j], t):
                tr.t_last = t
                if tr.kf.n_updates == 1:        # el Kalman se reinició: la fruta saltó
                    tr.hist = []
                tr.hist.append((t, dets[j].copy()))
                tr.hist = [(ti, z) for ti, z in tr.hist if ti >= t - self.vel_window]
                used_t.add(tid)
                used_d.add(j)
        v0 = self.belt_velocity()
        for j, z in enumerate(dets):
            if j in used_d or not on_belt(z):
                continue
            if any(np.linalg.norm(z - e) < exclude_radius for e in exclude):
                continue
            kf = FruitTracker()
            kf.update(z, t)
            if v0 is not None:
                kf.x[3:6] = v0
            self.tracks[self._next] = Track(self._next, kf)
            self._next += 1
        self.prune(t)

    def prune(self, t):
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            p, _ = tr.kf.predict(t)
            if t - tr.t_last > self.max_age or not on_belt(p, {**BELT_BOX, "z": (-0.30, 0.0)}):
                del self.tracks[tid]

    def estimates(self, t, min_updates=3):
        """{id: (p, v)} predichos al instante t, de las pistas confirmadas."""
        out = {}
        vb = self.belt_velocity() if self.shared_velocity else None
        for tid, tr in self.tracks.items():
            if tr.kf.n_updates < min_updates:
                continue
            if vb is None or not tr.hist:
                out[tid] = tr.kf.predict(t)
                continue
            h = [(ti, z) for ti, z in tr.hist if ti >= tr.hist[-1][0] - self.pos_window]
            p = np.mean([z + vb * (t - ti) for ti, z in h], axis=0)
            out[tid] = (p, vb.copy())
        return out
