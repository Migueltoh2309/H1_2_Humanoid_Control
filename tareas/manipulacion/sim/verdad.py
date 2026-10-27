"""Verdad de terreno del simulador para EVALUAR una tirada (no la usa el programa para decidir):
poses de los objetos, la pelvis y los pies desde /sim/ground_truth/<cuerpo>. El programa
principal la guarda en el registro al empezar, en cada cambio de estado (máximo de pies) y al
acabar; herramientas/metricas.py calcula con ella las métricas de la sección 9."""
import math
import threading

import numpy as np

PIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def guinada(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Verdad:
    def __init__(self, nodo_ros, objetos):
        from geometry_msgs.msg import PoseStamped
        self.cerrojo = threading.Lock()
        self.ult = {}
        self.pies0 = None
        self.pies_max = {p: 0.0 for p in PIES}
        for b in list(objetos) + ["pelvis", *PIES]:
            nodo_ros.create_subscription(PoseStamped, f"/sim/ground_truth/{b}", lambda m, b=b: self._al(b, m), 5)

    def _al(self, b, m):
        p, q = m.pose.position, m.pose.orientation
        pos, quat = np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z])
        with self.cerrojo:
            self.ult[b] = (pos, quat)
            if b in PIES:
                if self.pies0 is None or b not in self.pies0:
                    self.pies0 = dict(self.pies0 or {}, **{b: pos.copy()})
                self.pies_max[b] = max(self.pies_max[b], float(np.linalg.norm(pos[:2] - self.pies0[b][:2])))

    def foto(self):
        """{cuerpo: {pos, guinada_deg, inclinacion_deg}} y el desplazamiento máximo de cada pie."""
        with self.cerrojo:
            out = {}
            for b, (p, q) in self.ult.items():
                w, x, y, z = q
                zz = 1 - 2 * (x * x + y * y)               # componente z del eje z del cuerpo
                out[b] = {"pos": p.tolist(), "guinada_deg": math.degrees(guinada(q)),
                          "inclinacion_deg": math.degrees(math.acos(min(1.0, abs(zz))))}
            return {"cuerpos": out, "pies_max_m": dict(self.pies_max)}
