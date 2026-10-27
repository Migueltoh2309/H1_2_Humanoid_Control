"""El instructor del nivel 3, SOLO en el simulador (prueba del programa, no parte de él): mientras
el robot transporta un objeto, desplaza o gira otro entre 5 y 10 cm (sec. 4), con la verdad de
terreno del simulador (/sim/ground_truth/<cuerpo>) y /sim/poner_cuerpo. El programa no lo sabe:
lo tiene que ver en la siguiente observación.

El supervisor lo llama en cada cambio de estado (`al_estado`); actúa la primera vez que ve
TRANSPORTAR (o las `veces` primeras).
"""
import math
import threading

import numpy as np


class Instructor:
    def __init__(self, nodo_ros, objetos, zona, log=print, veces=1, semilla=0, destinos=None, holgura=0.10):
        from geometry_msgs.msg import PoseStamped
        self.nodo, self.objetos, self.zona, self.log = nodo_ros, list(objetos), zona, log
        self.veces, self.rng = veces, np.random.default_rng(semilla)
        self.gt, self.cerrojo = {}, threading.Lock()
        self.PoseStamped = PoseStamped
        for o in self.objetos:
            nodo_ros.create_subscription(PoseStamped, f"/sim/ground_truth/{o}", lambda m, o=o: self._gt(o, m), 2)
        self.pub = nodo_ros.create_publisher(PoseStamped, "/sim/poner_cuerpo", 2)
        self.hechos = []
        # el instructor no pone nada en el camino del brazo: lejos (holgura) del recorrido del
        # objeto transportado, de donde estaba a su destino (centro de su bandeja)
        self.destinos, self.holgura = destinos or {}, holgura
        self.inicio = {}

    def _gt(self, o, m):
        p, q = m.pose.position, m.pose.orientation
        with self.cerrojo:
            self.gt[o] = (np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z]))

    def _lejos(self, p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
        return np.linalg.norm(p - (a + t * ab)) >= self.holgura

    def al_estado(self, estado, objeto, terminados=()):
        if estado == "APROXIMAR" and objeto in self.gt:
            self.inicio[objeto] = self.gt[objeto][0][:2].copy()
        if estado != "TRANSPORTAR" or len(self.hechos) >= self.veces:
            return
        a = self.inicio.get(objeto)
        b = self.destinos.get(objeto)
        with self.cerrojo:
            libres = {o: v for o, v in self.gt.items() if o != objeto and o not in terminados
                      and self.zona.contiene(v[0][0], v[0][1], margen=0.03)
                      and (a is None or b is None or self._lejos(v[0][:2], a, b))}
        if not libres:
            return
        o = list(libres)[int(self.rng.integers(len(libres)))]
        p, q = libres[o]
        for _ in range(50):
            d = self.rng.uniform(0.05, 0.10)
            ang = self.rng.uniform(0, 2 * math.pi)
            p2 = p + np.array([d * math.cos(ang), d * math.sin(ang), 0.0])
            if self.zona.contiene(p2[0], p2[1], margen=0.04) and \
                    (a is None or b is None or self._lejos(p2[:2], a, b)):
                break
        else:
            return
        giro = self.rng.uniform(-0.6, 0.6)               # y a veces lo gira
        qz = np.array([math.cos(giro / 2), 0, 0, math.sin(giro / 2)])
        q2 = _qmul(qz, q)
        m = self.PoseStamped()
        m.header.frame_id = o
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p2[0]), float(p2[1]), float(p2[2] + 0.002)
        m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = (float(v) for v in q2)
        self.pub.publish(m)
        self.hechos.append({"objeto": o, "desde": p.tolist(), "hasta": p2.tolist(), "giro_deg": math.degrees(giro)})
        self.log(f"  [instructor] mueve {o} {100 * np.linalg.norm(p2 - p):.1f} cm y lo gira "
                 f"{math.degrees(giro):.0f} grados")


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
