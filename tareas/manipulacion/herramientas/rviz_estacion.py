#!/usr/bin/env python3
"""La estación del reto en RViz (solo simulador): mesa, zona de recogida, bandejas A-D,
marcadores ArUco del tablero, dianas o cruz del nivel, y los objetos donde están de verdad
(/sim/ground_truth/<objeto>), sobre el robot de /robot_description.

    python3 herramientas/rviz_estacion.py sim/estaciones/estacion_n3_s0.json

Publica /manip/estacion (MarkerArray, en `world`): la geometría fija una vez por segundo
y los objetos a 10 Hz. No necesita tf2 en Python (se evita: en el seguidor ahogaba el
GIL con robot_state_publisher en marcha).
"""
import json
import sys

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def color(r, g, b, a=1.0):
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


class Estacion(Node):
    def __init__(self, ruta):
        super().__init__("rviz_estacion")
        self.geo = json.load(open(ruta))
        self.pub = self.create_publisher(MarkerArray, "/manip/estacion", 2)
        self.poses = {}
        for n in self.geo["objetos"]:
            self.create_subscription(PoseStamped, f"/sim/ground_truth/{n}",
                                     lambda m, n=n: self.poses.__setitem__(n, m.pose), 5)
        self.fijos = self._fijos()
        self.create_timer(1.0, lambda: self.pub.publish(MarkerArray(markers=self.fijos)))
        self.create_timer(0.1, self._objetos)

    def _m(self, ns, k, tipo, rgba):
        m = Marker()
        m.header.frame_id = "world"
        m.ns, m.id, m.type, m.action = ns, k, tipo, Marker.ADD
        m.color = color(*rgba)
        m.pose.orientation.w = 1.0
        return m

    def _caja(self, ns, k, cx, cy, cz, sx, sy, sz, rgba):
        m = self._m(ns, k, Marker.CUBE, rgba)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(cx), float(cy), float(cz)
        m.scale.x, m.scale.y, m.scale.z = float(sx), float(sy), float(sz)
        return m

    def _fijos(self):
        e = self.geo["estacion"]
        zm = e["z_mesa"]
        t = e["tablero"]
        out = [self._caja("mesa", 0, t["cx"], t["cy"], zm - 0.015, t["fondo"], t["ancho"], 0.03, (0.9, 0.9, 0.88, 1))]
        z = e["zona"]
        lin = self._m("zona", 0, Marker.LINE_STRIP, (0.4, 0.4, 0.4, 1))
        lin.scale.x = 0.006
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
            lin.points.append(Point(x=z["cx"] + dx * z["fondo"] / 2, y=z["cy"] + dy * z["ancho"] / 2, z=zm + 0.001))
        out.append(lin)
        for k, (letra, b) in enumerate(e["bandejas"].items()):
            out.append(self._caja("bandejas", k, b["cx"], b["cy"], zm + 0.01, b["fondo"], b["ancho"], 0.02,
                                  (0.3, 0.32, 0.34, 0.8)))
            tx = self._m("letras", k, Marker.TEXT_VIEW_FACING, (0.1, 0.1, 0.1, 1))
            tx.pose.position.x, tx.pose.position.y, tx.pose.position.z = b["cx"], b["cy"], zm + 0.06
            tx.scale.z, tx.text = 0.05, letra
            out.append(tx)
        for k, a in enumerate(x for x in self.geo["aruco"] if x["cuerpo"] == "world"):
            out.append(self._caja("aruco", k, a["centro"][0], a["centro"][1], zm + 0.001, 0.04, 0.04, 0.002,
                                  (0.05, 0.05, 0.05, 1)))
        for k, (x, y, zz) in enumerate(self.geo["dianas"]):
            m = self._m("dianas", k, Marker.CYLINDER, (1.0, 0.45, 0.05, 1))
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, zz + 0.001
            m.scale.x, m.scale.y, m.scale.z = 0.02, 0.02, 0.002
            out.append(m)
        if self.geo.get("cruz"):
            x, y, zz = self.geo["cruz"]
            out += [self._caja("cruz", 0, x, y, zz + 0.001, 0.10, 0.008, 0.002, (0, 0, 0, 1)),
                    self._caja("cruz", 1, x, y, zz + 0.001, 0.008, 0.10, 0.002, (0, 0, 0, 1))]
        return out

    def _objetos(self):
        out = []
        for k, (n, o) in enumerate(self.geo["objetos"].items()):
            p = self.poses.get(n)
            if p is None:
                continue
            med = o["medidas"]
            forma = o["forma"]
            tipo = {"caja": Marker.CUBE, "esfera": Marker.SPHERE}.get(forma, Marker.CYLINDER)
            m = self._m("objetos", k, tipo, (0.9, 0.5, 0.2, 1))
            m.pose = p
            if forma == "caja":
                m.scale.x, m.scale.y, m.scale.z = med["medidas"]
            elif forma == "esfera":
                m.scale.x = m.scale.y = m.scale.z = 2 * med["radio"]
            elif forma == "cilindro":
                m.scale.x = m.scale.y = 2 * med["radio"]
                m.scale.z = med["alto"]
            else:
                m.scale.x = m.scale.y = 2 * med["radio"]
                m.scale.z = med["largo"]
            out.append(m)
            tx = self._m("nombres", k, Marker.TEXT_VIEW_FACING, (0.1, 0.1, 0.1, 1))
            tx.pose.position.x, tx.pose.position.y = p.position.x, p.position.y
            tx.pose.position.z = p.position.z + 0.08
            tx.scale.z, tx.text = 0.03, f"{n} -> {o['bandeja']}"
            out.append(tx)
        if out:
            self.pub.publish(MarkerArray(markers=out))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rclpy.init()
    nodo = Estacion(sys.argv[1])
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
