"""Marcadores para RViz (solo simulador, opcion --rviz de seguidor.py): lo que el
seguidor ve y piensa, encima del robot simulado.

  pista       (world)   la cinta real de la pista (del json del generador), en negro
  detectados  (pelvis)  puntos de linea de la ultima medida de percepcion, en verde
  memoria     (pelvis)  la linea recordada por la estimacion (cubre la zona ciega), en azul
  mira        (pelvis)  el punto de pure pursuit y la recta hasta el, en rojo
  texto       (pelvis)  estado del supervisor y ordenes

Los puntos del seguidor estan en el marco del robot (suelo bajo la pelvis): se
publican en `pelvis` con z = -altura de la pelvis.
"""
import json
import math

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

Z_PELVIS = 1.019     # pelvis sobre el suelo en la escena de la pista (de pie + 4 mm de la marcha cinematica)


def _color(r, g, b, a=1.0):
    return ColorRGBA(r=r, g=g, b=b, a=a)


class MarcadoresRviz:
    def __init__(self, nodo, ruta_pista=None):
        self.nodo = nodo
        self.pub = nodo.create_publisher(MarkerArray, "/seguidor/marcadores", 5)
        self.pista = None
        if ruta_pista:
            try:
                self.pista = json.load(open(ruta_pista))
            except OSError:
                pass
        self.n = 0

    def _m(self, ns, tipo, frame, stamp, color, escala, id_=0):
        m = Marker()
        m.header.frame_id, m.header.stamp = frame, stamp
        m.ns, m.id, m.type, m.action = ns, id_, tipo, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = escala
        m.color = color
        m.lifetime = Duration(sec=2)
        return m

    def _pista(self, stamp):
        out = []
        if self.pista is None:
            return out
        P, huecos = self.pista["linea"], self.pista.get("huecos", [])
        tramo, k = [], 0
        for x, y, _th, s in P[::4]:
            if any(a <= s <= b for a, b in huecos):
                if tramo:
                    m = self._m("pista", Marker.LINE_STRIP, "world", stamp, _color(0.05, 0.05, 0.05), 0.05, k)
                    m.points = tramo
                    out.append(m)
                    tramo, k = [], k + 1
                continue
            tramo.append(Point(x=x, y=y, z=0.002))
        if tramo:
            m = self._m("pista", Marker.LINE_STRIP, "world", stamp, _color(0.05, 0.05, 0.05), 0.05, k)
            m.points = tramo
            out.append(m)
        b = self.pista["barra"]
        th = b["rumbo_linea"] + math.pi / 2
        m = self._m("pista", Marker.LINE_LIST, "world", stamp, _color(0.05, 0.05, 0.05), 0.05, 999)
        m.points = [Point(x=b["x"] - 0.3 * math.cos(th), y=b["y"] - 0.3 * math.sin(th), z=0.002),
                    Point(x=b["x"] + 0.3 * math.cos(th), y=b["y"] + 0.3 * math.sin(th), z=0.002)]
        out.append(m)
        return out

    def publicar(self, medida, est_puntos, objetivo, estado, orden, confianza):
        self.n += 1
        stamp = self.nodo.get_clock().now().to_msg()
        arr = MarkerArray()
        if self.n % 10 == 1:
            arr.markers += self._pista(stamp)
        z = -Z_PELVIS + 0.01
        if medida is not None and len(medida.puntos):
            m = self._m("detectados", Marker.SPHERE_LIST, "pelvis", stamp, _color(0.0, 0.85, 0.0), 0.04)
            m.points = [Point(x=float(x), y=float(y), z=z) for x, y in medida.puntos]
            arr.markers.append(m)
        if len(est_puntos):
            m = self._m("memoria", Marker.POINTS, "pelvis", stamp, _color(0.1, 0.3, 1.0), 0.02)
            m.points = [Point(x=float(x), y=float(y), z=z + 0.005) for x, y in est_puntos]
            arr.markers.append(m)
        if objetivo is not None:
            m = self._m("mira", Marker.SPHERE, "pelvis", stamp, _color(1.0, 0.0, 0.0), 0.10)
            m.pose.position = Point(x=objetivo[0], y=objetivo[1], z=z)
            arr.markers.append(m)
            r = self._m("mira", Marker.LINE_LIST, "pelvis", stamp, _color(1.0, 0.0, 0.0, 0.7), 0.015, 1)
            r.points = [Point(x=0.0, y=0.0, z=z), Point(x=objetivo[0], y=objetivo[1], z=z)]
            arr.markers.append(r)
        t = self._m("texto", Marker.TEXT_VIEW_FACING, "pelvis", stamp, _color(0.1, 0.1, 0.1), 0.12)
        t.pose.position = Point(x=0.0, y=0.0, z=1.05)
        t.text = (f"{estado}  conf {confianza:.2f}\n"
                  f"vx {orden.vx:+.2f} m/s  vyaw {orden.vyaw:+.2f} rad/s" if not orden.parar else f"{estado}  (parado)")
        arr.markers.append(t)
        self.pub.publish(arr)
