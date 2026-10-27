#!/usr/bin/env python3
"""SOLO SIMULADOR: graba un vídeo de una tirada en la estación, para los entregables de los
hitos. Compone, a 1280x720 y en tiempo real:

  izquierda   el robot y la mesa en MuJoCo (render propio offscreen en la GPU: no toca el
              simulador ni su visor), con el estado del supervisor, el objeto y el peso de
              arm_sdk sobreimpresos (del lazo.csv que va escribiendo el programa)
  abajo       la consola de lo que se está ejecutando (--consola fichero)
  derecha     la imagen en color de la D435 y una vista cenital de la mesa

Con --solo-mujoco el cuadro entero es la vista de MuJoCo, como si se grabara solo su visor.

Lee los tópicos del simulador (/sim/joint_states, /sim/ground_truth/<cuerpo>,
/camera/color/image_raw, /sim/tapar_camara) y el registro más reciente de datos/tiradas.

    python3 herramientas/grabar_video.py --nivel 2 --semilla 0 --salida videos/x.mp4 \\
        --titulo "Hito 3" --consola /tmp/consola.log

Termina con Ctrl+C / SIGTERM (o --duracion).
"""
import argparse
import csv
import glob
import os
import signal
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2                     # noqa: E402
import numpy as np             # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(AQUI, "..")
sys.path.insert(0, AQUI)
from video_comun import (ANCHO, ALTO, COLOR_ESTADO, GRIS, EscritorVideo, Lienzo,   # noqa: E402
                         ultimas_lineas)

# disposición
TIT_H = 48
VW, VH = 848, 477                  # vista 3D
DER_X, DER_W = VW, ANCHO - VW      # columna derecha (432)
CAM_H = 324                        # D435 (640x480 -> 432x324)
CEN_Y, CEN_H = TIT_H + CAM_H, ALTO - TIT_H - CAM_H


class Topicos:
    def __init__(self):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Bool
        rclpy.init()
        self.rclpy, self.PoseStamped = rclpy, PoseStamped
        self.nodo = rclpy.create_node("grabar_video_manip")
        self.js = self.color = None
        self.poses = {}
        self.tapada = False
        n = self.nodo
        n.create_subscription(JointState, "/sim/joint_states", lambda m: setattr(self, "js", m), 5)
        n.create_subscription(Image, "/camera/color/image_raw", self._color, qos_profile_sensor_data)
        n.create_subscription(Bool, "/sim/tapar_camara", lambda m: setattr(self, "tapada", m.data), 5)
        threading.Thread(target=self._girar, daemon=True).start()

    def seguir(self, cuerpo):
        self.nodo.create_subscription(self.PoseStamped, f"/sim/ground_truth/{cuerpo}",
                                      lambda m, c=cuerpo: self.poses.__setitem__(c, m), 5)

    def _girar(self):
        try:
            self.rclpy.spin(self.nodo)
        except Exception:
            pass

    def _color(self, m):
        if m.encoding in ("rgb8", "bgr8"):
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            self.color = img[..., ::-1].copy() if m.encoding == "bgr8" else img.copy()


class Vista3D:
    """El modelo de la estación con las juntas y las poses de los cuerpos libres del simulador."""

    def __init__(self, modelo):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(modelo)
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, ANCHO)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, ALTO)
        self.d = mujoco.MjData(self.m)
        if self.m.nkey:
            self.d.qpos[:] = self.m.key_qpos[0]
        self.rend = {}
        self.adr = {self.m.joint(j).name: self.m.jnt_qposadr[j] for j in range(self.m.njnt)}
        # cuerpos con junta libre (pelvis y objetos): su pose viene de /sim/ground_truth/<cuerpo>
        self.libres = {}
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.libres[self.m.body(self.m.jnt_bodyid[j]).name] = self.m.jnt_qposadr[j]
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0               # la losa de colisión bajo el tablero, invisible

    def actualizar(self, js, poses):
        if js is not None:
            for nombre, q in zip(js.name, js.position):
                a = self.adr.get(nombre)
                if a is not None:
                    self.d.qpos[a] = q
        for cuerpo, m in poses.items():
            a = self.libres.get(cuerpo)
            if a is not None:
                p, o = m.pose.position, m.pose.orientation
                self.d.qpos[a:a + 7] = [p.x, p.y, p.z, o.w, o.x, o.y, o.z]
        self.mj.mj_forward(self.m, self.d)

    def render(self, w, h, lookat, distancia, azimut, elevacion):
        if (w, h) not in self.rend:
            self.rend[(w, h)] = self.mj.Renderer(self.m, height=h, width=w)
        cam = self.mj.MjvCamera()
        cam.type = self.mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distancia, azimut, elevacion
        r = self.rend[(w, h)]
        r.update_scene(self.d, camera=cam, scene_option=self.opt)
        return r.render()


class Lazo:
    """La última fila del lazo.csv de la tirada en curso: estado, objeto y peso de arm_sdk."""

    def __init__(self, desde):
        self.desde, self.ruta, self.fila = desde, None, None

    def leer(self):
        if self.ruta is None:
            dirs = [d for d in glob.glob(os.path.join(RAIZ, "datos", "tiradas", "nivel*"))
                    if os.path.getmtime(d) >= self.desde - 1]
            if not dirs:
                return None
            r = os.path.join(max(dirs, key=os.path.getmtime), "lazo.csv")
            if not os.path.isfile(r):
                return None
            with open(r) as f:
                self.cab = next(csv.reader(f), None)
            if not self.cab:                  # aún sin la cabecera (el programa la escribe con búfer)
                return None
            self.ruta = r
        try:
            with open(self.ruta, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4000))
                lineas = f.read().decode("utf-8", "replace").splitlines()
            for s in reversed(lineas[1:]):
                v = next(csv.reader([s]))
                if self.cab and len(v) == len(self.cab):
                    self.fila = dict(zip(self.cab, v))
                    break
        except OSError:
            pass
        return self.fila


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nivel", type=int, required=True)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--titulo", default="")
    ap.add_argument("--consola", default=None)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--duracion", type=float, default=1800.0)
    ap.add_argument("--solo-mujoco", action="store_true",
                    help="el cuadro entero es la vista de MuJoCo, sin paneles ni rótulos")
    ap.add_argument("--orbita", type=float, default=0.0, metavar="GRADOS_S",
                    help="con --solo-mujoco: la cámara gira alrededor de la estación")
    ap.add_argument("--camara", default=None, metavar="X,Y,Z,DIST,AZ,ELEV",
                    help="con --solo-mujoco: punto mirado, distancia, azimut y elevación")
    ap.add_argument("--simulacro", action="store_true",
                    help="la tirada es un simulacro: arm_sdk no se publica (no se muestra su peso)")
    a = ap.parse_args()

    # la vista del visor de MuJoCo: desde delante y a la izquierda del robot, mirando a la mesa
    cam_solo = [float(v) for v in a.camara.split(",")] if a.camara else [0.25, 0.0, 1.05, 2.1, 215.0, -24.0]
    vista = Vista3D(os.path.join(RAIZ, "sim", "estaciones", f"estacion_n{a.nivel}_s{a.semilla}.xml"))
    top = Topicos()
    for c in vista.libres:
        top.seguir(c)
    lazo = Lazo(time.time())
    vid = EscritorVideo(a.salida, a.fps)

    fin = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: fin.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: fin.__setitem__("v", True))
    t0 = time.monotonic()
    while "pelvis" not in top.poses and time.monotonic() - t0 < 30 and not fin["v"]:
        time.sleep(0.1)
    print(f"grabar_video: grabando {a.salida}", flush=True)
    t0 = time.monotonic()
    n = 0
    consola, fila = [], None
    salida_segura = None
    while not fin["v"] and time.monotonic() - t0 < a.duracion:
        t = time.monotonic() - t0
        vista.actualizar(top.js, dict(top.poses))
        if n % 3 == 0:
            fila = lazo.leer()
        if a.consola and n % 5 == 0:
            consola = ultimas_lineas(a.consola, 8, 92)
            for s in consola:
                if "SALIDA SEGURA" in s:
                    salida_segura = s.split("SALIDA SEGURA:")[-1].strip()

        if a.solo_mujoco:
            x, y, z, dist, az, el = cam_solo
            frame = vista.render(ANCHO, ALTO, (x, y, z), dist, az + a.orbita * t, el)
            debe = int((time.monotonic() - t0) * a.fps) + 1
            vid.escribir(frame, veces=max(1, debe - n))
            n = vid.n
            time.sleep(max(0.0, (n / a.fps) - (time.monotonic() - t0)))
            continue
        L = Lienzo()
        # ---- vista 3D: desde delante y a la izquierda del robot, mirando a la mesa
        L.pegar(0, TIT_H, vista.render(VW, VH, (0.28, 0.0, 1.02), 1.9, 215.0, -28.0))
        # ---- título
        L.rect(0, 0, ANCHO, TIT_H, (16, 18, 22))
        L.texto(16, 12, a.titulo, 22, negrita=True)
        L.texto(ANCHO - 16, 14, f"t = {t:5.1f} s", 18, GRIS, mono=True, ancla="ra")
        # ---- estado del supervisor, objeto y peso de arm_sdk
        if fila:
            e = fila.get("estado") or "?"
            col = COLOR_ESTADO.get(e, (120, 120, 120))
            w = int(L.ancho_texto(e, 22, True)) + 28
            L.rect(14, TIT_H + 14, 14 + w, TIT_H + 50, col)
            L.texto(28, TIT_H + 20, e, 22, negrita=True)
            L.rect(14, TIT_H + 56, 330, TIT_H + 116, (0, 0, 0), alfa=0.55)
            obj = fila.get("objeto") or "-"
            L.texto(26, TIT_H + 62, f"objeto: {obj}", 17, mono=True)
            try:
                peso = float(fila.get("peso") or 0.0)
            except ValueError:
                peso = 0.0
            if a.simulacro:                   # el peso del lazo es el que se publicaría
                L.texto(26, TIT_H + 88, "arm_sdk NO se publica", 17, (255, 210, 120), mono=True)
            else:
                L.texto(26, TIT_H + 88, f"peso arm_sdk {peso:4.2f}", 17, mono=True)
                L.rect(200, TIT_H + 92, 320, TIT_H + 106, (60, 60, 70))
                L.rect(200, TIT_H + 92, 200 + int(120 * max(0.0, min(1.0, peso))), TIT_H + 106, (40, 170, 90))
        if salida_segura:
            L.rect(14, TIT_H + VH - 56, VW - 14, TIT_H + VH - 14, (220, 50, 50))
            L.texto(VW // 2, TIT_H + VH - 48, f"SALIDA SEGURA · {salida_segura.replace('Parada: ', '')[:58]}",
                    18, negrita=True, ancla="ma")
        if top.tapada:
            L.rect(VW - 300, TIT_H + 14, VW - 14, TIT_H + 54, (220, 50, 50))
            L.texto(VW - 157, TIT_H + 22, "CÁMARA TAPADA", 22, negrita=True, ancla="ma")
        # ---- consola
        cy = TIT_H + VH
        L.rect(0, cy, VW, ALTO, (12, 13, 16))
        for k, s in enumerate(consola):
            L.texto(12, cy + 10 + k * 22, s, 15, (190, 225, 190), mono=True)
        # ---- D435 en color
        if top.color is not None:
            L.pegar(DER_X, TIT_H, cv2.resize(top.color, (DER_W, CAM_H), interpolation=cv2.INTER_AREA))
        L.rect(DER_X, TIT_H, DER_X + 170, TIT_H + 24, (0, 0, 0), alfa=0.6)
        L.texto(DER_X + 8, TIT_H + 4, "D435 · color", 14)
        # ---- cenital de la mesa (el robot abajo)
        L.pegar(DER_X, CEN_Y, vista.render(DER_W, CEN_H, (0.47, 0.0, 0.92), 1.05, 0.0, -89.9))
        L.rect(DER_X, CEN_Y, DER_X + 230, CEN_Y + 24, (0, 0, 0), alfa=0.6)
        L.texto(DER_X + 8, CEN_Y + 4, "Vista cenital de la mesa", 14)
        L.rect(DER_X, TIT_H, DER_X + 2, ALTO, (60, 60, 70))
        frame = L.final()
        # tiempo real: si el render se retrasa, se repite el cuadro
        debe = int((time.monotonic() - t0) * a.fps) + 1
        vid.escribir(frame, veces=max(1, debe - n))
        n = vid.n
        time.sleep(max(0.0, (n / a.fps) - (time.monotonic() - t0)))

    s = vid.cerrar()
    print(f"grabar_video: {a.salida} ({s:.0f} s)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
