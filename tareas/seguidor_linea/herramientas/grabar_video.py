#!/usr/bin/env python3
"""SOLO SIMULADOR: graba un video de lo que pasa en la pista, para los entregables de
los hitos. Compone, a 1280x720 y en tiempo real:

  izquierda   el robot en MuJoCo, con una camara que lo sigue por detras (render
              propio offscreen en la GPU: no toca el simulador ni su viewer)
              + estado del supervisor y ordenes (si corre seguidor.py --rviz)
  abajo       la consola de lo que se esta ejecutando (--consola fichero)
  derecha     la IR izquierda de la D435 con la linea detectada encima, y una vista
              cenital en el marco del robot: cinta real (gris), detectada (verde),
              memoria (azul), punto de pure pursuit (rojo)

Lee solo topicos ROS: /sim/joint_states + /sim/ground_truth/pelvis (pose del robot),
/camera/infra1/image_rect_raw, /seguidor/estado y /sim/tapar_camara.

    python3 herramientas/grabar_video.py --nivel 2 --salida videos/x.mp4 \\
        --titulo "Hito 3" --consola /tmp/consola.log [--final grafica.png] [--final texto.txt]

Termina con Ctrl+C / SIGTERM (o --duracion). Opcional: tarjeta inicial (--intro) y
tarjetas finales (--final png = imagen, txt = texto, 6 s cada una); por defecto el
video es solo la simulacion.
"""
import argparse
import json
import math
import os
import signal
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2                     # noqa: E402
import numpy as np             # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
sys.path.insert(0, AQUI)
from video_comun import (ANCHO, ALTO, BLANCO, COLOR_ESTADO, GRIS, EscritorVideo, Lienzo,   # noqa: E402
                         tarjeta, tarjeta_imagen, ultimas_lineas)

# disposicion
TIT_H = 48
VW, VH = 848, 477                  # vista 3D
DER_X, DER_W = VW, ANCHO - VW      # columna derecha (432)
IR_H = 324
CEN_Y, CEN_H = TIT_H + IR_H, ALTO - TIT_H - IR_H


class Topicos:
    def __init__(self):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Bool, String
        rclpy.init()
        self.rclpy = rclpy
        self.nodo = rclpy.create_node("grabar_video")
        self.js = self.pose = self.ir = self.estado = None
        self.t_estado = 0.0
        self.tapada = False
        n = self.nodo
        n.create_subscription(JointState, "/sim/joint_states", lambda m: setattr(self, "js", m), 5)
        n.create_subscription(PoseStamped, "/sim/ground_truth/pelvis", lambda m: setattr(self, "pose", m), 5)
        n.create_subscription(Image, "/camera/infra1/image_rect_raw", self._ir, qos_profile_sensor_data)
        n.create_subscription(String, "/seguidor/estado", self._estado, 5)
        n.create_subscription(Bool, "/sim/tapar_camara", lambda m: setattr(self, "tapada", m.data), 5)
        self.hilo = threading.Thread(target=self._girar, daemon=True)
        self.hilo.start()

    def _girar(self):
        try:
            self.rclpy.spin(self.nodo)
        except Exception:
            pass

    def _ir(self, m):
        if m.encoding in ("mono8", "8UC1"):
            self.ir = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width).copy()

    def _estado(self, m):
        self.estado = json.loads(m.data)
        self.t_estado = time.monotonic()


class Vista3D:
    def __init__(self, modelo):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(modelo)
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, VW)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, VH)
        for g in range(self.m.ngeom):      # la placa que da sombra (nivel 3), a 2.4 m: la camara la atraviesa
            if self.m.geom(g).name.startswith("placa_sombra"):
                self.m.geom_matid[g] = -1
                self.m.geom_rgba[g] = (0.25, 0.25, 0.3, 0.25)
        self.d = mujoco.MjData(self.m)
        if self.m.nkey:
            self.d.qpos[:] = self.m.key_qpos[0]
        self.rend = mujoco.Renderer(self.m, height=VH, width=VW)
        self.libre = next((j for j in range(self.m.njnt) if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE), None)
        self.adr = {}
        for j in range(self.m.njnt):
            self.adr[self.m.joint(j).name] = self.m.jnt_qposadr[j]
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation = 3.9, -34.0
        self.azim = None
        self.mirada = None

    def actualizar(self, js, pose):
        if js is not None:
            for nombre, q in zip(js.name, js.position):
                a = self.adr.get(nombre)
                if a is not None:
                    self.d.qpos[a] = q
        if pose is not None and self.libre is not None:
            a = self.m.jnt_qposadr[self.libre]
            p, o = pose.pose.position, pose.pose.orientation
            self.d.qpos[a:a + 7] = [p.x, p.y, p.z, o.w, o.x, o.y, o.z]
        self.mj.mj_forward(self.m, self.d)

    def base(self):
        """(x, y, yaw) de la pelvis en el mundo."""
        if self.libre is None:
            return 0.0, 0.0, 0.0
        a = self.m.jnt_qposadr[self.libre]
        x, y, _, w, qx, qy, qz = self.d.qpos[a:a + 7]
        return x, y, math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    def render(self, yaw):
        a = math.degrees(yaw)
        if self.azim is None:
            self.azim = a
        self.azim += ((a - self.azim + 180) % 360 - 180) * 0.04          # camara que sigue el rumbo, suave
        x, y, _ = self.base()
        obj = np.array([x + 1.1 * math.cos(yaw), y + 1.1 * math.sin(yaw), 0.35])  # mira por delante del robot
        self.mirada = obj if self.mirada is None else self.mirada + 0.1 * (obj - self.mirada)
        self.cam.lookat[:] = self.mirada
        self.cam.azimuth = self.azim
        self.rend.update_scene(self.d, camera=self.cam)
        return self.rend.render()


def a_robot(L, x, y, yaw):
    dx, dy = L[:, 0] - x, L[:, 1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack([dx * c + dy * s, -dx * s + dy * c])


def vista_cenital(est, cinta_robot, alto=CEN_H, ancho=DER_W):
    """Marco del robot: x adelante = arriba, y izquierda = izquierda."""
    img = np.full((alto, ancho, 3), 238, np.uint8)
    esc = 95.0                                  # px por m
    ox, oy = ancho // 2, alto - 40              # el robot

    def px(P):
        P = np.asarray(P, float).reshape(-1, 2)
        return np.column_stack([ox - P[:, 1] * esc, oy - P[:, 0] * esc]).astype(np.int32)
    for k in range(0, 5):                       # rejilla cada metro
        yv = oy - int(k * esc)
        cv2.line(img, (0, yv), (ancho, yv), (215, 215, 215), 1)
    for k in range(-2, 3):
        xv = ox + int(k * esc)
        cv2.line(img, (xv, 0), (xv, alto), (215, 215, 215), 1)
    cv2.rectangle(img, (0, oy - int(0.35 * esc)), (ancho, oy), (225, 225, 232), -1)   # zona ciega
    if cinta_robot is not None and len(cinta_robot):
        p = px(cinta_robot)
        d = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
        for i in np.nonzero(d < 15)[0]:         # sin unir los huecos de la cinta
            cv2.line(img, tuple(p[i]), tuple(p[i + 1]), (90, 90, 90), max(2, int(0.05 * esc)))
    if est:
        for x, y in px(est.get("memoria") or []):
            cv2.circle(img, (x, y), 2, (40, 90, 255), -1)
        for x, y in px(est.get("detectados") or []):
            cv2.circle(img, (x, y), 4, (20, 190, 60), -1)
        if est.get("mira"):
            m = px(est["mira"])[0]
            cv2.line(img, (ox, oy), tuple(m), (230, 40, 40), 2)
            cv2.circle(img, tuple(m), 7, (230, 40, 40), -1)
    pie = np.array([[ox, oy - 14], [ox - 10, oy + 10], [ox + 10, oy + 10]], np.int32)
    cv2.fillPoly(img, [pie], (30, 30, 30))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nivel", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--titulo", default="")
    ap.add_argument("--subtitulo", default="")
    ap.add_argument("--intro", action="append", default=[], help="linea de la tarjeta inicial (se repite)")
    ap.add_argument("--consola", default=None)
    ap.add_argument("--final", action="append", default=[], help="png o txt para tarjetas al final")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--duracion", type=float, default=600.0)
    a = ap.parse_args()

    pistas = os.path.join(AQUI, "..", "sim", "pistas")
    pista = json.load(open(os.path.join(pistas, f"pista_nivel{a.nivel}.json")))
    cinta = np.array(pista["linea"])[:, :2]
    vista = Vista3D(os.path.join(pistas, f"pista_nivel{a.nivel}.xml"))
    from seguidor import config
    from seguidor.geometria import ModeloCamara
    cam_ir = ModeloCamara.desde_config(config.cargar())
    top = Topicos()
    vid = EscritorVideo(a.salida, a.fps)
    if a.intro:
        vid.escribir(tarjeta(a.titulo, a.intro, a.subtitulo, pie="Simulador MuJoCo · H1-2 · reto R40-RT-H1_2-0001"),
                     veces=int(3.5 * a.fps))

    fin = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: fin.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: fin.__setitem__("v", True))
    t0 = time.monotonic()
    while top.pose is None and time.monotonic() - t0 < 30 and not fin["v"]:
        time.sleep(0.1)
    print(f"grabar_video: grabando {a.salida}", flush=True)
    t0 = time.monotonic()
    n = 0
    consola = []
    while not fin["v"] and time.monotonic() - t0 < a.duracion:
        t = time.monotonic() - t0
        vista.actualizar(top.js, top.pose)
        x, y, yaw = vista.base()
        est = top.estado if time.monotonic() - top.t_estado < 3.0 or (
            top.estado and top.estado.get("estado") in ("FIN", "PARADA")) else None

        L = Lienzo()
        L.pegar(0, TIT_H, vista.render(yaw))
        # ---- titulo
        L.rect(0, 0, ANCHO, TIT_H, (16, 18, 22))
        L.texto(16, 12, a.titulo, 22, negrita=True)
        L.texto(ANCHO - 16, 14, f"t = {t:5.1f} s", 18, GRIS, mono=True, ancla="ra")
        # ---- estado del seguidor sobre la vista 3D
        if est:
            e = est.get("estado", "?")
            col = COLOR_ESTADO.get(e, (120, 120, 120))
            w = int(L.ancho_texto(e, 22, True)) + 28
            L.rect(14, TIT_H + 14, 14 + w, TIT_H + 50, col)
            L.texto(28, TIT_H + 20, e, 22, negrita=True)
            L.rect(14, TIT_H + 56, 470, TIT_H + 150, (0, 0, 0), alfa=0.55)
            L.texto(26, TIT_H + 62, est.get("modo", ""), 15, (255, 210, 120) if "SIMULACRO" in est.get("modo", "")
                    else GRIS)
            L.texto(26, TIT_H + 84, f"vx {est['vx']:+.2f} m/s   vyaw {est['vyaw']:+.2f} rad/s   "
                    f"conf {est['conf']:.2f}", 17, mono=True)
            if est.get("desp") is not None:
                L.texto(26, TIT_H + 106, f"linea: desp {est['desp'] * 100:+5.1f} cm  ang "
                        f"{math.degrees(est.get('ang', 0)):+5.1f}°", 17, mono=True)
            L.texto(26, TIT_H + 128, str(est.get("motivo", ""))[:52], 15, GRIS)
        if top.tapada:
            L.rect(VW - 300, TIT_H + 14, VW - 14, TIT_H + 54, (220, 50, 50))
            L.texto(VW - 157, TIT_H + 22, "CÁMARA TAPADA", 22, negrita=True, ancla="ma")
        # ---- consola
        cy = TIT_H + VH
        L.rect(0, cy, VW, ALTO, (12, 13, 16))
        if a.consola and n % 5 == 0:
            consola = ultimas_lineas(a.consola, 8, 92)
        for k, s in enumerate(consola):
            L.texto(12, cy + 10 + k * 22, s, 15, (190, 225, 190), mono=True)
        # ---- IR con la deteccion
        ir = top.ir
        if ir is not None:
            img = cv2.cvtColor(ir, cv2.COLOR_GRAY2RGB)
            if est and est.get("detectados"):
                P = np.array(est["detectados"])
                uv = cam_ir.suelo_a_pixel(P[:, 0], P[:, 1], 0.0, 0.0)
                for u, v in uv:
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        cv2.circle(img, (int(u), int(v)), 5, (40, 230, 60), -1)
            L.pegar(DER_X, TIT_H, cv2.resize(img, (DER_W, IR_H), interpolation=cv2.INTER_AREA))
        L.rect(DER_X, TIT_H, DER_X + 190, TIT_H + 24, (0, 0, 0), alfa=0.6)
        L.texto(DER_X + 8, TIT_H + 4, "D435 · IR izquierda", 14)
        # ---- cenital
        cinta_r = a_robot(cinta, x, y, yaw)
        cerca = (cinta_r[:, 0] > -1.0) & (cinta_r[:, 0] < 4.5) & (np.abs(cinta_r[:, 1]) < 3.0)
        L.pegar(DER_X, CEN_Y, vista_cenital(est, cinta_r[cerca]))
        L.texto(DER_X + 8, CEN_Y + 6, "Vista cenital (marco del robot)", 14, (40, 40, 40))
        xl = DER_X + 8
        for nombre, col in (("cinta real", (90, 90, 90)), ("detectada", (20, 160, 50)), ("memoria", (40, 90, 255)),
                            ("mira", (230, 40, 40))):
            L.texto(xl, CEN_Y + 26, nombre, 13, col, negrita=True)
            xl += int(L.ancho_texto(nombre, 13, True)) + 14
        L.rect(DER_X, TIT_H, DER_X + 2, ALTO, (60, 60, 70))
        frame = L.final()
        # tiempo real: si el render se retrasa, se repite el cuadro
        debe = int((time.monotonic() - t0) * a.fps) + 1
        vid.escribir(frame, veces=max(1, debe - n))
        n = vid.n - (int(3.5 * a.fps) if a.intro else 0)
        time.sleep(max(0.0, (n / a.fps) - (time.monotonic() - t0)))

    for f in a.final:
        if not os.path.isfile(f):
            continue
        if f.endswith(".png"):
            c = tarjeta_imagen(f, a.titulo)
        else:
            lineas = open(f, encoding="utf-8").read().splitlines()
            c = tarjeta(lineas[0] if lineas else a.titulo, lineas[1:])
        vid.escribir(c, veces=6 * a.fps)
    s = vid.cerrar()
    print(f"grabar_video: {a.salida} ({s:.0f} s)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
