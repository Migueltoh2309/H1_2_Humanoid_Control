"""Fuentes de fotogramas, todas con la misma interfaz: leer(timeout) -> Fotograma | None.
La percepcion no sabe de cual viene (reto, sec. 5).

  FuenteRealSense  la D435i en el PC2 del robot (pyrealsense2, con sudo; sec. 7)
  FuenteROS        el simulador: /camera/infra1/image_rect_raw (+ camera_info, depth)
  FuenteDataset    un conjunto de datos grabado con herramientas/grabar_dataset.py

Fotograma.t es el reloj MONOTONIC de este proceso al recibir el fotograma (el
mismo que usan la IMU y el lazo), salvo en el dataset, que trae el suyo.
"""
import csv
import os
import threading
import time

import numpy as np

from .tipos import Fotograma


class FuenteRealSense:
    """IR izquierda 640x480 a 30 fps (+ profundidad opcional) con el emisor
    encendido o apagado (6.2.1). Se lanza con sudo (sec. 7)."""

    def __init__(self, emisor=True, profundidad=False, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, fps)
        if profundidad:
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        perfil = self.pipe.start(cfg)
        dev = perfil.get_device()
        sensor = dev.first_depth_sensor()
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1.0 if emisor else 0.0)
        self.escala = sensor.get_depth_scale()
        i = perfil.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
        self.K = (i.fx, i.fy, i.ppx, i.ppy)
        self.K_prof = None
        if profundidad:
            j = perfil.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            self.K_prof = (j.fx, j.fy, j.ppx, j.ppy)
        self.seq = 0

    def leer(self, timeout=1.0):
        try:
            fr = self.pipe.wait_for_frames(int(timeout * 1000))
        except RuntimeError:
            return None
        ir = fr.get_infrared_frame(1)
        if not ir:
            return None
        self.seq += 1
        prof = None
        d = fr.get_depth_frame()
        if d:
            prof = np.asanyarray(d.get_data()).astype(np.float32) * self.escala
        return Fotograma(time.monotonic(), np.asanyarray(ir.get_data()).copy(), self.K, self.seq, prof, self.K_prof)

    def cerrar(self):
        self.pipe.stop()


class FuenteROS:
    """Camaras del simulador (h1_2_sim2real). Nodo propio en un hilo. Ademas da la
    pose de verdad de terreno de la pelvis para evaluar, SOLO en simulacion.

    La verdad se lee de /sim/ground_truth/pelvis y NO de /tf: con RViz abierto,
    robot_state_publisher mete en /tf ~70 eslabones a 50 Hz, y un TransformListener
    de Python que lo procesa todo se comia el GIL; las respuestas del RPC de
    LocoClient tardaban ~4 s y el lazo bajaba de 20 Hz a ~1 Hz (medido con cProfile)."""

    def __init__(self, ir="/camera/infra1/image_rect_raw", info="/camera/infra1/camera_info",
                 profundidad=None, info_prof="/camera/depth/camera_info", verdad=True):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        if not rclpy.ok():
            rclpy.init()
        self.rclpy = rclpy
        self.nodo = rclpy.create_node("seguidor_fuente")
        self.cerrojo = threading.Condition()
        self.ultimo, self.K, self.K_prof, self.prof, self.seq, self.entregado = None, None, None, None, 0, 0
        self.nodo.create_subscription(Image, ir, self._al_ir, qos_profile_sensor_data)
        self.nodo.create_subscription(CameraInfo, info, self._al_info, 10)
        if profundidad:
            self.nodo.create_subscription(Image, profundidad, self._al_prof, qos_profile_sensor_data)
            self.nodo.create_subscription(CameraInfo, info_prof, self._al_info_prof, 10)
        self.verdad = None
        if verdad:
            from geometry_msgs.msg import PoseStamped
            self.nodo.create_subscription(PoseStamped, "/sim/ground_truth/pelvis", self._al_verdad, 10)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.nodo)
        self.hilo = threading.Thread(target=self._girar, daemon=True, name="fuente_ros")
        self.hilo.start()

    def _girar(self):
        try:
            self.ex.spin()
        except Exception:          # al cerrar, el executor puede salir con ExternalShutdownException
            pass

    @staticmethod
    def _imagen(msg):
        if msg.encoding in ("mono8", "8UC1"):
            return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width).copy()
        if msg.encoding == "32FC1":
            return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
        if msg.encoding == "rgb8":
            a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            return (a @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
        raise ValueError(f"encoding {msg.encoding} no soportado")

    def _al_info(self, m):
        self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

    def _al_info_prof(self, m):
        self.K_prof = (m.k[0], m.k[4], m.k[2], m.k[5])

    def _al_prof(self, m):
        self.prof = self._imagen(m)

    def _al_ir(self, m):
        img = self._imagen(m)
        with self.cerrojo:
            self.seq += 1
            self.ultimo = (time.monotonic(), img, self.seq)
            self.cerrojo.notify_all()

    def leer(self, timeout=1.0):
        with self.cerrojo:
            fin = time.monotonic() + timeout
            while self.ultimo is None or self.ultimo[2] == self.entregado or self.K is None:
                resto = fin - time.monotonic()
                if resto <= 0:
                    return None
                self.cerrojo.wait(resto)
            t, img, seq = self.ultimo
            self.entregado = seq
        return Fotograma(t, img, self.K, seq, self.prof, self.K_prof)

    def _al_verdad(self, m):
        import math
        q = m.pose.orientation
        self.verdad = (m.pose.position.x, m.pose.position.y,
                       math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def pose_verdad(self):
        """(x, y, yaw) de la pelvis en el mundo del simulador, o None."""
        return self.verdad

    def cerrar(self):
        """Cierre ordenado: sin el, rclpy deja hilos de C++ vivos y el proceso acaba
        con 'terminate called without an active exception'."""
        self.ex.shutdown(timeout_sec=1.0)
        self.hilo.join(timeout=1.0)
        self.nodo.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


class FuenteDataset:
    """Lee una carpeta de grabar_dataset.py: indice.csv + ir/*.png (+ prof/*.npy).
    Devuelve tambien la IMU y la verdad de terreno grabadas con cada fotograma."""

    def __init__(self, carpeta):
        import cv2
        self.cv2 = cv2
        self.carpeta = carpeta
        with open(os.path.join(carpeta, "indice.csv")) as f:
            self.filas = list(csv.DictReader(f))
        self.i = 0

    def __len__(self):
        return len(self.filas)

    def leer(self, timeout=0.0):
        if self.i >= len(self.filas):
            return None
        r = self.filas[self.i]
        self.i += 1
        ir = self.cv2.imread(os.path.join(self.carpeta, r["ir"]), self.cv2.IMREAD_GRAYSCALE)
        prof = None
        if r.get("prof"):
            prof = np.load(os.path.join(self.carpeta, r["prof"]))
        K = tuple(float(r[k]) for k in ("fx", "fy", "cx", "cy"))
        f = Fotograma(float(r["t"]), ir, K, int(r["seq"]), prof)
        f.fila = r
        return f
