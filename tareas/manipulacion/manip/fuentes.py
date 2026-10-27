"""Fuentes de imagen: todas entregan el mismo `Fotograma`, así que la percepción y la
calibración no saben de dónde viene (sec. 5: igual sobre datos grabados que en vivo).

  * FuenteRos: tópicos de realsense2_camera (simulador de h1_2_sim2real, o el robot con
    realsense-ros): color, IR1 y profundidad NATIVA (/camera/depth/image_rect_raw, marco de
    IR1). Un hilo propio hace spin de rclpy (no se usa tf2: la pose de la cámara sale SOLO
    del fichero de calibración; además, el /tf_static del simulador trae la pose verdadera
    de la cámara y leerla sería hacer trampa).
  * FuenteRealsense: pyrealsense2 en el PC2 (como Codigos/camara.py): color, depth z16
    nativa e IR1, con el proyector apagable (para ver el ArUco sin los puntos del emisor).
  * FuenteCarpeta: un conjunto grabado (npz por fotograma).
"""
import glob
import os
import threading
import time

import numpy as np

from .tipos import Fotograma


def _K(info):
    return np.array(info.k, dtype=float).reshape(3, 3)


def _img(m):
    if m.encoding == "32FC1":
        return np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
    if m.encoding == "16UC1":
        return np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
    can = {"rgb8": 3, "bgr8": 3, "mono8": 1}[m.encoding]
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, can).squeeze()
    return a[..., ::-1].copy() if m.encoding == "bgr8" else a.copy()


class FuenteRos:
    TOP = {"color": "/camera/color/image_raw", "ir": "/camera/infra1/image_rect_raw",
           "prof": "/camera/depth/image_rect_raw"}
    INFO = {"color": "/camera/color/camera_info", "prof": "/camera/depth/image_rect_raw/camera_info"}

    def __init__(self, con_color=True):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import CameraInfo, Image
        if not rclpy.ok():
            rclpy.init()
        self.nodo = rclpy.create_node("manip_camara")
        self.ult = {}
        self.K = {}
        self.cerrojo = threading.Lock()
        for k, t in self.TOP.items():
            if k == "color" and not con_color:
                continue
            self.nodo.create_subscription(Image, t, lambda m, k=k: self._img(k, m), 2)
        for k, t in self.INFO.items():
            self.nodo.create_subscription(CameraInfo, t, lambda m, k=k: self.K.__setitem__(k, _K(m)), 2)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.nodo)
        self.hilo = threading.Thread(target=self.ex.spin, daemon=True, name="ros_camara")
        self.hilo.start()

    def _img(self, k, m):
        with self.cerrojo:
            self.ult[k] = (time.monotonic(), _img(m))

    def edad(self):
        with self.cerrojo:
            return time.monotonic() - min((v[0] for v in self.ult.values()), default=-1e9)

    def fotograma(self, timeout=3.0, despues_de=None):
        """El último fotograma con IR y profundidad llegados después de `despues_de`."""
        t0 = time.monotonic()
        despues_de = t0 if despues_de is None else despues_de
        while time.monotonic() - t0 < timeout:
            with self.cerrojo:
                u = dict(self.ult)
            if "prof" in u and "ir" in u and u["prof"][0] > despues_de and u["ir"][0] > despues_de \
                    and "prof" in self.K:
                return Fotograma(t=u["prof"][0], color=u.get("color", (0, None))[1], prof=u["prof"][1],
                                 K_color=self.K.get("color"), K_prof=self.K["prof"], ir=u["ir"][1])
            time.sleep(0.01)
        return None

    def cerrar(self):
        import rclpy
        self.ex.shutdown()
        self.hilo.join(timeout=2.0)
        self.nodo.destroy_node()
        rclpy.try_shutdown()


class FuenteRealsense:
    def __init__(self, emisor=True, ancho=640, alto=480, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        c = rs.config()
        c.enable_stream(rs.stream.color, ancho, alto, rs.format.rgb8, fps)
        c.enable_stream(rs.stream.depth, ancho, alto, rs.format.z16, fps)
        c.enable_stream(rs.stream.infrared, 1, ancho, alto, rs.format.y8, fps)
        perfil = self.pipe.start(c)
        dev = perfil.get_device()
        sensor = dev.first_depth_sensor()
        self.escala = sensor.get_depth_scale()
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1.0 if emisor else 0.0)

        def K(stream, idx=0):
            i = perfil.get_stream(stream, idx).as_video_stream_profile().get_intrinsics()
            return np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])
        self.K_color, self.K_prof = K(rs.stream.color), K(rs.stream.depth)
        ext = perfil.get_stream(rs.stream.depth).get_extrinsics_to(perfil.get_stream(rs.stream.color))
        self.T_color_prof = np.eye(4)
        self.T_color_prof[:3, :3] = np.array(ext.rotation).reshape(3, 3).T
        self.T_color_prof[:3, 3] = ext.translation
        self.t_ultimo = 0.0

    def edad(self):
        return time.monotonic() - self.t_ultimo

    def fotograma(self, timeout=3.0, despues_de=None):
        f = self.pipe.wait_for_frames(int(timeout * 1000))
        c, d, i = f.get_color_frame(), f.get_depth_frame(), f.get_infrared_frame(1)
        if not d or not i:
            return None
        self.t_ultimo = time.monotonic()
        return Fotograma(t=self.t_ultimo, color=np.asanyarray(c.get_data()).copy() if c else None,
                         prof=np.asanyarray(d.get_data()).astype(np.float32) * self.escala,
                         K_color=self.K_color, K_prof=self.K_prof, ir=np.asanyarray(i.get_data()).copy())

    def cerrar(self):
        self.pipe.stop()


class FuenteCarpeta:
    """Fotogramas grabados (grabar(): un .npz por fotograma con color, prof, ir, K_color, K_prof)."""

    def __init__(self, carpeta):
        self.ficheros = sorted(glob.glob(os.path.join(carpeta, "*.npz")))
        self.k = 0

    def __len__(self):
        return len(self.ficheros)

    def fotograma(self, timeout=None, despues_de=None):
        if self.k >= len(self.ficheros):
            return None
        z = np.load(self.ficheros[self.k], allow_pickle=False)
        self.k += 1
        g = {k: z[k] for k in z.files}
        return Fotograma(t=float(g.get("t", 0.0)), color=g.get("color"), prof=g["prof"], K_color=g.get("K_color"),
                         K_prof=g["K_prof"], ir=g.get("ir"),
                         q_brazos=g.get("q_brazos"), cintura=float(g.get("cintura", 0.0)))


def grabar(ruta, f: Fotograma):
    datos = {"t": f.t, "prof": f.prof, "K_prof": f.K_prof, "cintura": f.cintura}
    for k in ("color", "K_color", "ir", "q_brazos"):
        v = getattr(f, k)
        if v is not None:
            datos[k] = v
    np.savez_compressed(ruta, **datos)
