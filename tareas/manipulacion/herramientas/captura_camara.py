#!/usr/bin/env python3
"""Guarda un fotograma de la D435 (simulador o robot con realsense-ros): color, profundidad
alineada al color y profundidad nativa (IR1), en PNG (vista) y NPZ (datos en metros).

    python3 herramientas/captura_camara.py datos/capturas/mesa_n3
"""
import os
import sys
import time

import numpy as np
import rclpy
from sensor_msgs.msg import Image

TOPICOS = {"color": "/camera/color/image_raw", "prof_alineada": "/camera/depth/image_raw",
           "prof_nativa": "/camera/depth/image_rect_raw", "ir1": "/camera/infra1/image_rect_raw"}


def a_numpy(m):
    if m.encoding == "32FC1":
        return np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
    if m.encoding == "16UC1":
        return np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
    canales = {"rgb8": 3, "bgr8": 3, "mono8": 1}[m.encoding]
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, canales).squeeze()
    return a[..., ::-1].copy() if m.encoding == "bgr8" else a.copy()


def main():
    salida = sys.argv[1] if len(sys.argv) > 1 else "datos/capturas/captura"
    rclpy.init()
    nodo = rclpy.create_node("captura_camara")
    imgs = {}
    for k, t in TOPICOS.items():
        nodo.create_subscription(Image, t, lambda m, k=k: imgs.__setitem__(k, a_numpy(m)), 2)
    t0 = time.monotonic()
    while len(imgs) < len(TOPICOS) and time.monotonic() - t0 < 15.0:
        rclpy.spin_once(nodo, timeout_sec=0.2)
    nodo.destroy_node()
    rclpy.shutdown()
    if not imgs:
        sys.exit("no llega ninguna imagen")
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    np.savez_compressed(salida + ".npz", **imgs)
    for k, a in imgs.items():
        if k.startswith("prof"):
            v = np.clip((a - 0.5) / 1.5, 0, 1)
            vis = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_TURBO)
            vis[a <= 0] = 0
        else:
            vis = a[..., ::-1] if a.ndim == 3 else a
        cv2.imwrite(f"{salida}_{k}.png", vis)
    print("guardado:", salida, {k: a.shape for k, a in imgs.items()}, "faltan:", set(TOPICOS) - set(imgs))


if __name__ == "__main__":
    main()
