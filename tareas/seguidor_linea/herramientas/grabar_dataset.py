#!/usr/bin/env python3
"""Graba un conjunto de datos (reto, hito "Conjunto de datos"): imagenes IR con
marca de tiempo + IMU, mientras un operador lleva el robot por la pista con
teleop.py (o wasd.py en el robot real). Sobre esto se desarrolla y se prueba la
percepcion sin mover el robot (FuenteDataset, evaluar_percepcion.py).

    python3 herramientas/grabar_dataset.py --sim [--nivel 3] [--profundidad]
    sudo -E ~/teleop_venv/bin/python herramientas/grabar_dataset.py --iface eth0 --emisor off

Carpeta datos/dataset_<nivel>_<fecha>/:
  indice.csv   un fotograma por fila: t, seq, ir, prof, fx, fy, cx, cy, roll, pitch,
               yaw, gyro_z (IMU en ese instante) y gt_x, gt_y, gt_yaw (SOLO simulador:
               verdad de terreno de la pelvis)
  imu.csv      rt/lowstate a ~100 Hz (registrar a menos de 3 Hz solapa con la cadencia
               de 1.43 Hz: sec. 7 del reto)
  ir/*.png  prof/*.npy
"""
import argparse
import csv
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from seguidor.robot_io import FuenteImu, iniciar_dds   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--nivel", default="x")
    ap.add_argument("--profundidad", action="store_true")
    ap.add_argument("--emisor", choices=["on", "off"], default="on")
    ap.add_argument("--segundos", type=float, default=0.0, help="0 = hasta Ctrl+C")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    carpeta = a.salida or os.path.join(AQUI, "..", "datos", f"dataset_n{a.nivel}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(carpeta, "ir"), exist_ok=True)
    if a.profundidad:
        os.makedirs(os.path.join(carpeta, "prof"), exist_ok=True)
    iniciar_dds(a.sim, a.iface)
    imu = FuenteImu()
    if not imu.esperar(5.0):
        sys.exit("No llega rt/lowstate")
    if a.sim:
        from seguidor.fuentes import FuenteROS
        f = FuenteROS(profundidad="/camera/depth/image_raw" if a.profundidad else None)
    else:
        from seguidor.fuentes import FuenteRealSense
        f = FuenteRealSense(emisor=a.emisor == "on", profundidad=a.profundidad)
    vivo = {"si": True}

    def grabar_imu():
        with open(os.path.join(carpeta, "imu.csv"), "w", newline="") as fi:
            w = csv.writer(fi)
            w.writerow(["t", "roll", "pitch", "yaw", "gx", "gy", "gz", "motores_ok"])
            while vivo["si"]:
                u = imu.leer()
                w.writerow([f"{time.monotonic():.4f}", f"{u.roll:.5f}", f"{u.pitch:.5f}", f"{u.yaw:.5f}",
                            *[f"{g:.5f}" for g in u.gyro], int(u.motores_ok)])
                time.sleep(0.01)
    threading.Thread(target=grabar_imu, daemon=True).start()

    campos = ["t", "seq", "ir", "prof", "fx", "fy", "cx", "cy", "roll", "pitch", "yaw", "gyro_z", "gt_x", "gt_y", "gt_yaw"]
    fi = open(os.path.join(carpeta, "indice.csv"), "w", newline="")
    w = csv.DictWriter(fi, fieldnames=campos)
    w.writeheader()
    # SIGTERM igual que Ctrl+C (lanzado en segundo plano desde un script, bash le
    # hace ignorar SIGINT y no habria forma limpia de pararlo)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    print(f"grabando en {carpeta} (Ctrl+C para terminar)")
    t0, n = time.monotonic(), 0
    try:
        while not a.segundos or time.monotonic() - t0 < a.segundos:
            fr = f.leer(1.0)
            if fr is None:
                print("sin fotogramas")
                continue
            u = imu.leer()
            nombre = f"ir/{fr.seq:06d}.png"
            cv2.imwrite(os.path.join(carpeta, nombre), fr.ir)
            prof = ""
            if a.profundidad and fr.profundidad is not None:
                prof = f"prof/{fr.seq:06d}.npy"
                np.save(os.path.join(carpeta, prof), fr.profundidad.astype(np.float16))
            gt = f.pose_verdad() if hasattr(f, "pose_verdad") else None
            w.writerow({"t": f"{fr.t:.4f}", "seq": fr.seq, "ir": nombre, "prof": prof,
                        "fx": fr.K[0], "fy": fr.K[1], "cx": fr.K[2], "cy": fr.K[3],
                        "roll": f"{u.roll:.5f}", "pitch": f"{u.pitch:.5f}", "yaw": f"{u.yaw:.5f}",
                        "gyro_z": f"{u.gyro[2]:.5f}",
                        "gt_x": f"{gt[0]:.4f}" if gt else "", "gt_y": f"{gt[1]:.4f}" if gt else "",
                        "gt_yaw": f"{gt[2]:.5f}" if gt else ""})
            n += 1
            if n % 30 == 0:
                print(f"\r{n} fotogramas, {n / (time.monotonic() - t0):.1f} fps", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        vivo["si"] = False
        fi.close()
        if hasattr(f, "cerrar"):
            f.cerrar()
        print(f"\n{n} fotogramas en {carpeta}")


if __name__ == "__main__":
    main()
