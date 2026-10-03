#!/usr/bin/env python3
"""Demo — Método A (color+profundidad) vs Método B (YOLO+profundidad) del
plan de percepción (ver docs/PERCEPTION_PLAN.md, secciones 4, 6 y
7), medidos contra el ground truth EXACTO del simulador (posición real de
la mandarina, sin necesidad de motion capture).

NO usa ROS/el bridge: instancia MuJoCo directamente (mismo patrón que
demos/benchmark_ik.py), leyendo/escribiendo el estado del simulador a mano.
Por qué así y no lanzando los dos launch files reales: acá el ground truth
y el timing quedan exactos (data.xpos, sin el ruido de sincronización de
tópicos/TF de un pipeline en vivo) — la comparación algorítmica queda
aislada de efectos de sistema (esos ya se midieron aparte, en vivo, con
carga de CPU real). La detección en sí (perception_common.detect_color /
detect_yolo) es EXACTAMENTE la misma que corren color_depth_detector_node.py
y yolo_color_depth_detector_node.py — no hay una tercera implementación.

El robot se queda en su pose de arranque del MJCF durante todo el script
(no hace falta moverlo, solo importa la cámara del torso + la mandarina).

Dos experimentos:
  1. Barrido espacial estático: la mandarina se coloca en N posiciones
     conocidas a lo largo de la faja (fijando qpos directamente, sin
     avanzar física) — mide precisión en todo el rango de trabajo.
  2. Paso dinámico: keyframe "mandarina_moving", física real avanzando
     (mj_step) — mide lo mismo mientras el objeto se mueve de verdad. Sin
     gravedad (ver run_moving_pass): este script no controla los 27 motores
     del robot, así que sin gravedad los brazos no caen en caída libre y
     tapan la cámara durante los ~16s del experimento.

Uso:
    python3 demos/evaluate_perception.py                  # todo por defecto
    python3 demos/evaluate_perception.py --n-static 21
    python3 demos/evaluate_perception.py --no-yolo         # solo Método A
    python3 demos/evaluate_perception.py --device cuda:0   # si hay torch+CUDA

Guarda CSVs en results/percepcion/ (mismo patrón que
results/trayectorias/ del paquete) para graficar después.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ament_index_python.packages import get_package_share_directory  # noqa: E402

from h1_2_mujoco_sim_bridge.mujoco_sim import MujocoSim               # noqa: E402
from h1_2_mujoco_sim_bridge.sim_bridge_node import DEFAULT_JOINT_NAMES  # noqa: E402

from h1_2_algoritms.vision.perception_common import (                        # noqa: E402
    detect_color, detect_yolo, deproject,
    DEFAULT_YOLO_MODEL_VARIANT, DEFAULT_YOLO_TARGET_CLASSES)

CAMERA_NAME = "robot_rgbd_camera"
MANDARINA_BODY = "mandarina"
MANDARINA_JOINT = "mandarina_slide_joint"
MOVING_KEYFRAME = "mandarina_moving"

# 180° sobre X local: el mismo flip que mujoco_sim.py aplica para publicar
# el TF óptico de la cámara (ver ahí "_ROS_OPTICAL_FLIP_WXYZ", verificado
# numéricamente, error < 1e-15). Acá se usa para llevar un punto ya
# deproyectado (convención óptica ROS: X der., Y abajo, Z adelante) a la
# convención de cámara de MuJoCo (X der., Y arriba, mira hacia -Z) antes de
# pasarlo a mundo con cam_xpos/cam_xmat — así el ground truth y la
# estimación quedan en el mismo frame (mundo) sin pasar por TF/ROS.
_OPTICAL_TO_MUJOCO_CAM = np.diag([1.0, -1.0, -1.0])

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "percepcion")


def camera_intrinsics(sim):
    fy = (sim.camera_height / 2.0) / np.tan(np.radians(sim.camera_fovy) / 2.0)
    fx = fy  # píxeles cuadrados, misma convención que msg_builders.camera_info_msg
    return fx, fy, sim.camera_width / 2.0, sim.camera_height / 2.0


def camera_point_to_world(sim, point_optical):
    cam_xpos = sim.data.cam_xpos[sim.camera_id]
    cam_xmat = sim.data.cam_xmat[sim.camera_id].reshape(3, 3)
    point_muj_cam = _OPTICAL_TO_MUJOCO_CAM @ np.asarray(point_optical)
    return cam_xpos + cam_xmat @ point_muj_cam


def make_sim(mjcf_path, initial_keyframe=None):
    sim = MujocoSim(
        model_path=mjcf_path,
        joint_names=DEFAULT_JOINT_NAMES,
        base_body="pelvis",
        camera_name=CAMERA_NAME,
        camera_width=640,
        camera_height=480,
        initial_keyframe=initial_keyframe,
    )
    sim.warmup_renderer()
    return sim


class Method:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def detect(self, rgb, depth, bgr):
        return self.fn(rgb, depth, bgr)


def build_methods(use_yolo, device):
    methods = [Method("A (color)", lambda rgb, depth, bgr: detect_color(rgb, depth))]
    if not use_yolo:
        return methods
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics no está instalado — salto el Método B "
              "(ver PERCEPTION_PLAN.md sección 7). Sigo solo con el Método A.\n")
        return methods

    weights = os.path.join(os.path.expanduser("~/.cache/h1_2_algoritms"), DEFAULT_YOLO_MODEL_VARIANT)
    print(f"Cargando YOLO ({weights}, device={device})...")
    model = YOLO(weights)
    model.to(device)
    name_to_id = {v: k for k, v in model.names.items()}
    class_ids = [name_to_id[n] for n in DEFAULT_YOLO_TARGET_CLASSES if n in name_to_id]
    has_masks = "seg" in DEFAULT_YOLO_MODEL_VARIANT

    def yolo_fn(rgb, depth, bgr):
        return detect_yolo(model, class_ids, has_masks, bgr, depth, device=device)

    methods.append(Method(f"B (YOLO/{device})", yolo_fn))
    return methods


def evaluate_frame(sim, methods, gt_world):
    rgb = sim.render_rgb()
    depth = sim.render_depth()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    fx, fy, cx, cy = camera_intrinsics(sim)

    rows = []
    for method in methods:
        t0 = time.time()
        det = method.detect(rgb, depth, bgr)
        dt_ms = (time.time() - t0) * 1000.0
        row = dict(method=method.name, ok=det is not None, dt_ms=dt_ms, gt=gt_world)
        if det is not None:
            point_world = camera_point_to_world(sim, deproject(det.u, det.v, det.z, fx, fy, cx, cy))
            row.update(est=point_world, err=point_world - gt_world, score=det.score, label=det.label)
        rows.append(row)
    return rows


def run_static_sweep(sim, methods, n_points):
    jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, MANDARINA_JOINT)
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, MANDARINA_BODY)
    qadr = sim.model.jnt_qposadr[jid]
    lo, hi = sim.model.jnt_range[jid]
    margin = 0.03  # no clavar la mandarina justo en el tope mecánico
    xs = np.linspace(lo + margin, hi - margin, n_points)

    rows = []
    for x in xs:
        sim.data.qpos[qadr] = x
        sim.data.qvel[qadr] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        gt_world = sim.data.xpos[bid].copy()
        for row in evaluate_frame(sim, methods, gt_world):
            row["x_joint"], row["t"] = float(x), None
            rows.append(row)
        print(".", end="", flush=True)
    print()
    return rows


def run_moving_pass(mjcf_path, methods, duration_s, sample_dt):
    sim = make_sim(mjcf_path, initial_keyframe=MOVING_KEYFRAME)
    sim.model.opt.gravity[:] = 0.0  # ver docstring del módulo

    jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, MANDARINA_JOINT)
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, MANDARINA_BODY)
    qadr = sim.model.jnt_qposadr[jid]

    rows = []
    t_next = 0.0
    while sim.data.time < duration_s:
        if sim.data.time >= t_next:
            gt_world = sim.data.xpos[bid].copy()
            for row in evaluate_frame(sim, methods, gt_world):
                row["t"], row["x_joint"] = float(sim.data.time), float(sim.data.qpos[qadr])
                rows.append(row)
            t_next += sample_dt
            print(".", end="", flush=True)
        mujoco.mj_step(sim.model, sim.data)
    print()
    sim.close_viewer()
    return rows


def summarize(rows, method_name):
    sub = [r for r in rows if r["method"] == method_name]
    ok = [r for r in sub if r["ok"]]
    lat = np.array([r["dt_ms"] for r in sub]) if sub else np.array([])
    err3 = np.array([np.linalg.norm(r["err"]) for r in ok]) * 1000.0 if ok else np.array([])
    errxyz = (np.array([r["err"] for r in ok]) * 1000.0 if ok
              else np.zeros((0, 3)))
    return dict(
        n=len(sub), recall=len(ok) / len(sub) if sub else float("nan"),
        lat_mean=lat.mean() if lat.size else float("nan"),
        lat_median=np.median(lat) if lat.size else float("nan"),
        err3_mean=err3.mean() if err3.size else float("nan"),
        err3_median=np.median(err3) if err3.size else float("nan"),
        err3_max=err3.max() if err3.size else float("nan"),
        errx=np.abs(errxyz[:, 0]).mean() if errxyz.size else float("nan"),
        erry=np.abs(errxyz[:, 1]).mean() if errxyz.size else float("nan"),
        errz=np.abs(errxyz[:, 2]).mean() if errxyz.size else float("nan"),
    )


def print_table(rows, methods, title):
    print(f"\n{title}")
    print(f"{'método':<16}{'recall':>8}{'err 3D mm':>12}{'(mediana)':>11}{'(máx)':>9}"
          f"{'|x|':>8}{'|y|':>8}{'|z|':>8}{'lat ms':>10}")
    print("-" * 90)
    for m in methods:
        s = summarize(rows, m.name)
        print(f"{m.name:<16}{100*s['recall']:>7.0f}%{s['err3_mean']:>12.1f}"
              f"{s['err3_median']:>11.1f}{s['err3_max']:>9.1f}"
              f"{s['errx']:>8.1f}{s['erry']:>8.1f}{s['errz']:>8.1f}{s['lat_mean']:>10.1f}")


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["method", "t", "x_joint", "ok", "dt_ms",
              "gt_x", "gt_y", "gt_z", "est_x", "est_y", "est_z",
              "err_x", "err_y", "err_z", "score", "label"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {"method": r["method"], "t": r["t"], "x_joint": r["x_joint"],
                   "ok": r["ok"], "dt_ms": f"{r['dt_ms']:.2f}"}
            out["gt_x"], out["gt_y"], out["gt_z"] = r["gt"]
            if r["ok"]:
                out["est_x"], out["est_y"], out["est_z"] = r["est"]
                out["err_x"], out["err_y"], out["err_z"] = r["err"]
                out["score"], out["label"] = r["score"], r["label"]
            w.writerow(out)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-static", type=int, default=15,
                     help="puntos del barrido estático (def. 15)")
    ap.add_argument("--moving-seconds", type=float, default=16.0,
                     help="duración del paso dinámico [s] (def. 16, la faja tarda ~15s en cruzar)")
    ap.add_argument("--sample-dt", type=float, default=0.4,
                     help="cada cuánto muestrear durante el paso dinámico [s] (def. 0.4)")
    ap.add_argument("--no-yolo", action="store_true", help="solo Método A")
    ap.add_argument("--device", default="cpu", help="device de YOLO: cpu (def.) o cuda:0")
    ap.add_argument("--mjcf", default="", help="ruta MJCF (def.: h1_2_description instalado)")
    args = ap.parse_args()

    mjcf_path = args.mjcf or os.path.join(
        get_package_share_directory("h1_2_description"), "mjcf", "h1_2_scene_surgery_table.xml")

    print("#" * 90)
    print("#  EVALUACIÓN DE PERCEPCIÓN — Método A (color) vs Método B (YOLO)")
    print("#  contra el ground truth exacto del simulador (PERCEPTION_PLAN.md, sección 4)")
    print("#" * 90)
    print(f"\nMJCF: {mjcf_path}")

    methods = build_methods(use_yolo=not args.no_yolo, device=args.device)
    print(f"Métodos: {[m.name for m in methods]}\n")

    print(f"=== EXPERIMENTO 1 — barrido espacial estático ({args.n_static} posiciones) ===")
    sim1 = make_sim(mjcf_path)
    static_rows = run_static_sweep(sim1, methods, args.n_static)
    sim1.close_viewer()
    print_table(static_rows, methods, "Resultado — barrido estático")
    save_csv(static_rows, os.path.join(RESULTS_DIR, "estatico.csv"))

    print(f"\n=== EXPERIMENTO 2 — mandarina en movimiento "
          f"({args.moving_seconds:.0f}s, cada {args.sample_dt:.1f}s) ===")
    moving_rows = run_moving_pass(mjcf_path, methods, args.moving_seconds, args.sample_dt)
    print_table(moving_rows, methods, "Resultado — objeto en movimiento")
    save_csv(moving_rows, os.path.join(RESULTS_DIR, "movimiento.csv"))

    print("\n" + "=" * 90)
    print("RESUMEN (ambos experimentos juntos)")
    print("=" * 90)
    print_table(static_rows + moving_rows, methods, "Estático + movimiento combinados")
    print("\nNota sobre el error en Z: es esperable que sea mayor que en X/Y en ambos métodos —")
    print("la profundidad medida es la de la superficie más cercana de la mandarina, no su")
    print("centro (sesgo geométrico conocido, ver PERCEPTION_PLAN.md sección 6). No es ruido.")
    print()


if __name__ == "__main__":
    main()
