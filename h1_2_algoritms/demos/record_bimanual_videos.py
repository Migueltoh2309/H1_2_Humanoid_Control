#!/usr/bin/env python3
"""Videos comparativos del control bimanual, para presentaciones.

Cada video pone lado a lado dos corridas del MISMO escenario con distintos
modos (p.ej. "sin evasión" vs "con capa 1"), usando exactamente el código de
demos/evaluate_bimanual_avoidance.py (`run(..., on_step=...)`): lo que se ve
es lo que se evalúa, no una animación aparte.

Qué se dibuja en cada panel:
  * el robot y la escena de MuJoCo, desde una cámara externa;
  * esferas en los objetivos de cada mano (roja = izquierda, azul = derecha);
  * los vóxeles de la cámara que están cerca de los brazos (morado): lo que
    el QP considera obstáculo en ese instante;
  * el camino planificado por la capa 3 (verde), cuando lo hay;
  * los puntos de contacto reales de MuJoCo (rojo) y un aviso "CHOQUE";
  * distancias mínimas y el último evento de las capas 2 y 3.

Formato: MP4 H.264 (yuv420p), 1920x1080, 20 fps = tiempo real (un cuadro por
ciclo de control), que PowerPoint reproduce directamente.

Uso:
    MUJOCO_GL=egl python3 demos/record_bimanual_videos.py            # todos
    MUJOCO_GL=egl python3 demos/record_bimanual_videos.py --only 3   # uno
    python3 demos/record_bimanual_videos.py --out ~/mis_videos

Requiere imageio-ffmpeg (está en requirements.txt del workspace).
"""
import argparse
import os
import sys
import textwrap
import time
import unicodedata

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_bimanual_avoidance as E                                  # noqa: E402
from h1_2_algoritms.bimanual.collision_model import arm_capsules, points_to_segment  # noqa: E402
from h1_2_algoritms.movimiento.fk_functions import fkine_arm_left_unitree, fkine_arm_right_unitree  # noqa: E402

DEFAULT_OUT = os.path.join(E.PKG_DIR, "videos")
W, H = 1920, 1080
PANEL = 960                 # cada panel es cuadrado: 960x960
HEADER = H - PANEL          # 120 px de título arriba
FPS = 20                    # = 1/dt del control: tiempo real

MODE_LABEL = {
    "off": "Sin evasion (QP sin restricciones)",
    "self": "Capa 1: QP + velocity dampers (sin camara)",
    "self+frame": "Capa 1 + camara (solo el ultimo cuadro)",
    "self+cam": "Capa 1 + camara con memoria",
    "full": "Capas 1 + 2 (coordinacion por prioridad)",
    "full+plan": "Capas 1 + 2 + 3 (planificador global)",
}

# (archivo, escenario, modo izquierda, modo derecha, segundos, cámara, título)
# cámara = (lookat xyz en MUNDO, azimut, elevación, distancia)
CAM_FRONT = ((0.30, 0.0, 1.10), 180, -18, 1.35)
CAM_TRAY = ((0.30, -0.15, 1.15), 125, -8, 1.35)
CAM_BELT = ((0.30, 0.0, 0.95), 165, -30, 1.45)
VIDEOS = [
    ("01_choque_vs_evasion_brazos.mp4", "circles", "off", "self", 12.0, CAM_FRONT,
     "Las manos se cruzan en circulos: sin evasion chocan; con la capa 1 nunca"),
    ("02_zona_compartida_capa2.mp4", "shared", "self+cam", "full", 10.0, CAM_FRONT,
     "Ambas manos quieren el mismo punto: solo capa 1 se bloquea; la capa 2 decide quien pasa"),
    ("03_bandeja_capa3.mp4", "obstacle", "full", "full+plan", 12.0, CAM_TRAY,
     "Bandeja sobre la mano: capas 1+2 se traban debajo; la capa 3 planifica y la rodea"),
    ("04_oclusion_memoria_camara.mp4", "table", "self+frame", "self+cam", 8.0, CAM_BELT,
     "Manos hacia la faja: la mano tapa a la camara; la memoria de voxeles evita el choque"),
    ("05_sin_camara_atraviesa.mp4", "obstacle", "self", "self+cam", 6.0, CAM_TRAY,
     "Sin camara el modelo no conoce la bandeja y la atraviesa; con camara la evita"),
]


# ============================================================ dibujo 3D

def _add_geom(scn, gtype, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                        np.eye(3).flatten() if mat is None else mat, np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def _torso_to_world(scene, p):
    tid = scene.tid
    return scene.d.xmat[tid].reshape(3, 3) @ np.asarray(p) + scene.d.xpos[tid]


def decorate(scene, scn, ctx, ctrl_d_i_env=0.20, voxel=0.03):
    # objetivos
    for side, rgba in (("left", (1, 0.1, 0.1, 0.6)), ("right", (0.1, 0.3, 1, 0.6))):
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0],
                  _torso_to_world(scene, np.asarray(ctx["goals"][side])[:3]), rgba)
    # vóxeles cercanos a los brazos (lo que el QP considera obstáculo)
    P = ctx["obstacles"]
    if P is not None and len(P):
        q = ctx["q"]
        caps = arm_capsules("left", q[:7]) + arm_capsules("right", q[7:])
        near = np.zeros(len(P), dtype=bool)
        for c in caps:
            d, _ = points_to_segment(P, c["a"], c["b"])
            near |= d - c["r"] < ctrl_d_i_env
        for p in P[near][:600]:
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [voxel / 2] * 3, _torso_to_world(scene, p),
                      (0.6, 0.2, 0.9, 0.45))
    # camino de la capa 3
    pl = ctx["planner"]
    if pl is not None and pl.plan is not None:
        side = pl.plan["side"]
        fk = fkine_arm_left_unitree if side == "left" else fkine_arm_right_unitree
        for i, qw in enumerate(pl.plan["path"]):
            T = fk(qw)
            hand = T[:3, 3] + T[:3, 0] * 0.10
            done = i < pl.plan["i"]
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], _torso_to_world(scene, hand),
                      (0.2, 0.9, 0.2, 0.35 if done else 0.9))


# ============================================================ texto 2D
# OpenCV dibuja en BGR (no RGB): rojo = (B, G, R) = (30, 30, 200).
WHITE = (255, 255, 255)
RED = (30, 30, 200)
LIGHT_RED = (120, 120, 255)
LIGHT_GREEN = (180, 255, 180)
YELLOW = (120, 230, 255)

def put(img, text, org, scale=0.8, color=(255, 255, 255), thick=2, bg=None):
    # Las fuentes Hershey de OpenCV solo tienen ASCII: "planificación" -> "planificacion".
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    if bg is not None:
        x, y = org
        cv2.rectangle(img, (x - 8, y - h - 10), (x + w + 8, y + base + 6), bg, -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def overlay(img, ctx, label, events, crash_steps):
    gt = ctx["gt"]
    put(img, label, (20, 45), 0.85, (255, 255, 255), 2, bg=(40, 40, 40))
    put(img, f"t = {ctx['t']:4.1f} s", (20, 95), 0.8, (255, 255, 255), 2, bg=(40, 40, 40))
    crash = gt["n_arm_arm"] + gt["n_arm_env"] + gt["n_arm_torso"] > 0
    if crash:
        put(img, "CHOQUE", (PANEL - 190, 60), 1.3, WHITE, 3, bg=RED)
    y = PANEL - 110
    d_arm = min(gt["d_arms"], 0.25)
    line = f"dist. brazo-brazo: {1000 * d_arm:4.0f} mm"
    if not np.isnan(gt.get("d_obst", np.nan)):
        line += f"   dist. a la bandeja: {1000 * max(gt['d_obst'], 0):4.0f} mm"
    elif gt["d_belt"] < 0.2:
        line += f"   altura sobre la faja: {1000 * gt['d_belt']:4.0f} mm"
    put(img, line, (20, y), 0.7, (255, 255, 255), 2, bg=(40, 40, 40))
    tasks = ctx["tasks"]
    status = f"pasos con choque: {crash_steps}"
    if tasks is not None:
        status += f"   tareas completas: {sum(x.done for x in tasks.values())}/2"
    put(img, status, (20, y + 45), 0.7, LIGHT_RED if crash_steps else LIGHT_GREEN, 2,
        bg=(40, 40, 40))
    # último evento de capas 2/3 durante 3 s
    recent = [(t_, m_) for t_, m_ in events if ctx["t"] - t_ < 3.0]
    if recent:
        wrapped = textwrap.wrap(recent[-1][1], 62)[:2]
        for i, ln in enumerate(wrapped):
            put(img, ln, (20, 150 + 38 * i), 0.62, YELLOW, 2, bg=(40, 40, 40))


# ============================================================ grabación

def record_run(scenario, mode, seconds, camdef, mjcf):
    m = E.build_model(mjcf, with_tray=(scenario == "obstacle"))
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
    scene = E.Scene(m, has_tray=(scenario == "obstacle"))
    rend = mujoco.Renderer(m, PANEL, PANEL, max_geom=12000)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    m.vis.scale.contactwidth, m.vis.scale.contactheight = 0.06, 0.02
    m.vis.rgba.contactpoint[:] = (1, 0, 0, 1)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = camdef
    frames, flags, crash = [], [], [0]

    def on_step(ctx):
        gt = ctx["gt"]
        hit = gt["n_arm_arm"] + gt["n_arm_env"] + gt["n_arm_torso"] > 0
        flags.append(hit)
        if hit:
            crash[0] += 1
        rend.update_scene(scene.d, camera=cam, scene_option=opt)
        decorate(scene, rend.scene, ctx)
        img = rend.render()[:, :, ::-1].copy()      # RGB -> BGR para OpenCV
        events = ([] if ctx["coord"] is None else [(t, "[capa 2] " + e) for t, e in ctx["coord"].events]) + \
                 ([] if ctx["planner"] is None else [(t, "[capa 3] " + e) for t, e in ctx["planner"].events])
        events.sort(key=lambda x: x[0])
        overlay(img, ctx, MODE_LABEL[mode], events, crash[0])
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames.append(jpg)

    E.run(scene, scenario, mode, seconds, 0.05, 3, on_step=on_step)   # mismo dt y cámara que la evaluación
    rend.close()
    scene.close()
    return frames, flags


FREEZE_S = 1.5   # pausa en el primer choque: dura 0.25 s y en una diapositiva no se ve


def compose(path, title, left, right):
    """left/right = (cuadros JPEG, choque por cuadro). En el PRIMER cuadro en
    que cualquiera de los dos paneles choca, ambos se congelan FREEZE_S s con
    un aviso, para que el choque se vea en la presentación."""
    (left, fl), (right, fr) = left, right
    n = max(len(left), len(right))
    hits = [i for i in range(n) if (i < len(fl) and fl[i]) or (i < len(fr) and fr[i])]
    freeze_at = hits[0] if hits else None
    writer = imageio_ffmpeg.write_frames(path, (W, H), fps=FPS, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
    writer.send(None)
    header = np.full((HEADER, W, 3), 25, np.uint8)
    put(header, title, (30, 75), 1.05, (255, 255, 255), 2)
    for i in range(n):
        a = cv2.imdecode(left[min(i, len(left) - 1)], cv2.IMREAD_COLOR)
        b = cv2.imdecode(right[min(i, len(right) - 1)], cv2.IMREAD_COLOR)
        frame = np.vstack((header, np.hstack((a, b))))
        cv2.line(frame, (PANEL, HEADER), (PANEL, H), (255, 255, 255), 3)
        writer.send(np.ascontiguousarray(frame[:, :, ::-1]))   # BGR -> RGB
        if i == freeze_at:
            paused = frame.copy()
            put(paused, "|| PAUSA: primer choque (puntos de contacto en rojo)",
                (W // 2 - 420, H - 30), 0.9, WHITE, 2, bg=RED)
            for _ in range(int(FREEZE_S * FPS)):
                writer.send(np.ascontiguousarray(paused[:, :, ::-1]))
    writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", type=int, nargs="+", help="números de video (1..%d)" % len(VIDEOS))
    ap.add_argument("--mjcf", default=E.DEFAULT_MJCF)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for idx, (fname, sc, m_left, m_right, secs, camdef, title) in enumerate(VIDEOS, start=1):
        if args.only and idx not in args.only:
            continue
        t0 = time.time()
        print(f"[{idx}] {fname}: {sc}  {m_left}  vs  {m_right}  ({secs:.0f} s)", flush=True)
        left = record_run(sc, m_left, secs, camdef, args.mjcf)
        right = record_run(sc, m_right, secs, camdef, args.mjcf)
        path = os.path.join(args.out, fname)
        compose(path, title, left, right)
        print(f"    -> {path}  ({os.path.getsize(path) / 1e6:.1f} MB, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
