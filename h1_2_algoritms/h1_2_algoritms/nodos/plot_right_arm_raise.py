#!/usr/bin/env python3
"""
plot_right_arm_raise.py — Gráficas del ensayo de `fk_right_arm_raise`.

Produce DOS figuras a partir del CSV que deja el nodo:

  1. <csv>_motores.png  — los 7 motores del brazo derecho: para cada uno,
     la referencia comandada (/joint_cmd) contra lo que realmente hizo
     MuJoCo (/joint_states), más un panel con el error de seguimiento de
     los 7 juntos.
  2. <csv>_efector.png  — el efector final calculado con la cinemática
     directa del paquete (fkine_arm_right_unitree): x/y/z contra el tiempo,
     rapidez, el camino en el plano sagital (X-Z, que es el plano en el que
     el brazo se levanta hacia adelante) y visto desde arriba (X-Y).

El mismo script sirve para el CSV del robot real: basta con que tenga las
mismas columnas. Con dos CSV compara los dos ensayos en los mismos ejes:

    ros2 run h1_2_algoritms plot_right_arm_raise sim.csv --compare real.csv \
        --labels simulador,real
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")          # sin ventana: guarda PNG y ya
import matplotlib.pyplot as plt
import numpy as np

JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
          "wrist_roll", "wrist_pitch", "wrist_yaw"]


def load(path):
    """Lee el CSV en un dict {columna: array}."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [[float(v) for v in row] for row in r if row]
    if not rows:
        raise SystemExit(f"{path}: el CSV no tiene datos")
    data = np.array(rows)
    return {name: data[:, i] for i, name in enumerate(header)}


def _phase_marks(ax, t_rise, t_hold, t_max):
    """Líneas verticales que separan subida / sostenido / bajada."""
    for t in (t_rise, t_rise + t_hold, 2 * t_rise + t_hold):
        if t is not None and 0 < t < t_max:
            ax.axvline(t, color="0.75", ls=":", lw=0.9, zorder=0)


def detect_phases(d):
    """Deduce (t_rise, t_hold) del propio comando: la subida termina cuando
    la velocidad de referencia del hombro vuelve a ~0 por primera vez."""
    if "dq_des_shoulder_pitch" not in d:
        return None, None
    moving = np.abs(d["dq_des_shoulder_pitch"]) > 1e-6
    if not moving.any():
        return None, None
    t = d["t"]
    idx = np.where(moving)[0]
    # fin del primer tramo continuo de movimiento
    brk = np.where(np.diff(idx) > 1)[0]
    i_end = idx[brk[0]] if brk.size else idx[-1]
    t_rise = t[i_end]
    t_hold = (t[idx[brk[0] + 1]] - t_rise) if brk.size else None
    return t_rise, t_hold


# ============================================================ FIGURA 1: MOTORES
def plot_motors(sets, labels, out_path):
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 13), sharex=True)
    fig.suptitle("Brazo derecho H1-2 — los 7 motores: comando vs. simulador",
                 fontsize=15, fontweight="bold")

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for k, jname in enumerate(JOINTS):
        ax = axes[k // 2][k % 2]
        for s, (d, lab) in enumerate(zip(sets, labels)):
            t = d["t"]
            c = colors[s % len(colors)]
            ax.plot(t, np.degrees(d[f"q_des_{jname}"]), ls="--", lw=1.6,
                    color=c, alpha=0.65,
                    label=f"comando ({lab})" if len(sets) > 1 else "comando (/joint_cmd)")
            ax.plot(t, np.degrees(d[f"q_meas_{jname}"]), lw=2.0, color=c,
                    label=f"real ({lab})" if len(sets) > 1 else "real (/joint_states)")
        t_rise, t_hold = detect_phases(sets[0])
        if t_rise:
            _phase_marks(ax, t_rise, t_hold or 0.0, sets[0]["t"][-1])
        ax.set_title(f"{k + 1}. right_{jname}_joint", fontsize=11)
        ax.set_ylabel("ángulo [°]")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="best")

    # ===== Panel 8: error de seguimiento de los 7 =====
    ax = axes[3][1]
    d = sets[0]
    for k, jname in enumerate(JOINTS):
        e = np.degrees(d[f"q_meas_{jname}"] - d[f"q_des_{jname}"])
        ax.plot(d["t"], e, lw=1.5, label=jname)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_title(f"8. Error de seguimiento (real − comando){'' if len(sets) == 1 else f' — {labels[0]}'}",
                 fontsize=11)
    ax.set_ylabel("error [°]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")

    for ax in axes[3]:
        ax.set_xlabel("tiempo [s]")

    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ============================================================ FIGURA 2: EFECTOR
def plot_end_effector(sets, labels, out_path):
    fig = plt.figure(figsize=(13.5, 10))
    fig.suptitle("Brazo derecho H1-2 — efector final (cinemática directa, frame torso_link)",
                 fontsize=15, fontweight="bold")

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    axis_colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}

    # ---- (1) x, y, z contra el tiempo -------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    for s, (d, lab) in enumerate(zip(sets, labels)):
        for comp in "xyz":
            suf = f" ({lab})" if len(sets) > 1 else ""
            ax1.plot(d["t"], d[f"ee_des_{comp}"], ls="--", lw=1.4,
                     color=axis_colors[comp], alpha=0.55,
                     label=f"{comp} comando{suf}")
            ax1.plot(d["t"], d[f"ee_meas_{comp}"], lw=2.0,
                     color=axis_colors[comp], label=f"{comp} real{suf}")
    t_rise, t_hold = detect_phases(sets[0])
    if t_rise:
        _phase_marks(ax1, t_rise, t_hold or 0.0, sets[0]["t"][-1])
    ax1.set_title("Posición del efector contra el tiempo", fontsize=11)
    ax1.set_xlabel("tiempo [s]")
    ax1.set_ylabel("posición [m]")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, ncol=2, loc="best")

    # ---- (2) rapidez y desvío comando-real --------------------------------
    # Dos escalas distintas (m/s y mm), así que eje gemelo: si se dibujan en
    # el mismo eje, una de las dos curvas queda aplastada contra el cero.
    ax2 = fig.add_subplot(2, 2, 2)
    ax2b = ax2.twinx()
    handles = []
    for s, (d, lab) in enumerate(zip(sets, labels)):
        p = np.column_stack([d["ee_meas_x"], d["ee_meas_y"], d["ee_meas_z"]])
        t = d["t"]
        dt = np.gradient(t)
        v = np.linalg.norm(np.gradient(p, axis=0) / dt[:, None], axis=1)
        pd_ = np.column_stack([d["ee_des_x"], d["ee_des_y"], d["ee_des_z"]])
        err = np.linalg.norm(p - pd_, axis=1) * 1000.0
        c = colors[s % len(colors)]
        suf = f" ({lab})" if len(sets) > 1 else ""
        handles += ax2.plot(t, v, lw=2.0, color=c, label=f"rapidez real{suf} [m/s]")
        handles += ax2b.plot(t, err, lw=1.6, ls="-.", color=c, alpha=0.75,
                             label=f"desvío comando−real{suf} [mm]")
    if t_rise:
        _phase_marks(ax2, t_rise, t_hold or 0.0, sets[0]["t"][-1])
    ax2.set_title("Rapidez del efector y desvío respecto del comando", fontsize=11)
    ax2.set_xlabel("tiempo [s]")
    ax2.set_ylabel("rapidez [m/s]")
    ax2b.set_ylabel("desvío comando−real [mm]")
    ax2.set_ylim(bottom=0.0)
    ax2b.set_ylim(bottom=0.0)
    ax2.grid(alpha=0.3)
    ax2.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="best")

    # ---- (3) camino en el plano sagital X-Z (el del levantamiento) --------
    ax3 = fig.add_subplot(2, 2, 3)
    for s, (d, lab) in enumerate(zip(sets, labels)):
        c = colors[s % len(colors)]
        suf = f" ({lab})" if len(sets) > 1 else ""
        ax3.plot(d["ee_des_x"], d["ee_des_z"], ls="--", lw=1.5, color=c,
                 alpha=0.6, label=f"comando{suf}")
        ax3.plot(d["ee_meas_x"], d["ee_meas_z"], lw=2.2, color=c, label=f"real{suf}")
        ax3.plot(d["ee_meas_x"][0], d["ee_meas_z"][0], "o", color=c, ms=9,
                 mfc="white", mew=2, zorder=5)
        ax3.plot(d["ee_meas_x"][-1], d["ee_meas_z"][-1], "s", color=c, ms=9, zorder=5)
        ax3.plot(d["ee_des_x"][-1], d["ee_des_z"][-1], "s", color=c, ms=8,
                 mfc="none", mew=1.6, alpha=0.7, zorder=5)
    ax3.set_title("Camino en el plano sagital X-Z  (○ inicio, □ final)", fontsize=11)
    ax3.set_xlabel("X [m]  (hacia adelante)")
    ax3.set_ylabel("Z [m]  (hacia arriba)")
    ax3.grid(alpha=0.3)
    ax3.set_aspect("equal", adjustable="datalim")
    ax3.margins(0.12)
    ax3.legend(fontsize=8, loc="best")

    # ---- (4) camino visto desde arriba X-Y --------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    for s, (d, lab) in enumerate(zip(sets, labels)):
        c = colors[s % len(colors)]
        suf = f" ({lab})" if len(sets) > 1 else ""
        ax4.plot(d["ee_des_x"], d["ee_des_y"], ls="--", lw=1.5, color=c,
                 alpha=0.6, label=f"comando{suf}")
        ax4.plot(d["ee_meas_x"], d["ee_meas_y"], lw=2.2, color=c, label=f"real{suf}")
        ax4.plot(d["ee_meas_x"][0], d["ee_meas_y"][0], "o", color=c, ms=9,
                 mfc="white", mew=2, zorder=5)
        ax4.plot(d["ee_meas_x"][-1], d["ee_meas_y"][-1], "s", color=c, ms=9, zorder=5)
        ax4.plot(d["ee_des_x"][-1], d["ee_des_y"][-1], "s", color=c, ms=8,
                 mfc="none", mew=1.6, alpha=0.7, zorder=5)
    ax4.set_title("Camino visto desde arriba X-Y  (○ inicio, □ final)", fontsize=11)
    ax4.set_xlabel("X [m]  (hacia adelante)")
    ax4.set_ylabel("Y [m]  (+ izquierda del robot)")
    ax4.grid(alpha=0.3)
    ax4.set_aspect("equal", adjustable="datalim")
    ax4.margins(0.12)
    ax4.legend(fontsize=8, loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def summary(d, label):
    q_err = np.array([d[f"q_meas_{j}"][-1] - d[f"q_des_{j}"][-1] for j in JOINTS])
    p0 = np.array([d["ee_meas_x"][0], d["ee_meas_y"][0], d["ee_meas_z"][0]])
    pf = np.array([d["ee_meas_x"][-1], d["ee_meas_y"][-1], d["ee_meas_z"][-1]])
    pdf = np.array([d["ee_des_x"][-1], d["ee_des_y"][-1], d["ee_des_z"][-1]])
    k = int(np.argmax(np.abs(q_err)))
    print(f"\n[{label}]  {len(d['t'])} muestras, {d['t'][-1]:.2f} s")
    print(f"  error articular final máximo : {np.degrees(np.abs(q_err).max()):.2f}° "
          f"en right_{JOINTS[k]}_joint")
    print(f"  efector inicial -> final     : {np.round(p0, 4)} -> {np.round(pf, 4)} m")
    print(f"  desplazamiento del efector   : {np.linalg.norm(pf - p0) * 1000:.0f} mm "
          f"(Δx={pf[0] - p0[0]:+.3f}, Δy={pf[1] - p0[1]:+.3f}, Δz={pf[2] - p0[2]:+.3f} m)")
    print(f"  desvío final comando-real    : {np.linalg.norm(pf - pdf) * 1000:.1f} mm")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV del ensayo (el que deja fk_right_arm_raise)")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="CSV(s) extra para superponer (p.ej. el del robot real)")
    ap.add_argument("--labels", default="",
                    help="etiquetas separadas por coma, una por CSV")
    ap.add_argument("--out-prefix", default="",
                    help="prefijo de salida (por defecto, el del primer CSV)")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    paths = [args.csv] + list(args.compare)
    sets = [load(p) for p in paths]
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(paths):
            raise SystemExit(f"--labels tiene {len(labels)} etiquetas y hay {len(paths)} CSV")
    else:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    prefix = args.out_prefix or os.path.splitext(os.path.abspath(paths[0]))[0]
    f1 = plot_motors(sets, labels, prefix + "_motores.png")
    f2 = plot_end_effector(sets, labels, prefix + "_efector.png")

    for d, lab in zip(sets, labels):
        summary(d, lab)

    print(f"\nGráficas guardadas:\n  1. {f1}\n  2. {f2}")


if __name__ == "__main__":
    main()
