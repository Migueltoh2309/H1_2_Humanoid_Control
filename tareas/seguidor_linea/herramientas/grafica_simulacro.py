#!/usr/bin/env python3
"""Hito 3 (simulacro): del registro de una tirada con --simulacro, la grafica que
muestra que las ordenes tienen el signo correcto y que el vigilante para.

  arriba   donde ve el seguidor la linea (desplazamiento lateral, + = a la izquierda)
           y el punto de pure pursuit (obj_y)
  medio    vyaw y vx calculados (NO enviados): vyaw > 0 gira a la izquierda
  abajo    estado del supervisor; la PARADA del vigilante, con su motivo

Signo correcto: con el punto de mira a la izquierda (obj_y > 0) la orden tiene que
girar a la izquierda (vyaw > 0), y al reves. Se cuenta sobre los ciclos con |obj_y| > 3 cm.

    python3 herramientas/grafica_simulacro.py datos/tiradas/nivel2_XXXX.csv
"""
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ESTADOS = ["ESPERA", "SEGUIMIENTO", "LINEA_PERDIDA", "GIRO_ESQUINA", "LLEGADA", "FIN", "PARADA"]


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return np.nan


def main():
    ruta = sys.argv[1]
    F = list(csv.DictReader(open(ruta)))
    t = np.array([num(r, "t") for r in F])
    desp = np.array([num(r, "est_desp") for r in F]) * 100
    oy = np.array([num(r, "obj_y") for r in F]) * 100
    vyaw = np.array([num(r, "vyaw") for r in F])
    vx = np.array([num(r, "vx") for r in F])
    est = [r["estado"] for r in F]
    sel = np.isfinite(oy) & (np.abs(oy) > 3) & (np.abs(vyaw) > 1e-3)
    ok = np.sign(vyaw[sel]) == np.sign(oy[sel])
    pct = 100.0 * ok.mean() if sel.any() else float("nan")
    meta = {}
    try:
        meta = json.load(open(ruta[:-4] + ".json"))
    except (OSError, ValueError):
        pass
    fin = meta.get("motivo_fin") or meta.get("resumen", {}).get("motivo_fin") or ""

    fig, ax = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})
    ax[0].plot(t, desp, label="desplazamiento de la línea (cm, + izquierda)")
    ax[0].plot(t, oy, alpha=0.7, label="punto de mira, lateral (cm)")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_ylabel("cm")
    ax[1].plot(t, vyaw, "C3", label="vyaw calculado (rad/s, + izquierda)")
    ax[1].plot(t, vx, "C2", label="vx calculado (m/s)")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_ylabel("órdenes (NO enviadas)")
    y = [ESTADOS.index(e) if e in ESTADOS else -1 for e in est]
    ax[2].plot(t, y, "k", drawstyle="steps-post")
    ax[2].set_yticks(range(len(ESTADOS)))
    ax[2].set_yticklabels(ESTADOS, fontsize=7)
    ax[2].set_xlabel("t (s)")
    if "PARADA" in est:
        tp = t[est.index("PARADA")]
        for a_ in ax:
            a_.axvline(tp, color="r", ls="--")
        ax[2].annotate("PARADA: " + str(F[est.index("PARADA")].get("motivo", ""))[:60], (tp, 6),
                       xytext=(-10, -14), textcoords="offset points", ha="right", color="r", fontsize=8)
    ax[0].set_title(f"Simulacro: signo de vyaw correcto en {pct:.1f} % de {int(sel.sum())} ciclos "
                    f"(|mira| > 3 cm) · {fin}", fontsize=10)
    fig.tight_layout()
    png = ruta[:-4] + "_simulacro.png"
    fig.savefig(png, dpi=100)
    print(f"signo de vyaw correcto: {pct:.1f} % de {int(sel.sum())} ciclos con |mira| > 3 cm")
    print(f"grafica: {png}")


if __name__ == "__main__":
    main()
