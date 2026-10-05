#!/usr/bin/env python3
"""Metricas de una tirada (reto, sec. 9) a partir de su CSV + JSON.

Con una tirada del SIMULADOR, ademas, metricas con VERDAD DE TERRENO (la pose real
de la pelvis, TF world -> pelvis, grabada en gt_x, gt_y, gt_yaw) contra la
geometria exacta de la pista (sim/pistas/pista_nivelN.json):
  * error lateral REAL del centro del robot respecto de la linea (medio, maximo);
  * criterio de exito: el pie (centro +-0.12 m) no sale mas de 30 cm de la linea;
  * distancia de parada REAL de la punta del pie a la barra (+ = se paso);
  * error de la PERCEPCION y de la ESTIMACION: desplazamiento y angulo medidos
    frente a los reales en cada ciclo.

    python3 herramientas/metricas.py datos/tiradas/nivel1_XXXX.csv [--pista sim/pistas/pista_nivel1.json] [--png]
"""
import argparse
import csv
import json
import math
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
SEMI_ANCHO_PIES = 0.12       # m: los pies quedan a +-12 cm del centro del robot
PIE_DELANTE = 0.10


def leer_csv(ruta):
    with open(ruta) as f:
        filas = list(csv.DictReader(f))
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan
    cols = {k: np.array([num(r[k]) for r in filas]) for k in filas[0] if k not in ("estado", "motivo", "med_esquina", "est_esquina")}
    cols["estado"] = np.array([r["estado"] for r in filas])
    return cols


def verdad_linea(pista, x, y, yaw):
    """Para la pose (x, y, yaw): (desplazamiento con signo de la linea respecto del
    robot, angulo de la linea respecto del robot, s del punto mas cercano)."""
    L = np.array(pista["linea"])
    d = np.hypot(L[:, 0] - x, L[:, 1] - y)
    i = int(np.argmin(d))
    lx, ly, th, s = L[i]
    # componente lateral (en el marco del robot) del punto mas cercano de la linea
    dx, dy = lx - x, ly - y
    lat = -dx * math.sin(yaw) + dy * math.cos(yaw)
    # desplazamiento en x = 0 del robot: corrige por el angulo relativo
    ang = (th - yaw + math.pi) % (2 * math.pi) - math.pi
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    desp = lat - fwd * math.tan(ang) if abs(ang) < 1.3 else lat
    dist = float(d[i])
    if i == 0 or i == len(L) - 1:
        # antes del inicio de la linea (al marcar el paso en ESPERA el robot deriva unos cm
        # hacia atras) o pasado el final: distancia LATERAL a su prolongacion, no al extremo
        dist = abs(-(x - lx) * math.sin(th) + (y - ly) * math.cos(th))
    return desp, ang, s, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--pista", default=None)
    ap.add_argument("--png", action="store_true", help="grafica de trayectoria y errores (matplotlib)")
    a = ap.parse_args()
    C = leer_csv(a.csv)
    with open(a.csv[:-4] + ".json") as f:
        res = json.load(f)
    print(f"Tirada {os.path.basename(a.csv)}: {res['motivo_fin']}")
    for k in ("tiempo_recorrido_s", "recorrido_estimado_m", "error_lateral_medio_m", "error_lateral_max_m",
              "pct_confianza_sobre_umbral", "roll_max_deg", "pitch_max_deg", "percepcion_ms_media", "percepcion_ms_p95"):
        print(f"  {k:30s} {res.get(k)}")
    if np.all(np.isnan(C["gt_x"])):
        print("  (sin verdad de terreno: tirada del robot real; la distancia de parada se mide con cinta)")
        return
    nivel = str(res.get("nivel"))
    ruta = a.pista or os.path.join(AQUI, "..", "sim", "pistas", f"pista_nivel{nivel}.json")
    with open(ruta) as f:
        pista = json.load(f)
    verdad = {"tirada": os.path.basename(a.csv), "nivel": nivel, "motivo_fin": res["motivo_fin"],
              "tiempo_s": res["tiempo_recorrido_s"]}
    gx, gy, gyaw = C["gt_x"], C["gt_y"], C["gt_yaw"]
    ok = ~np.isnan(gx)
    V = [verdad_linea(pista, x, y, w) if k else (math.nan,) * 4 for x, y, w, k in zip(gx, gy, gyaw, ok)]
    desp_v = np.array([v[0] for v in V])
    ang_v = np.array([v[1] for v in V])
    dist_v = np.array([v[3] for v in V])
    andando = ok & np.isin(C["estado"], ["SEGUIMIENTO", "LINEA_PERDIDA", "LLEGADA", "GIRO_ESQUINA"])
    print("\n  VERDAD DE TERRENO (simulador)")
    if andando.any():
        e = dist_v[andando]
        print(f"  error lateral real del centro   medio {e.mean():.3f} m, max {e.max():.3f} m")
        pie = e.max() + SEMI_ANCHO_PIES
        print(f"  pie fuera de la linea (max)     {pie:.3f} m  -> {'OK (< 0.30)' if pie <= 0.30 else 'FALLA (> 0.30)'}")
        verdad.update(error_medio_m=round(float(e.mean()), 4), error_max_m=round(float(e.max()), 4),
                      pie_max_m=round(float(pie), 3))
    # parada: punta del pie respecto de la barra, a lo largo del rumbo final de la linea
    b = pista["barra"]
    i_fin = np.flatnonzero(ok)[-1]
    px = gx[i_fin] + PIE_DELANTE * math.cos(gyaw[i_fin])
    py = gy[i_fin] + PIE_DELANTE * math.sin(gyaw[i_fin])
    paso = (px - b["x"]) * math.cos(b["rumbo_linea"]) + (py - b["y"]) * math.sin(b["rumbo_linea"])
    print(f"  parada: punta del pie a {paso:+.3f} m de la barra ({'se paso' if paso > 0 else 'antes'}; "
          f"el reto exige no pasarse mas de 0.30 y llegar)  -> {'OK' if -0.30 <= paso <= 0.30 else 'FALLA'}")
    llego = res["motivo_fin"].startswith("FIN: pie sobre la barra")
    verdad.update(parada_m=round(paso, 3), llego=llego,
                  exito=bool(llego and verdad.get("pie_max_m", 9) <= 0.30 and -0.30 <= paso <= 0.30))
    for nombre, col_d, col_a in (("percepcion", "med_desp", "med_ang"), ("estimacion", "est_desp", "est_ang")):
        conf_min = res.get("config", {}).get("estimacion", {}).get("confianza_min", 0.35)
        col_c = "med_conf" if nombre == "percepcion" else "est_conf"
        # en el giro sobre el eje (esquina) la "linea respecto del robot" es ambigua: fuera
        m = andando & (C["estado"] != "GIRO_ESQUINA") & ~np.isnan(C[col_d]) & (C[col_c] >= conf_min)
        if m.sum() < 3:
            continue
        ed = C[col_d][m] - desp_v[m]
        ea = np.degrees(((C[col_a][m] - ang_v[m]) + math.pi) % (2 * math.pi) - math.pi)
        verdad[f"{nombre}_rms_cm"] = round(float(np.sqrt((ed ** 2).mean()) * 100), 2)
        print(f"  {nombre:11s} (en x=0, extrapolado) desplazamiento: sesgo {ed.mean() * 100:+.1f} cm, RMS {np.sqrt((ed ** 2).mean()) * 100:.1f} cm"
              f" | angulo: sesgo {ea.mean():+.1f} grados, RMS {np.sqrt((ea ** 2).mean()):.1f} grados   ({m.sum()} ciclos)")
    print(f"  EXITO (reto, sec. 9): {'SI' if verdad['exito'] else 'NO'}")
    with open(a.csv[:-4] + "_verdad.json", "w") as f:
        json.dump(verdad, f, indent=1)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        L = np.array(pista["linea"])
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(L[:, 0], L[:, 1], "k-", lw=3, label="linea")
        ax[0].plot(gx[ok], gy[ok], "C1-", label="robot (real)")
        ax[0].plot(b["x"], b["y"], "rs", label="barra")
        ax[0].axis("equal"); ax[0].legend(); ax[0].set_title("trayectoria")
        ax[1].plot(C["t"][andando], dist_v[andando] * 100, label="error lateral real [cm]")
        ax[1].plot(C["t"], C["med_conf"] * 100, ".", ms=2, label="confianza [%]")
        ax[1].legend(); ax[1].set_xlabel("t [s]")
        png = a.csv[:-4] + ".png"
        fig.tight_layout(); fig.savefig(png, dpi=90)
        print(f"  grafica: {png}")


if __name__ == "__main__":
    main()
