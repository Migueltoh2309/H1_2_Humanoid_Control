#!/usr/bin/env python3
"""Métricas de la sección 9 de una o varias tiradas del SIMULADOR, con la verdad de terreno
que el programa guarda en el registro (meta.verdad_inicio / verdad_final) y la geometría de la
escena (sim/estaciones/estacion_n<N>_s<S>.json). No se fía de la percepción del robot.

  * nivel 1: error de la yema (FK de las juntas medidas en el contacto) al centro de la diana
    más cercana, fuerza de contacto; superado si las 4 a <= 2 cm y <= 300 gf;
  * niveles 2-3: objeto COLOCADO = al final dentro de su bandeja, quieto y sin volcar (el cubo
    sobre cualquier cara; el vaso y el prisma de pie); cogidos a la primera; tiempo por objeto;
  * nivel 4: centro a la cruz (<= 1.5 cm) y giro respecto del inicial (<= 15 grados de 90);
  * todos: pasos (un pie a más de 5 cm de su huella), giro y fuerza máximos, salida segura.

    python3 herramientas/metricas.py datos/tiradas/nivel2_20261014_100615
    python3 herramientas/metricas.py datos/tiradas/nivel*_2026101[4]*      # varias -> tabla
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
from manip.config import cargar, cargar_objetos  # noqa: E402
from manip.estacion import Estacion  # noqa: E402

PASO_M = 0.05


def _escena(nivel, semilla):
    ruta = os.path.join(RAIZ, "sim", "estaciones", f"estacion_n{nivel}_s{semilla}.json")
    return json.load(open(ruta)) if os.path.isfile(ruta) else None


def tirada(dirr, est, cat):
    d = json.load(open(os.path.join(dirr, "eventos.json")))
    meta, met = d["meta"], d.get("metricas", {})
    nivel = meta["nivel"]
    esc = meta.get("escena", {})
    geo = _escena(nivel, esc.get("semilla")) if esc.get("semilla") is not None else None
    v0, v1 = meta.get("verdad_inicio"), meta.get("verdad_final")
    out = {"tirada": os.path.basename(dirr), "nivel": nivel, "semilla": esc.get("semilla"), "s": met.get("s"),
           "giro_max_deg": met.get("giro_max_deg"), "salida_segura": met.get("salida_segura") or "",
           "fuerza_max_gf": max((max(v) for v in d.get("fuerza_max_gf", {}).values()), default=None)}
    if v1:
        pies = v1.get("pies_max_m", {})
        out["pie_max_cm"] = 100 * max(pies.values()) if pies else None
        out["pasos"] = sum(1 for v in pies.values() if v > PASO_M)
    pel = np.array(v0["cuerpos"]["pelvis"]["pos"][:2]) if v0 and "pelvis" in v0["cuerpos"] else np.zeros(2)
    if nivel == 1 and geo:
        dianas = np.array(geo["dianas"])[:, :2]
        errs, fz = [], []
        for o in d["objetos"]:
            if not o.get("tocada"):
                errs.append(float("nan"))
                continue
            yema = np.array(o["yema"][:2]) + pel          # torso -> mundo (el torso está sobre la pelvis)
            errs.append(100 * float(np.min(np.linalg.norm(dianas - yema, axis=1))))
            fz.append(o.get("fuerza_gf") or 0.0)
        out["dianas_tocadas"] = f"{sum(1 for e in errs if e == e)}/{len(dianas)}"
        out["error_cm"] = [round(e, 2) for e in errs]
        out["fuerza_contacto_gf"] = [round(f) for f in fz]
        out["superado"] = len(errs) == 4 and all(e == e and e <= 2.0 for e in errs) and all(f <= 300 for f in fz)
    elif nivel in (2, 3) and v1:
        col, detalle = [], {}
        for k, o in cat.items():
            c = v1["cuerpos"].get(k)
            if c is None:
                continue
            b = est.bandejas[o["bandeja"]]
            dentro = bool(b.contiene(c["pos"][0], c["pos"][1], margen=0.0)) and c["pos"][2] > est.z_mesa - 0.02
            incl = c["inclinacion_deg"]
            if o["forma"] == "caja" and abs(o["medidas"][0] - o["medidas"][2]) < 1e-3:
                incl = min(incl, abs(incl - 90))           # el cubo sobre cualquier cara
            de_pie = o["forma"] == "esfera" or incl < 20
            ok = dentro and de_pie
            detalle[k] = {"en_bandeja": dentro, "inclinacion_deg": round(c["inclinacion_deg"], 1), "colocado": ok}
            if ok:
                col.append(k)
        intentos = {}
        for o in d["objetos"]:
            if o.get("entrega"):
                continue
            intentos.setdefault(o["clase"], []).append(o)
        out["colocados"] = f"{len(col)}/{len(cat)}"
        out["a_la_primera"] = sum(1 for k, v in intentos.items() if k in col and len(v) == 1)
        out["detalle"] = detalle
        t = [o["s"] for o in d["objetos"] if o.get("colocado")]
        out["s_por_objeto"] = round(float(np.mean(t)), 1) if t else None
        out["superado"] = len(col) == len(cat) and not out.get("pasos")
    elif nivel == 4 and v0 and v1 and geo:
        clase = next((o["clase"] for o in d["objetos"] if o.get("nivel4")), None)
        if clase:
            c0, c1 = v0["cuerpos"][clase], v1["cuerpos"][clase]
            cruz = np.array(geo["cruz"][:2])
            e_pos = 100 * float(np.linalg.norm(np.array(c1["pos"][:2]) - cruz))
            o = cat[clase]
            sim = 180.0 if (o["forma"] == "caja" and abs(o["medidas"][0] - o["medidas"][1]) > 1e-3) else 90.0
            giro = c1["guinada_deg"] - c0["guinada_deg"]
            e_gui = abs(math.degrees(math.remainder(math.radians(giro - 90.0), math.radians(sim))))
            out.update(objeto=clase, error_cruz_cm=round(e_pos, 2), giro_real_deg=round(giro, 1),
                       error_giro_deg=round(e_gui, 1), de_pie=c1["inclinacion_deg"] < 20,
                       superado=e_pos <= 1.5 and e_gui <= 15 and c1["inclinacion_deg"] < 20)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiradas", nargs="+")
    ap.add_argument("--json", help="guardar todo aquí")
    a = ap.parse_args()
    est, cat = Estacion(cargar()), cargar_objetos()
    filas = []
    for pat in a.tiradas:
        for dirr in sorted(glob.glob(pat)):
            if os.path.isfile(os.path.join(dirr, "eventos.json")):
                try:
                    filas.append(tirada(dirr, est, cat))
                except Exception as e:
                    print(f"{dirr}: {e}")
    for f in filas:
        base = f"{f['tirada']:28s} N{f['nivel']} s{f['semilla']}  {'SUPERADO' if f.get('superado') else 'no      '}"
        extra = {k: v for k, v in f.items() if k not in ("tirada", "nivel", "semilla", "superado", "detalle")}
        print(base, " ".join(f"{k}={v}" for k, v in extra.items() if v not in (None, "")))
    if a.json:
        json.dump(filas, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
