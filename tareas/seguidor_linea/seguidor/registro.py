"""Bloque de REGISTRO (reto, sec. 5 y 9): CSV por ciclo + resumen JSON por tirada,
suficientes para reconstruir la tirada (como datos_cuadrado/ en Codigos).
Las metricas de la sec. 9 se calculan al cerrar."""
import csv
import json
import math
import os
import time

import numpy as np

COLUMNAS = ["t", "estado", "motivo",
            "med_valida", "med_desp", "med_ang", "med_curv", "med_conf", "med_barra", "med_esquina", "med_ms",
            "est_valida", "est_desp", "est_ang", "est_curv", "est_conf", "est_barra", "est_esquina",
            "sin_linea_m", "recorrido", "yaw", "roll", "pitch", "obj_x", "obj_y",
            "vx", "vy", "vyaw", "parar", "edad_camara", "gt_x", "gt_y", "gt_yaw"]


def _f(v, n=4):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (tuple, list)):
        return ";".join(str(_f(x, n)) for x in v)
    if isinstance(v, float):
        return "" if math.isnan(v) else round(v, n)
    return v


class Registro:
    def __init__(self, carpeta, nombre, meta):
        os.makedirs(carpeta, exist_ok=True)
        self.base = os.path.join(carpeta, f"{nombre}_{time.strftime('%Y%m%d_%H%M%S')}")
        self.f = open(self.base + ".csv", "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=COLUMNAS)
        self.w.writeheader()
        self.meta = meta
        self.filas = []

    def fila(self, **k):
        r = {c: _f(k.get(c)) for c in COLUMNAS}
        self.w.writerow(r)
        self.filas.append(k)
        if len(self.filas) % 20 == 0:
            self.f.flush()              # que el CSV se pueda mirar durante la tirada

    def cerrar(self, eventos, motivo_fin, conf_min):
        self.f.close()
        F = self.filas
        con_linea = [r for r in F if r.get("est_valida") and r.get("estado") in ("SEGUIMIENTO", "LLEGADA")]
        err = np.array([abs(r["est_desp"]) for r in con_linea]) if con_linea else np.zeros(0)
        conf = np.array([r.get("med_conf") or 0.0 for r in F if r.get("estado") not in ("ESPERA",)])
        andando = [r for r in F if r.get("estado") not in ("ESPERA", "FIN", "PARADA")]
        ms = np.array([r["med_ms"] for r in F if r.get("med_ms")])
        resumen = {
            **self.meta,
            "motivo_fin": motivo_fin,
            "tiempo_recorrido_s": round(andando[-1]["t"] - andando[0]["t"], 2) if andando else 0.0,
            "recorrido_estimado_m": round(F[-1].get("recorrido", 0.0), 3) if F else 0.0,
            "error_lateral_medio_m": round(float(err.mean()), 4) if len(err) else None,
            "error_lateral_max_m": round(float(err.max()), 4) if len(err) else None,
            "pct_confianza_sobre_umbral": round(100.0 * float((conf >= conf_min).mean()), 1) if len(conf) else None,
            "roll_max_deg": round(max((abs(math.degrees(r.get("roll") or 0)) for r in F), default=0.0), 2),
            "pitch_max_deg": round(max((abs(math.degrees(r.get("pitch") or 0)) for r in F), default=0.0), 2),
            "percepcion_ms_media": round(float(ms.mean()), 2) if len(ms) else None,
            "percepcion_ms_p95": round(float(np.percentile(ms, 95)), 2) if len(ms) else None,
            "eventos": eventos,
            "csv": os.path.basename(self.base + ".csv"),
        }
        with open(self.base + ".json", "w") as f:
            json.dump(resumen, f, indent=2, ensure_ascii=False)
        return resumen
