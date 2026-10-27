#!/usr/bin/env python3
"""Planificación comprobada sin robot contra el modelo (entregable de F3, preguntas 6.3.2 y
6.3.5). Por cada escena grabada (datos/dataset/, ver evaluar_percepcion.py):
  percepción (con la calibración del hito 2) -> por cada objeto, el plan completo (RRT de la
  tesis para los tránsitos, rectas para el descenso) -> comprobación contra el MJCF de la
  estación vacía -> duración de cada tramo con la regla del bloque del brazo (mínima
  sacudida: T >= max(duracion_min, 1.875·L/v_max)).
Sale: planes válidos, por qué falla el resto (fuera de alcance, otro lado: entrega de mano a
mano...), holgura mínima con la mesa y el presupuesto de tiempo por objeto y por tirada.

    python3 herramientas/probar_plan.py                 # escenas n2..n4, sin ruido
    python3 herramientas/probar_plan.py --escenas n2_s0_r1 n3_s1_r1
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sim"))
from manip import calibracion as CAL  # noqa: E402
from manip.config import _leer, cargar, cargar_objetos  # noqa: E402
from manip.estacion import Estacion  # noqa: E402
from manip.fuentes import FuenteCarpeta  # noqa: E402
from manip.modelo import Modelo  # noqa: E402
from manip.percepcion import Percepcion  # noqa: E402
from manip.planificacion import Planificador  # noqa: E402

DATOS = os.path.join(RAIZ, "datos")
T_MANO = 3.0      # s: cerrar y comprobar el agarre, o abrir (Inspire a velocidad 150)


def duracion(tramo, cfg_brazo):
    ruta = np.vstack([tramo.q0, tramo.q_ruta])
    L = float(np.linalg.norm(np.diff(ruta, axis=0), axis=1).sum())
    return max(cfg_brazo["duracion_min"], 1.875 * L / cfg_brazo["v_max"]), L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--escenas", nargs="+")
    ap.add_argument("--duracion-min", type=float, help="sustituye la de config/brazo.yaml (p. ej. la medida)")
    a = ap.parse_args()
    cfg, cat = cargar(), cargar_objetos()
    est = Estacion(cfg)
    cfg_plan = _leer(os.path.join(RAIZ, "config", "planificacion.yaml"))
    cfg_brazo = _leer(os.path.join(RAIZ, "config", "brazo.yaml"))
    if a.duracion_min:
        cfg_brazo["duracion_min"] = a.duracion_min
    agarres = _leer(os.path.join(RAIZ, "config", "agarres.yaml"))
    poses = _leer(os.path.join(RAIZ, "config", "poses.yaml"))
    from construir_estacion import construir
    xml, _ = construir(0, 0, cfg, cat, salida=tempfile.mkdtemp())       # estación VACÍA
    mod = Modelo(xml)
    plan = Planificador(est, agarres, cat, cfg_plan, mod, poses["observacion"])
    per = Percepcion(CAL.cargar(os.path.join(DATOS, "calibracion")), cat, est)
    escenas = a.escenas or sorted(os.path.basename(d) for d in os.listdir(os.path.join(DATOS, "dataset"))
                                  if d.endswith("_r0"))
    filas = []
    for esc in escenas:
        fuente = FuenteCarpeta(os.path.join(DATOS, "dataset", esc))
        lista = per([fuente.fotograma() for _ in range(len(fuente))])
        q_act = {k: np.asarray(v) for k, v in poses["observacion"].items()}
        for o in lista.objetos:
            if o.clase == "desconocido":
                continue
            t0 = time.perf_counter()
            otros = [x for x in lista.objetos if x is not o]
            p = plan.planificar(o, q_act, otros=otros, plano=lista.plano_mesa)
            if not p.ok and "entrega" in p.motivo:
                # cambio de mano por la mesa: la mano de su lado lo deja en la franja común
                pc = plan.punto_comun(o, lista.objetos, q_act)
                if pc is None:
                    p.motivo = "entrega: no hay sitio común libre al alcance de las dos manos"
                else:
                    p1 = plan.planificar(o, q_act, bandeja=pc, otros=otros, plano=lista.plano_mesa)
                    from dataclasses import replace
                    o2 = replace(o, centro=np.array([pc[0], pc[1], o.centro[2]]))
                    otra = "der" if p1.mano == "izq" else "izq"
                    p2 = plan.planificar(o2, q_act, mano=otra, otros=otros, plano=lista.plano_mesa)
                    if p1.ok and p2.ok:
                        p = p2
                        p.tramos = p1.tramos + p2.tramos
                        p.motivo = f"con cambio de mano por la mesa en ({pc[0]:.2f}, {pc[1]:+.2f})"
                    else:
                        p.motivo = "entrega por la mesa: " + (p1.motivo if not p1.ok else p2.motivo)
            t_plan = time.perf_counter() - t0
            fila = {"escena": esc, "objeto": o.clase, "lado_obj": "izq" if o.centro[1] > 0 else "der",
                    "bandeja": cat[o.clase]["bandeja"], "ok": p.ok, "motivo": p.motivo, "t_plan_s": t_plan,
                    "centro": o.centro.round(3).tolist()}
            if p.ok:
                ts = {t.nombre: duracion(t, cfg_brazo)[0] for t in p.tramos}
                fila.update(holgura_mesa_mm=1000 * p.holgura_min_mesa, tramos_s=ts,
                            total_s=sum(duracion(t, cfg_brazo)[0] for t in p.tramos)
                            + 2 * T_MANO * (len(p.tramos) // 7),
                            az_agarre=p.az_agarre, az_suelta=p.az_suelta, nota=p.motivo)
            filas.append(fila)
            print(f"{esc:10s} {o.clase:10s} {'OK ' if p.ok else 'NO '} {t_plan:5.1f} s de cálculo  " +
                  (f"{fila['total_s']:5.0f} s por objeto, holgura mesa {fila['holgura_mesa_mm']:.0f} mm, "
                   f"az {p.az_agarre:.0f} -> {p.az_suelta:.0f} {p.motivo}" if p.ok else p.motivo), flush=True)
    os.makedirs(os.path.join(DATOS, "planes"), exist_ok=True)
    json.dump(filas, open(os.path.join(DATOS, "planes", "resultados.json"), "w"), indent=1)
    ok = [f for f in filas if f["ok"]]
    print(f"\nplanes válidos: {len(ok)}/{len(filas)}")
    motivos = {}
    for f in filas:
        if not f["ok"]:
            motivos[f["motivo"]] = motivos.get(f["motivo"], 0) + 1
    for m, n in motivos.items():
        print(f"  {n:3d} x {m}")
    if ok:
        tot = np.array([f["total_s"] for f in ok])
        print(f"tiempo por objeto (duración mínima {cfg_brazo['duracion_min']} s por tramo): medio {tot.mean():.0f} s, "
              f"máx {tot.max():.0f} s -> 4 objetos ~{4 * tot.mean() / 60:.1f} min")
        nombres = list(ok[0]["tramos_s"])
        print("  por tramo (medio): " +
              ", ".join(f"{n} {np.mean([f['tramos_s'][n] for f in ok]):.0f} s" for n in nombres))


if __name__ == "__main__":
    main()
