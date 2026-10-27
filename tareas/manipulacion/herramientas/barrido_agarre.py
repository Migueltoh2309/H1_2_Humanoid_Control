#!/usr/bin/env python3
"""Barrido de parámetros del agarre con el robot simulado completo (probar_agarre.probar), en
paralelo: cada combinación x cada semilla es una prueba independiente (un proceso por núcleo).

    python3 herramientas/barrido_agarre.py --objeto pelota --param prof_palma 0.066 0.08 \\
        --param avance 0.215 0.235 --semillas 0 1 2
    python3 herramientas/barrido_agarre.py --objeto pelota --fijar obj.radio=0.03 altura_agarre=0.03
    python3 herramientas/barrido_agarre.py --objeto vaso --fijar tipo=lado --param altura_agarre 0.05 0.06

Imprime, por combinación, cuántas semillas llegan a la bandeja y el detalle; guarda
datos/agarre/barrido_<objeto>_<fecha>.json.
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
sys.path.insert(0, RAIZ)

_CTX = {}


def _iniciar():
    import tempfile

    import numpy as np
    import probar_agarre as PA
    from manip.config import _leer, cargar, cargar_objetos
    from manip.modelo import Modelo
    cfg, cat = cargar(), cargar_objetos()
    _CTX.update(PA=PA, cfg=cfg, cat=cat, np=np,
                cfg_plan=_leer(os.path.join(RAIZ, "config", "planificacion.yaml")),
                cfg_brazo=_leer(os.path.join(RAIZ, "config", "brazo.yaml")),
                poses=_leer(os.path.join(RAIZ, "config", "poses.yaml")),
                agarres=_leer(os.path.join(RAIZ, "config", "agarres.yaml")),
                mod=Modelo(PA.construir(0, 0, cfg, cat, salida=tempfile.mkdtemp())[0]))


def _una(trabajo):
    objeto, nivel, semilla, combo, error = trabajo
    c = _CTX
    agarres = {k: dict(v) for k, v in c["agarres"].items()}
    agarres[objeto].update({k: v for k, v in combo.items() if not k.startswith("obj.")})
    cat = {k: dict(v) for k, v in c["cat"].items()}           # obj.<clave>: el propio objeto (radio...)
    cat[objeto].update({k[4:]: v for k, v in combo.items() if k.startswith("obj.")})
    t0 = time.time()
    try:
        r = c["PA"].probar(objeto, nivel, semilla, error, c["cfg"], cat, agarres, c["cfg_plan"],
                           c["cfg_brazo"], c["poses"], c["mod"], c["np"].random.default_rng(semilla), False)
    except Exception as e:                      # una combinación rota no para el barrido
        r = {"plan": False, "motivo": f"excepción: {e!r}"}
    r["t_s"] = round(time.time() - t0, 1)
    return combo, semilla, r


def main():
    import yaml
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objeto", required=True)
    ap.add_argument("--nivel", type=int, default=2)
    ap.add_argument("--semillas", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fijar", nargs="*", default=[], help="clave=valor fijos")
    ap.add_argument("--param", nargs="+", action="append", default=[], help="clave v1 v2 ...")
    ap.add_argument("--error", type=float, default=0.0)
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    fijos = {}
    for kv in a.fijar:
        k, v = kv.split("=", 1)
        fijos[k] = yaml.safe_load(v)
    claves = [p[0] for p in a.param]
    valores = [[yaml.safe_load(v) for v in p[1:]] for p in a.param]
    combos = [dict(fijos, **dict(zip(claves, vs))) for vs in itertools.product(*valores)] or [dict(fijos)]
    trabajos = [(a.objeto, a.nivel, s, cb, a.error) for cb in combos for s in a.semillas]
    print(f"{len(combos)} combinaciones x {len(a.semillas)} semillas = {len(trabajos)} pruebas en "
          f"{a.procesos} procesos", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.procesos, initializer=_iniciar) as pool:
        res = pool.map(_una, trabajos)
    print(f"({time.time() - t0:.0f} s)")
    filas = []
    for cb in combos:
        rs = [(s, r) for c2, s, r in res if c2 == cb]
        ok = sum(bool(r.get("ok")) for _, r in rs)
        variable = {k: cb[k] for k in claves}
        print(f"{ok}/{len(rs)}  {variable}")
        for s, r in rs:
            if not r.get("plan"):
                print(f"     s{s}: sin plan ({r.get('motivo', '')[:80]})")
            else:
                print(f"     s{s}: {'OK ' if r['ok'] else 'MAL'} mano {r['agarre_mano']} "
                      f"sube {r['levantado_mm']:.0f} mm, dedos {r.get('dedos_elevado')}, "
                      f"bandeja {r['en_bandeja']}, incl {r['inclinacion_deg']:.0f}, ángulos {r['angulos'][:5]}, "
                      f"fuerzas {r['fuerzas'][:5]}")
        filas.append({"combo": cb, "ok": ok, "pruebas": [{"semilla": s, **r} for s, r in rs]})
    os.makedirs(os.path.join(RAIZ, "datos", "agarre"), exist_ok=True)
    ruta = os.path.join(RAIZ, "datos", "agarre", f"barrido_{a.objeto}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(filas, open(ruta, "w"), indent=1, default=str)
    print("guardado", ruta)


if __name__ == "__main__":
    main()
