#!/usr/bin/env python3
"""Fase A del LfD: DMP de alcance sobre la faja (docs/LFD_DMP.md).

    generar   demostraciones con la máquina de estados + verdad de terreno
              (oráculo), en un conjunto de ENTRENAMIENTO de posiciones y
              velocidades -> results/dmp/demos/*.csv
    entrenar  ajusta la DMP agrupada y la paramétrica, valida dejando una
              demo fuera (LOO) y guarda results/dmp/dmp_alcance*.npz
    evaluar   lazo cerrado con física y percepción REAL (esfera + ruido
              D435) en posiciones/velocidades NO demostradas: máquina de
              estados vs. DMP con varias duraciones τ
              -> results/dmp/evaluacion.csv

Uso:
    MUJOCO_GL=egl python3 demos/dmp_alcance.py generar
    python3 demos/dmp_alcance.py entrenar
    MUJOCO_GL=egl python3 demos/dmp_alcance.py evaluar [--seeds 2]
"""
import argparse
import csv
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h1_2_algoritms.aprendizaje.dmp import DMP                          # noqa: E402
from h1_2_algoritms.aprendizaje.demostraciones import demo_from_trace   # noqa: E402

OUT = os.path.join(PKG_DIR, "results", "dmp")
DEMOS = os.path.join(OUT, "demos")
TRAZAS = os.path.join(OUT, "trazas")
MODEL = {"param": os.path.join(OUT, "dmp_alcance.npz"),
         "pooled": os.path.join(OUT, "dmp_alcance_agrupada.npz")}

# Entrenamiento y prueba DISJUNTOS (la prueba usa otras x, y y velocidades).
TRAIN_STATIC = [(x, y) for x in (0.30, 0.34, 0.40) for y in (-0.25, -0.15, -0.05, 0.10, 0.20)]
TRAIN_MOVING = [(x, v) for x in (0.30, 0.40) for v in (0.03, 0.05, 0.07)]
TEST_STATIC = [(x, y) for x in (0.32, 0.37) for y in (-0.20, -0.10, 0.0, 0.15, 0.25)]
TEST_MOVING = [(x, v) for x in (0.32, 0.37) for v in (0.04, 0.06, 0.08)]


# =====================================================================
def _gen_job(a):
    import visual_servoing_grasp as V
    sc, x, p = a
    try:
        return V.run_trial(sc, "gt", p, fruit_x=x, log_dir=DEMOS)
    except Exception as e:
        return {"scenario": sc, "param": p, "fruit_x": x, "success": False, "final_phase": f"ERROR {e!r}"}


def generar(args):
    jobs = [("static", x, y) for x, y in TRAIN_STATIC] + [("moving", x, v) for x, v in TRAIN_MOVING]
    t0 = time.time()
    with Pool(args.workers) as P:
        res = P.map(_gen_job, jobs)
    ok = sum(bool(r.get("success")) for r in res)
    print(f"{len(jobs)} demostraciones en {time.time() - t0:.0f} s, {ok} con agarre verificado")
    for r in res:
        if not r.get("success"):
            print("  sin agarre:", {k: r.get(k) for k in ("scenario", "fruit_x", "param", "final_phase")})


# =====================================================================
def load_demos(solo_estaticas=False):
    """Todas las demos, relativas a la meta MÓVIL (la de GraspServoing por
    omisión). Para la opción de intercepción (meta fija = p(t_cierre)) hay
    que usar solo las de fruta quieta: las de la faja muestran a la máquina
    de estados persiguiendo la fruta, que desde el punto de cita es un
    barrido lateral (con ellas, LOO 45 mm en vez de 7 mm)."""
    demos, names = [], []
    for f in sorted(glob.glob(os.path.join(DEMOS, "*.csv"))):
        if solo_estaticas and not os.path.basename(f).startswith("static"):
            continue
        d = demo_from_trace(f)
        if d is not None:
            demos.append(d)
            names.append(os.path.basename(f)[:-4])
    return demos, names


def repro_err(m, d, tau_scale=1.0):
    """Error medio y final [m] entre la reproducción desde el inicio de la
    demo (con su duración) y la demo."""
    t, Y, _ = m.rollout(d[1][0], tau=d[0][-1] * tau_scale)
    ti = np.clip(t / tau_scale, 0, d[0][-1])
    yi = np.array([np.interp(ti, d[0], d[1][:, k]) for k in range(3)]).T
    e = np.linalg.norm(Y - yi, axis=1)
    return float(e.mean()), float(np.linalg.norm(Y[-1]))


def entrenar(args):
    demos, names = load_demos()
    print(f"{len(demos)} demostraciones (convención brazo derecho, relativas a la meta)")
    taus = np.array([d[0][-1] for d in demos])
    print(f"duración del alcance demostrado: {taus.mean():.2f} ± {taus.std():.2f} s")
    rows = []
    for kind, par in (("pooled", False), ("param", True)):
        m = DMP(n_basis=args.n_basis, parametric=par).fit(demos)
        m.save(MODEL[kind])
        fit = [repro_err(m, d) for d in demos]
        loo = []
        for i in range(len(demos)):
            mi = DMP(n_basis=args.n_basis, parametric=par).fit(demos[:i] + demos[i + 1:])
            loo.append(repro_err(mi, demos[i]))
        for n, (ef, _), (el, eend) in zip(names, fit, loo):
            rows.append({"modelo": kind, "demo": n, "err_ajuste_mm": round(1e3 * ef, 1),
                         "err_loo_mm": round(1e3 * el, 1), "err_final_loo_mm": round(1e3 * eend, 1)})
        print(f"{kind:7s} ajuste {1e3 * np.mean([e for e, _ in fit]):5.1f} mm | "
              f"LOO medio {1e3 * np.mean([e for e, _ in loo]):5.1f} mm, "
              f"máx {1e3 * np.max([e for e, _ in loo]):5.1f} mm")
    with open(os.path.join(OUT, "entrenamiento.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    _figura(demos, names)


def _figura(demos, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = DMP.load(MODEL["param"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for d in demos:
        t, Y, _ = m.rollout(d[1][0], tau=d[0][-1])
        for k, a in enumerate("xyz"):
            ax[k].plot(d[0], 1e3 * d[1][:, k], color="0.6", lw=1)
            ax[k].plot(t, 1e3 * Y[:, k], color="C0", lw=1, ls="--")
    for k, a in enumerate("xyz"):
        ax[k].set_title(f"{a} relativo a la meta (brazo der.)")
        ax[k].set_xlabel("t [s]")
        ax[k].set_ylabel("mm")
    ax[0].plot([], [], color="0.6", label="demostración")
    ax[0].plot([], [], color="C0", ls="--", label="DMP paramétrica")
    ax[0].legend()
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, "figuras"), exist_ok=True)
    fig.savefig(os.path.join(OUT, "figuras", "ajuste_dmp.png"), dpi=120)


# =====================================================================
def _eval_job(a):
    import visual_servoing_grasp as V
    sc, x, p, cfg, seed = a
    kind, scale = cfg
    reach = None if kind == "fsm" else DMP.load(MODEL[kind])
    tau = None if reach is None else reach.tau_demo * scale
    tag = "" if reach is None else f"_{kind}_{scale}"
    try:
        r = V.run_trial(sc, "sphere", p, fruit_x=x, seed=seed, reach=reach, reach_tau=tau, tag=tag,
                        log_dir=TRAZAS)
    except Exception as e:
        r = {"scenario": sc, "param": p, "fruit_x": x, "seed": seed, "success": False,
             "final_phase": f"ERROR {e!r}"}
    r["config"] = "fsm" if kind == "fsm" else f"dmp_{kind}_tau{scale}"
    return r


def evaluar(args):
    cfgs = [("fsm", 1.0)] + [("param", s) for s in args.tau_scales] + \
        ([("pooled", 1.0)] if args.pooled else [])
    cases = [("static", x, y) for x, y in TEST_STATIC] + [("moving", x, v) for x, v in TEST_MOVING]
    jobs = [(sc, x, p, c, s) for c in cfgs for sc, x, p in cases for s in range(args.seeds)]
    t0 = time.time()
    with Pool(args.workers) as P:
        res = P.map(_eval_job, jobs)
    keys = ["config"] + [k for k in res[0].keys() if k != "config"]
    keys += sorted({k for r in res for k in r} - set(keys))
    with open(os.path.join(OUT, args.out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(res)
    print(f"{len(jobs)} ensayos en {time.time() - t0:.0f} s -> results/dmp/{args.out}\n")
    resumen(res)


def resumen(res):
    print("real = la fruta terminó levantada (verdad de terreno); verif = además lo confirmó la cámara")
    print(f"{'config':22s} {'escenario':9s} {'real':>7s} {'verif':>7s} {'t alcance':>9s} {'t ciclo':>8s} "
          f"{'err cierre':>10s} {'jerk':>7s} {'camino':>7s} {'d brazos':>8s}")
    print(f"{'':22s} {'':9s} {'':>7s} {'':>7s} {'[s]':>9s} {'[s]':>8s} {'[mm]':>10s} {'[m/s³]':>7s} "
          f"{'[m]':>7s} {'[mm]':>8s}")
    for c in dict.fromkeys(r["config"] for r in res):
        for sc in ("static", "moving", "todos"):
            r = [x for x in res if x["config"] == c and (sc == "todos" or x["scenario"] == sc)]
            if not r:
                continue
            ok = [x for x in r if str(x.get("success")) == "True"]
            held = [x for x in r if str(x.get("held")) == "True"]

            def f(k, rr=r):
                v = [float(x.get(k, np.nan)) for x in rr]
                return np.nanmean(v) if np.isfinite(v).any() else np.nan
            print(f"{c:22s} {sc:9s} {len(held):3d}/{len(r):<3d} {len(ok):3d}/{len(r):<3d} "
                  f"{f('t_grasp'):9.2f} {f('t_cycle', ok):8.2f} {f('err_close_mm'):10.1f} "
                  f"{f('jerk_rms'):7.1f} {f('path_m'):7.3f} {f('d_arms_min_mm'):8.0f}")
    for x in res:
        if str(x.get("held")) != "True":
            keys = ("config", "scenario", "fruit_x", "param", "seed", "final_phase", "err_close_mm", "why")
            print("  sin agarre:", {k: x.get(k) for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["generar", "entrenar", "evaluar", "resumen"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tau-scales", type=float, nargs="+", default=[1.0, 0.7, 0.5])
    ap.add_argument("--out", default="evaluacion.csv")
    ap.add_argument("--pooled", action="store_true", help="evaluar también la DMP agrupada")
    args = ap.parse_args()
    os.makedirs(DEMOS, exist_ok=True)
    if args.accion == "generar":
        generar(args)
    elif args.accion == "entrenar":
        entrenar(args)
    elif args.accion == "evaluar":
        evaluar(args)
    else:
        with open(os.path.join(OUT, args.out)) as f:
            resumen(list(csv.DictReader(f)))


if __name__ == "__main__":
    main()
