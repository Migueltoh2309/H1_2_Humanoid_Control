#!/usr/bin/env python3
"""Diagnóstico de los agarres fallidos y PREDICTOR DE ÉXITO del agarre
(docs/MAPA_LFD.md §14).

Lee los intentos registrados por `bc_faja.py` con BC_INTENTOS=<carpeta>
(un .pkl por episodio: estado simbólico al decidir, brazo, fruta elegida y
resultado: ok | fallo | perdida | caida) y:

    python predictor_exito.py diagnostico --dir ../results/bc/intentos_ancho_dx3
    python predictor_exito.py entrenar    --dir ../results/bc/intentos_ancho_dx3 \
           --out ../results/bc/predictor_exito_ancho_suave_DR_dx3.pt

El predictor es P(ok | φ_fruta, φ_ctx): las MISMAS características de la
BC (alto_nivel.features, relativas al brazo) de la fruta elegida, más el
contexto. Validación cruzada agrupada por episodio (AUC, Brier).
Los tiempos (TIMING) tienen que ser los de la evaluación: --timing.
"""
import argparse
import glob
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

from h1_2_algoritms.aprendizaje import alto_nivel as AN
from h1_2_algoritms.aprendizaje.habilidades import TIMING

NOMBRES = ["x-0.35", "y", "10vy", "y_agarre", "s_min-t", "s_max-t", "factible_ya", "s_min_otro-t",
           "s_max_otro-t", "factible_otro", "dy_otro", "urgencia",
           "n/6", "otro_ocupado", "otro_resta/10", "n_fact/3", "n_fact_otro/3", "urg_min/10", "ocupado"]


def cargar(d):
    A = []
    for f in sorted(glob.glob(os.path.join(d, "*.pkl"))):
        with open(f, "rb") as fh:
            A += pickle.load(fh)
    return A


def tabla(A):
    """X (n×19), y (1 = ok), seed, política, resultado."""
    X, y, g, pol, res = [], [], [], [], []
    for a in A:
        ids, Xc, ctx = AN.features(a["state"], a["side"])
        if a["act"] not in ids:
            continue
        X.append(np.hstack([Xc[ids.index(a["act"])], ctx]))
        y.append(a["resultado"] == "ok")
        g.append(a["seed"])
        pol.append(a["politica"])
        res.append(a["resultado"])
    return np.array(X, np.float32), np.array(y, np.float32), np.array(g), np.array(pol), np.array(res)


def auc(p, y):
    """AUC (Mann-Whitney) de p como puntaje de y = 1."""
    o = np.argsort(p)
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)


def diagnostico(args):
    A = cargar(args.dir)
    X, y, g, pol, res = tabla(A)
    print(f"{len(y)} intentos, {len(set(g))} episodios\n")
    for p in sorted(set(pol)):
        m = pol == p
        c = Counter(res[m])
        print(f"{p:8s} intentos {m.sum():4d}  ok {c['ok']:4d} ({100 * c['ok'] / m.sum():.1f}%)  "
              + "  ".join(f"{k} {v}" for k, v in sorted(c.items()) if k != "ok"))
    whys = defaultdict(Counter)
    for a in A:
        if a["resultado"] in ("fallo", "perdida"):
            w = (a.get("why") or "").split("(")[0].strip()
            whys[a["politica"]][f"{a['resultado']}: {w}"] += 1
    print("\nmotivos:")
    for p, c in sorted(whys.items()):
        for k, v in c.most_common():
            print(f"  {p:8s} {v:3d}  {k}")

    def por(col, bins, nombre, f=lambda v: v):
        print(f"\nfracaso por {nombre}:")
        v = f(X[:, col])
        print("  " + " " * 16 + "".join(f"{p:>22s}" for p in sorted(set(pol))))
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (v >= lo) & (v < hi)
            fila = ""
            for p in sorted(set(pol)):
                mm = m & (pol == p)
                fila += f"{int((1 - y[mm]).sum()):5d}/{int(mm.sum()):<5d} ({100 * (1 - y[mm]).mean() if mm.any() else 0:4.1f}%)  "
            print(f"  [{lo:6.2f},{hi:6.2f})  {fila}")
    por(0, [-0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12], "x de la fruta (relativo a 0.35, sin dx)")
    por(5, [-1, 0.5, 1.0, 1.5, 2.5, 4, 16], "holgura s_max - t (s)")
    por(2, [0, 0.6, 0.7, 0.8, 1.0], "|vy| (×10 m/s)", f=np.abs)
    por(3, [-0.4, -0.2, -0.1, 0.0, 0.1, 0.3], "y al agarrar (conv. brazo derecho)")
    por(13, [-0.5, 0.5, 1.5], "otro brazo ocupado (0/1)")
    por(10, [0, 0.15, 0.3, 0.6, 1.01], "distancia en y al agarre del otro brazo")


def _mlp(d, h):
    import torch.nn as nn
    if h == 0:
        return nn.Linear(d, 1)
    return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))


def _fit(X, y, h=32, epochs=400, wd=1e-3, seed=0):
    import torch
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    net = _mlp(X.shape[1], h)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=wd)
    Xt, yt = torch.from_numpy((X - mu) / sd), torch.from_numpy(y)
    for _ in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt).squeeze(1), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net, mu, sd


def _pred(model, X):
    import torch
    net, mu, sd = model
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy((X - mu) / sd)).squeeze(1)).numpy()


def entrenar(args):
    import torch
    A = cargar(args.dir)
    X, y, g, pol, _ = tabla(A)
    seeds = np.array(sorted(set(g)))
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(seeds), 5)
    print(f"{len(y)} intentos, tasa de fracaso {100 * (1 - y.mean()):.1f}%  (Brier de la constante "
          f"{np.mean((y.mean() - y) ** 2):.4f})")
    for h in (0, 16, 32):
        p = np.zeros(len(y))
        for f in folds:
            te = np.isin(g, f)
            p[te] = _pred(_fit(X[~te], y[~te], h=h), X[te])
        print(f"  {'logística' if h == 0 else f'MLP {h}':10s} AUC {auc(p, y):.3f}  Brier {np.mean((p - y) ** 2):.4f}  "
              f"| P(ok) de los fracasos: med {np.median(p[y == 0]):.2f}, de los ok: med {np.median(p[y == 1]):.2f}")
        for pp in sorted(set(pol)):
            m = pol == pp
            print(f"      {pp:8s} AUC {auc(p[m], y[m]):.3f}")
        if h == args.oculta:
            p_cv = p
    # cuánto se evitaría: fracasos/ok con P(ok) bajo un umbral (validación cruzada)
    for u in (0.5, 0.7, 0.8, 0.9):
        m = p_cv < u
        print(f"  P(ok) < {u}: {int((y[m] == 0).sum())} fracasos y {int(y[m].sum())} ok de {int((y == 0).sum())} / {int(y.sum())}")
    model = _fit(X, y, h=args.oculta)
    torch.save({"state": model[0].state_dict(), "mu": model[1], "sd": model[2], "oculta": args.oculta,
                "nombres": NOMBRES, "timing": dict(TIMING)}, args.out)
    print("->", args.out)


class Predictor:
    """P(ok) de cada fruta candidata para el brazo `side` (lo usa BCExitoPolicy)."""

    def __init__(self, path):
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        net = _mlp(len(d["nombres"]), d["oculta"])
        net.load_state_dict(d["state"])
        self.model = (net.eval(), d["mu"], d["sd"])

    def __call__(self, X, ctx):
        Z = np.hstack([X, np.repeat(ctx[None], len(X), 0)]).astype(np.float32)
        return _pred(self.model, Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["diagnostico", "entrenar"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="predictor_exito.pt")
    ap.add_argument("--oculta", type=int, default=32)
    ap.add_argument("--timing", default=None)
    args = ap.parse_args()
    if args.timing:
        TIMING.update({k: float(v) for k, v in (kv.split("=") for kv in args.timing.split(","))})
    {"diagnostico": diagnostico, "entrenar": entrenar}[args.accion](args)


if __name__ == "__main__":
    main()
