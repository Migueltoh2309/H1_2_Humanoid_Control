"""Comparación pareada por semilla entre evaluaciones de bc_faja (doc §6.6).

  python3 comparar.py A.csv:politica B.csv:politica [...]
Compara cada par con el primero: diferencia media de depositadas por episodio,
IC 95% bootstrap y Wilcoxon, usando solo las semillas comunes.
Ej.: comparar.py results/bc/evaluacion_exigente_espera.csv:bc results/bc/evaluacion_exigente.csv:bc
"""
import sys, csv, numpy as np
from scipy.stats import wilcoxon

def load(arg):
    path, pol = arg.rsplit(":", 1)
    with open(path) as f:
        return {int(r["seed"]): int(r["depositadas"]) for r in csv.DictReader(f) if r["politica"] == pol}

ref = load(sys.argv[1])
for arg in sys.argv[2:]:
    oth = load(arg)
    seeds = sorted(set(ref) & set(oth))
    d = np.array([ref[s] - oth[s] for s in seeds])
    rng = np.random.default_rng(0)
    boot = rng.choice(d, (10000, len(d))).mean(1)
    p = wilcoxon(d).pvalue if np.any(d) else 1.0
    print(f"{sys.argv[1]} − {arg}: n={len(d)}  {sum(ref[s] for s in seeds)} vs {sum(oth[s] for s in seeds)}  "
          f"Δ={d.mean():+.2f}/ep  IC95 [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]  "
          f"gana/empata/pierde {(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}  p={p:.3g}")
