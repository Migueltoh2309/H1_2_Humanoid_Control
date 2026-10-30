"""Oscilación de la muñeca tras 'volver' (1.0 s): inversiones de la velocidad en y,
amplitud pico a pico del vaivén y razón recorrido / desplazamiento neto."""
import numpy as np, sys
for f in sys.argv[1:]:
    A = np.load(f, allow_pickle=True)
    t = A[:, 0].astype(float); right = A[:, 1].astype(bool); sk = A[:, 2]
    R = []
    for msk in (~right, right):
        T = t[msk]; K = sk[msk]; X = A[msk, 17:20].astype(float)
        for i in [i for i in range(1, len(K)) if K[i-1] == "volver" and K[i] != "volver"]:
            j1 = np.searchsorted(T, T[i] + 1.0)
            x = X[i:j1:5]                         # 10 ms
            v = np.diff(x, axis=0)
            rev = int(np.sum(np.diff(np.sign(v[np.abs(v[:, 1]) > 1e-4, 1])) != 0))
            path = np.linalg.norm(v, axis=1).sum(); net = np.linalg.norm(x[-1] - x[0])
            R.append((rev, np.ptp(x[:, 1]), path / max(net, 1e-3), path))
    R = np.array(R)
    print(f"{f.split('/')[-1]:22s} n={len(R)}  inversiones vy={R[:,0].mean():.1f}  "
          f"recorrido en 1 s={100*R[:,3].mean():.1f} cm  recorrido/neto={np.median(R[:,2]):.1f}")
# aceleración de la muñeca (todo el episodio): un vaivén subamortiguado la dispara
for f in sys.argv[1:]:
    A = np.load(f, allow_pickle=True); t = A[:, 0].astype(float); right = A[:, 1].astype(bool)
    acc = []
    for msk in (~right, right):
        X = A[msk, 17:20].astype(float)[::5]; dt = 0.01
        a = np.diff(X, 2, axis=0) / dt**2
        acc.append(np.linalg.norm(a, axis=1))
    a = np.concatenate(acc)
    print(f"{f.split('/')[-1]:22s} acel muñeca: RMS {np.sqrt(np.mean(a**2)):.2f}  p99 {np.percentile(a, 99):.1f} m/s²")
