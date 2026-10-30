import numpy as np, sys
A = np.load(sys.argv[1], allow_pickle=True)
t = A[:, 0].astype(float); right = A[:, 1].astype(bool); sk = A[:, 2]
for side, msk in (("left", ~right), ("right", right)):
    T = t[msk]; K = sk[msk]; Q = A[msk, 3:10].astype(float); QR = A[msk, 10:17].astype(float); X = A[msk, 17:20].astype(float)
    ends = [i for i in range(1, len(K)) if K[i-1] == "volver" and K[i] != "volver"]
    for i in ends[:3]:
        j0 = np.searchsorted(T, T[i] - 0.5); j1 = np.searchsorted(T, T[i] + 1.2)
        print(f"--- {side} fin volver t={T[i]:.2f} (siguiente: {K[i]})")
        for j in range(j0, j1, 25):   # cada 50 ms (dt 2 ms)
            v = np.linalg.norm(X[min(j+1, j1)] - X[j]) / (T[min(j+1, j1)] - T[j] + 1e-9)
            print(f"  t={T[j]-T[i]:+.2f} {str(K[j]):9s} |q-qref|max={np.abs(Q[j]-QR[j]).max():.3f}  "
                  f"x={X[j].round(3)}  v={v:.2f}  qref-q[j1..4]={np.round(QR[j,:4]-Q[j,:4],2)}")
