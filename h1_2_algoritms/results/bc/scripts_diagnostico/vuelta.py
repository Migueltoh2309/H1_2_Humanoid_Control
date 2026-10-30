"""Mide el seguimiento articular durante depositar/volver: q medido vs q_ref (PD)."""
import sys, os, numpy as np
PERF = os.environ.get("PERF", "espera"); SEED = int(os.environ.get("SEED", "10002"))
sys.argv = ['x']; PK = os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0, PK + '/demos')
import bc_faja as B
from h1_2_mujoco_sim_bridge.mujoco_sim import MujocoSim
from h1_2_algoritms.aprendizaje import habilidades as H
from h1_2_algoritms.vision import grasp_geometry as GG
B.ESCENARIO, B.PERFIL = "exigente", PERF
H.TIMING.update(B.PERFILES[PERF].get("timing", {}))
KD = os.environ.get("KD"); FF = os.environ.get("FF") == "1"
LOG = []; EX = {}
orig_init = H.ArmExecutor.__init__
def init(self, side, *a, **k):
    orig_init(self, side, *a, **k); EX[side] = self
H.ArmExecutor.__init__ = init
orig_step = MujocoSim.step
st = {"qref_prev": None, "n": 0}
def step(self, pd):
    if not hasattr(self, "_idx"):
        self._idx = {s: [self.mapping.index[k] for k in B.V.ARM_JOINTS[s]] for s in ("left", "right")}
        if KD:
            pd.kd = pd.kd * float(KD)
    q, dq = self.get_q_dq()
    if FF:
        qd = pd.q_des.copy()
        if st["qref_prev"] is not None and np.any(qd != st["qref_prev"]):
            pd.dq_des = (qd - st["qref_prev"]) / 0.05
        st["qref_prev"] = qd
    t = self.data.time
    for s in ("left", "right"):
        e = EX.get(s)
        i = self._idx[s]
        LOG.append((t, s, e.skill if e else None, q[i].copy(), pd.q_des[i].copy(), dq[i].copy(),
                    GG.FK[s](q[i])[0:3, 3].copy()))
    orig_step(self, pd)
MujocoSim.step = step
res, _ = B.run_episode(B.make_policy("experto"), SEED)
print(PERF, "kd x", KD or 1, "ff", FF, "-> dep", res["depositadas"])
# por cada tramo 'volver': sobrepaso y oscilación de la muñeca tras terminar el servo
import itertools
for s in ("left", "right"):
    L = [r for r in LOG if r[1] == s]
    segs = [list(g) for k, g in itertools.groupby(L, key=lambda r: r[2]) if k in ("depositar", "volver")]
    for g in segs:
        ts = np.array([r[0] for r in g]); q = np.array([r[3] for r in g]); qr = np.array([r[4] for r in g])
        dq = np.array([r[5] for r in g]); x = np.array([r[6] for r in g])
        err = np.abs(q - qr).max(); lag = np.abs(q - qr).max(0)
        # inversiones de signo de la velocidad articular (oscilación)
        rev = sum(int(np.sum(np.diff(np.sign(dq[:, j][np.abs(dq[:, j]) > 0.05])) != 0)) for j in range(7))
        v = np.linalg.norm(np.diff(x, axis=0), axis=1) / np.diff(ts)
        print(f"{s:5s} {g[0][2]:9s} t={ts[0]:5.2f} dur={ts[-1]-ts[0]:4.2f}s  err_max={err:.3f} rad  "
              f"|dq|max={np.abs(dq).max():.2f}  inversiones={rev:3d}  v_muñeca max={v.max():.2f} m/s")
np.save(os.environ.get("OUT", "/home/mito/Github/H1_2_Humanoid_Control/h1_2_algoritms/results/bc/scripts_diagnostico/vuelta.npy"),
        np.array([(r[0], r[1] == "right", str(r[2]), *r[3], *r[4], *r[6]) for r in LOG], dtype=object), allow_pickle=True)
