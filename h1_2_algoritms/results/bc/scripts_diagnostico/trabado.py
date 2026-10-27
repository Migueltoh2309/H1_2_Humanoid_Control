import sys, os, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.bimanual import bimanual_avoidance as BA
from h1_2_algoritms.vision import grasp_geometry as GG
orig = BA.BimanualAvoidanceController.step
st={'k':0}
def step(self, q, targets, obstacles=None, **kw):
    dq, diag = orig(self, q, targets, obstacles=obstacles, **kw)
    st['k']+=1; t=st['k']*0.05
    if t0 <= t <= t1 and st['k'] % 5 == 0:
        x = GG.FK['right'](q[7:])[0:3,3]
        env = sorted(diag['env'], key=lambda e: e[1])[:3]
        qr=q[7:]; lim = [(i, round(qr[i],2)) for i in range(7) if min(qr[i]-self.q_min[7+i], self.q_max[7+i]-qr[i])<0.05]
        print(f"t={t:5.1f} xR={np.round(x,3)} tgtR={np.round(targets['right'][:3],3)} errR={diag['err']['right'][0]*1e3:4.0f}mm/{diag['err']['right'][1]:.2f}rad "
              f"nobs={0 if obstacles is None else len(obstacles)} env={[(e[0], round(e[1]*1e3)) for e in env]} lim={lim} dself={diag['d_arms_min']*1e3:.0f}")
    return dq, diag
BA.BimanualAvoidanceController.step = step
B.BimanualAvoidanceController = BA.BimanualAvoidanceController
pol, seed, t0, t1 = sys.argv[1:] if False else (os.environ['POL'], int(os.environ['SEED']), float(os.environ['T0']), float(os.environ['T1']))
r,_ = B.run_episode(B.make_policy(pol), seed)
for s,log in r['_eventos'].items():
    for e in log: print(s,*e)
