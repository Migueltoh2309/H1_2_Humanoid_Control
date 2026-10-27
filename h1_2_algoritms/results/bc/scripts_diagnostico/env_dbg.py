import sys, os, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje import habilidades as H
from h1_2_algoritms.bimanual import bimanual_avoidance as BA
from h1_2_algoritms.aprendizaje.habilidades import TIMING
B.ESCENARIO='exigente'; B.PERFIL=os.environ.get('PERF','tactil'); TIMING.update(B.PERFILES[B.PERFIL].get('timing',{}))
EX={}
st=H.ArmExecutor.step
def step(self, t, q_meas, est=None, vel=None):
    EX[self.side]=(self,t); return st(self, t, q_meas, est, vel)
H.ArmExecutor.step=step
cs=BA.BimanualAvoidanceController.step
def cstep(self, q, targets, obstacles=None, **kw):
    dq, diag = cs(self, q, targets, obstacles=obstacles, **kw)
    for s,(e,t) in EX.items():
        if e.skill=='agarrar' and e.phase in ('CLOSE',) :
            near=[(n,round(d*1e3)) for n,d,*_ in diag['env'] if d<0.15]
            print(f"t={t:.2f} {s} {e.phase} nobs={0 if obstacles is None else len(obstacles)} env<15cm={near[:4]} self={round(diag['d_arms_min']*1e3)}")
    return dq, diag
BA.BimanualAvoidanceController.step=cstep
B.BimanualAvoidanceController=BA.BimanualAvoidanceController
r,_=B.run_episode(B.make_policy('experto'), int(os.environ['SEED']))
