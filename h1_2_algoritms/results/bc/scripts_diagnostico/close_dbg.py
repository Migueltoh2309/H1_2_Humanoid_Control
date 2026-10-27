import sys, os, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje import habilidades as H
from h1_2_algoritms.vision import grasp_geometry as GG
from h1_2_algoritms.aprendizaje.habilidades import TIMING
B.ESCENARIO='exigente'; B.PERFIL=os.environ.get('PERF','tactil'); TIMING.update(B.PERFILES[B.PERFIL].get('timing',{}))
EX={}; X0={}
st=H.ArmExecutor.step
def step(self, t, q_meas, est=None, vel=None):
    EX[self.side]=self
    out=st(self, t, q_meas, est, vel)
    q = q_meas[:7] if self.side=='left' else q_meas[7:]
    x=GG.FK[self.side](q)[0:3,3]
    if self.skill=='agarrar' and self.phase=='CLOSE':
        if 'cierra' in out[3]: X0[self.side]=(x.copy(), out[0][0:3].copy() if np.ndim(out[0])==1 else None)
        o='left' if self.side=='right' else 'right'
        oe=EX.get(o)
        if self.side in X0:
            dx=(x-X0[self.side][0])*1e3
            print(f"t={t:.2f} {self.side} CLOSE dx={np.round(dx)} |{np.linalg.norm(dx):.0f}| otro={oe.phase if oe else None}")
    return out
H.ArmExecutor.step=step
r,_=B.run_episode(B.make_policy('experto'), int(os.environ['SEED']))
