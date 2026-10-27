import sys, os, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje import escena_faja as EF, habilidades as H
from h1_2_algoritms.vision import grasp_geometry as GG
from h1_2_algoritms.aprendizaje.habilidades import TIMING
B.ESCENARIO='exigente'; B.PERFIL='tactil'; TIMING.update(B.PERFILES['tactil']['timing'])
FAJA=[None]
gf=EF.FajaMulti.grip_force
def grip(self, side):
    FAJA[0]=self; return gf(self, side)
EF.FajaMulti.grip_force=grip
st=H.ArmExecutor.step
def step(self, t, q_meas, est=None, vel=None):
    ph0=self.phase
    out=st(self, t, q_meas, est, vel)
    if ('fallo' in out[3] or 'verificado' in out[3]) and FAJA[0] is not None:
        f=FAJA[0]; i=f.held.get(self.side) if hasattr(f,'held') else None
        print(f"t={t:.2f} {self.side} {out[3]} held={i} f={gf(f,self.side):.2f} why={self.log[-1][3] if len(self.log[-1])>3 else ''}", flush=True)
    return out
H.ArmExecutor.step=step
r,_=B.run_episode(B.make_policy('experto'), int(os.environ['SEED']))
print({k:v for k,v in r.items() if k!='_eventos'})
