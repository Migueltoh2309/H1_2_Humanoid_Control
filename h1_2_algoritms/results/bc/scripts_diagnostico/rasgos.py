"""Rasgos del cierre (velocidad de muñeca, deriva x/z a 0.4 s, error) por resultado."""
import sys, os, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje import habilidades as H
from h1_2_algoritms.vision import grasp_geometry as GG
from h1_2_algoritms.aprendizaje.habilidades import TIMING
seeds=[int(x) for x in os.environ['SEEDS'].split(',')]
B.ESCENARIO='exigente'; B.PERFIL=os.environ.get('PERF','tactil'); TIMING.update(B.PERFILES[B.PERFIL].get('timing',{}))
st=H.ArmExecutor.step
def step(self, t, q_meas, est=None, vel=None):
    q = q_meas[:7] if self.side=='left' else q_meas[7:]
    x=GG.FK[self.side](q)[0:3,3].copy()
    prev=getattr(self,'_xprev',x); self._xprev=x
    out=st(self, t, q_meas, est, vel)
    if 'cierra' in out[3]:
        self._c=(t,x,(x-prev)/0.05)
    if getattr(self,'_c',None) and t-self._c[0]>=0.4:
        t0,x0,v0=self._c; self._c=None
        vb=np.asarray(self.fsm.v_ref)
        self.log.append((t,'rasgo',self.track,'',np.linalg.norm(v0-vb), (x-x0)-vb*(t-t0)))
    return out
H.ArmExecutor.step=step
res=B._pool([('experto',s,False,'','sphere',True) for s in seeds],8)
rows=[]
for r in res:
    for s,log in r['_eventos'].items():
        for i,e in enumerate(log):
            if e[1]=='rasgo':
                nxt=next((x[1] for x in log[i+1:] if x[1] in ('verificado','fallo')),'?')
                rows.append((nxt,e[4],e[5]))
for k in ('verificado','fallo'):
    R=[x for x in rows if x[0]==k]
    if not R: continue
    v=np.array([x[1] for x in R]); dr=np.array([x[2] for x in R])*1e3
    print(B.PERFIL,k,len(R),'v_rel cierre med/p90 %.3f/%.3f'%(np.median(v),np.percentile(v,90)),'deriva mm med',np.round(np.median(dr,0)),'p90|d|',np.round(np.percentile(np.linalg.norm(dr,axis=1),90)))
