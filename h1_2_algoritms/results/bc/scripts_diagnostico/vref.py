"""v_ref (velocidad estimada de la fruta al cerrar) vs velocidad real, por resultado."""
import sys, os, collections, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje import habilidades as H
from h1_2_algoritms.aprendizaje.habilidades import TIMING
seeds=[int(x) for x in os.environ['SEEDS'].split(',')]
B.ESCENARIO='exigente'; B.PERFIL=os.environ.get('PERF','tactil'); TIMING.update(B.PERFILES[B.PERFIL].get('timing',{}))
st=H.ArmExecutor.step
def step(self, t, q_meas, est=None, vel=None):
    out=st(self, t, q_meas, est, vel)
    if 'cierra' in out[3]:
        self.log.append((t,'vref',self.track,'',np.asarray(self.fsm.v_ref).copy(), np.asarray(self.fsm.fruit_ref).copy()))
    return out
H.ArmExecutor.step=step
res=B._pool([('experto',s,False,'','sphere',True) for s in seeds],8)
rows=[]
for r in res:
    v=r['v_faja']
    for s,log in r['_eventos'].items():
        for i,e in enumerate(log):
            if e[1]=='vref':
                nxt=next((x[1] for x in log[i+1:] if x[1] in ('verificado','fallo')),'?')
                rows.append((nxt, e[4], v, e[5]))
for k in ('verificado','fallo'):
    R=[x for x in rows if x[0]==k]
    if not R: continue
    dv=np.array([x[1] for x in R]); vb=np.array([x[2] for x in R]); p=np.array([x[3] for x in R])
    print(k, len(R), 'vref med', np.round(np.median(dv,0),3), '|v| err med', np.round(np.median(np.abs(np.linalg.norm(dv,axis=1)-vb)),3),
          'p90', np.round(np.percentile(np.abs(np.linalg.norm(dv,axis=1)-vb),90),3), 'x med', np.round(np.median(p,0),3))
    if k=='fallo':
        for x in R: print('   ', np.round(x[1],3), x[2], np.round(x[3],3))
