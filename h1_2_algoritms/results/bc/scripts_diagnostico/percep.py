"""Al entrar a CLOSE: verdad (fruta más cercana en la faja), estimación de la
pista (fruit_ref) y punto de agarre de la mano; y si hubo weld."""
import sys, os, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje import escena_faja as EF, habilidades as H
from h1_2_algoritms.vision import grasp_geometry as GG
REC=[]; FAJA=[None]
_init=EF.FajaMulti.__init__
def init(self,*a,**k):
    _init(self,*a,**k); FAJA[0]=self
EF.FajaMulti.__init__=init; B.FajaMulti=EF.FajaMulti
_st=H.ArmExecutor.step
def astep(self,t,q,est=None,vel=None):
    out=_st(self,t,q,est,vel)
    f=FAJA[0]
    if 'cierra' in out[3]:
        qs=q[:7] if self.side=='left' else q[7:]
        hand=GG.wrist_to_fruit(self.side,qs)
        belt=[i for i in range(f.n) if f.state[i]=='belt']
        i=min(belt,key=lambda i: np.linalg.norm(f.pos_torso(i)-hand))
        REC.append({'side':self.side,'t':t,'i':i,'truth':f.pos_torso(i).copy(),'est':self.fsm.fruit_ref.copy(),'hand':hand,'v':f.v,'rec':self})
    for r in REC:
        if r['rec'] is self and 'ok' not in r and ('verificado' in out[3] or 'fallo' in out[3]):
            r['ok']='verificado' in out[3]
    return out
H.ArmExecutor.step=astep
_job=B._job
def job(a):
    REC.clear(); r=_job(a); r['_rec']=[{k:v for k,v in x.items() if k!='rec'} for x in REC]; return r
B._job=job
from multiprocessing import Pool
seeds=[int(x) for x in os.environ['SEEDS'].split(',')]
with Pool(8,maxtasksperchild=1) as P: res=P.map(job,[(os.environ.get('POL','fifo'),s,False,'','sphere',True) for s in seeds],chunksize=1)
rows=[x for r in res for x in r['_rec'] if 'ok' in x]
for side in ('left','right'):
    for ok in (True,False):
        R=[x for x in rows if x['side']==side and x['ok']==ok]
        if not R: continue
        pe=np.array([x['est']-x['truth'] for x in R])*1e3; he=np.array([x['hand']-x['truth'] for x in R])*1e3
        print(f"{side:5s} {'OK ' if ok else 'MAL'} n={len(R):2d}  est-verdad {np.round(pe.mean(0))} (std {np.round(pe.std(0))})  mano-verdad {np.round(he.mean(0))} (std {np.round(he.std(0))})  y_verdad {np.mean([x['truth'][1] for x in R]):+.3f}")
print('depositadas',sum(r['depositadas'] for r in res),'/',8*len(res))
import pickle; pickle.dump(rows,open(os.environ.get('OUTP','/dev/null'),'wb'))
