import sys, os, collections, numpy as np, pickle
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje import escena_faja as EF, habilidades as H
REC=[]
tw = EF.FajaMulti.try_weld
def try_weld(self, side, expected, tol=0.02):
    st=getattr(self,'_cl',{})
    self._cl=st
    r = tw(self, side, expected, tol)
    belt=[i for i in range(self.n) if self.state[i]=='belt']
    key=side
    if key not in st or self.d.time-st[key][0]>1.5:   # primer intento de este cierre
        if belt:
            i=min(belt,key=lambda i: np.linalg.norm(self.pos_torso(i)-expected))
            st[key]=[self.d.time, side, (self.pos_torso(i)-expected).copy(), self.v]
            REC.append(st[key])
    if r is not None: st[key].append('weld')
    return r
EF.FajaMulti.try_weld=try_weld; B.FajaMulti=EF.FajaMulti
LEAD={'left':float(os.environ.get('LL','0')),'right':float(os.environ.get('LR','0'))}
_st=H.ArmExecutor.step
def astep(self,t,q,est=None,vel=None):
    if est is not None and vel is not None: est=est+LEAD[self.side]*vel
    return _st(self,t,q,est,vel)
H.ArmExecutor.step=astep
_job=B._job
def job(a):
    REC.clear(); r=_job(a); r['_rec']=[x[1:] for x in REC]; return r
B._job=job
from multiprocessing import Pool
pol=os.environ.get('POL','fifo'); seeds=[int(x) for x in os.environ['SEEDS'].split(',')]
with Pool(8,maxtasksperchild=1) as P: res=P.map(job,[(pol,s,False,'','sphere',True) for s in seeds],chunksize=1)
rows=[x for r in res for x in r['_rec']]
for side in ('left','right'):
    for ok in (True,False):
        D=np.array([x[1] for x in rows if x[0]==side and ((isinstance(x[-1],str))==ok)])*1e3
        if len(D): print(side,'weld' if ok else 'SIN weld',len(D),'d medio [x y z] mm',np.round(D.mean(0)),'std',np.round(D.std(0)))
print('depositadas',sum(r['depositadas'] for r in res),'/',8*len(res))
pickle.dump(rows,open(os.environ.get('OUTP','/dev/null'),'wb'))
