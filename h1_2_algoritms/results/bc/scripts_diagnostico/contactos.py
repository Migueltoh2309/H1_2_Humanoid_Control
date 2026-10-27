import sys, os, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje import escena_faja as EF
from h1_2_algoritms.aprendizaje.habilidades import TIMING
B.ESCENARIO='exigente'; B.PERFIL=os.environ.get('PERF','tactil'); TIMING.update(B.PERFILES[B.PERFIL].get('timing',{}))
W=[tuple(map(float,w.split(':'))) for w in os.environ['WIN'].split(',')]  # t0:t1:fruta
af=EF.FajaMulti.after_step; last={}
def after(self, t):
    af(self, t)
    for t0,t1,fi in W:
        if t0<=t<=t1:
            m,d=self.m,self.d; fb=self.bid[int(fi)]; names=set()
            for k in range(d.ncon):
                c=d.contact[k]; b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
                if fb in (b1,b2): names.add(m.body(b2 if b1==fb else b1).name)
            key=(fi,tuple(sorted(names)))
            if last.get(fi)!=key:
                last[fi]=key; print(f"t={t:.2f} fruta {int(fi)} toca {sorted(names)} p={np.round(self.pos_torso(int(fi)),3)} st={self.state[int(fi)]}")
EF.FajaMulti.after_step=after
r,_=B.run_episode(B.make_policy('experto'), int(os.environ['SEED']))
