import sys, os, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje import escena_faja as EF, habilidades as H
from h1_2_algoritms.vision import grasp_geometry as GG
rel = EF.FajaMulti.release
def release(self, side, deposited):
    if not deposited:
        i = self.held[side]
        print(f"  release {side} deposited={deposited} held={i}", "fruta z_torso=%s" % (np.round(self.pos_torso(i),3) if i is not None else None))
    return rel(self, side, deposited)
EF.FajaMulti.release = release; B.FajaMulti = EF.FajaMulti
st = H.ArmExecutor.step
def step(self, t, q_meas, est=None, vel=None):
    out = st(self, t, q_meas, est, vel)
    if 'fallo' in out[3] or 'cierra' in out[3]:
        q = q_meas[:7] if self.side=='left' else q_meas[7:]
        x = GG.FK[self.side](q)[0:3,3]
        tgt = GG.show_wrist_target(self.side)[0:3]
        print(f"t={t:.2f} {self.side} {out[3]} track={self.track} wrist={np.round(x,3)} show_err={np.linalg.norm(x-tgt)*1e3:.0f}mm hits={self.fsm._hits if self.fsm else None}")
    return out
H.ArmExecutor.step = step
tw = EF.FajaMulti.try_weld
last={}
def try_weld(self, side, expected, tol=0.02):
    import mujoco
    r = tw(self, side, expected, tol)
    belt=[i for i in range(self.n) if self.state[i]=='belt']
    if belt:
        i=min(belt,key=lambda i: np.linalg.norm(self.pos_torso(i)-expected))
        dv=self.pos_torso(i)-expected
        m,d=self.m,self.d; gs=set()
        for k in range(d.ncon):
            c=d.contact[k]; b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
            for fb,o in ((b1,b2),(b2,b1)):
                if fb==self.bid[i]:
                    nm=m.body(o).name
                    for g in ("thumb","index","middle","ring","pinky","wrist_yaw"):
                        if g in nm: gs.add(g)
        key=(side,i)
        if r is None and (key not in last or abs(self.d.time-last[key])>0.4):
            last[key]=self.d.time
            print(f"   weld? {side} t={self.d.time:.2f} fruta {i} d={np.round(dv*1e3)}mm |d|={np.linalg.norm(dv)*1e3:.0f} grupos={sorted(gs)}")
    return r
EF.FajaMulti.try_weld = try_weld
r,_ = B.run_episode(B.make_policy(os.environ['POL']), int(os.environ['SEED']))
print({k:v for k,v in r.items() if k!='_eventos'})
