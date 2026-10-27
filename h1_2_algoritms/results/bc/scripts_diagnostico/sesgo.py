import sys, os, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B, mujoco
import visual_servoing_grasp as V
from h1_2_mujoco_sim_bridge.mujoco_sim import MujocoSim
from h1_2_algoritms.aprendizaje.escena_faja import build_xml, FajaMulti, BELT_TOP, FRUIT_R
from h1_2_algoritms.vision.multi_tracking import detect_fruits
from h1_2_algoritms.vision import grasp_geometry as GG
sim = MujocoSim(build_xml(V.MJCF, 1), V.DEFAULT_JOINT_NAMES); m,d=sim.model,sim.data
for s in ('left','right'):
    q,_=GG.solve_arm_ik(s,V.HOME[s]); d.qpos[[m.jnt_qposadr[m.joint(k).id] for k in V.ARM_JOINTS[s]]]=q
f=FajaMulti(m,d,1,0.05); f.state[0]='belt'
cams=V.Cameras(m,d,noise=os.environ.get('NOISE','0')=='1',seed=0)
R_cam,p_cam=cams.T_torso("robot_rgbd_camera")
a=f.qadr[0]
for x in (0.30,0.35,0.40):
    row=[]
    for y in np.arange(-0.55,0.56,0.1):
        d.qpos[a:a+3]=[x,y,BELT_TOP+FRUIT_R+0.001]; d.qpos[a+3:a+7]=[1,0,0,0]; mujoco.mj_forward(m,d)
        rgb,_,depth,_,_=cams.grab(); dets=detect_fruits(rgb,depth,cams.K,R_cam,p_cam)
        tr=f.pos_torso(0)
        if dets:
            e=(min(dets,key=lambda z: np.linalg.norm(z-tr))-tr)*1e3; row.append(f"y={y:+.2f}: [{e[0]:+4.0f}{e[1]:+4.0f}{e[2]:+4.0f}]")
        else: row.append(f"y={y:+.2f}: --")
    print(f"x={x:.2f}  ", "  ".join(row))
