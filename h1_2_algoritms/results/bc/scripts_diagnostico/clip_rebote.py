"""Clip en cámara lenta de la primera vuelta del depósito (un episodio de bc_faja).

Renderiza a 60 Hz con una cámara cercana al brazo, desde PRE s antes de soltar
la fruta hasta POST s después de que el brazo queda libre, y dibuja la estela
de la muñeca. Se reproduce a 15 fps (4x más lento).
  PERF=espera SEED=10018 POL=bc OUT=clip.mp4 python3 clip_rebote.py
"""
import sys, os, collections, numpy as np, mujoco, cv2
PERF, SEED, POL, OUT = os.environ["PERF"], int(os.environ["SEED"]), os.environ.get("POL", "bc"), os.environ["OUT"]
PRE, POST, HZ = 0.3, 1.2, 60
sys.argv = ['x']; sys.path.insert(0, os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms/demos'))
import bc_faja as B
from h1_2_mujoco_sim_bridge.mujoco_sim import MujocoSim
from h1_2_algoritms.aprendizaje import habilidades as H
B.ESCENARIO, B.PERFIL = "exigente", PERF
H.TIMING.update(B.PERFILES[PERF].get("timing", {}))
if POL == "bc":
    B.NET_PATH = os.path.join(B.OUT, f"politica_bc_exigente_{PERF}.pt")
EX = {}
_init = H.ArmExecutor.__init__
def init(self, side, *a, **k):
    _init(self, side, *a, **k); EX[side] = self
H.ArmExecutor.__init__ = init
st = {"buf": collections.deque(maxlen=int(PRE * HZ)), "rec": [], "side": None, "t_libre": None, "done": False,
      "trail": []}
_step = MujocoSim.step
def step(self, pd):
    _step(self, pd)
    if st["done"]:
        return
    m, d = self.model, self.data
    if not hasattr(self, "_r"):
        self._r = mujoco.Renderer(m, 540, 720); self._cam = mujoco.MjvCamera()
        self._k = max(1, int(round(1 / (HZ * m.opt.timestep)))); self._n = 0
    self._n += 1
    if self._n % self._k:
        return
    if st["side"] is None:
        for s, e in EX.items():
            if e.skill == "volver":
                st["side"] = s
    s = st["side"] or "right"
    e = EX.get(s)
    if st["side"] is not None and st["t_libre"] is None and e.skill != "volver":
        st["t_libre"] = d.time
    if st["t_libre"] is not None and d.time > st["t_libre"] + POST:
        st["done"] = True
        return
    sg = 1 if s == "left" else -1
    cam = self._cam
    cam.lookat[:] = [0.30, sg * 0.22, 0.95]; cam.distance, cam.azimuth, cam.elevation = 1.0, 180 + sg * 35, -20
    self._r.update_scene(d, cam)
    img = self._r.render()[:, :, ::-1].copy()
    # estela de la muñeca (proyección de la cámara libre)
    wid = m.body(f"{s}_wrist_yaw_link").id
    p = d.xpos[wid].copy()
    if st["side"] is not None:
        st["trail"].append(p)
    con = self._r.scene.camera[0]
    fwd = np.array(con.forward); up = np.array(con.up); pos = np.array(con.pos); right = np.cross(fwd, up)
    fy = 0.5 * 540 / np.tan(np.radians(m.vis.global_.fovy) / 2)
    pts = []
    for q in st["trail"]:
        v = q - pos; z = v @ fwd
        pts.append((int(360 + fy * (v @ right) / z), int(270 - fy * (v @ up) / z)))
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, (0, 220, 255), 2, cv2.LINE_AA)
    phase = e.phase if e else ""
    cv2.putText(img, f"t = {d.time:5.2f} s   brazo {'izq' if s == 'left' else 'der'}: {phase}   (4x lento)",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    (st["rec"] if st["side"] is not None else st["buf"]).append(img)
MujocoSim.step = step
class Stop(Exception): pass
_old = B.FajaMulti.after_step
def after(self, t):
    _old(self, t)
    if st["done"]:
        raise Stop
B.FajaMulti.after_step = after
try:
    B.run_episode(B.make_policy(POL), SEED)
except Stop:
    pass
frames = list(st["buf"]) + st["rec"]
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 15, (720, 540))
for f in frames:
    vw.write(f)
vw.release()
print(OUT, len(frames), "cuadros; brazo", st["side"], "libre en", st["t_libre"])
