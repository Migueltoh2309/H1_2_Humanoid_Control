"""Límites del equilibrio de pie del H1-2 (cuerpo_pie: arranque con el arnés + estrategia de tobillo
con la IMU) ante perturbaciones de intensidad creciente.

Cada prueba parte de la misma instantánea (robot de pie, brazos en la pose de espera de la tesis),
aplica una perturbación con una magnitud y deja `--recuperar` s para recuperarse. Las magnitudes se
recorren de menor a mayor; al primer "cae" se refina el límite por bisección.

Perturbaciones (magnitud):
  empuje_adelante / empuje_atras / empuje_lado   fuerza horizontal en el pecho durante 0.2 s [N]
  brazos_arriba    los dos brazos de la espera a arriba (sobre la cabeza) y vuelta; magnitud =
                   velocidad pico del hombro [rad/s] (min-jerk: pico = 1.875 * amplitud / duración)
  brazos_frente    ídem, brazos estirados al frente (el centro de masas va hacia delante)
  brazo_lado       solo el brazo izquierdo, estirado al costado (asimétrico: alabeo y guiñada)
  carga_frente     brazos estirados al frente y una carga en cada mano que aparece en 0.5 s [kg]
                   (el control no la conoce: la gravedad del PC y el modelo del equilibrio no la ven)
  cintura          giro de la cintura de 0 a 0.8 rad y vuelta; magnitud = velocidad pico [rad/s]

Resultado de cada prueba:
  estable    inclinación del torso < 3 grados y las dos suelas planas en el suelo (< 3 grados) todo
             el tiempo
  recupera   se movió más (o un pie quedó de canto o despegó) pero no cae
  cae        el arnés recibe > 15 % del peso, o el torso se inclina > 15 grados, o la pelvis
             baja > 8 cm (con --sin-arnes no hay quien lo sujete y se ve la caída entera)

La inclinación es el ángulo entre el eje z del torso y la vertical respecto de la de partida (sin
guiñada). Los brazos van por el agente arm_sdk emulado (como en la tesis), con g(q) en tau_ff; la
velocidad del agente no se limita (la trayectoria la impone la prueba).

Uso:
  python prueba_equilibrio.py                                   # todo, perfiles base y rigido
  python prueba_equilibrio.py --pruebas empuje_lado carga_frente --perfiles rigido
  python prueba_equilibrio.py --video empuje_adelante 150 --perfiles rigido   # una prueba en vídeo
"""
import argparse
import csv
import math
import os
import time

os.environ["H12_CUERPO"] = "pie"
os.environ.setdefault("H12_FAJA_DX", "0.03")
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco                                                         # noqa: E402
import numpy as np                                                    # noqa: E402

import visual_servoing_grasp as V                                     # noqa: E402
from h1_2_algoritms.aprendizaje import escena_faja as EF              # noqa: E402
from h1_2_algoritms.aprendizaje import cuerpo_pie as CP               # noqa: E402
from h1_2_algoritms.aprendizaje import comandos_robot as CR           # noqa: E402
from h1_2_algoritms.vision import grasp_geometry as GG                # noqa: E402
from h1_2_mujoco_sim_bridge.mujoco_sim import MujocoSim               # noqa: E402
from h1_2_mujoco_sim_bridge.pd_controller import JointPD              # noqa: E402

SIDES = ("left", "right")
DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "..", "results", "equilibrio")

# poses del brazo IZQUIERDO (el derecho en espejo, CP.ESPEJO)
POSE = {"arriba": [-2.9, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
        "frente": [-1.57, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
        "lado": [0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0]}

PRUEBAS = {
    # nombre: (unidad, magnitudes)
    "empuje_adelante": ("N", [25, 50, 75, 100, 150, 200, 250, 300, 400]),
    "empuje_atras": ("N", [25, 50, 75, 100, 150, 200, 250, 300, 400]),
    "empuje_lado": ("N", [25, 50, 75, 100, 150, 200, 250, 300, 400]),
    "brazos_arriba": ("rad/s", [1, 2, 3, 4, 6, 8, 10, 13]),
    "brazos_frente": ("rad/s", [1, 2, 3, 4, 6, 8, 10, 13]),
    "brazo_lado": ("rad/s", [1, 2, 3, 4, 6, 8, 10, 13]),
    "carga_frente": ("kg", [0, 1, 2, 3, 4, 5, 7, 10, 15]),
    "cintura": ("rad/s", [0.5, 1, 2, 3, 4, 6, 8]),
}
T_EMPUJE = 0.2
AMPL_CINTURA = 0.8
LIM_ESTABLE_DEG, LIM_CAE_DEG, LIM_ARNES, LIM_BAJA = 3.0, 15.0, 0.15, 0.08
LIM_SUELA_DEG = 3.0         # suela inclinada más que esto: el pie quedó de canto (no apoya plano)


def _minjerk(x):
    x = min(max(x, 0.0), 1.0)
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5


def escena(con_faja):
    """La escena de la faja de pie; sin la faja ni la mesa (por omisión): nada en que apoyarse."""
    tree, root = EF._preparar(V.MJCF)
    if not con_faja:
        quitar = set(EF.BELT_BODIES) | {"mandarina"}
        for padre in root.iter():
            for h in list(padre):
                if h.tag in ("body", "geom") and h.get("name") in quitar:
                    padre.remove(h)
        # lo que nombraba a lo quitado: soldaduras, motores y sensores de la faja, exclusiones
        quedan = {e.get("name") for e in root.iter() if e.get("name")}
        for padre in root.iter():
            if padre.tag not in ("equality", "actuator", "sensor", "contact", "tendon"):
                continue
            for h in list(padre):
                refs = [h.get(a) for a in ("body1", "body2", "joint", "site", "objname", "geom1", "geom2")
                        if h.get(a)]
                if any(r not in quedan for r in refs):
                    padre.remove(h)
    return EF._escribir(tree, "h1_2_equilibrio_")


class Banco:
    """Un robot de pie listo (instantánea del arranque) para un perfil de equilibrio."""

    def __init__(self, perfil, con_faja=False, sin_arnes=False):
        CR.aplicar_limites_robot()
        self.perfil = perfil
        self.cfg, self.esc_cintura = CP.cfg_perfil(perfil)
        self.sin_arnes = sin_arnes
        self.xml = escena(con_faja)
        N = V.DEFAULT_JOINT_NAMES
        self.sim = sim = MujocoSim(self.xml, N, joint_armature=0.01, joint_damping=0.05, gravity_comp=False)
        self.m, self.d = m, d = sim.model, sim.data
        G = V.DEFAULT_GAINS
        self.pd = JointPD(sim.mapping, np.array([G[k][0] for k in N]), np.array([G[k][1] for k in N]),
                          np.array([G[k][2] for k in N]), np.zeros(len(N)))
        import bc_faja as B
        home = B.home_pose(B.PERFILES["suave_DR"].get("espera"))
        self.qh = np.concatenate([GG.solve_arm_ik(s, home[s])[0] for s in SIDES])
        self.hand_ids = {s: [m.actuator(k).id for k in GG.hand_actuator_names(s)] for s in SIDES}
        for s in SIDES:
            d.ctrl[self.hand_ids[s]] = GG.HAND_PRESHAPE
        self.arm_idx = [sim.mapping.index[k] for s in SIDES for k in V.ARM_JOINTS[s]]
        self.i_cint = sim.mapping.index["torso_joint"]
        self.torso = m.body("torso_link").id
        self.pelvis = m.body("pelvis").id
        self.munecas = [m.body(f"{s}_wrist_yaw_link").id for s in SIDES]
        self.masa0 = m.body_mass[self.munecas].copy()
        self.ipos0 = m.body_ipos[self.munecas].copy()
        self.suelas = [m.geom(f"{s}_suela").id for s in SIDES]
        self.cc = CP.ControlCuerpo(sim, self.pd, self.xml, self.qh, cfg=self.cfg, log=lambda *a: None)
        t0 = time.time()
        self.t0 = self.cc.arrancar(t_max=60)
        self.foto = self.cc.guardar()
        self.peso = float(sum(m.body_mass[b] for b in range(m.nbody) if m.body_rootid[b] == self.pelvis)) * 9.81
        print(f"[{perfil}] de pie en {self.t0:.1f} s de simulación ({time.time() - t0:.0f} s reales)", flush=True)

    # ------------------------------------------------------------------ una prueba
    def brazos_obj(self, prueba):
        q = self.qh.copy()
        if prueba == "brazo_lado":
            q[:7] = POSE["lado"]
            return q
        pose = {"brazos_arriba": "arriba", "brazos_frente": "frente", "carga_frente": "frente"}[prueba]
        q[:7] = POSE[pose]
        q[7:] = np.array(POSE[pose]) * CP.ESPEJO
        return q

    def correr(self, prueba, mag, recuperar=3.0, al_paso=None):
        m, d, sim, cc = self.m, self.d, self.sim, self.cc
        cc.restaurar(self.foto)
        m.body_mass[self.munecas] = self.masa0
        m.body_ipos[self.munecas] = self.ipos0
        d.qfrc_applied[:] = 0.0
        if self.sin_arnes:
            cc.arnes.soltar()
        dt = m.opt.timestep
        agente = CR.AgenteEmulado(hz=100.0, vel_max=100.0, tau_max=15.0)
        agente.kp[12] *= self.esc_cintura
        agente.kd[12] *= self.esc_cintura
        q_all, _ = sim.get_q_dq()
        q27 = np.zeros(CR.J.N)
        q27[CR.IDX_DDS] = q_all[self.arm_idx]
        q27[12] = q_all[self.i_cint]
        agente.iniciar(q27)
        n_ag = int(round(0.01 / dt))

        # perfil temporal de la perturbación
        t_ini = 0.5
        q0 = q_all[self.arm_idx].copy()
        tray = None                # (q_obj, duración de ida, espera arriba)
        if prueba in ("brazos_arriba", "brazos_frente", "brazo_lado"):
            q1 = self.brazos_obj(prueba)
            T = 1.875 * float(np.max(np.abs(q1 - q0))) / mag
            tray = (q1, T, 1.0)
            t_pert = 2 * T + 1.0
        elif prueba == "carga_frente":
            q1 = self.brazos_obj(prueba)
            tray = (q1, 2.0, None)              # va al frente en 2 s y se queda
            t_pert = 2.0 + 1.0 + 0.5            # 1 s quieto, la carga aparece en 0.5 s
        elif prueba == "cintura":
            T = 1.875 * AMPL_CINTURA / mag
            t_pert = 2 * T + 0.5
        else:
            t_pert = T_EMPUJE
        t_fin = t_ini + t_pert + recuperar

        R0 = d.xmat[self.torso].reshape(3, 3).copy()
        p0 = d.xpos[self.torso].copy()
        zp0 = float(d.xpos[self.pelvis][2])
        sf0 = d.geom_xpos[self.suelas].copy()
        desliz_max = suela_max = 0.0
        inc_max = gui_max = desp_max = arnes_max = baja_max = cop_max = 0.0
        despegue = 0.0
        sin_pie = np.zeros(2)
        cae, t_cae = False, None
        cmd = None
        f_emp = {"empuje_adelante": [mag, 0, 0], "empuje_atras": [-mag, 0, 0], "empuje_lado": [0, mag, 0]}
        for k in range(int(t_fin / dt)):
            t = k * dt
            tp = t - t_ini
            # --- brazos / cintura (agente arm_sdk)
            if k % n_ag == 0:
                q14 = q0
                if tray is not None and tp > 0:
                    q1, T, espera = tray
                    if espera is None or tp < T + espera:
                        q14 = q0 + (q1 - q0) * _minjerk(tp / T)
                    else:
                        q14 = q1 + (q0 - q1) * _minjerk((tp - T - espera) / T)
                if prueba == "cintura":
                    T = 1.875 * AMPL_CINTURA / mag
                    a = _minjerk(tp / T) if tp < T + 0.5 else 1.0 - _minjerk((tp - T - 0.5) / T)
                    agente.q_sostener[12] = q27[12] + AMPL_CINTURA * (a if tp > 0 else 0.0)
                agente.orden(q14, None, sim.gravity_torque()[self.arm_idx])
                cmd = agente.ciclo()
            # --- carga en las manos (en la palma, donde iría la fruta)
            if prueba == "carga_frente" and tp > 3.0:
                kg = mag * min(1.0, (tp - 3.0) / 0.5)
                for j, (b, s) in enumerate(zip(self.munecas, SIDES)):
                    m0 = self.masa0[j]
                    m.body_mass[b] = m0 + kg
                    m.body_ipos[b] = (m0 * self.ipos0[j] + kg * GG.GRASP_OFFSET[s]) / (m0 + kg)
            # --- empujón en el pecho
            d.qfrc_applied[:] = 0.0
            if prueba in f_emp and 0.0 <= tp < T_EMPUJE:
                punto = d.xpos[self.torso] + d.xmat[self.torso].reshape(3, 3) @ [0.0, 0.0, 0.25]
                mujoco.mj_applyFT(m, d, np.array(f_emp[prueba], float), np.zeros(3), punto, self.torso,
                                  d.qfrc_applied)
            cc.paso(self.t0 + t, cmd)
            sim.step(self.pd)
            if al_paso is not None:
                al_paso(t, prueba, mag)
            # --- medidas
            if k % 5:
                continue
            R = d.xmat[self.torso].reshape(3, 3)
            Rr = R0.T @ R
            inc = math.degrees(math.acos(min(1.0, max(-1.0, float(Rr[2, 2])))))
            gui = abs(math.degrees(math.atan2(Rr[1, 0], Rr[0, 0])))
            inc_max, gui_max = max(inc_max, inc), max(gui_max, gui)
            desp_max = max(desp_max, float(np.linalg.norm(d.xpos[self.torso] - p0)))
            desliz_max = max(desliz_max, float(np.max(np.linalg.norm((d.geom_xpos[self.suelas] - sf0)[:, :2], axis=1))))
            # suela de canto: el pie gira sobre su borde (sigue tocando el suelo, pero ya no apoya plano)
            suela_max = max(suela_max, max(math.degrees(math.acos(min(1.0, float(d.geom_xmat[g_][8]))))
                                           for g_ in self.suelas))
            baja = zp0 - float(d.xpos[self.pelvis][2])
            baja_max = max(baja_max, baja)
            carga = cc.arnes.tension / self.peso if cc.arnes.enganchado else 0.0
            arnes_max = max(arnes_max, carga)
            cop_max = max(cop_max, cc.eq.cop_max())
            en_suelo = np.zeros(2, bool)
            for i in range(d.ncon):
                c = d.contact[i]
                for j, g in enumerate(self.suelas):
                    if g in (c.geom1, c.geom2):
                        en_suelo[j] = True
            sin_pie = np.where(en_suelo, 0.0, sin_pie + 5 * dt)
            despegue = max(despegue, float(sin_pie.max()))
            if not cae and (carga > LIM_ARNES or inc > LIM_CAE_DEG or baja > LIM_BAJA):
                cae, t_cae = True, t
                if al_paso is None:
                    break
        if cae:
            res = "cae"
        elif inc_max < LIM_ESTABLE_DEG and suela_max < LIM_SUELA_DEG and despegue < 0.05:
            res = "estable"
        else:
            res = "recupera"
        return {"perfil": self.perfil, "prueba": prueba, "magnitud": mag, "unidad": PRUEBAS[prueba][0],
                "resultado": res, "inclinacion_max_deg": round(inc_max, 2), "guinada_max_deg": round(gui_max, 2),
                "torso_desp_max_mm": round(1000 * desp_max, 1), "pelvis_baja_max_mm": round(1000 * baja_max, 1),
                "pie_despegado_max_s": round(despegue, 3), "pies_desliz_max_mm": round(1000 * desliz_max, 1), "suela_incl_max_deg": round(suela_max, 1), "arnes_max_peso": round(arnes_max, 3),
                "cop_modelo_max": round(cop_max, 2), "t_cae": None if t_cae is None else round(t_cae - t_ini, 2)}


def barrer(banco, prueba, mags, recuperar, refinar, log):
    filas, ok, mal = [], None, None
    for mag in mags:
        r = banco.correr(prueba, mag, recuperar)
        filas.append(r)
        log(r)
        if r["resultado"] == "cae":
            mal = mag
            break
        ok = mag
    if mal is not None and ok is not None:
        for _ in range(refinar):
            mid = round(0.5 * (ok + mal), 3)
            r = banco.correr(prueba, mid, recuperar)
            r["bisección"] = True
            filas.append(r)
            log(r)
            if r["resultado"] == "cae":
                mal = mid
            else:
                ok = mid
    return filas, ok, mal


def _trabajo(a):
    perfil, pruebas, args = a
    banco = Banco(perfil, con_faja=args.con_faja, sin_arnes=args.sin_arnes)
    out, lim = [], []

    def log(r):
        print(f"  [{r['perfil']:6s}] {r['prueba']:15s} {r['magnitud']:7g} {r['unidad']:5s} -> {r['resultado']:8s} "
              f"incl {r['inclinacion_max_deg']:5.1f}°  guiñ {r['guinada_max_deg']:4.1f}°  "
              f"torso {r['torso_desp_max_mm']:5.0f} mm  suela de canto {r['suela_incl_max_deg']:4.1f}°  pies resbalan {r['pies_desliz_max_mm']:3.0f} mm  "
              f"arnés {r['arnes_max_peso']:.0%}", flush=True)
    for p in pruebas:
        mags = PRUEBAS[p][1]
        filas, ok, mal = barrer(banco, p, mags, args.recuperar, args.refinar, log)
        out += filas
        estables = [r["magnitud"] for r in filas if r["resultado"] == "estable"]
        lim.append({"perfil": perfil, "prueba": p, "unidad": PRUEBAS[p][0],
                    "estable_hasta": max(estables) if estables else None,
                    "aguanta_hasta": ok, "cae_desde": mal})
    return out, lim


def video(args):
    import cv2
    prueba, mag = args.video[0], float(args.video[1])
    perfil = args.perfiles[0]
    banco = Banco(perfil, con_faja=args.con_faja, sin_arnes=args.sin_arnes)
    r_ = mujoco.Renderer(banco.m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.9]
    cam.distance, cam.azimuth, cam.elevation = 3.0, 135 if "lado" not in prueba else 180, -10
    frames, k = [], {"n": 0}
    n_cam = max(1, int(round((1 / 30.0) / banco.m.opt.timestep)))

    def al_paso(t, p, mg):
        k["n"] += 1
        if k["n"] % n_cam:
            return
        r_.update_scene(banco.d, cam)
        banco.cc.arnes.dibujar(r_.scene)
        f = r_.render().copy()
        cv2.putText(f, f"{perfil}  {p} {mg:g} {PRUEBAS[p][0]}  t={t:4.2f}s", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        frames.append(f[:, :, ::-1])
    r = banco.correr(prueba, mag, args.recuperar, al_paso=al_paso)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{perfil}_{prueba}_{mag:g}.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for f in frames:
        w.write(f)
    w.release()
    print(r)
    print("vídeo:", os.path.normpath(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perfiles", nargs="+", default=list(CP.PERFILES_EQUILIBRIO), choices=CP.PERFILES_EQUILIBRIO)
    ap.add_argument("--pruebas", nargs="+", default=list(PRUEBAS), choices=list(PRUEBAS))
    ap.add_argument("--recuperar", type=float, default=3.0, help="s de recuperación tras la perturbación")
    ap.add_argument("--refinar", type=int, default=3, help="pasos de bisección del límite")
    ap.add_argument("--workers", type=int, default=2, help="procesos (~1.5 GB cada uno)")
    ap.add_argument("--con-faja", action="store_true", help="con la faja y la mesa delante (por omisión, sin)")
    ap.add_argument("--sin-arnes", action="store_true", help="suelta el arnés tras el arranque")
    ap.add_argument("--video", nargs=2, metavar=("PRUEBA", "MAGNITUD"), help="graba una prueba (el 1er perfil)")
    ap.add_argument("--out", default=None, help="CSV (por omisión results/equilibrio/limites.csv)")
    args = ap.parse_args()
    if args.video:
        return video(args)
    # un trabajo por (perfil, prueba): cada proceso hace su arranque
    trabajos = [(p, [x], args) for p in args.perfiles for x in args.pruebas]
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            res = pool.map(_trabajo, trabajos, chunksize=1)
    else:
        res = [_trabajo(t) for t in trabajos]
    filas = [r for f, _ in res for r in f]
    lims = [x for _, li in res for x in li]
    os.makedirs(OUT, exist_ok=True)
    out = args.out or os.path.join(OUT, "limites.csv")
    campos = list(dict.fromkeys(k for r in filas for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    with open(out.replace(".csv", "_resumen.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lims[0]))
        w.writeheader()
        w.writerows(lims)
    print(f"\n{len(filas)} pruebas en {time.time() - t0:.0f} s -> {os.path.normpath(out)}\n")
    print(f"{'prueba':16s} {'unidad':6s} " + " ".join(f"{p + ' estable/aguanta/cae':>30s}" for p in args.perfiles))
    for x in args.pruebas:
        celdas = []
        for p in args.perfiles:
            li = next(l_ for l_ in lims if l_["perfil"] == p and l_["prueba"] == x)
            f = lambda v: "-" if v is None else f"{v:g}"     # noqa: E731
            celdas.append(f"{f(li['estable_hasta']):>8s} / {f(li['aguanta_hasta']):>7s} / {f(li['cae_desde']):>7s}")
        print(f"{x:16s} {PRUEBAS[x][0]:6s} " + " ".join(f"{c:>30s}" for c in celdas))
    print("\n(estable: incl del torso < 3° y suelas planas < 3°; aguanta: no cae; cae: arnés > 15 % del peso, "
          "incl > 15° o la pelvis baja > 8 cm; '-' = no llegó a caer en el rango probado)")


if __name__ == "__main__":
    main()
