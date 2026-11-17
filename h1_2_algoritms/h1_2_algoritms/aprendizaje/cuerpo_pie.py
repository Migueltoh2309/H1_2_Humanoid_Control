"""El H1-2 DE PIE frente a la faja: arranque con el arnés y equilibrio propio (modo debug).

Hasta 2026-11-15 la escena de la faja tenía la pelvis SOLDADA al mundo y las piernas rectas. En el
laboratorio el robot no está así: cuelga de la grúa en modo debug (L2+R2) y, como en la tarea de
manipulación (tareas/manipulacion: manip/cuerpo.py + herramientas/arranque_cuerpo.py), el arranque es:

  1. COLGADO   el robot cuelga del arnés por el cuello, brazos colgando, pies a `alto_colgado` del
               suelo. Las piernas van a la postura de pie (rodillas 0.46 rad) en `t_postura` s.
  2. BAJADA    el operador baja la grúa despacio (`bajada_vel`, 1 cm/s): el contacto de los pies se
               detecta por el par de las rodillas y la ayuda de equilibrio entra en rampa
               (APOYANDO); con el cable flojo (`holgura_final`, sigue enganchado) queda DE_PIE.
  3. EQUILIBRIO  de pie y estable `t_espera_brazos` s (estrategia de tobillo con la IMU,
               aprendizaje/equilibrio.py, copia de la de manipulación).
  4. BRAZOS    suben por el lado (`brazos_subida`: abrir el hombro colgando, doblar el codo) y
               entran a la pose de ESPERA de la tesis (la `home` de bc_faja), por encima de la faja.
  5. PICKING   desde ahí manda rt/arm_sdk (el agente emulado de comandos_robot.py) con peso 1, que
               este control mezcla como el de Unitree; las piernas y la cintura siguen en equilibrio.

Todo en el MISMO proceso y paso a paso (determinista: sirve en el Pool de bc_faja), con el motor
del robot como PD puro + tau_ff (sin la compensación ideal de MujocoSim: gravity_comp=False). Las
piernas llevan la gravedad del modelo del equilibrio en tau; los brazos la del PC (g(q) en tau_ff).

Constantes medidas con este arranque (ver `medir()` / `python -m ... cuerpo_pie`):
  TORSO_Z_PIE   z de torso_link de pie: 2 cm más abajo que con la pelvis soldada (rodillas
                dobladas). La percepción y las poses relativas a la faja la usan (grasp_geometry,
                H12_CUERPO=pie).
  ADELANTO_X    cuánto se cuelga adelantado para que, al aterrizar, la pelvis quede en x = 0 (colgado
                del cuello el cuerpo se va hacia atrás, bajo la polea).
"""
import math
import os

import numpy as np

from h1_2_algoritms.aprendizaje.arnes import Arnes
from h1_2_algoritms.aprendizaje.equilibrio import Equilibrio, _rpy

# Configuración: la de tareas/manipulacion/config/cuerpo.yaml (2026-11-16), salvo la pose final de los
# brazos (la de espera de la tesis, no la de observación de la mesa).
CFG = {
    "masa_torso": 27.789,
    "postura_piernas": [0.0, -0.20, 0.0, 0.46, -0.26, 0.0, 0.0, -0.20, 0.0, 0.46, -0.26, 0.0],
    "cintura": 0.0,
    "kp_piernas": [200, 200, 200, 300, 40, 40, 200, 200, 200, 300, 40, 40],
    "kd_piernas": [5, 5, 5, 8, 2, 2, 5, 5, 5, 8, 2, 2],
    "kp_cintura": 200.0, "kd_cintura": 5.0,
    "kp_brazos": [100.0] * 14, "kd_brazos": [2.0] * 14,
    # el brazo izquierdo; el derecho en espejo
    "brazos_colgando": [0.0, 0.15, 0.0, 1.5, 0.0, 0.0, 0.0],
    "brazos_subida": [[0.0, 1.0, 0.0, 1.5, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    "t_tramo_brazos": 2.0,
    "t_espera_brazos": 2.0,
    "imu_kp": 800.0, "imu_kd": 100.0, "imu_t_integral": 2.0, "imu_max_deg": 5.0,
    "imu_factor_alabeo": 0.5, "par_imu_max": 60.0, "fc_derivada_hz": 15.0,
    "inclinacion_ref_deg": [0.0, 0.0],
    "umbral_contacto": 0.3, "rampa_apoyo": 1.5, "t_postura": 3.0,
    "inclinacion_max_deg": 3.0, "cop_max": 0.8,
    # arnés (simulador)
    "alto_colgado": 0.05, "bajada_vel": 0.01, "holgura_final": 0.03,
}
# Medido 2026-11-16 (faja, RH56, suelas de caja): con 0.024 (el de manipulación) la pelvis aterriza
# en x = -0.005 -> 0.029 la deja en ~0.
ADELANTO_X = 0.029
# Perfil de equilibrio (env H12_EQUILIBRIO):
#   "base"    el de manipulación (cuerpo.yaml), todo hasta 2026-11-17;
#   "rigido"  piernas (kp, kd) x2, estrategia de tobillo con la IMU x2 y la cintura de rt/arm_sdk x3
#             (ESCALA_CINTURA: la aplica quien arma el arm_sdk). Banco de balanceo 2026-11-17 (alcances
#             alternados de los dos brazos): el balanceo NO es hacia delante sino lateral (~5 mm) y de
#             guiñada (~2 grados, la mitad en la cintura); el punto de agarre, fijo al torso, se movía
#             11.4 mm (p99) durante el cierre de la mano -> 3.5 mm. Bajar las rodillas solo (0.85 rad)
#             daba 6.3 mm y baja el torso 5 cm (cambia el alcance a la faja).
#             En dev (40 semillas, 2026-11-17): experto 393 y BC 394 de 400, como la pelvis soldada.
PERFILES_EQUILIBRIO = ("base", "rigido")
CFG_BASE = dict(CFG)


def cfg_perfil(nombre):
    """(cfg, escala de la cintura de rt/arm_sdk) del perfil de equilibrio `nombre`."""
    import copy
    cfg = copy.deepcopy(CFG_BASE)
    if nombre == "rigido":
        cfg["kp_piernas"] = [2 * x for x in cfg["kp_piernas"]]
        cfg["kd_piernas"] = [2 * x for x in cfg["kd_piernas"]]
        cfg["imu_kp"], cfg["imu_kd"] = 1600.0, 200.0
        return cfg, 3.0
    if nombre != "base":
        raise ValueError(f"perfil de equilibrio {nombre!r}: {' | '.join(PERFILES_EQUILIBRIO)}")
    return cfg, 1.0


EQUILIBRIO = os.environ.get("H12_EQUILIBRIO", "base")
CFG, ESCALA_CINTURA = cfg_perfil(EQUILIBRIO)
# TORSO_Z_PIE (z de torso_link de pie) vive en vision/grasp_geometry.py: lo usa la percepción.

FASES = ("POSTURA", "COLGADO", "APOYANDO", "DE_PIE", "INESTABLE")
POSTURA, COLGADO, APOYANDO, DE_PIE, INESTABLE = range(len(FASES))
BRAZOS = ("ABAJO", "SUBIENDO", "ARRIBA")
ESPEJO = np.array([1, -1, -1, 1, -1, 1, -1], float)
PIERNAS_CINTURA = [f"{s}_{j}" for s in ("left", "right") for j in
                   ("hip_yaw_joint", "hip_pitch_joint", "hip_roll_joint", "knee_joint",
                    "ankle_pitch_joint", "ankle_roll_joint")] + ["torso_joint"]


def _minjerk(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


def escena_pie(root):
    """En el árbol MJCF de la escena de la faja: la pelvis con base flotante (freejoint, el primero
    del modelo: qpos[0:7], como espera el equilibrio)."""
    import xml.etree.ElementTree as ET
    pelvis = next(b for b in root.iter("body") if b.get("name") == "pelvis")
    if pelvis.find("freejoint") is None:
        pelvis.insert(0, ET.Element("freejoint", {"name": "floating_base_joint"}))
    # Suela: una caja plana en vez de la malla del tobillo (como la escena de manipulación,
    # construir_estacion: la misma que usa el equilibrio, PLANTA y MEDIO_PIE). Con la malla el pie
    # apoya en pocos puntos y la fricción blanda lo deja girar: de pie, la guiñada crecía sin parar
    # (20 grados en 60 s, 2026-11-16).
    for side in ("left", "right"):
        b = next(x for x in root.iter("body") if x.get("name") == f"{side}_ankle_roll_link")
        for g in b.findall("geom"):
            if g.get("contype") != "0":
                g.set("contype", "0")
                g.set("conaffinity", "0")
                g.set("group", "1")
        if not any(g.get("name") == f"{side}_suela" for g in b.findall("geom")):
            ET.SubElement(b, "geom", {"name": f"{side}_suela", "type": "box", "pos": "0.044 0 -0.035",
                                      "size": "0.13 0.043 0.01", "rgba": "0.2 0.2 0.2 0", "group": "3",
                                      "friction": "1.0 0.02 0.01"})


class ControlCuerpo:
    """El control de cuerpo entero (lo que en el robot hace manip/cuerpo.py a 500 Hz) más el arnés
    y el operador de la grúa del simulador. Uso, por cada paso de física:

        cc.paso(t, arm_sdk)      # escribe q_des, dq_des, kp, kd, tau_ff de los 27 motores en el PD
        sim.step(pd)

    `arm_sdk`: None, o dict(q, dq, kp, kd, tau) de los 15 motores de rt/arm_sdk (cintura + 14 de
    brazo, el ciclo de AgenteEmulado); se mezcla con peso 1 solo DE PIE y con los brazos ARRIBA."""

    def __init__(self, sim, pd, xml, brazos_final, cfg=CFG, adelanto_x=None, log=None):
        import mujoco
        self.mj, self.sim, self.pd, self.cfg = mujoco, sim, pd, cfg
        self.m, self.d = sim.model, sim.data
        m = self.m
        self.log = log or (lambda *a: None)
        assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "la escena no tiene base flotante (escena_pie)"
        self.torso = m.body("torso_link").id
        m.body_mass[self.torso] = float(cfg["masa_torso"])
        brazos = [f"{s}_{j}" for s in ("left", "right") for j in
                  ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
                   "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")]
        self.nombres = PIERNAS_CINTURA + brazos           # orden DDS (h1_2_juntas.NOMBRES)
        self.i27 = np.array([sim.mapping.index[n] for n in self.nombres])
        self.i_brazos = self.i27[13:]
        self.eq = Equilibrio(xml, self.nombres, cfg)
        self.arnes = Arnes(mujoco, m, self.d)
        # el peso que cuelga es el del ROBOT (Arnes suma todos los cuerpos: aquí, mesa, faja y frutas)
        pelvis = m.body("pelvis").id
        self.masa_robot = float(sum(m.body_mass[b] for b in range(m.nbody) if m.body_rootid[b] == pelvis))
        self.arnes.fijar_peso(self.masa_robot)
        self.kp_b = np.array(cfg["kp_brazos"], float)
        self.kd_b = np.array(cfg["kd_brazos"], float)
        colg = np.array(cfg["brazos_colgando"], float)
        self.q_colgando = np.concatenate([colg, colg * ESPEJO])
        self.brazos_final = np.asarray(brazos_final, float)
        self.ruta = [self.q_colgando] + [np.concatenate([np.array(p, float), np.array(p, float) * ESPEJO])
                                         for p in cfg["brazos_subida"]] + [self.brazos_final]
        self.t_b = (len(self.ruta) - 1) * float(cfg["t_tramo_brazos"])
        self.adelanto = ADELANTO_X if adelanto_x is None else float(adelanto_x)
        self.fase, self.t_fase = POSTURA, 0.0
        self.est_b, self.t_sub = 0, 0.0
        self.q_ini = None
        self.bajando = False
        self.tau_m = np.zeros(27)
        self.historia = []          # (t, fase, brazos, soporte, carga del cable, cabeceo, alabeo)

    # ------------------------------------------------------------------ arranque
    def colgar(self):
        """El robot como al encender en debug, colgado de la grúa: pies `alto_colgado` sobre el
        suelo, adelantado `adelanto`, brazos colgando, piernas rectas."""
        m, d, mj = self.m, self.d, self.mj
        d.qpos[2] += float(self.cfg["alto_colgado"])
        d.qpos[0] += self.adelanto
        d.qpos[[m.jnt_qposadr[m.joint(n).id] for n in self.nombres[13:]]] = self.q_colgando
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        self.arnes.colgar()
        q, _ = self.sim.get_q_dq()
        self.q_ini = q[self.i27].copy()

    def enderezar(self):
        """El operador gira el robot colgado para que los pies caigan sobre sus marcas: la guiñada
        de la base a 0 (al llevar las piernas a la postura colgado, la reacción lo gira sobre el
        cable). Como grua('enderezar') del simulador de manipulación. Devuelve lo corregido [grados]."""
        m, d, mj = self.m, self.d, self.mj
        w, x, y, z = d.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        qz = np.array([math.cos(-yaw / 2), 0.0, 0.0, math.sin(-yaw / 2)])
        out = np.zeros(4)
        mj.mju_mulQuat(out, qz, d.qpos[3:7].copy())
        d.qpos[3:7] = out
        d.qvel[3:6] = 0.0
        mj.mj_forward(m, d)
        return math.degrees(yaw)

    def guinada(self):
        w, x, y, z = self.d.qpos[3:7]
        return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @property
    def listo(self):
        """De pie, estable y con los brazos en la pose de espera: empieza el picking."""
        return self.fase in (DE_PIE, INESTABLE) and self.est_b == 2

    def imu(self):
        """(quat wxyz, giróscopo en el marco del torso) de torso_link, como imu_state."""
        v = np.zeros(6)
        self.mj.mj_objectVelocity(self.m, self.d, self.mj.mjtObj.mjOBJ_BODY, self.torso, v, 1)
        return self.d.xquat[self.torso].copy(), v[:3].copy()

    def _en_ruta(self, t):
        x = min(max(t / self.t_b, 0.0), 1.0) * (len(self.ruta) - 1)
        i = min(int(x), len(self.ruta) - 2)
        return self.ruta[i] + (self.ruta[i + 1] - self.ruta[i]) * _minjerk(x - i)

    # ------------------------------------------------------------------ un paso
    def paso(self, t, arm_sdk=None):
        cfg, eq, pd, dt = self.cfg, self.eq, self.pd, self.m.opt.timestep
        q_all, dq_all = self.sim.get_q_dq()
        q, dq = q_all[self.i27], dq_all[self.i27]
        quat, gyro = self.imu()
        # ---- el operador de la grúa: baja despacio cuando las piernas ya están en la postura
        if self.fase == COLGADO and not self.bajando and t - self.t_fase > 0.5:
            self.bajando = True
            self.guinada_corregida = self.enderezar()
            self.log(f"[cuerpo] t={t:.2f} el operador endereza ({self.guinada_corregida:+.2f} grados) "
                     f"y baja la grúa")
        if self.bajando and self.arnes.enganchado:
            if self.arnes.holgura() < float(cfg["holgura_final"]):
                self.arnes.bajar(float(cfg["bajada_vel"]) * dt)
            else:
                self.bajando = False
        self.arnes.aplicar(dt)
        # ---- piernas y cintura (equilibrio)
        apoyado = self.fase != POSTURA
        sop = eq.carga_pies(self.tau_m, dt) if apoyado else 0.0
        ql, dql, kpl, kdl, taul = eq.orden(q, dq, quat, gyro, dt, apoyado=apoyado, soporte=sop)
        # ---- brazos: quietos colgando hasta estar de pie y estable; luego suben a la de espera
        if self.est_b == 0 and self.fase == DE_PIE and t - self.t_fase > float(cfg["t_espera_brazos"]):
            self.est_b, self.t_sub = 1, t
            self.log(f"[cuerpo] t={t:.2f} de pie y estable: suben los brazos")
        if self.est_b == 0:
            q_brazos = self.q_colgando
        elif self.est_b == 1:
            q_brazos = self._en_ruta(t - self.t_sub)
            if t - self.t_sub >= self.t_b:
                self.est_b = 2
                self.log(f"[cuerpo] t={t:.2f} brazos en la pose de espera: empieza el picking")
        else:
            q_brazos = self.brazos_final
        qc = np.concatenate([ql, q_brazos])
        if self.fase == POSTURA:
            a = _minjerk(t / float(cfg["t_postura"]))
            qc[:13] = self.q_ini[:13] + (ql - self.q_ini[:13]) * a
            dql, taul = np.zeros(13), np.zeros(13)
            if a >= 1.0:
                self.fase, self.t_fase = COLGADO, t
                self.log(f"[cuerpo] t={t:.2f} piernas en la postura de pie (colgado)")
        # ---- fases
        if self.fase in (APOYANDO, DE_PIE, INESTABLE) and not eq.contacto:
            self.fase, self.t_fase = COLGADO, t
        if self.fase == COLGADO and eq.contacto:
            self.fase, self.t_fase = APOYANDO, t
            self.log(f"[cuerpo] t={t:.2f} contacto de los pies: entra el equilibrio")
        if self.fase == APOYANDO and sop >= 1.0 and t - self.t_fase > float(cfg["rampa_apoyo"]) + 1.0:
            self.fase, self.t_fase = DE_PIE, t
            self.log(f"[cuerpo] t={t:.2f} DE PIE (cable {self.arnes.carga():.0%} del peso)")
        if self.fase in (DE_PIE, INESTABLE):
            mal = np.max(np.abs(eq.error_incl)) > math.radians(float(cfg["inclinacion_max_deg"])) \
                or eq.cop_max() > float(cfg["cop_max"])
            if mal and self.fase == DE_PIE:
                self.fase = INESTABLE
            elif not mal and self.fase == INESTABLE:
                self.fase = DE_PIE
        dqc = np.zeros(27)
        kpc = np.concatenate([kpl, self.kp_b])
        kdc = np.concatenate([kdl, self.kd_b])
        tauc = np.zeros(27)
        dqc[:13], tauc[:13] = dql, taul
        # brazos del arranque: PD puro sin par (como manip/cuerpo.py); en el picking la gravedad la
        # pone el PC en el tau_ff de arm_sdk
        # ---- rt/arm_sdk con peso 1 (solo de pie y con los brazos arriba)
        if arm_sdk is not None and self.listo:
            b = slice(12, 27)              # cintura + brazos: los motores 12..26 de rt/arm_sdk
            qc[b], dqc[b], kpc[b], kdc[b], tauc[b] = (arm_sdk[k] for k in ("q", "dq", "kp", "kd", "tau"))
        pd.q_des[self.i27] = qc
        pd.dq_des[self.i27] = dqc
        pd.kp[self.i27] = kpc
        pd.kd[self.i27] = kdc
        pd.tau_ff[self.i27] = tauc
        # par del motor (tau_est de rt/lowstate) para el próximo carga_pies
        self.tau_m = np.clip(tauc + kpc * (qc - q) + kdc * (dqc - dq), -pd.tau_max[self.i27], pd.tau_max[self.i27])
        if int(round(t / dt)) % 50 == 0:
            r, p = _rpy(*quat)
            self.historia.append((t, self.fase, self.est_b, sop, self.arnes.carga(), math.degrees(p), math.degrees(r)))

    # ------------------------------------------------------------------ marco fijo para la percepción
    # De pie el torso (y la cámara, que va en él) se mueve al mover los brazos: 13-20 mm y 3-4 grados
    # (p99) en la faja, 2026-11-16. En el marco del torso la faja deja de estar quieta: la fruta
    # sube/baja ~2.4 cm, sale de la caja de la faja y el ajuste en recta del tracker falla. Los pies
    # no se mueven: la pose del torso respecto de los pies sale de la CINEMÁTICA DE LAS PIERNAS con
    # los ángulos medidos (encoders; sin verdad de terreno), como en el robot. Marco N = el torso al
    # terminar el arranque (en reposo): fijo al suelo y casi igual al del torso.
    def _torso_en_pies(self, q27):
        eq, mj = self.eq, self.mj
        if getattr(self, "_dk", None) is None:
            self._dk = mj.MjData(eq.m)
        m, d = eq.m, self._dk
        d.qpos[:] = m.qpos0
        d.qpos[eq.qadr] = q27
        d.qpos[0:3] = 0.0
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_kinematics(m, d)
        from h1_2_algoritms.aprendizaje.equilibrio import PLANTA
        pf = np.mean([d.xpos[b] + d.xmat[b].reshape(3, 3) @ PLANTA for b in eq.pies], axis=0)
        Rf = d.xmat[eq.pies[0]].reshape(3, 3)
        return Rf.T @ d.xmat[eq.torso].reshape(3, 3), Rf.T @ (d.xpos[eq.torso] - pf)

    def fijar_marco(self):
        """El marco N: el torso ahora (llamar con el robot quieto, al terminar el arranque)."""
        q, _ = self.sim.get_q_dq()
        self._ref_N = self._torso_en_pies(q[self.i27])

    def torso_en_N(self):
        """(R, p) del torso actual en el marco N: p_N = R p_torso + p."""
        q, _ = self.sim.get_q_dq()
        R1, p1 = self._torso_en_pies(q[self.i27])
        R0, p0 = self._ref_N
        return R0.T @ R1, R0.T @ (p1 - p0)

    # ------------------------------------------------------------------ instantánea
    # El arranque es el mismo en cada episodio (frutas estacionadas, sin ruido): se hace una vez y
    # los episodios empiezan desde ahí (física + estado interno del control, el equilibrio y el arnés).
    _NO_COPIAR = {"mj", "m", "d", "sim", "pd", "eq", "arnes", "log", "cfg", "_dk"}

    def guardar(self):
        import copy
        mj = self.mj
        n = mj.mj_stateSize(self.m, mj.mjtState.mjSTATE_INTEGRATION)
        fis = np.zeros(n)
        mj.mj_getState(self.m, self.d, fis, mj.mjtState.mjSTATE_INTEGRATION)
        return {"fisica": fis,
                "cc": {k: copy.deepcopy(v) for k, v in vars(self).items() if k not in self._NO_COPIAR},
                "eq": {k: copy.deepcopy(v) for k, v in vars(self.eq).items() if k not in ("mj", "m", "d", "cfg")},
                "arnes": {k: copy.deepcopy(v) for k, v in vars(self.arnes).items() if k not in ("mj", "m", "d")}}

    def restaurar(self, st):
        import copy
        mj = self.mj
        mj.mj_setState(self.m, self.d, st["fisica"], mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_forward(self.m, self.d)
        for obj, k in ((self, "cc"), (self.eq, "eq"), (self.arnes, "arnes")):
            vars(obj).update(copy.deepcopy(st[k]))

    def arrancar(self, t_max=60.0, asentar=0.5, cada=None, al_paso=None):
        """Toda la secuencia (colgar -> ... -> brazos en espera) y `asentar` s más. `al_paso(t)`
        cada paso (p. ej. para el vídeo). Devuelve el instante de fin (reloj del control)."""
        dt = self.m.opt.timestep
        self.colgar()
        t_listo = None
        for k in range(int(t_max / dt)):
            t = k * dt
            self.paso(t)
            self.sim.step(self.pd)
            if al_paso is not None:
                al_paso(t)
            if self.listo and t_listo is None:
                t_listo = t
            if t_listo is not None and t - t_listo >= asentar:
                self.fijar_marco()
                return t + dt
        raise RuntimeError(f"el arranque no terminó en {t_max} s: {self.estado()}")

    def estado(self):
        r, p = _rpy(*self.d.xquat[self.torso])
        return {"fase": FASES[self.fase], "brazos": BRAZOS[self.est_b], "soporte": self.eq.soporte,
                "cable": self.arnes.carga(), "holgura": self.arnes.holgura(),
                "cabeceo_deg": math.degrees(p), "alabeo_deg": math.degrees(r), "guinada_deg": self.guinada(),
                "pelvis": self.d.qpos[0:3].copy(), "torso_z": float(self.d.xpos[self.torso][2])}
