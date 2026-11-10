"""Biblioteca de habilidades de un brazo (fase B del LfD, docs/LFD_BC.md).

La política de alto nivel no mueve articulaciones: elige HABILIDADES con
parámetros simbólicos. Este ejecutor las corre, una por brazo, para que
los dos brazos trabajen a la vez sobre frutas distintas:

    agarrar(pista)  DMP de alcance (fase A) -> cerrar -> levantar ->
                    mostrar a la cámara y verificar     (GraspServoing)
    depositar       llevar la fruta a la pose de depósito al costado y abrir
    volver          a home con la mano en preforma -> libre

Cada habilidad tiene precondición (brazo libre; para depositar, fruta
verificada en la mano) y efecto (la fruta sale de la faja / entra al
depósito). Las metas cartesianas de los dos ejecutores van al MISMO QP
bimanual, que impone límites y distancias brazo-brazo y con el entorno.

Depositar y volver van en ESPACIO ARTICULAR (`joint_space`, por omisión):
servo hacia posturas precalculadas por IK en la rama de home
(`joint_target`, el `joint_targets` del QP). Con metas cartesianas, el giro
de muñeca al depósito (−90° de azimut) llevaba al QP local a otra rama: las
articulaciones 2 y 4 quedaban en sus límites, el brazo ya no volvía a home
(timeout de 3 s, a 38 mm y 0.24 rad) y todos los alcances siguientes de ese
brazo fallaban a 260-290 mm (medido en los episodios de varias frutas).
"""
import warnings

import numpy as np

from h1_2_algoritms.vision import grasp_geometry as GG
from h1_2_algoritms.vision.visual_servoing import GraspServoing

# Depósito al costado, fuera de la faja: dedos 60° hacia abajo y hacia
# afuera (barrido de IK: alcanzable en ambos brazos).
# (z relativa a la faja: GG.DZ_MESA)
DEPOSIT_FRUIT = {"left": np.array([0.15, 0.40, -0.05 + GG.DZ_MESA]),
                 "right": np.array([0.15, -0.40, -0.05 + GG.DZ_MESA])}
DEPOSIT_AZ = {"left": np.deg2rad(90.0), "right": np.deg2rad(-90.0)}
DEPOSIT_ELEV = np.deg2rad(-60.0)

# Modelo de tiempos de las habilidades [s]. Lo usan el experto y las
# características del estado. Medianas medidas con `demos/bc_faja.py
# calibrar` (16 episodios, ~100 ciclos, 2026-10-23, depositar y volver en
# espacio articular, servo Kp = 6): alcance = τ + 0.33 (3.25 s con
# τ = 2.92), cierra->verificado 2.90, depositar 1.45, volver 0.90.
TIMING = {"reach_extra": 0.33, "close_lift": 1.9, "show_verify": 1.0, "deposit": 1.45, "home": 0.9}


# Depósito CORTO (opcional, `ArmExecutor(deposit_pose=DEPOSIT_SHORT)`):
# dedos 60° hacia abajo pero apuntando al frente (azimut 0), delante del
# borde de la faja (x = 0.20 < 0.275 − radio). Desde home la articulación
# que más gira lo hace 0.77 rad, contra 1.65 con el azimut de ±90°.
# (x, y, z de la fruta del brazo derecho, elevación, azimut); la izquierda
# se refleja en y y en azimut.
# Mesa u_table (faja 14 cm más alta): con los dedos a -60° el depósito no
# tiene IK dentro de los límites del robot (muñeca pitch <= 0.30, yaw <= 0.96;
# comandos_robot.aplicar_limites_robot) y el IK caía en otra rama (hombro
# yaw -2.6, 2 rad desde home): el brazo barría la faja al depositar y al
# volver (35 frutas tiradas en 180 episodios --robot). A -45° hay solución
# a <= 0.75 rad de home con los dos juegos de límites.
DEPOSIT_SHORT = ((0.20, -0.40, -0.05 + GG.DZ_MESA), -45.0, 0.0)


def deposit_wrist_target(side, pose=None):
    if pose is None:
        R = GG.grasp_rotation(side, elev=DEPOSIT_ELEV, az=DEPOSIT_AZ[side])
        return GG.grasp_wrist_target(side, DEPOSIT_FRUIT[side], R)
    (x, y, z), elev, az = pose
    sg = 1.0 if side == "right" else -1.0
    R = GG.grasp_rotation(side, elev=np.deg2rad(elev), az=np.deg2rad(sg * az))
    return GG.grasp_wrist_target(side, np.array([x, sg * y, z]), R)


def cycle_time(reach_tau, timing=TIMING):
    """Duración de un ciclo completo agarrar + depositar + volver."""
    return reach_tau + sum(timing.values())


class ArmExecutor:

    def __init__(self, side, home, reach_dmp=None, reach_tau=None, timing=TIMING,
                 deposit_tol=0.03, deposit_timeout=3.5, open_time=0.5, home_tol=0.03, home_timeout=3.0,
                 joint_space=True, joint_tol=0.08, open_tol=None, grasp_kwargs=None, deposit_pose=None,
                 home_vel_tol=None):
        self.side = side
        self.home = np.asarray(home[side], dtype=float)
        self._home_all = home
        self.reach_dmp = reach_dmp
        self.reach_tau = reach_tau if reach_tau is not None else \
            (reach_dmp.tau_demo if reach_dmp is not None else 3.6)
        self.timing = timing
        self.deposit = deposit_wrist_target(side, deposit_pose)
        self.deposit_tol = deposit_tol
        self.deposit_timeout = deposit_timeout
        self.open_time = open_time
        self.home_tol = home_tol
        self.home_timeout = home_timeout
        self.joint_space = joint_space
        self.joint_tol = joint_tol
        self.open_tol = joint_tol if open_tol is None else open_tol   # empieza a abrir a esta distancia del depósito
        # volver termina además con el brazo asentado (|q̇| < home_vel_tol rad/s,
        # por diferencias de q entre ciclos); sin esto se pasa al control
        # cartesiano a media marcha (~0.5 m/s en la muñeca) y el brazo rebota
        self.home_vel_tol = home_vel_tol
        self._q_prev = None
        self.grasp_kwargs = dict(grasp_kwargs or {})                # parámetros de GraspServoing (perfil de tiempos)
        self.q_home = self.q_deposit = None
        if joint_space:
            self.q_home, _ = GG.solve_arm_ik(side, self.home)
            self.q_deposit, ok = GG.solve_arm_ik(side, self.deposit, q0=self.q_home)
            if not ok:
                warnings.warn(f"{side}: el depósito no tiene IK dentro de los límites articulares; "
                              f"la postura usada está a {np.max(np.abs(self.q_deposit - self.q_home)):.2f} rad de home")
        self.joint_target = None    # postura articular para el QP (depositar / volver) o None
        self.skill = None           # None (libre) | "agarrar" | "depositar" | "volver"
        self.fsm = None
        self.track = None
        self.target_p = None        # última meta (fruta) del agarre, torso_link
        self.t_skill = 0.0
        self._opening = None
        self.log = []               # (t, evento, pista)

    # ------------------------------------------------------------ estado
    @property
    def busy(self):
        return self.skill is not None

    @property
    def phase(self):
        if self.skill == "agarrar":
            return self.fsm.phase
        return self.skill or "LIBRE"

    def remaining(self, t):
        """Tiempo estimado hasta quedar libre (modelo de tiempos)."""
        T = self.timing
        el = t - self.t_skill
        if self.skill is None:
            return 0.0
        if self.skill == "volver":
            return max(0.0, T["home"] - el)
        if self.skill == "depositar":
            return max(0.0, T["deposit"] - el) + T["home"]
        ph = self.fsm.phase
        el = t - self.fsm.t_phase
        rest = T["deposit"] + T["home"]
        if ph in ("REACH", "APPROACH", "DESCEND"):
            return max(0.0, self.reach_tau + T["reach_extra"] - (t - self.t_skill)) + \
                T["close_lift"] + T["show_verify"] + rest
        if ph in ("CLOSE", "LIFT"):
            return max(0.0, T["close_lift"] - el) + T["show_verify"] + rest
        if ph in ("RETREAT", "DONE"):
            return max(0.0, T["show_verify"] - el) + rest
        return T["home"]

    # ------------------------------------------------------------ habilidades
    def pick(self, t, track_id):
        assert self.skill is None, "precondición: brazo libre"
        self.fsm = GraspServoing(self._home_all, reach_dmp=self.reach_dmp, reach_tau=self.reach_tau,
                                 auto_start=False, **self.grasp_kwargs)
        self.fsm.start(t, self.side)
        self.skill, self.track, self.t_skill = "agarrar", track_id, t
        self.log.append((t, "agarrar", track_id))

    def _go(self, skill, t):
        self.skill, self.t_skill, self._opening = skill, t, None

    def step(self, t, q_meas, est=None, vel=None):
        """Un ciclo de control. q_meas: 14 (izq + der). est, vel: predicción
        de la pista asignada (o None si no se ve / ya no hace falta).

        Devuelve (meta 7, feedforward 3 o None, mano 6, eventos), con eventos
        en {"cierra", "verificado", "fallo", "perdida", "suelta", "libre"}."""
        q = q_meas[:7] if self.side == "left" else q_meas[7:]
        qd = None if self._q_prev is None or t <= self._q_prev[0] else (q - self._q_prev[1]) / (t - self._q_prev[0])
        self._q_prev = (t, q.copy())
        ev = []
        self.joint_target = None
        if self.skill is None:
            return self.home.copy(), None, GG.HAND_PRESHAPE.copy(), ev
        if self.skill == "agarrar":
            ph0 = self.fsm.phase
            n = 10 if est is not None else 0
            out = self.fsm.step(t, q_meas, est, vel, n)
            ph = out["phase"]
            if est is not None and ph in ("REACH", "APPROACH", "DESCEND"):
                self.target_p = np.asarray(est).copy()
            if ph != ph0:
                if ph == "CLOSE":
                    ev.append("cierra")
                elif ph == "DONE":
                    ev.append("verificado")
                    self._go("depositar", t)
                elif ph == "FAILED":
                    ev.append("fallo")
                    self._go("volver", t)
                elif ph == "MISSED":
                    ev.append("perdida")
                    self._go("volver", t)
            for e in ev:
                extra = (self.fsm.events.get("err_close"),) if e == "cierra" else ()
                self.log.append((t, e, self.track, self.fsm.events.get("why", ""), *extra))
            if self.skill == "agarrar":
                return out["targets"][self.side], out["x_dot_ff"][self.side], out["hand"][self.side], ev
        if self.skill == "depositar":
            x = GG.FK[self.side](q)[0:3, 3]
            near = np.linalg.norm(x - self.deposit[0:3]) < self.deposit_tol
            if self.joint_space:
                near = np.max(np.abs(q - self.q_deposit)) < self.open_tol
            if self._opening is None and (near or t - self.t_skill > self.deposit_timeout):
                self._opening = t
            if self._opening is not None and t - self._opening > self.open_time:
                ev.append("suelta")
                self.log.append((t, "suelta", self.track))
                self._go("volver", t)
            else:
                hand = GG.HAND_OPEN.copy() if self._opening is not None else GG.HAND_CLOSED.copy()
                self.joint_target = self.q_deposit
                return self.deposit.copy(), None, hand, ev
        if self.skill == "volver":
            x = GG.FK[self.side](q)[0:3, 3]
            home = np.max(np.abs(q - self.q_home)) < self.joint_tol if self.joint_space else \
                np.linalg.norm(x - self.home[0:3]) < self.home_tol
            if self.home_vel_tol is not None:
                home = home and qd is not None and np.max(np.abs(qd)) < self.home_vel_tol
            self.joint_target = self.q_home
            if home or t - self.t_skill > self.home_timeout:
                ev.append("libre")
                self.log.append((t, "libre", self.track))
                self.skill, self.track, self.fsm, self.target_p = None, None, None, None
                self.joint_target = None
            return self.home.copy(), None, GG.HAND_PRESHAPE.copy(), ev
        return self.home.copy(), None, GG.HAND_PRESHAPE.copy(), ev

    def observe(self, t, z):
        """Medida cruda de una fruta (para la verificación del agarre)."""
        if self.skill == "agarrar":
            self.fsm.observe(t, z)
