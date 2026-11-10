"""Comandos del ROBOT REAL para la faja (paso sim2real del LfD, docs/LFD_ROBOT.md).

En el sim de la tesis (bc_faja.py) el QP escribe `q_des` en un PD ideal: con
compensación de gravedad perfecta en el motor, las ganancias de
`DEFAULT_GAINS` (hombro kp 60), sin límite de velocidad, y las manos reciben
radianes. En el H1-2 la misma decisión viaja así (h1_2_sim2real, README §4):

    PC: /joint_cmd (q, dq, tau_ff)  --UDP-->  agente_sdk.py  --DDS-->  rt/arm_sdk
                                                  │ recorta a límites (margen 0.05),
                                                  │ |tau_ff| <= --tau-max, rampa a --vel-max,
                                                  │ kp/kd de h1_2_juntas, a --hz (100)
                                                  └-- Modbus --> Inspire ANGLE_SET 0-1000

y el motor del robot es un PD PURO: tau = tau_ff + kp·(q - q_m) + kd·(dq - dq_m),
saturado al par del URDF; la gravedad la tiene que poner el PC en tau_ff.

Este módulo reproduce ese camino paso a paso, CON LA MISMA TABLA de juntas,
límites y ganancias que usa el agente (`h1_2_sim2real/comun/h1_2_juntas.py`,
se carga de ahí, no se copia):

    AgenteEmulado   lo que hace agente_sdk.py con cada orden y en cada ciclo
                    (_rx, _avanzar_objetivo, _publicar), y lo que publicaría
                    en rt/arm_sdk: q, dq, tau, kp, kd de las juntas 12-26.
    ManosInspire    rad del sim <-> registros 0-1000 de la RH56 (orden
                    [meñique, anular, medio, índice, pulgar flexión, pulgar
                    rotación]), con la velocidad de SPEED_SET (recorrido
                    completo en 0.6 s a 1000, como ManoSim).
    Registro        el flujo de comandos tal cual iría al robot (.npz), para
                    revisarlo y para reproducirlo contra el agente
                    (demos/reproducir_comandos.py).
"""
import importlib.util
import os

import numpy as np

from h1_2_algoritms.vision import grasp_geometry as GG


def _cargar_comun(nombre):
    """Módulo maestro de h1_2_sim2real/comun (sin ROS ni numpy)."""
    aqui = os.path.dirname(os.path.realpath(__file__))
    cands = [os.environ.get("H1_2_SIM2REAL_COMUN", ""),
             os.path.normpath(os.path.join(aqui, "..", "..", "..", "h1_2_sim2real", "comun")),
             os.path.expanduser("~/humanoid_ws/src/h1_2_sim2real/comun"),
             os.path.expanduser("~/Github/H1_2_Humanoid_Control/h1_2_sim2real/comun")]
    for c in cands:
        p = os.path.join(c, nombre + ".py")
        if c and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location(f"s2r_{nombre}", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(f"no encuentro h1_2_sim2real/comun/{nombre}.py (H1_2_SIM2REAL_COMUN=...)")


J = _cargar_comun("h1_2_juntas")
P = _cargar_comun("protocolo_sim2real")

SIDES = ("left", "right")
_ARM = ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
        "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")
ARM_NAMES = {s: [f"{s}_{j}" for j in _ARM] for s in SIDES}
# índice DDS de cada una de las 14 juntas de brazo del QP (izq + der)
IDX_DDS = np.array([J.INDICE[n] for s in SIDES for n in ARM_NAMES[s]])
BRAZOS = np.array(J.BRAZOS)                    # 12..26: lo que manda rt/arm_sdk
LADO_MANO = {"left": "izq", "right": "der"}
# columna de GG.HAND_ACTUATORS (orden del sim) para cada registro de la Inspire
_COL_GG = [GG.HAND_ACTUATORS.index(n) for n, _ in J.DOF_MANO]


# ---------------------------------------------------------------- límites
def limites_robot(side):
    """(7, 2) [lo, hi] del brazo `side` en el orden de joint_limits: los del
    agente (J.LIMITES, ya con J.MARGEN), que son los que llegan al motor."""
    return np.array([[J.LIMITES[J.INDICE[n]][0] + J.MARGEN, J.LIMITES[J.INDICE[n]][1] - J.MARGEN]
                     for n in ARM_NAMES[side]])


def aplicar_limites_robot():
    """QP e IK con los límites del agente en vez de los del MJCF. Si el QP
    planifica con límites más anchos, el agente recorta y el q_ref que integra
    el QP se separa del robot (el codo llegó a pedir 2.85 rad con tope 1.55):
    el brazo ya no vuelve a home. Llamar antes de crear controladores/IK."""
    from h1_2_algoritms.movimiento import joint_limits as JL
    antes = {s: JL.ARM_LIMITS[s].copy() for s in SIDES}
    for s in SIDES:
        JL.ARM_LIMITS[s] = limites_robot(s)
    return antes


# ---------------------------------------------------------------- manos
def rad_a_inspire(side, hand_rad):
    """6 rad en el orden del sim (GG.HAND_ACTUATORS) -> 6 enteros ANGLE_SET."""
    return [J.rad_a_angulo(k, float(hand_rad[_COL_GG[k]])) for k in range(6)]


def inspire_a_rad(angulos):
    """6 ANGLE (0 cerrado .. 1000 abierto) -> 6 rad en el orden del sim."""
    out = np.zeros(6)
    for k in range(6):
        out[_COL_GG[k]] = J.angulo_a_rad(k, angulos[k])
    return out


class ManosInspire:
    """Las dos RH56DFTP vistas desde el sim: consigna entera 0-1000 (lo que
    escribe el agente en ANGLE_SET, ~30 Hz) y ángulo que avanza a SPEED_SET."""

    T_RECORRIDO = 0.6     # s de 0 a 1000 a SPEED_SET = 1000 (ManoSim, RH56)

    def __init__(self, velocidad=1000, hz=30.0):
        self.vel = velocidad
        self.dt_bus = 1.0 / hz
        self.pedido = {s: [1000] * 6 for s in SIDES}
        self.consigna = {s: [1000] * 6 for s in SIDES}    # lo último que llegó a la mano
        self.actual = {s: np.full(6, 1000.0) for s in SIDES}
        self._t_bus = -np.inf

    def iniciar(self, side, hand_rad):
        a = rad_a_inspire(side, hand_rad)
        self.pedido[side] = list(a)
        self.consigna[side] = list(a)
        self.actual[side] = np.array(a, dtype=float)

    def mandar(self, side, hand_rad):
        self.pedido[side] = rad_a_inspire(side, hand_rad)

    def paso(self, t, dt):
        """Avanza dt. Devuelve {side: 6 rad (orden del sim)} para d.ctrl."""
        if t - self._t_bus >= self.dt_bus - 1e-9:      # el hilo Modbus del agente escribe a ~30 Hz
            self._t_bus = t
            for s in SIDES:
                self.consigna[s] = list(self.pedido[s])
        paso = 1000.0 * dt / self.T_RECORRIDO * self.vel / 1000.0
        out = {}
        for s in SIDES:
            d = np.array(self.consigna[s], dtype=float) - self.actual[s]
            self.actual[s] += np.clip(d, -paso, paso)
            out[s] = inspire_a_rad(np.round(self.actual[s]))
        return out

    def angulos12(self):
        return [*self.consigna["left"], *self.consigna["right"]]


# ---------------------------------------------------------------- agente
class AgenteEmulado:
    """agente_sdk.py en modo arm_sdk, sin red: misma tabla, mismas reglas.

    kp/kd por omisión = J.KP / J.KD (los del agente: brazos 100 / 2).
    `kp_escala` / `kd_escala` (solo brazos) = `--kp-escala` / `--kd-escala`
    del agente."""

    def __init__(self, hz=100.0, vel_max=1.0, tau_max=10.0, kp_escala=1.0, kd_escala=1.0,
                 kp=None, kd=None):
        self.hz, self.dt = hz, 1.0 / hz
        self.vel_max, self.tau_max = vel_max, tau_max
        brazo = np.isin(np.arange(J.N), J.IZQ + J.DER)      # como --kp/--kd-escala del agente
        self.kp = np.array(J.KP, float) * np.where(brazo, kp_escala, 1.0) if kp is None else np.asarray(kp, float)
        self.kd = np.array(J.KD, float) * np.where(brazo, kd_escala, 1.0) if kd is None else np.asarray(kd, float)
        self.pedido_q = np.zeros(J.N)
        self.pedido_dq = np.zeros(J.N)
        self.pedido_tau = np.zeros(J.N)
        self.comandada = np.zeros(J.N, bool)
        self.obj = np.zeros(J.N)
        self.q_sostener = np.zeros(J.N)
        self.n_recortes = 0          # órdenes recortadas por límite articular
        self.n_tau_sat = 0           # tau_ff saturado a tau_max

    def iniciar(self, q27):
        """esperar_lowstate(): sostiene la postura medida (y el peso ya en 1)."""
        self.q_sostener = np.array(q27, float)
        self.obj = self.q_sostener.copy()

    def orden(self, q14, dq14=None, tau14=None):
        """_rx(): una orden del PC para las 14 juntas de brazo."""
        for k, i in enumerate(IDX_DDS):
            q = J.recortar(int(i), float(q14[k]))
            self.n_recortes += abs(q - q14[k]) > 1e-9
            self.pedido_q[i] = q
            self.pedido_dq[i] = 0.0 if dq14 is None else dq14[k]
            tau = 0.0 if tau14 is None else float(tau14[k])
            self.n_tau_sat += abs(tau) > self.tau_max
            self.pedido_tau[i] = max(-self.tau_max, min(self.tau_max, tau))
            self.comandada[i] = True

    def ciclo(self):
        """_avanzar_objetivo() + _publicar(): el LowCmd de rt/arm_sdk (juntas 12-26)."""
        paso = self.vel_max * self.dt
        dest = np.where(self.comandada, self.pedido_q, self.q_sostener)
        b = BRAZOS
        self.obj[b] += np.clip(dest[b] - self.obj[b], -paso, paso)
        lim = np.abs(self.obj[b] - self.pedido_q[b]) > 1e-4
        com = self.comandada[b]
        return {"q": self.obj[b].copy(),
                "dq": np.where(com & ~lim, self.pedido_dq[b], 0.0),
                "tau": np.where(com, self.pedido_tau[b], 0.0),
                "kp": self.kp[b].copy(), "kd": self.kd[b].copy(),
                "limitada": lim}


# ---------------------------------------------------------------- registro
class Registro:
    """Flujo de comandos tal cual sale al robot. Arrays por ciclo del agente
    (rt/arm_sdk, juntas 12-26) y por orden del PC (la orden UDP)."""

    def __init__(self):
        self.ag = {k: [] for k in ("t", "q", "dq", "tau", "kp", "kd", "q_medida", "dq_medida")}
        self.pc = {k: [] for k in ("t", "q", "dq", "tau", "mano")}

    def ciclo_agente(self, t, cmd, q15, dq15):
        a = self.ag
        a["t"].append(t)
        for k in ("q", "dq", "tau", "kp", "kd"):
            a[k].append(cmd[k])
        a["q_medida"].append(q15)
        a["dq_medida"].append(dq15)

    def orden_pc(self, t, q14, dq14, tau14, mano12):
        p = self.pc
        p["t"].append(t)
        p["q"].append(np.asarray(q14, float).copy())
        p["dq"].append(np.zeros(14) if dq14 is None else np.asarray(dq14, float).copy())
        p["tau"].append(np.zeros(14) if tau14 is None else np.asarray(tau14, float).copy())
        p["mano"].append(list(mano12))

    def guardar(self, path, **meta):
        out = {f"ag_{k}": np.asarray(v) for k, v in self.ag.items()}
        out.update({f"pc_{k}": np.asarray(v) for k, v in self.pc.items()})
        out["idx_dds_brazos"] = BRAZOS
        out["idx_dds_orden"] = IDX_DDS
        out["nombres"] = np.array(J.NOMBRES)
        for k, v in meta.items():
            out[f"meta_{k}"] = np.asarray(v)
        np.savez_compressed(path, **out)


def resumen_registro(reg, dt_ag):
    """Métricas del flujo: lo que el robot vería y si el agente lo deforma."""
    a = {k: np.asarray(v) for k, v in reg.ag.items()}
    p = {k: np.asarray(v) for k, v in reg.pc.items()}
    brazos = slice(1, 15)        # 12 es el torso
    v_ped = np.abs(np.diff(p["q"], axis=0)) / np.diff(p["t"])[:, None] if len(p["t"]) > 1 else np.zeros((1, 14))
    err = np.abs(a["q"][:, brazos] - a["q_medida"][:, brazos])
    return {"vel_pedida_p99": float(np.percentile(v_ped, 99)), "vel_pedida_max": float(v_ped.max()),
            "tau_ff_max": float(np.abs(p["tau"]).max()),
            "err_seguimiento_p99_rad": float(np.percentile(err, 99)),
            "err_seguimiento_med_rad": float(np.median(err))}
