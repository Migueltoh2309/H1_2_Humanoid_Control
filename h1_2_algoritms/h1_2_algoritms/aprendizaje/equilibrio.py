"""[COPIA para la tesis, 2026-11-16, de tareas/manipulacion/manip/equilibrio.py (tarea de manipulación, sin commit allá); la lógica es la
misma: si allá se corrige, copiarla de nuevo. La usa aprendizaje/cuerpo_pie.py.]

Equilibrio de pie del H1-2 en modo debug (rt/lowcmd de cuerpo entero), sin dar pasos.

En debug no hay política de Unitree que sostenga el cuerpo: lo sostenemos nosotros. Al extender
los brazos el centro de masas se va hacia delante y, con un PD de posición solo, el cuerpo cede
(el par del tobillo sale del error de posición) y se va hacia delante. Lo que hace este bloque:

  1. POSTURA con las rodillas algo dobladas (la de pie de Unitree, rodilla 0.36 rad = 21 grados):
     el cuerpo queda más bajo y la rodilla tiene margen para absorber.
  2. COMPENSACIÓN DE GRAVEDAD estática de piernas y cintura con el modelo del robot y el estado
     medido (juntas + IMU): la gravedad en coordenadas generalizadas (qfrc_bias con dq = 0) se
     reparte entre los dos pies (la llave de cada pie que equilibra la base flotante, la de
     mínima norma) y el par de cada junta es tau = g_j - J_j^T w. Con ese par por delante el PD
     solo corrige errores: las juntas no ceden aunque los brazos lleven el centro de masas hacia
     delante, y el torso (y la cámara) no se mueven.
  3. IMU en los TOBILLOS (estrategia de tobillo): el cuerpo es un péndulo invertido sobre los
     tobillos y con su kp bajo (40) el PD no tiene rigidez para sostenerlo (m g h ~ 680 N m/rad):
     al par del tobillo se suma K·(inclinación del torso + su integral lenta) + D·giróscopo.
  4. Vigila el centro de presiones de cada pie (de la llave calculada): si se acerca al borde del
     pie, el supervisor no debería seguir extendiendo el brazo.

Solo calcula: lo publica manip/cuerpo.py. El modelo es el MJCF del robot (el del simulador o el
mismo con la escena: solo se usan los cuerpos del robot).
"""
import math

import numpy as np

PIERNAS = list(range(12))
CINTURA = 12
TOBILLO_PITCH = (4, 10)
TOBILLO_ROLL = (5, 11)
PIES = ("left_ankle_roll_link", "right_ankle_roll_link")
PLANTA = np.array([0.044, 0.0, -0.045])      # centro de la suela en el marco del tobillo (caja del MJCF)
MEDIO_PIE = (0.13, 0.043)                    # semilargo y semiancho de la suela [m]


def _rpy(qw, qx, qy, qz):
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    return roll, pitch


class Equilibrio:
    def __init__(self, xml, nombres, cfg):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(xml)
        if cfg.get("masa_torso"):
            self.m.body_mass[self.m.body("torso_link").id] = float(cfg["masa_torso"])
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.cfg = cfg
        self.qadr = np.array([m.joint(n).qposadr[0] for n in nombres])
        self.vadr = np.array([m.joint(n).dofadr[0] for n in nombres])
        self.pelvis = m.body("pelvis").id
        self.torso = m.body("torso_link").id
        self.pies = [m.body(p).id for p in PIES]
        self.nv = m.nv
        self.q_nom = np.array(cfg["postura_piernas"] + [cfg.get("cintura", 0.0)], dtype=float)
        self.kp = np.array(cfg["kp_piernas"] + [cfg["kp_cintura"]], dtype=float)
        self.kd = np.array(cfg["kd_piernas"] + [cfg["kd_cintura"]], dtype=float)
        self.k_imu = float(cfg["imu_kp"])
        self.t_int = float(cfg["imu_t_integral"])          # s
        self.f_alabeo = float(cfg.get("imu_factor_alabeo", 0.5))
        self.ref = np.radians(np.array(cfg.get("inclinacion_ref_deg", [0.0, 0.0]), dtype=float))
        self.kd_imu = float(cfg["imu_kd"])
        self.lim_imu = math.radians(float(cfg["imu_max_deg"]))
        self.integral = np.zeros(2)            # pitch, roll
        self.fc_derivada = float(cfg.get("fc_derivada_hz", 15.0))
        self.par_max = float(cfg.get("par_imu_max", 40.0))
        self._e_ant = None
        self.de = np.zeros(2)
        self.par_imu = np.zeros(2)
        cy, cx = math.cos(self.ref[0] / 2), math.cos(self.ref[1] / 2)
        sy, sx = math.sin(self.ref[0] / 2), math.sin(self.ref[1] / 2)
        self.quat_ref = np.array([cx * cy, sx * cy, cx * sy, -sx * sy])     # alabeo y cabeceo de referencia
        self.tau = np.zeros(13)
        self.cop = np.zeros((2, 2))            # centro de presiones de cada pie respecto del centro de la suela
        self.reparto = np.array([0.5, 0.5])
        self.error_incl = np.zeros(2)
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))
        self._sesgo = np.zeros(self.nv)
        self.soporte = 0.0
        self.contacto = False
        self.t_contacto = 0.0
        self.t_descarga = 0.0
        self.frac_rodilla = 0.0
        self.rampa_apoyo = float(cfg.get("rampa_apoyo", 1.5))
        self.umbral_contacto = float(cfg.get("umbral_contacto", 0.3))
        # descarga: el operador SUBE la grúa y los pies dejan de cargar -> la ayuda baja en rampa
        self.umbral_descarga = float(cfg.get("umbral_descarga", 0.15))
        self.t_descarga_min = float(cfg.get("t_descarga", 0.5))
        # par de las rodillas de pie en la postura nominal con los brazos de qpos0 (para carga_pies)
        q0 = np.array([float(self.m.qpos0[a]) for a in self.qadr])
        q0[:13] = self.q_nom
        self._poner(q0, self.quat_ref)
        J, w = self._llaves()
        tg = self._sesgo.copy()
        for Ji, wi in zip(J, (w[:6], w[6:])):
            tg -= Ji.T @ wi
        self.tau_rodilla_pie = (float(tg[self.vadr[3]]), float(tg[self.vadr[9]]))

    # ------------------------------------------------------------------ modelo
    def _poner(self, q, quat_torso):
        """El modelo en la postura medida; la orientación de la pelvis sale de la IMU del torso
        y la cintura (la IMU va en torso_link)."""
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = m.qpos0
        d.qpos[self.qadr] = q
        d.qpos[0:3] = 0.0
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[:] = 0.0
        mj.mj_kinematics(m, d)
        # R_pelvis = R_torso_medida · (R_torso en el marco de la pelvis)^-1
        R_tp = d.xmat[self.torso].reshape(3, 3)            # con la pelvis en la identidad
        R_t = np.zeros(9)
        mj.mju_quat2Mat(R_t, np.asarray(quat_torso, dtype=float))
        R_p = R_t.reshape(3, 3) @ R_tp.T
        qp = np.zeros(4)
        mj.mju_mat2Quat(qp, R_p.ravel())
        d.qpos[3:7] = qp
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        return R_p

    def _inclinacion(self):
        """(cabeceo, alabeo) del centro de masas visto desde el centro de las dos suelas [rad]."""
        d = self.d
        c = np.zeros(3)
        for b in self.pies:
            c += 0.5 * (d.xpos[b] + d.xmat[b].reshape(3, 3) @ PLANTA)
        v = d.subtree_com[self.pelvis] - c
        # en el marco de los pies (guiñada de la pelvis)
        Rp = d.xmat[self.pelvis].reshape(3, 3)
        yaw = math.atan2(Rp[1, 0], Rp[0, 0])
        cy, sy = math.cos(yaw), math.sin(yaw)
        x, y = cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]
        return np.array([math.atan2(x, v[2]), -math.atan2(y, v[2])])

    def _llaves(self):
        """Gravedad generalizada y la llave de cada pie (fuerza y momento en el centro de la suela)
        que la equilibra en la base flotante (mínima norma)."""
        mj, m, d = self.mj, self.m, self.d
        mj.mj_rne(m, d, 0, self._sesgo)
        J = []
        for b in self.pies:
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ PLANTA
            mj.mj_jac(m, d, self._jacp, self._jacr, p, b)
            J.append(np.vstack([self._jacp, self._jacr]).copy())
        A = np.hstack([Ji[:, :6].T for Ji in J])          # 6 x 12
        w = np.linalg.lstsq(A, self._sesgo[:6], rcond=None)[0]
        return J, w

    # ------------------------------------------------------------------ ley
    def carga_pies(self, tau_medido, dt):
        """Cuánto entra la ayuda de equilibrio (0 colgado de la grúa, 1 de pie). No hay sensor de
        fuerza en los pies: el contacto se detecta por el par medido en las rodillas (`tau_est` de
        rt/lowstate) cuando pasa del 30 % del que hacen de pie durante 0.2 s; desde ahí la ayuda
        sube en rampa (`rampa_apoyo` s) y ya no baja (el operador sigue bajando la grúa)."""
        ref = 0.5 * (self.tau_rodilla_pie[0] + self.tau_rodilla_pie[1])
        med = 0.5 * (float(tau_medido[3]) + float(tau_medido[9]))
        frac = med / ref if abs(ref) > 1e-3 else 0.0
        if not self.contacto:
            self.t_contacto = self.t_contacto + dt if frac > self.umbral_contacto else 0.0
            self.contacto = self.t_contacto >= 0.2
        else:
            # colgado otra vez (subieron la grúa): las rodillas por debajo del 15 % durante 0.5 s -> la
            # ayuda baja en rampa y, a 0, vuelve a "sin contacto" (PD puro, como colgado)
            self.t_descarga = self.t_descarga + dt if frac < self.umbral_descarga else 0.0
            if self.t_descarga >= self.t_descarga_min:
                self.soporte = max(0.0, self.soporte - dt / self.rampa_apoyo)
                if self.soporte <= 0.0:
                    self.contacto, self.t_contacto, self.t_descarga = False, 0.0, 0.0
            else:
                self.soporte = min(1.0, self.soporte + dt / self.rampa_apoyo)
        self.frac_rodilla = frac
        return self.soporte

    def orden(self, q, dq, quat_torso, gyro, dt, apoyado=True, soporte=1.0):
        """(q, dq, kp, kd, tau) de piernas + cintura (13) para el rt/lowcmd. `q`, `dq`: los 27 motores."""
        q = np.asarray(q, dtype=float)
        q_cmd = self.q_nom.copy()
        if not apoyado:
            self.integral[:] = 0.0
            self._e_ant = None
        dq_cmd = np.zeros(13)
        tau = np.zeros(13)
        if apoyado:
            # 1) estado: inclinación del centro de masas sobre los pies (péndulo invertido), con la
            #    cinemática medida y la orientación de la IMU
            self._poner(q, quat_torso)
            lean_m = self._inclinacion()
            # 2) gravedad en la postura DESEADA (piernas y cintura nominales, cuerpo derecho) con los
            #    brazos medidos: cubre la carga estática de los brazos sin anular la rigidez natural
            qm = q.copy()
            qm[:13] = self.q_nom
            self._poner(qm, self.quat_ref)
            lean_n = self._inclinacion()
            J, w = self._llaves()
            tau_g = self._sesgo.copy()
            for Ji, wi in zip(J, (w[:6], w[6:])):
                tau_g -= Ji.T @ wi
            tau = tau_g[self.vadr[:13]].copy()
            for k, (b, wi) in enumerate(zip(self.pies, (w[:6], w[6:]))):
                R = self.d.xmat[b].reshape(3, 3)
                f, mo = R.T @ wi[:3], R.T @ wi[3:]
                fz = max(f[2], 1e-6)
                self.cop[k] = (-mo[1] / fz, mo[0] / fz)
            fz = np.array([w[2], w[8]])
            self.reparto = fz / max(fz.sum(), 1e-6)
            # 3) estrategia de tobillo: el cuerpo se ha inclinado (lean_m) respecto de lo que daría la
            #    postura nominal (lean_n) -> par en los tobillos K (e + integral lenta) + D de/dt
            e = lean_m - lean_n
            if self._e_ant is None:
                self._e_ant = e.copy()
            de = (e - self._e_ant) / max(dt, 1e-4)
            self._e_ant = e.copy()
            a_f = min(1.0, dt * 2 * math.pi * self.fc_derivada)
            self.de = self.de + a_f * (de - self.de)
            self.error_incl = e
            if soporte > 0.9:
                self.integral = np.clip(self.integral + e * dt / self.t_int, -self.lim_imu, self.lim_imu)
            par = self.k_imu * (e + self.integral) + self.kd_imu * self.de
            par[1] *= self.f_alabeo
            par = np.clip(par, -self.par_max, self.par_max) * min(max(soporte, 0.0), 1.0)
            tau = tau * min(max(soporte, 0.0), 1.0)
            self.par_imu = par
            for j in TOBILLO_PITCH:
                tau[j] += par[0]
            for j in TOBILLO_ROLL:
                tau[j] += par[1]
            self.tau = tau
        return q_cmd, dq_cmd, self.kp.copy(), self.kd.copy(), tau

    def cop_max(self):
        """Fracción del semipié que usa el centro de presiones (1 = en el borde)."""
        return float(max(np.max(np.abs(self.cop[:, 0])) / MEDIO_PIE[0], np.max(np.abs(self.cop[:, 1])) / MEDIO_PIE[1]))
