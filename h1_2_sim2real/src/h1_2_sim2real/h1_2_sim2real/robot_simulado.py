"""El H1-2 simulado en MuJoCo, visto desde fuera IGUAL que el robot real.

  * DDS del SDK de Unitree (dominio 1 sobre lo, como unitree_mujoco; lo publica
    y recibe el proceso_dds, ver compartido.py):
      rt/lowstate   <- lo publica el simulador (motor_state, imu_state)
      rt/lowcmd     -> cuerpo entero, como el robot en Debug
      rt/arm_sdk    -> torso + brazos mezclados con el control interno segun el
                       peso del slot 27, como el robot en FSM 201. unitree_mujoco
                       NO lo implementa (por eso caja_cuadrado.Brazo tenia un modo
                       sim aparte); aqui si, para que el codigo del robot corra igual.
  * Manos Inspire por Modbus TCP: 127.0.1.211:6000 (izq) y 127.0.1.210:6000 (der),
    con los registros de la mano real (ANGLE_SET, ANGLE_ACT, FORCE_ACT, SPEED_SET...).

El "control interno" emula al de Unitree con el robot de pie: sostiene la postura
de arranque con las ganancias de h1_2_juntas y publica rt/lowcmd (marcado en
reserve[0], el propio simulador lo ignora) para que la comprobacion de
"control interno activo" de los scripts de arm_sdk funcione tambien aqui.

Modo del robot (parametro modo_robot):
  auto   (por defecto) si llega rt/lowcmd externo en los ultimos 100 ms manda el
         (Debug); si no, control interno + arm_sdk (FSM 201).
  ai     solo control interno + arm_sdk; rt/lowcmd externo se ignora, como el
         robot real con CheckMode en 'ai' (asi fallo el selector el 2026-09-25).
  debug  solo rt/lowcmd externo; sin el, motores sin par (como el robot en Debug).

La base esta soldada (pelvis fija, escena de la faja), equivalente al robot colgado
o de pie sin dar pasos: la IMU es constante.
"""
import math
import threading
import time

import mujoco
import numpy as np

from .comun import J, modbus_mini

MARCA_INTERNO = 0x51A     # reserve[0] de los rt/lowcmd del control interno simulado

# Ganancias de PIERNAS del control interno con BASE FLOTANTE (escena del suelo).
# Con los pies apoyados el cuerpo es un pendulo invertido sobre los tobillos: para
# sostenerse como estatua la rigidez tiene que superar m*g*h ~ 67 kg * 9.81 * 0.9 m
# ~ 590 N·m/rad. Con las de h1_2_juntas (tobillo 80) se cae de bruces en 1.6 s
# (medido). Con estas aguanta quieto (inclinacion max ~1.3 grados), pero un empujon
# de 100 N x 0.2 s lo tira: no da pasos. En el robot real lo que lo sostiene en
# FSM 201 es la politica de Unitree, no un PD.
KP_PIERNA_PIE = [300, 400, 400, 500, 800, 400] * 2
KD_PIERNA_PIE = [6, 8, 8, 10, 12, 8] * 2

# Cinta elastica (como unitree_mujoco): muelle-amortiguador de torso_link a un
# punto fijo por encima. Solo tira (es una cuerda). Teclas del viewer:
# 7 sube (acorta la cuerda), 8 baja, 9 engancha / suelta.
BANDA_K, BANDA_C, BANDA_PASO = 5000.0, 800.0, 0.05


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


# camara MuJoCo (mira a -Z, Y arriba) -> frame optico ROS (Z adelante, Y abajo)
_FLIP_OPTICO = np.array([0.0, 1.0, 0.0, 0.0])


class MarchaCinematica:
    """Sustituto de la marcha para lo que NO es caminar: la base se desplaza como un
    solido rigido con las velocidades de LocoClient.Move (servicio "loco" simulado),
    las piernas quietas en la postura de pie, un poco por encima del suelo (no da
    pasos ni puede caerse: hace de arnes). Sirve para cerrar el lazo de
    percepcion -> control -> Move y probar supervisor, perdida de linea, fin...

    Modela lo que el reto describe de la planta real (R40-RT-H1_2-0001, sec. 1 y 6.3),
    con valores SUPUESTOS hasta medirlos en el robot:
      * retardo puro (RPC + control de marcha) y respuesta de primer orden;
      * deriva de rumbo de ~2 grados/s andando, de signo aleatorio en cada arranque;
      * balanceo de la camara a la cadencia de 1.43 Hz (roll a f, pitch y altura a 2f).
    Solo FSM 201/204 anda (como el robot: Move fuera de 201 no hace nada)."""

    def __init__(self, sim, retardo=0.30, tau=0.35, deriva_deg=2.0, cadencia=1.43, roll_deg=1.5,
                 pitch_deg=0.8, z_mm=8.0, elevacion=0.004, semilla=None):
        self.sim = sim
        d = sim.data
        self.x, self.y = float(d.qpos[0]), float(d.qpos[1])
        qw, qx, qy, qz = d.qpos[3:7]
        self.yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        self.z0 = float(d.qpos[2]) + elevacion
        self.retardo, self.tau, self.cadencia = retardo, tau, cadencia
        self.deriva = math.radians(deriva_deg)
        self.a_roll, self.a_pitch, self.a_z = math.radians(roll_deg), math.radians(pitch_deg), z_mm / 1000.0
        self.rng = np.random.default_rng(semilla)
        self.cola = []                       # (t, v_cmd) para el retardo
        self.v = np.zeros(3)                 # vx, vy, vyaw reales (marco del robot)
        self.v_mundo_ant = np.zeros(3)
        self.acc_mundo = np.zeros(3)
        self.fase = 0.0
        self.amp = 0.0
        self.signo = 1.0
        self.andaba = False

    def paso(self, ahora, c, dt):
        activo = int(c["fsm"][0]) in (201, 204) and ahora < c["loco_t_fin"][0]
        v_cmd = c["loco_v"].copy() if activo else np.zeros(3)
        self.cola.append((ahora, v_cmd))
        while len(self.cola) > 1 and self.cola[1][0] <= ahora - self.retardo:
            self.cola.pop(0)
        v_ret = self.cola[0][1] if self.cola[0][0] <= ahora - self.retardo else np.zeros(3)
        self.v += (v_ret - self.v) * min(1.0, dt / self.tau)
        andando = math.hypot(self.v[0], self.v[1]) > 0.03 or abs(self.v[2]) > 0.05
        if andando and not self.andaba:
            self.signo = float(self.rng.choice([-1.0, 1.0]))     # deriva con signo nuevo en cada tirada
        self.andaba = andando
        self.amp += ((1.0 if andando else 0.0) - self.amp) * min(1.0, dt / 0.4)
        if self.amp > 1e-3:
            self.fase = (self.fase + 2 * math.pi * self.cadencia * dt) % (2 * math.pi)
        deriva = self.signo * self.deriva * min(1.0, abs(self.v[0]) / 0.2)
        wz = self.v[2] + deriva
        c_, s_ = math.cos(self.yaw), math.sin(self.yaw)
        vxm, vym = self.v[0] * c_ - self.v[1] * s_, self.v[0] * s_ + self.v[1] * c_
        self.x += vxm * dt
        self.y += vym * dt
        self.yaw += wz * dt
        roll = self.amp * self.a_roll * math.sin(self.fase)
        pitch = self.amp * self.a_pitch * math.sin(2 * self.fase)
        dz = self.amp * self.a_z * math.sin(2 * self.fase)
        w2 = 2 * math.pi * self.cadencia * 2
        vz = self.amp * self.a_z * w2 * math.cos(2 * self.fase)
        v_mundo = np.array([vxm, vym, vz])
        self.acc_mundo = (v_mundo - self.v_mundo_ant) / dt
        self.v_mundo_ant = v_mundo
        d = self.sim.data
        d.qpos[0:3] = (self.x, self.y, self.z0 + dz)
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.array([roll, pitch, self.yaw]), "XYZ")
        d.qpos[3:7] = q
        d.qvel[0:3] = v_mundo
        d.qvel[3:6] = (self.amp * self.a_roll * 2 * math.pi * self.cadencia * math.cos(self.fase),
                       self.amp * self.a_pitch * w2 * math.cos(2 * self.fase), wz)


class MarchaPolitica:
    """La marcha de VERDAD en el simulador: la politica de unitree_rl_gym para el H1-2
    (politica_marcha.py) hace de control interno de Unitree en FSM 201. Recibe el comando
    de LocoClient.Move (servicio "loco" simulado, con su duration) y decide a 50 Hz los
    objetivos de las 12 juntas de las piernas, con su PD (kp 200/200/200/300/40/40).
    Torso y brazos los sostiene el control interno con las ganancias de deploy_real de
    Unitree (300 cintura, 120/80 brazos), y rt/arm_sdk se mezcla encima como en el robot.
    Fisica completa: puede tropezar y caerse; la deriva y el balanceo salen solos.
    La observacion usa el estado de la pelvis (como deploy_mujoco); el control real de
    Unitree usa sus propios sensores."""

    def __init__(self, sim, decimacion=10):
        from .politica_marcha import PoliticaMarcha
        self.sim = sim
        self.pol = PoliticaMarcha()
        self.objetivo = self.pol.q0.copy()
        self.decimacion = decimacion
        self.n = 0
        self.cmd = np.zeros(3)
        self.kp_brazos = np.array(self.pol.meta["arm_waist_kps"], float)
        self.kd_brazos = np.array(self.pol.meta["arm_waist_kds"], float)

    def actualizar(self, ahora, c):
        activo = int(c["fsm"][0]) in (201, 204) and ahora < c["loco_t_fin"][0]
        self.cmd = np.clip(c["loco_v"], -self.pol.max_cmd, self.pol.max_cmd) if activo else np.zeros(3)
        if self.n % self.decimacion == 0:
            d, s = self.sim.data, self.sim
            self.objetivo = self.pol.paso(d.qpos[3:7], d.qvel[3:6], d.qpos[s.qadr[:12]], d.qvel[s.vadr[:12]],
                                          self.cmd, self.decimacion * s.dt)
        self.n += 1


def info_camara(m, nombre):
    """Resolucion, fovy y pose del frame optico ROS de una camara del MJCF."""
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, nombre)
    if cid < 0:
        return None
    w, h = (int(v) for v in m.cam_resolution[cid])
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.cam_bodyid[cid]))
    qw, qx, qy, qz = _quat_mul(m.cam_quat[cid], _FLIP_OPTICO)
    return {"id": cid, "w": w, "h": h, "fovy": float(m.cam_fovy[cid]), "body": body,
            "pos": m.cam_pos[cid].copy(), "quat_xyzw": [qx, qy, qz, qw]}


class ManoSim:
    """Mapa de registros Modbus de una Inspire, respaldado por los actuadores del MJCF."""

    # tiempo de recorrido completo (0 -> 1000) a SPEED_SET = 1000. La RH56 tarda
    # del orden de 0.5-1 s en cerrar del todo a velocidad maxima.
    T_RECORRIDO = 0.6
    BRAZO_PALANCA = 0.055    # m: par del dedo (N·m) -> fuerza en la yema (1 N·m ~ 18 N)

    def __init__(self, sim, lado):
        self.sim, self.lado = sim, lado
        pre = J.PREFIJO[lado]
        m = sim.model
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, pre + j) for j, _ in J.DOF_MANO]
        self.qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, pre + j)])
                     for j, _ in J.DOF_MANO]
        if min(self.act) < 0:
            raise RuntimeError(f"el modelo no tiene la mano {lado} ({pre}*_proximal_joint)")
        self.regs = {}
        self.objetivo = [1000.0] * 6         # abierta
        self.actual = [1000.0] * 6           # consigna que se va moviendo a SPEED_SET
        self._escribir_bloque(J.SPEED_SET, [1000] * 6)
        self._escribir_bloque(J.FORCE_SET, [2000] * 6)   # de fabrica: sin limite
        self._escribir_bloque(J.ANGLE_SET, [1000] * 6)
        self._escribir_bloque(J.TEMP, [32, 32, 32])
        self.cerrojo = threading.Lock()

    def _escribir_bloque(self, d, vals):
        for k, v in enumerate(vals):
            self.regs[d + k] = int(v) & 0xFFFF

    # --- interfaz de modbus_mini
    def leer(self, d, n):
        with self.cerrojo:
            out = []
            for a in range(d, d + n):
                if J.ANGLE_ACT <= a < J.ANGLE_ACT + 6:
                    k = a - J.ANGLE_ACT
                    out.append(J.rad_a_angulo(k, float(self.sim.data.qpos[self.qadr[k]])))
                elif J.FORCE_ACT <= a < J.FORCE_ACT + 6:
                    k = a - J.FORCE_ACT
                    f = abs(float(self.sim.data.actuator_force[self.act[k]])) / self.BRAZO_PALANCA
                    out.append(int(min(f * 101.97, 3000)))          # gramos-fuerza
                else:
                    v = self.regs.get(a, 0)
                    out.append(v - 0x10000 if v >= 0x8000 else v)
            return out

    def escribir(self, d, vals):
        with self.cerrojo:
            for k, v in enumerate(vals):
                a = d + k
                if J.ANGLE_SET <= a < J.ANGLE_SET + 6:
                    if v == 0xFFFF:              # -1: no tocar ese dedo (Mano.mover)
                        continue
                    self.objetivo[a - J.ANGLE_SET] = float(min(max(v, 0), 1000))
                self.regs[a] = int(v) & 0xFFFF

    # --- en el hilo de fisica
    def paso(self, dt):
        with self.cerrojo:
            for k in range(6):
                vel = max(1, self.regs.get(J.SPEED_SET + k, 1000)) / 1000.0 * 1000.0 / self.T_RECORRIDO
                d = self.objetivo[k] - self.actual[k]
                self.actual[k] += max(-vel * dt, min(vel * dt, d))
                self.sim.data.ctrl[self.act[k]] = J.angulo_a_rad(k, self.actual[k])


class RobotSimulado:
    def __init__(self, modelo, modo_robot="auto", keyframe="", armature=0.01, damping=0.05,
                 grasp_assist=False, banda=False, marcha=None, politica=False, log=print):
        self.log = log
        self.model = mujoco.MjModel.from_xml_path(modelo)
        m = self.model
        self.dt = float(m.opt.timestep)
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in J.NOMBRES]
        acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in J.NOMBRES]
        if min(ids) < 0 or min(acts) < 0:
            faltan = [n for n, i, a in zip(J.NOMBRES, ids, acts) if i < 0 or a < 0]
            raise RuntimeError(f"faltan juntas/actuadores en el MJCF: {faltan}")
        self.qadr = np.array([m.jnt_qposadr[i] for i in ids])
        self.vadr = np.array([m.jnt_dofadr[i] for i in ids])
        self.act = np.array(acts)
        for d in self.vadr:                       # mismo arreglo de estabilidad que el sim_bridge
            m.dof_armature[d] = max(m.dof_armature[d], armature)
            m.dof_damping[d] = max(m.dof_damping[d], damping)
        self.tau_max = np.array(J.TAU_MAX, dtype=float)

        self.data = mujoco.MjData(m)
        # base flotante = la pelvis cuelga del mundo por un freejoint (escena del suelo)
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.flotante = any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == self.pelvis
                            for j in range(m.njnt))
        if not keyframe and self.flotante and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "de_pie") >= 0:
            keyframe = "de_pie"
        self.keyframe = keyframe
        if keyframe:
            k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if k < 0:
                raise RuntimeError(f"keyframe '{keyframe}' no existe")
            mujoco.mj_resetDataKeyframe(m, self.data, k)
        mujoco.mj_forward(m, self.data)
        self.cerrojo = threading.Lock()           # mj_step vs lecturas de otros hilos

        self.modo_robot = modo_robot
        self.kp = np.array(J.KP, dtype=float)
        self.kd = np.array(J.KD, dtype=float)
        if self.flotante:
            self.kp[J.PIERNAS] = KP_PIERNA_PIE
            self.kd[J.PIERNAS] = KD_PIERNA_PIE
        self.q_interno = self.data.qpos[self.qadr].copy()   # postura que sostiene el control interno

        self.peso_arm = 0.0
        self.fuente = "interno"

        self.manos = {}
        for lado in J.LADOS:
            try:
                self.manos[lado] = ManoSim(self, lado)
            except RuntimeError as e:
                log(f"sin mano {lado}: {e}")

        # base soldada: posiciones relativas a la pelvis (raiz del TF); flotante: mundo
        self.origen = np.zeros(3) if self.flotante else self.data.xpos[self.pelvis].copy()
        self.frame_mundo = "world" if self.flotante else "pelvis"

        # IMU: sensores del MJCF si estan (escena del suelo: sitio imu de torso_link)
        def sensor(nombre):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, nombre)
            return None if i < 0 else slice(m.sensor_adr[i], m.sensor_adr[i] + m.sensor_dim[i])
        self.s_quat = sensor("imu_quat")
        self.s_gyro = sensor("imu-angular-velocity")
        self.s_acc = sensor("imu-linear-acceleration")

        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.banda = bool(banda)
        self.banda_ancla = self.data.xipos[self.torso] + np.array([0.0, 0.0, 1.0])
        self.banda_largo = 1.0 - (0.30 if banda else 0.0)      # colgado: los pies ~10 cm sobre el suelo
        self._vel6 = np.zeros(6)
        self._banda_f = None

        # marcha cinematica (dict de parametros de MarchaCinematica, o None)
        self.marcha = MarchaCinematica(self, **marcha) if (marcha is not None and self.flotante) else None
        self.politica = MarchaPolitica(self) if (politica and self.flotante) else None
        if politica and not self.flotante:
            log("marcha:=politica necesita base flotante (escena suelo o una pista): se ignora")
        if self.marcha is not None or self.politica is not None:
            self.banda = False
        self._assist = {}
        if grasp_assist:
            self._activar_assist()

    # ------------------------------------------------------------------ control
    def _comando(self, ahora, c):
        """(q, dq, kp, kd, tau, activo) de los 27 motores segun la fuente activa.
        `c` = copia de los comandos DDS de la memoria compartida (lc_*, arm_*)."""
        n = J.N
        ext = ahora - c["lc_t"][0] < 0.1
        if self.modo_robot == "debug" or (self.modo_robot == "auto" and ext):
            self.fuente = "lowcmd" if ext else "debug_sin_ordenes"
            self.peso_arm = 0.0
            if not ext:
                return None
            return c["lc_q"], c["lc_dq"], c["lc_kp"], c["lc_kd"], c["lc_tau"], c["lc_mode"] == 1

        q, kp, kd = self._base_interna()
        dq, tau = np.zeros(n), np.zeros(n)
        edad = ahora - c["arm_t"][0]
        w = min(max(float(c["arm_w"][0]), 0.0), 1.0)
        if edad > 0.5:                          # el publicador de arm_sdk se callo: devolver en 1 s
            w *= max(0.0, 1.0 - (edad - 0.5))
        self.peso_arm = w
        if w > 0.0:
            b = J.BRAZOS
            q[b] = (1 - w) * q[b] + w * c["arm_q"][b]
            dq[b] = w * c["arm_dq"][b]
            kp[b] = (1 - w) * kp[b] + w * c["arm_kp"][b]
            kd[b] = (1 - w) * kd[b] + w * c["arm_kd"][b]
            tau[b] = w * c["arm_tau"][b]
        self.fuente = "interno+arm_sdk" if w > 0 else "interno"
        return q, dq, kp, kd, tau, np.ones(n, dtype=bool)

    def _base_interna(self):
        """(q, kp, kd) del control interno: postura de arranque y, con la politica, sus
        objetivos y ganancias en las piernas y las de deploy_real en torso y brazos."""
        q, kp, kd = self.q_interno.copy(), self.kp.copy(), self.kd.copy()
        if self.politica is not None:
            P = J.PIERNAS
            q[P], kp[P], kd[P] = self.politica.objetivo, self.politica.pol.kp, self.politica.pol.kd
            kp[J.BRAZOS], kd[J.BRAZOS] = self.politica.kp_brazos, self.politica.kd_brazos
        return q, kp, kd

    def paso(self, ahora, c):
        d = self.data
        if self.politica is not None:
            self.politica.actualizar(ahora, c)
        cmd = self._comando(ahora, c)
        if cmd is None:
            par = np.zeros(J.N)
        else:
            q_d, dq_d, kp, kd, tau_ff, activo = cmd
            q, dq = d.qpos[self.qadr], d.qvel[self.vadr]
            par = tau_ff + kp * (q_d - q) + kd * (dq_d - dq)
            par = np.where(activo, np.clip(par, -self.tau_max, self.tau_max), 0.0)
        d.ctrl[self.act] = par
        if self.marcha is not None:
            self.marcha.paso(ahora, c, self.dt)
        elif self.flotante:
            self._aplicar_banda()
        for mano in self.manos.values():
            mano.paso(self.dt)
        with self.cerrojo:
            mujoco.mj_step(self.model, d)
            if self._assist:
                self._actualizar_assist()

    def _aplicar_banda(self):
        # solo se quita la fuerza que puso la propia cinta: los empujones con el raton
        # del viewer tambien van por xfrc_applied y no hay que pisarlos
        d = self.data
        if self._banda_f is not None:
            d.xfrc_applied[self.torso, :] -= self._banda_f
            self._banda_f = None
        if not self.banda:
            return
        delta = self.banda_ancla - d.xipos[self.torso]
        dist = float(np.linalg.norm(delta))
        if dist <= self.banda_largo:
            return
        u = delta / dist
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.torso, self._vel6, 0)
        v = self._vel6[3:]
        f = BANDA_K * (dist - self.banda_largo) * u - BANDA_C * v
        self._banda_f = np.concatenate([f, -20.0 * self._vel6[0:3]])   # par: que no gire como una peonza
        d.xfrc_applied[self.torso, :] += self._banda_f

    def tecla(self, codigo):
        """Teclas del viewer, como unitree_mujoco: 7 sube, 8 baja, 9 engancha/suelta la cinta."""
        if not self.flotante:
            return
        if codigo == ord("9"):
            self.banda = not self.banda
            if self.banda:      # re-anclar encima de donde esta ahora el torso
                self.banda_ancla = self.data.xipos[self.torso] + np.array([0.0, 0.0, 1.0])
                self.banda_largo = 1.0
            self.log(f"cinta elastica {'ENGANCHADA' if self.banda else 'SUELTA'}")
        elif codigo == ord("7") and self.banda:
            self.banda_largo = max(0.2, self.banda_largo - BANDA_PASO)
        elif codigo == ord("8") and self.banda:
            self.banda_largo += BANDA_PASO

    # ------------------------------------------------------------------ estado
    def estado_motores(self):
        d = self.data
        return (d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy(), d.actuator_force[self.act].copy())

    def imu(self):
        """(quat wxyz, rpy, gyro, acc). Con los sensores del MJCF (escena del suelo:
        sitio imu de torso_link, como el h1_2.xml de Unitree) si estan; si no, la
        pelvis (base soldada: constante)."""
        d = self.data
        if self.s_quat is not None:
            qw, qx, qy, qz = d.sensordata[self.s_quat]
        else:
            qw, qx, qy, qz = d.xquat[self.pelvis]
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        if self.marcha is not None:
            # base prescrita: el acelerometro del MJCF veria la dinamica de un solido
            # que se teletransporta; se calcula a partir del movimiento impuesto
            R = d.xmat[self.torso].reshape(3, 3)
            acc = R.T @ (self.marcha.acc_mundo + np.array([0.0, 0.0, 9.81]))
            gyro = d.sensordata[self.s_gyro] if self.s_gyro is not None else np.zeros(3)
        elif self.s_gyro is not None and self.s_acc is not None:
            gyro, acc = d.sensordata[self.s_gyro], d.sensordata[self.s_acc]
        else:
            R = d.xmat[self.pelvis].reshape(3, 3)
            acc = R.T @ np.array([0.0, 0.0, 9.81])
            gyro = R.T @ d.cvel[self.pelvis][:3]
        return [qw, qx, qy, qz], [roll, pitch, yaw], [float(x) for x in gyro], [float(x) for x in acc]

    def pose_base(self):
        """(pos, quat wxyz) de la pelvis en el mundo: verdad de terreno (no hay odometria en el robot)."""
        with self.cerrojo:
            return self.data.xpos[self.pelvis].copy(), self.data.xquat[self.pelvis].copy()

    def contactos(self, f_min=1e-3):
        m, d = self.model, self.data
        out, f6 = [], np.zeros(6)
        with self.cerrojo:
            for i in range(d.ncon):
                c = d.contact[i]
                mujoco.mj_contactForce(m, d, i, f6)
                if abs(f6[0]) < f_min:
                    continue
                b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1])) or "?"
                b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2])) or "?"
                out.append((np.array(c.pos) - self.origen, float(abs(f6[0])), b1, b2))
        return out

    def pose_cuerpo(self, nombre):
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nombre)
        if b < 0:
            return None
        with self.cerrojo:
            return self.data.xpos[b] - self.origen, self.data.xquat[b].copy()

    # --------------------------------------------- asistencia de agarre (solo sim)
    def _activar_assist(self, objeto="mandarina", junta="index_proximal_joint", on=0.6, off=0.3):
        """Weld palma-fruta al cerrar la mano con >= 2 grupos de la mano tocando la
        fruta (mismo criterio que h1_2_mujoco_sim_bridge.MujocoSim.enable_grasp_assist)."""
        m = self.model
        obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, objeto)
        for side, pre in (("left", "L_"), ("right", "R_")):
            eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{side}_grasp_assist")
            act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, pre + junta)
            wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
            if min(obj, eq, act, wrist) < 0:
                continue
            grupo = {}
            for b in range(m.nbody):
                nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if b == wrist:
                    grupo[b] = "palm"
                elif nb.startswith(pre):
                    for g in ("thumb", "index", "middle", "ring", "pinky"):
                        if g in nb:
                            grupo[b] = g
            self._assist[side] = dict(eq=eq, act=act, wrist=wrist, obj=obj, grupo=grupo, on=on, off=off)
        if self._assist:
            self.log(f"asistencia de agarre (solo simulacion) en {sorted(self._assist)}")

    def _actualizar_assist(self):
        m, d = self.model, self.data
        for a in self._assist.values():
            cierre = d.ctrl[a["act"]]
            if d.eq_active[a["eq"]]:
                if cierre < a["off"]:
                    d.eq_active[a["eq"]] = 0
                continue
            if cierre < a["on"]:
                continue
            tocan = set()
            for i in range(d.ncon):
                c = d.contact[i]
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if a["obj"] == b1 and b2 in a["grupo"]:
                    tocan.add(a["grupo"][b2])
                elif a["obj"] == b2 and b1 in a["grupo"]:
                    tocan.add(a["grupo"][b1])
            if len(tocan) < 2:
                continue
            w, o = a["wrist"], a["obj"]
            R1 = d.xmat[w].reshape(3, 3)
            qi, rq = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(qi, d.xquat[w])
            mujoco.mju_mulQuat(rq, qi, d.xquat[o])
            m.eq_data[a["eq"]][0:3] = 0.0
            m.eq_data[a["eq"]][3:6] = R1.T @ (d.xpos[o] - d.xpos[w])
            m.eq_data[a["eq"]][6:10] = rq
            m.eq_data[a["eq"]][10] = 1.0
            d.eq_active[a["eq"]] = 1

    # ------------------------------------------------------------------ manos
    def servir_manos(self, ips=None):
        ips = ips or J.MANO_IP_SIM
        srv = []
        for lado, mano in self.manos.items():
            srv.append(modbus_mini.servir(ips[lado], J.MANO_PUERTO, mano))
        return srv
