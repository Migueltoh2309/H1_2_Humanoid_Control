"""Compensación del error de seguimiento del brazo (pregunta 6.3.4), solo con rt/arm_sdk.

Con kp 100 la junta se queda corta lo que hace falta para que kp·e equilibre la gravedad (y
en el robot, además, lo que tire la política): 1-2 grados en el simulador, ~5 en el robot,
13-16 mm en la muñeca al agarrar. Subir kp lo reduce pero empuja más al cuerpo (sec. 10: kp
<= 150). Aquí, dos capas, sin tocar kp:

  * MODELO del error: la consigna publicada es la deseada más g(q)/kp, con g el par de
    gravedad de las juntas del brazo calculado con el MJCF (mj_rne con velocidad y
    aceleración cero, torso vertical) y la carga del objeto cogido como masa puntual en el
    centro de agarre;
  * LAZO con las juntas medidas: en las paradas (con el brazo quieto) se suma al desvío lo
    que aún falta según los encoders. Recoge lo que el modelo no sabe (la política, la
    fricción, la masa real de la mano) y se aprende pose a pose. Con tope, por seguridad.

La consigna "deseada" es la del plan; el error de seguimiento se mide contra ella.

Además, el CUERPO: al extender el brazo el robot se inclina (0.3-2.4 grados en el simulador,
con las piernas rígidas; en el robot lo mueve además la política), y el objeto, que se midió
en el marco del torso desde la pose de observación, ya no está en el mismo sitio de ese marco
(hasta 2.4 cm). Las juntas del brazo no lo ven. `por_cuerpo` lo corrige con la IMU: el cuerpo
como sólido rígido que gira alrededor de un PIVOTE efectivo fijo en el marco del torso
(config/brazo.yaml: pivote_cuerpo; simulador: 45 cm sobre el suelo, ajustado con 9 agarres,
error <= 0.1 mm; en el robot se ajusta igual en el hito 2 con la IMU y los ArUco del tablero).
"""
import numpy as np

from . import cinematica as C

G = 9.81


class Compensador:
    def __init__(self, xml, kp, tope_lazo=0.10, tope_total=0.20):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        k = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "de_pie")
        if k >= 0:
            mujoco.mj_resetDataKeyframe(self.m, self.d, k)
        self.d.qpos[3:7] = [1, 0, 0, 0]            # torso vertical
        self.qadr = {lado: [int(self.m.jnt_qposadr[self.m.joint(n).id]) for n in C.NOMBRES_JUNTAS[lado]]
                     for lado in C.LADOS}
        self.vadr = {lado: [int(self.m.jnt_dofadr[self.m.joint(n).id]) for n in C.NOMBRES_JUNTAS[lado]]
                     for lado in C.LADOS}
        self.cintura = int(self.m.jnt_qposadr[self.m.joint("torso_joint").id])
        self.kp = float(kp)
        self.tope_lazo, self.tope_total = float(tope_lazo), float(tope_total)
        self.lazo = {lado: np.zeros(7) for lado in C.LADOS}
        self.carga = {lado: (0.0, None) for lado in C.LADOS}   # (kg, punto en el marco de la muñeca)
        self.activo = True

    def poner_carga(self, lado, masa=0.0, punto=None):
        """Objeto cogido: masa (kg) en `punto` (marco de *_wrist_yaw_link; el centro de agarre)."""
        self.carga[lado] = (float(masa), None if punto is None else np.asarray(punto, float))

    def gravedad(self, lado, q, cintura=0.0):
        """Par de gravedad (N·m) en las 7 juntas del brazo `lado` en la postura q (más la carga)."""
        m, d = self.m, self.d
        d.qpos[self.qadr[lado]] = q
        d.qpos[self.cintura] = cintura
        d.qvel[:] = 0.0
        self.mj.mj_forward(m, d)                       # qfrc_bias = gravedad (velocidad 0)
        tau = d.qfrc_bias[self.vadr[lado]].copy()
        masa, punto = self.carga[lado]
        if masa > 0 and punto is not None:
            # J^T · (m g) de la masa puntual: misma convención que qfrc_bias (par que hay que dar)
            b = m.body(("left" if lado == "izq" else "right") + "_wrist_yaw_link").id
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ punto
            jp = np.zeros((3, m.nv))
            self.mj.mj_jac(m, d, jp, None, p, b)
            tau += jp[:, self.vadr[lado]].T @ np.array([0.0, 0.0, masa * G])
        return tau

    def desvio(self, lado, q):
        """Lo que se suma a la consigna deseada q para que la junta llegue a q."""
        if not self.activo:
            return np.zeros(7)
        e = self.gravedad(lado, q) / self.kp + self.lazo[lado]
        return np.clip(e, -self.tope_total, self.tope_total)

    def corregir(self, lado, q_deseada, q_medida, ganancia=1.0):
        """Lazo: con el brazo quieto, añade lo que falta según los encoders. Devuelve el error
        que había (rad)."""
        err = np.asarray(q_deseada, float) - np.asarray(q_medida, float)
        if self.activo:
            self.lazo[lado] = np.clip(self.lazo[lado] + ganancia * err, -self.tope_lazo, self.tope_lazo)
        return err


def por_cuerpo(p, R0, R1, pivote):
    """Punto `p` (marco del torso cuando la IMU daba R0) en el marco del torso ahora (IMU R1),
    con el cuerpo girando como un sólido alrededor de `pivote` (marco del torso)."""
    P = np.asarray(pivote, float)
    return np.asarray(R1).T @ np.asarray(R0) @ (np.asarray(p, float) - P) + P


def rot_rpy(rpy):
    """R (mundo <- torso) de roll, pitch, yaw de la IMU (convención ZYX de Unitree)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])
