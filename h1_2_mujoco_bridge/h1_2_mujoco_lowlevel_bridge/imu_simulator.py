"""
imu_simulator.py — IMU simulada para unitree_hg/msg/IMUState.

Fuente de datos
---------------
Este MJCF YA define una IMU nativa: los sensores 'imu-angular-velocity'
(mjSENS_GYRO) e 'imu-linear-acceleration' (mjSENS_ACCELEROMETER) montados en
el site "imu" del body torso_link (misma ubicación que la IMU del H1-2 real).
Se leen directamente de data.sensordata, que es lo más fiel físicamente:

  * gyro MuJoCo  -> velocidad angular EN EL MARCO LOCAL del site [rad/s].
  * accelerometer MuJoCo -> aceleración PROPIA en el marco local del site
    [m/s²], es decir incluye la reacción a la gravedad: en reposo con el site
    alineado al mundo mide (0, 0, +9.81). Esto coincide con lo que reporta un
    acelerómetro MEMS real (y la IMU de Unitree).

Si el modelo no tuviera estos sensores, hay un fallback documentado que
calcula lo mismo desde el body pelvis (orientación de xquat, velocidad
angular local vía mj_objectVelocity y aceleración por diferencias finitas de
la velocidad lineal en mundo menos gravedad, rotada al marco local).

Convenciones verificadas (MuJoCo vs unitree_hg/IMUState)
--------------------------------------------------------
  * Cuaternión: MuJoCo usa (w, x, y, z); IMUState.quaternion también es
    (w, x, y, z)  -> se copia sin reordenar. Representa la rotación del marco
    de la IMU respecto al mundo (Z hacia arriba).
  * rpy: (roll, pitch, yaw) en radianes, convención ZYX intrínseca
    (yaw-pitch-roll), igual que el SDK de Unitree.
  * Ejes: el site "imu" del MJCF está alineado con el body torso_link
    (X adelante, Y izquierda, Z arriba en la pose home), consistente con la
    convención del robot.
  * Unidades: rad, rad/s, m/s².

Nota sobre esta escena: la pelvis está SOLDADA al mundo (no hay freejoint),
así que en la práctica la orientación es constante, el gyro ~0 y el
acelerómetro ~(0,0,9.81) más las vibraciones que el movimiento de brazos
induce en el torso. El código es genérico y funcionaría igual con base libre.
"""
from __future__ import annotations

import numpy as np
import mujoco


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    """Cuaternión (w,x,y,z) -> (roll, pitch, yaw) ZYX en radianes."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(s, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class ImuSimulator:

    def __init__(self, model: mujoco.MjModel, fallback_body: str = "pelvis"):
        self.model = model
        self._gyro_adr = -1
        self._acc_adr = -1
        self._site_id = -1

        for s in range(model.nsensor):
            stype = model.sensor_type[s]
            if stype == mujoco.mjtSensor.mjSENS_GYRO and self._gyro_adr < 0:
                self._gyro_adr = int(model.sensor_adr[s])
                self._site_id = int(model.sensor_objid[s])
            elif stype == mujoco.mjtSensor.mjSENS_ACCELEROMETER and self._acc_adr < 0:
                self._acc_adr = int(model.sensor_adr[s])
                if self._site_id < 0:
                    self._site_id = int(model.sensor_objid[s])

        self.use_sensors = self._gyro_adr >= 0 and self._acc_adr >= 0

        # Fallback: body pelvis
        self._body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, fallback_body
        )
        self._prev_v_world = np.zeros(3)
        self._gravity = np.array(model.opt.gravity)  # normalmente (0,0,-9.81)

        # Buffers
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._vel6 = np.zeros(6)

    def describe(self) -> str:
        if self.use_sensors:
            site = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, self._site_id)
            body = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.site_bodyid[self._site_id]),
            )
            return f"sensores nativos del MJCF (site '{site}' en body '{body}')"
        return "fallback por cinemática del body pelvis (el MJCF no trae sensores IMU)"

    # ------------------------------------------------------------------
    def read(self, data: mujoco.MjData, dt: float):
        """
        Devuelve (quaternion_wxyz[4], rpy[3], gyro_local[3], accel_local[3]).
        Llamar DESPUÉS de mj_step (sensordata actualizado).
        """
        if self.use_sensors:
            # Orientación del site de la IMU (matriz -> cuaternión w,x,y,z)
            mujoco.mju_mat2Quat(self._quat, data.site_xmat[self._site_id])
            gyro = data.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
            accel = data.sensordata[self._acc_adr:self._acc_adr + 3].copy()
        else:
            # --- Fallback documentado -------------------------------------
            self._quat = data.xquat[self._body_id].copy()      # (w,x,y,z)
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                self._body_id, self._vel6, 1,                   # 1 = marco local
            )
            gyro = self._vel6[0:3].copy()                       # ang. local

            # Aceleración propia: a_medida = R^T * (a_mundo - g)
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                self._body_id, self._vel6, 0,                   # 0 = marco mundo
            )
            v_world = self._vel6[3:6].copy()
            a_world = (v_world - self._prev_v_world) / max(dt, 1e-9)
            self._prev_v_world = v_world
            R = data.xmat[self._body_id].reshape(3, 3)
            accel = R.T @ (a_world - self._gravity)

        rpy = quat_to_rpy(self._quat)
        return self._quat.copy(), rpy, gyro, accel
