"""
lowstate_builder.py — Construcción de unitree_hg/msg/LowState desde MuJoCo.

Definición oficial del mensaje (unitree_ros2, unitree_hg):

  LowState:
    uint32[2] version         -> [0, 0] (sin significado en simulación)
    uint8 mode_pr             -> eco del último LowCmd recibido (PR=0 default)
    uint8 mode_machine        -> parámetro del bridge (los controladores lo
                                 leen y lo repiten; su valor es arbitrario
                                 dentro de la simulación)
    uint32 tick               -> milisegundos de simulación (mod 2^32)
    IMUState imu_state        -> ver imu_simulator.py
    MotorState[35] motor_state
    uint8[40] wireless_remote -> 40 ceros (no hay mando en simulación)
    uint32[4] reserve         -> ceros
    uint32 crc                -> 0. Documentado: los scripts de referencia NO
                                 verifican el CRC de LowState (solo el robot
                                 verifica el de LowCmd), así que no se calcula
                                 para no gastar CPU a 500 Hz. Si algún
                                 controlador lo exigiera, puede añadirse con
                                 la misma técnica de crc.py.

  MotorState (por motor):
    mode      -> modo efectivo aplicado por el bridge (0/1)
    q, dq     -> posición/velocidad simuladas [rad, rad/s]
    ddq       -> aceleración por diferencia finita de dq entre ticks de
                 control [rad/s²]
    tau_est   -> torque aplicado por el actuador de MuJoCo
                 (data.actuator_force == torque de control tras saturaciones;
                 documentado en mujoco_simulator.py)
    temperature[2] -> [0, 0]   vol -> 0.0   sensor[2] -> [0,0]
    motorstate -> 0            reserve[4] -> ceros
    (valores seguros: no existen en la simulación)

Los índices 0..26 corresponden a las articulaciones controladas; los índices
27..34 se rellenan con ceros y mode=0 (motores que el H1-2 no tiene pero que
el array de 35 exige). El mensaje SIEMPRE lleva los 35 elementos, por lo que
test_leer.py puede leer cualquier índice sin modificar su lógica.

Duck-typed: `msg` puede ser un LowState real de rclpy o un doble de pruebas
con la misma estructura (los tests corren sin ROS).
"""
from __future__ import annotations

import numpy as np

from .joint_mapping import NUM_CMD_MOTORS

NUM_STATE_MOTORS = 35


def fill_lowstate(
    msg,
    snap: dict,
    imu_quat: np.ndarray,
    imu_rpy: np.ndarray,
    imu_gyro: np.ndarray,
    imu_accel: np.ndarray,
    mode_pr: int,
    mode_machine: int,
) -> None:
    """Rellena `msg` (LowState) in-place a partir del snapshot del simulador."""
    msg.version = [0, 0]
    msg.mode_pr = int(mode_pr) & 0xFF
    msg.mode_machine = int(mode_machine) & 0xFF
    msg.tick = int(snap["sim_time"] * 1000.0) & 0xFFFFFFFF

    # ---- IMU ---------------------------------------------------------------
    imu = msg.imu_state
    imu.quaternion = [float(imu_quat[0]), float(imu_quat[1]),
                      float(imu_quat[2]), float(imu_quat[3])]   # (w,x,y,z)
    imu.gyroscope = [float(v) for v in imu_gyro]                # rad/s local
    imu.accelerometer = [float(v) for v in imu_accel]           # m/s² local
    imu.rpy = [float(v) for v in imu_rpy]                       # rad
    imu.temperature = 0

    # ---- Motores 0..26 (simulados) ------------------------------------------
    q = snap["q"]; dq = snap["dq"]; ddq = snap["ddq"]
    tau = snap["tau_est"]; mode = snap["mode"]

    ms = msg.motor_state
    for i in range(NUM_CMD_MOTORS):
        s = ms[i]
        s.mode = int(mode[i]) & 0xFF
        s.q = float(q[i])
        s.dq = float(dq[i])
        s.ddq = float(ddq[i])
        s.tau_est = float(tau[i])
        s.temperature = [0, 0]
        s.vol = 0.0
        s.sensor = [0, 0]
        s.motorstate = 0
        s.reserve = [0, 0, 0, 0]

    # ---- Motores 27..34 (no existen en el H1-2): valores seguros -------------
    for i in range(NUM_CMD_MOTORS, NUM_STATE_MOTORS):
        s = ms[i]
        s.mode = 0
        s.q = 0.0
        s.dq = 0.0
        s.ddq = 0.0
        s.tau_est = 0.0
        s.temperature = [0, 0]
        s.vol = 0.0
        s.sensor = [0, 0]
        s.motorstate = 0
        s.reserve = [0, 0, 0, 0]

    # ---- Resto --------------------------------------------------------------
    msg.wireless_remote = [0] * 40
    msg.reserve = [0, 0, 0, 0]
    msg.crc = 0   # documentado arriba: no verificado por los consumidores
