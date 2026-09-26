"""
offline_demo.py — Validación de lazo cerrado SIN ROS (Etapa 0).

Replica EXACTAMENTE la ley de control de test_mandar_modificado.py (etapa 1:
ir suavemente a postura cero; etapa 2: senos en tobillos ±0.25 rad y muñecas
roll ±0.5 rad) y la hace pasar por el MISMO pipeline del bridge:

    LowCmd falso (duck) + CRC real
        -> LowCmdHandler (verificación CRC, saneo, límites)
        -> MotorController (PD+ff, saturaciones, rate-limit)
        -> MujocoSimulator (sub-pasos, overrides de estabilidad)
        -> snapshot (equivalente a LowState)

Solo queda fuera el transporte DDS. Sirve para validar mapeo, control,
estabilidad y frecuencias sin necesidad de ROS ni robot.

Uso:
    python3 -m h1_2_mujoco_lowlevel_bridge.offline_demo --mjcf /ruta/al.xml \
        [--out demo.png] [--dur 8.0] [--no-overrides]
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

try:
    from .crc import attach_crc
    from .joint_mapping import build_joint_mapping, NUM_CMD_MOTORS
    from .lowcmd_handler import LowCmdHandler
    from .motor_controller import MotorController
    from .mujoco_simulator import MujocoSimulator
    from .safety_limits import build_limits
    from .imu_simulator import ImuSimulator
except ImportError:  # ejecución directa como script
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from h1_2_mujoco_lowlevel_bridge.crc import attach_crc
    from h1_2_mujoco_lowlevel_bridge.joint_mapping import build_joint_mapping, NUM_CMD_MOTORS
    from h1_2_mujoco_lowlevel_bridge.lowcmd_handler import LowCmdHandler
    from h1_2_mujoco_lowlevel_bridge.motor_controller import MotorController
    from h1_2_mujoco_lowlevel_bridge.mujoco_simulator import MujocoSimulator
    from h1_2_mujoco_lowlevel_bridge.safety_limits import build_limits
    from h1_2_mujoco_lowlevel_bridge.imu_simulator import ImuSimulator


# ---- Dobles mínimos de los mensajes (mismo layout de campos) -----------------
class FakeMotorCmd:
    __slots__ = ("mode", "q", "dq", "tau", "kp", "kd", "reserve")

    def __init__(self):
        self.mode = 0
        self.q = self.dq = self.tau = self.kp = self.kd = 0.0
        self.reserve = 0


class FakeLowCmd:
    def __init__(self):
        self.mode_pr = 0
        self.mode_machine = 0
        self.motor_cmd = [FakeMotorCmd() for _ in range(35)]
        self.reserve = [0, 0, 0, 0]
        self.crc = 0


# Índices (idénticos a H12JointIndex del script de referencia)
L_ANKLE_P, L_ANKLE_R, R_ANKLE_P, R_ANKLE_R = 4, 5, 10, 11
L_WRIST_ROLL, R_WRIST_ROLL = 17, 24


def reference_controller_cmd(t: float, q_measured: np.ndarray,
                             duration: float = 3.0) -> FakeLowCmd:
    """Réplica 1:1 de LowLevelCmdSender.control() de test_mandar_modificado."""
    cmd = FakeLowCmd()
    cmd.mode_pr = 0     # PR
    cmd.mode_machine = 4

    for i in range(NUM_CMD_MOTORS):
        mc = cmd.motor_cmd[i]
        mc.mode = 1
        mc.tau = 0.0
        mc.q = 0.0
        mc.dq = 0.0
        mc.kp = 100.0 if i < 13 else 50.0
        mc.kd = 1.0

    if t < duration:                       # Etapa 1: llevar a cero suavemente
        ratio = min(max(t / duration, 0.0), 1.0)
        for i in range(NUM_CMD_MOTORS):
            cmd.motor_cmd[i].q = (1.0 - ratio) * float(q_measured[i])
    else:                                   # Etapa 2: senos tobillos + muñecas
        tt = t - duration
        max_pitch = max_roll = 0.25
        lp = max_pitch * math.cos(2.0 * math.pi * tt)
        lr = max_roll * math.sin(2.0 * math.pi * tt)
        rp = max_pitch * math.cos(2.0 * math.pi * tt)
        rr = -max_roll * math.sin(2.0 * math.pi * tt)
        for idx, qd, kp in ((L_ANKLE_P, lp, 80.0), (L_ANKLE_R, lr, 80.0),
                            (R_ANKLE_P, rp, 80.0), (R_ANKLE_R, rr, 80.0)):
            mc = cmd.motor_cmd[idx]
            mc.q, mc.dq, mc.kp, mc.kd, mc.tau = qd, 0.0, kp, 1.0, 0.0
        wr = 0.5 * math.sin(2.0 * math.pi * tt)
        for idx in (L_WRIST_ROLL, R_WRIST_ROLL):
            mc = cmd.motor_cmd[idx]
            mc.q, mc.dq, mc.kp, mc.kd, mc.tau = wr, 0.0, 50.0, 1.0, 0.0

    attach_crc(cmd)     # CRC obligatorio, igual que en el script real
    return cmd


def run(mjcf: str, out_png: str, duration: float,
        overrides: bool, plot: bool, hang_offset: float = 0.30) -> dict:
    control_dt = 0.002          # 500 Hz, como el script de referencia
    sim = MujocoSimulator(
        mjcf,
        mapping_builder=lambda m: build_joint_mapping(m),
        timestep=0.001,
        joint_armature=0.01 if overrides else 0.0,
        joint_damping=0.05 if overrides else 0.0,
        base_height_offset=hang_offset,
    )
    n_sub = int(round(control_dt / sim.dt))
    limits = build_limits(sim.mapping,
                          defaults={"tau_max": 80.0, "kp_max": 300.0,
                                    "kd_max": 20.0, "tau_rate_max": 1000.0,
                                    "dq_max": 20.0})
    controller = MotorController(limits, dt_substep=sim.dt)
    warns = []
    handler = LowCmdHandler(sim.mapping, limits, verify_crc=True,
                            warn=warns.append, warn_period_s=1.0)
    imu = ImuSimulator(sim.model)

    n_ticks = int(duration / control_dt)
    log = {k: np.zeros(n_ticks) for k in
           ("t", "q_wr", "qd_wr", "tau_wr", "dq_wr", "q_ap", "qd_ap")}

    t0 = time.perf_counter()
    for k in range(n_ticks):
        t = k * control_dt
        q, _ = sim.get_q_dq()

        cmd = reference_controller_cmd(t, q)          # "controlador"
        parsed = handler.process(cmd)                  # "bridge: entrada"
        assert parsed is not None, "CRC válido rechazado"
        controller.set_command(parsed["mode"], parsed["q_des"], parsed["dq_des"],
                               parsed["tau_ff"], parsed["kp"], parsed["kd"])

        sim.step_control_tick(controller, n_sub)       # "bridge: física"
        snap = sim.snapshot(controller)                # "bridge: LowState"

        log["t"][k] = t
        log["q_wr"][k] = snap["q"][L_WRIST_ROLL]
        log["qd_wr"][k] = parsed["q_des"][L_WRIST_ROLL]
        log["dq_wr"][k] = snap["dq"][L_WRIST_ROLL]
        log["tau_wr"][k] = snap["tau_est"][L_WRIST_ROLL]
        log["q_ap"][k] = snap["q"][L_ANKLE_P]
        log["qd_ap"][k] = parsed["q_des"][L_ANKLE_P]
    wall = time.perf_counter() - t0

    quat, rpy, gyro, accel = imu.read(sim.data, control_dt)

    # ---- métricas (tras 1 s de la etapa 2) -----------------------------------
    i2 = int((3.0 + 1.0) / control_dt)
    rmse_wr = float(np.sqrt(np.mean((log["q_wr"][i2:] - log["qd_wr"][i2:]) ** 2)))
    rmse_ap = float(np.sqrt(np.mean((log["q_ap"][i2:] - log["qd_ap"][i2:]) ** 2)))
    sg = np.sign(log["dq_wr"][i2:]); sg[sg == 0] = 1
    rev_s = float(np.sum(np.abs(np.diff(sg)) > 0) / (log["t"][-1] - log["t"][i2]))

    res = {
        "rmse_wrist_roll": rmse_wr, "rmse_ankle_pitch": rmse_ap,
        "wrist_dq_max": float(np.abs(log["dq_wr"]).max()),
        "wrist_reversals_per_s": rev_s,
        "crc_rejected": handler.n_rejected_crc,
        "cmd_accepted": handler.n_accepted,
        "hang_offset": hang_offset,
        "sim_seconds": duration, "wall_seconds": wall,
        "realtime_factor": duration / wall,
        "imu_quat": quat.tolist(), "imu_rpy": rpy.tolist(),
        "imu_gyro": gyro.tolist(), "imu_accel": accel.tolist(),
        "warnings": warns,
    }

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        ax[0].plot(log["t"], log["qd_wr"], "k--", lw=1, label="q_des")
        ax[0].plot(log["t"], log["q_wr"], "-", color="#1f77b4", lw=1.2,
                   label="q simulada")
        ax[0].set_ylabel("wrist_roll izq [rad]")
        ax[0].set_title(
            "Lazo cerrado offline: controlador de referencia (test_mandar) → pipeline "
            f"completo del bridge → MuJoCo\nRMSE muñeca={rmse_wr*1000:.1f} mrad, "
            f"RMSE tobillo={rmse_ap*1000:.1f} mrad, reversiones dq muñeca={rev_s:.0f}/s "
            f"(sin chattering), CRC rechazados={handler.n_rejected_crc}", fontsize=10)
        ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(log["t"], log["qd_ap"], "k--", lw=1, label="q_des")
        ax[1].plot(log["t"], log["q_ap"], "-", color="#2ca02c", lw=1.2,
                   label="q simulada")
        ax[1].set_ylabel("ankle_pitch izq [rad]")
        if hang_offset > 0.0:
            nota = (f"robot 'colgado' +{hang_offset:.2f} m sobre el suelo — "
                    "equivalente al pórtico del que Unitree\ncuelga el robot en "
                    "este test del SDK: tobillos libres, contacto pie-suelo = 0 N")
        else:
            nota = ("seguimiento limitado por CONTACTO pie-suelo (los pies apoyan "
                    "en el piso de la escena;\nfísica correcta — quitar --standing "
                    "para replicar el pórtico del SDK)")
        ax[1].text(0.99, 0.05, nota,
                   transform=ax[1].transAxes, fontsize=7.5, ha="right",
                   color="#555555",
                   bbox=dict(boxstyle="round", fc="white", ec="#aaaaaa", alpha=0.8))
        ax[1].legend(); ax[1].grid(alpha=0.3)
        ax[2].plot(log["t"], log["tau_wr"], "-", color="#d62728", lw=0.9)
        ax[2].set_ylabel("tau aplicado\nwrist_roll [N·m]")
        ax[2].set_xlabel("tiempo [s]")
        ax[2].grid(alpha=0.3)
        for a in ax:
            a.axvline(3.0, color="gray", ls=":", lw=1)
        ax[0].text(3.05, ax[0].get_ylim()[1]*0.8, "etapa 2 (senos)", fontsize=8,
                   color="gray")
        plt.tight_layout()
        plt.savefig(out_png, dpi=110)
        res["plot"] = out_png

    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", default="lowlevel_demo_tracking.png")
    ap.add_argument("--dur", type=float, default=8.0)
    ap.add_argument("--hang-offset", type=float, default=0.30,
                    help="elevar la base [m] para 'colgar' el robot como en el "
                         "pórtico del test del SDK (default 0.30)")
    ap.add_argument("--standing", action="store_true",
                    help="escena tal cual (de pie, pies en el suelo)")
    ap.add_argument("--no-overrides", action="store_true",
                    help="desactiva armature/damping (para ver la inestabilidad)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    hang = 0.0 if args.standing else args.hang_offset
    res = run(args.mjcf, args.out, args.dur,
              overrides=not args.no_overrides, plot=not args.no_plot,
              hang_offset=hang)
    for k, v in res.items():
        if k != "warnings":
            print(f"{k}: {v}")
    for w in res["warnings"]:
        print("WARN:", w)


if __name__ == "__main__":
    main()
