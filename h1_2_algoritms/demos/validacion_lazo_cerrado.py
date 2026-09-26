#!/usr/bin/env python3
"""Demo 2 — Lazo cerrado IK -> /lowcmd -> MuJoCo -> /lowstate.

NECESITA el bridge corriendo:
    ros2 launch h1_2_mujoco_lowlevel_bridge mujoco_lowlevel_bridge.launch.py

Para cada objetivo mide el error del efector en tres puntos:
  1. la solución de la IK (error puramente numérico),
  2. lo que consigue el robot con PD puro,
  3. lo que consigue con PD + término integral.

Uso:
    python3 demos/validacion_lazo_cerrado.py [N]    # N objetivos (def. 4)
"""
import sys
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from unitree_hg.msg import LowCmd, LowState

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h1_2_mujoco_lowlevel_bridge.crc import attach_crc          # noqa: E402
from h1_2_algoritms.fk_functions import (                       # noqa: E402
    fkine_arm_left_unitree as FK, TF2xyzquat)
from h1_2_algoritms.ik_functions import (                       # noqa: E402
    ik_solve_limited, pose_error)
from h1_2_algoritms import joint_limits as JL                   # noqa: E402
from h1_2_algoritms.ik_lowcmd_node import (                     # noqa: E402
    DEFAULT_KP, DEFAULT_KD, DEFAULT_KI, DEFAULT_TAU_I_MAX)

SIDE = "left"
IDX = JL.ARM_MOTOR_IDX[SIDE]
LO, HI = JL.get_limits(SIDE)
KP = np.array(DEFAULT_KP)
KD = np.array(DEFAULT_KD)
KI = np.array(DEFAULT_KI)
TAU_I_MAX = np.array(DEFAULT_TAU_I_MAX)


class Probe(Node):
    def __init__(self):
        super().__init__('validacion_lazo_cerrado')
        self.pub = self.create_publisher(LowCmd, '/lowcmd', 10)
        self.create_subscription(LowState, '/lowstate', self.cb,
                                 qos_profile_sensor_data)
        self.q = None
        self.tau_i = np.zeros(7)
        self.q_cmd = None
        self.t_prev = None

    def cb(self, msg):
        self.q = np.array([msg.motor_state[i].q for i in IDX])

    def send(self, q_des, use_integral):
        now = time.monotonic()
        dt = 0.0 if self.t_prev is None else now - self.t_prev
        self.t_prev = now
        if use_integral and 0.0 < dt < 0.1:
            self.tau_i = np.clip(self.tau_i + KI * (q_des - self.q) * dt,
                                 -TAU_I_MAX, TAU_I_MAX)
        m = LowCmd()
        m.mode_pr = 0
        m.mode_machine = 4
        for i in range(35):
            c = m.motor_cmd[i]
            c.mode = 0
            c.q = c.dq = c.tau = c.kp = c.kd = 0.0
        for k, i in enumerate(IDX):
            c = m.motor_cmd[i]
            c.mode = 1
            c.q = float(q_des[k])
            c.dq = 0.0
            c.tau = float(self.tau_i[k]) if use_integral else 0.0
            c.kp = float(KP[k])
            c.kd = float(KD[k])
        attach_crc(m)
        self.pub.publish(m)
        self.q_cmd = q_des

    def goto(self, q_goal, ramp, settle, use_integral):
        q0 = self.q.copy()
        t0 = time.time()
        nxt = 0.0
        while rclpy.ok() and time.time() - t0 < ramp + settle:
            if time.time() >= nxt:
                s = min((time.time() - t0) / ramp, 1.0)
                a = 0.5 * (1.0 - np.cos(np.pi * s))
                self.send(np.clip(q0 + a * (q_goal - q0), LO, HI), use_integral)
                nxt = time.time() + 1.0 / 250.0
            rclpy.spin_once(self, timeout_sec=0.004)

    def wait(self, secs):
        end = time.time() + secs
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)


def main():
    n_targets = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    # Poses de referencia moderadas: dentro de rango, con el brazo separado
    # del torso y sin cruzar la mesa de la escena.
    rng = np.random.default_rng(5)
    q_refs = [np.array([rng.uniform(-0.9, 0.2), rng.uniform(0.25, 1.0),
                        rng.uniform(-0.5, 0.5), rng.uniform(0.2, 1.2),
                        rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3),
                        rng.uniform(-0.6, 0.6)]) for _ in range(n_targets)]

    rclpy.init()
    node = Probe()
    print("\nEsperando /lowstate del bridge...")
    node.wait(2.0)
    if node.q is None:
        print("\n  ERROR: no llega /lowstate. ¿Está corriendo el bridge?")
        print("  ros2 launch h1_2_mujoco_lowlevel_bridge "
              "mujoco_lowlevel_bridge.launch.py\n")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print()
    print("#" * 74)
    print("#  DEMO 2 — LAZO CERRADO: IK -> /lowcmd -> MuJoCo -> /lowstate")
    print("#" * 74)
    print("\nBrazo izquierdo. Error del efector medido con la q real de la")
    print("simulación (la FK está validada contra MuJoCo a menos de 1 um).\n")
    print(f"{'#':>2} {'IK dentro lim.':>15} {'error IK':>11} "
          f"{'robot: PD solo':>16} {'robot: PD+I':>14}")
    print("-" * 74)

    rows = []
    for i, q_ref in enumerate(q_refs):
        xd = TF2xyzquat(FK(q_ref))
        r = ik_solve_limited(FK, TF2xyzquat, node.q.copy(), xd, LO, HI)
        ok_lim = JL.within_limits(r["q"], SIDE)

        node.tau_i = np.zeros(7)
        node.goto(r["q"], 3.0, 3.0, use_integral=False)
        e_pd = np.linalg.norm(pose_error(xd, TF2xyzquat(FK(node.q)))[:3])

        node.goto(r["q"], 0.5, 7.0, use_integral=True)
        e_pi = np.linalg.norm(pose_error(xd, TF2xyzquat(FK(node.q)))[:3])

        print(f"{i:>2} {('SI' if ok_lim else 'NO'):>15} "
              f"{r['e_pos']*1000:9.4f}mm {e_pd*1000:14.2f}mm "
              f"{e_pi*1000:12.4f}mm")
        rows.append((ok_lim, r["e_pos"], e_pd, e_pi))

    print("-" * 74)
    e_pd_m = np.mean([r[2] for r in rows]) * 1000
    e_pi_m = np.mean([r[3] for r in rows]) * 1000
    print(f"  soluciones dentro de límites : {sum(r[0] for r in rows)}/{len(rows)}")
    print(f"  error medio con PD puro      : {e_pd_m:8.2f} mm  "
          f"(caída por gravedad: el PD solo equilibra con error)")
    print(f"  error medio con PD + integral: {e_pi_m:8.4f} mm  "
          f"(mejora x{e_pd_m/max(e_pi_m, 1e-9):.0f})")
    print()

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
