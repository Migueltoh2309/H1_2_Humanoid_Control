#!/usr/bin/env python3
"""Cierra el lazo IK -> /lowcmd -> bridge MuJoCo -> /lowstate.

Conecta la IK con cualquier consumidor de /lowcmd que publique /lowstate:
el robot real o un bridge de bajo nivel (el bridge MuJoCo de bajo nivel,
`h1_2_mujoco_lowlevel_bridge`, ya no se incluye en este repositorio). El nodo:

  1. Lee la q real de los brazos desde /lowstate.
  2. Resuelve la IK con límites hacia la pose objetivo de cada brazo.
  3. Interpola suavemente desde la q medida hasta la solución y publica
     /lowcmd con PD por articulación, con el CRC que el bridge verifica.

Arranque seguro: hasta que no llega el primer /lowstate no se publica
nada, y la primera consigna parte de la q medida, así que no hay salto
inicial. Al terminar (Ctrl-C) se dejan de enviar comandos y el watchdog
del bridge libera el torque en rampa.

Uso (con el robot o el bridge de bajo nivel publicando /lowstate):
    ros2 run h1_2_algoritms ik_lowcmd_node

Parámetros principales:
    arms              brazos a controlar: ["left"], ["right"] o ambos
    left_target       pose objetivo [x y z qw qx qy qz] en torso_link
    right_target      idem para el brazo derecho
    approach_time     segundos de la rampa hasta la pose objetivo
"""
# ===== ROS =====
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# ===== LIBS =====
import numpy as np
from sensor_msgs.msg import JointState
from unitree_hg.msg import LowCmd, LowState

# ===== CUSTOM =====
from h1_2_algoritms.movimiento.fk_functions import (
    fkine_arm_left_unitree, fkine_arm_right_unitree, TF2xyzquat)
from h1_2_algoritms.movimiento.ik_functions import ik_solve_limited, pose_error
from h1_2_algoritms.movimiento import joint_limits as JL

# El CRC vive en el paquete del bridge: es el mismo que verifica el nodo
# receptor, así que se reutiliza en lugar de duplicarlo.
from h1_2_algoritms.unitree_crc import attach_crc

NUM_MOTORS = 35

FK = {"left": fkine_arm_left_unitree, "right": fkine_arm_right_unitree}

# Ganancias por articulación: hombro y codo cargan el peso del brazo, las
# muñecas van suaves (su kd_max en el bridge es 5.0 por estabilidad).
DEFAULT_KP = [80.0, 80.0, 40.0, 40.0, 20.0, 20.0, 20.0]
DEFAULT_KD = [3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0]

# Ganancia integral [N·m/(rad·s)] y tope del término integral [N·m].
# Un PD puro deja error estacionario: la gravedad tira del brazo y solo se
# equilibra con kp·Δq ≠ 0. El integral aporta el par de sostén sin
# necesidad de un modelo dinámico, así que sirve igual en el robot real.
# El tope va holgadamente por debajo del tau_max de cada articulación
# (hombro 38, yaw/codo 17, muñecas 18 N·m en motor_limits.yaml).
DEFAULT_KI = [80.0, 80.0, 40.0, 40.0, 20.0, 20.0, 20.0]
DEFAULT_TAU_I_MAX = [25.0, 25.0, 12.0, 12.0, 8.0, 8.0, 8.0]


class IkLowCmdNode(Node):

    def __init__(self):
        super().__init__('ik_lowcmd_node')

        # ===== Parámetros =====
        self.declare_parameter('arms', ['left', 'right'])
        self.declare_parameter('left_target', [0.25, 0.25, 0.10, 1.0, 0.0, 0.0, 0.0])
        self.declare_parameter('right_target', [0.25, -0.25, 0.10, 1.0, 0.0, 0.0, 0.0])
        self.declare_parameter('kp', DEFAULT_KP)
        self.declare_parameter('kd', DEFAULT_KD)
        self.declare_parameter('ki', DEFAULT_KI)
        self.declare_parameter('tau_i_max', DEFAULT_TAU_I_MAX)
        self.declare_parameter('use_integral', True)
        self.declare_parameter('command_frequency', 250.0)
        self.declare_parameter('approach_time', 4.0)
        self.declare_parameter('mode_machine', 4)
        self.declare_parameter('publish_joint_states', False)

        gp = self.get_parameter
        self.arms = list(gp('arms').value)
        self.kp = np.array(gp('kp').value, dtype=float)
        self.kd = np.array(gp('kd').value, dtype=float)
        self.ki = np.array(gp('ki').value, dtype=float)
        self.tau_i_max = np.array(gp('tau_i_max').value, dtype=float)
        self.use_integral = bool(gp('use_integral').value)
        self.cmd_freq = float(gp('command_frequency').value)
        self.approach_time = float(gp('approach_time').value)
        self.mode_machine = int(gp('mode_machine').value)

        for side in self.arms:
            if side not in FK:
                raise ValueError(f"Brazo desconocido: '{side}' (usa left/right)")

        self.targets = {}
        for side in self.arms:
            self.targets[side] = np.array(
                gp(f'{side}_target').value, dtype=float)

        # ===== Estado por brazo =====
        self.state = {}
        for side in self.arms:
            lo, hi = JL.get_limits(side)
            self.state[side] = {
                'idx': JL.ARM_MOTOR_IDX[side],
                'q_min': lo, 'q_max': hi,
                'q_meas': None,     # última q medida en /lowstate
                'q_start': None,    # q al resolver la IK (origen de la rampa)
                'q_goal': None,     # solución de la IK (destino)
                'tau_i': np.zeros(7),   # término integral acumulado
            }

        # ===== ROS I/O =====
        # /lowcmd RELIABLE: la suscripción BEST_EFFORT del bridge lo acepta.
        self.pub_cmd = self.create_publisher(LowCmd, '/lowcmd', 10)
        self.create_subscription(LowState, '/lowstate', self.on_lowstate,
                                 qos_profile_sensor_data)

        self.pub_js = None
        if gp('publish_joint_states').value:
            # Nombre distinto de /joint_states: el bridge ya publica ahí.
            self.pub_js = self.create_publisher(JointState, 'ik_joint_states', 10)

        self.solved = False
        self.t_start = None
        self.t_prev = None
        self.create_timer(1.0 / self.cmd_freq, self.on_timer)
        self.create_timer(1.0, self.on_report)

        self.get_logger().info(
            f"Nodo IK->/lowcmd iniciado. Brazos: {self.arms}. "
            f"Esperando /lowstate del bridge...")

    # ================================================================ ROS
    def on_lowstate(self, msg: LowState):
        for side, st in self.state.items():
            st['q_meas'] = np.array([msg.motor_state[i].q for i in st['idx']])

    def _ready(self):
        return all(st['q_meas'] is not None for st in self.state.values())

    # ================================================================= IK
    def solve_all(self):
        """Resuelve la IK de cada brazo partiendo de la q medida."""
        for side, st in self.state.items():
            q_meas = st['q_meas'].copy()
            xd = self.targets[side]

            res = ik_solve_limited(
                FK[side], TF2xyzquat, q_meas, xd, st['q_min'], st['q_max'])

            st['q_start'] = q_meas
            st['q_goal'] = res['q']

            reach = "alcanzable" if res['ok'] else "NO alcanzable exactamente"
            lvl = self.get_logger().info if res['ok'] else self.get_logger().warn
            lvl(f"[{side}] objetivo {np.round(xd[:3], 3)} -> {reach}: "
                f"e_pos={res['e_pos']*1000:.1f} mm, e_ori={res['e_ori']:.4f}, "
                f"rearranques={res['restarts']}")
            self.get_logger().info(
                f"[{side}] q_solucion = {np.round(res['q'], 3)}")

            if not JL.within_limits(res['q'], side):
                # No debería ocurrir: ik_solve_limited satura siempre.
                self.get_logger().error(
                    f"[{side}] la solución excede los límites: "
                    f"{np.round(JL.violations(res['q'], side), 4)}")

    # ============================================================== LOOP
    def on_timer(self):
        if not self._ready():
            return

        now = self.get_clock().now()

        if not self.solved:
            self.solve_all()
            self.solved = True
            self.t_start = now
            self.t_prev = now
            return

        # Rampa suave (coseno) desde la q de arranque hasta la solución.
        elapsed = (now - self.t_start).nanoseconds * 1e-9
        s = min(elapsed / self.approach_time, 1.0) if self.approach_time > 0 else 1.0
        alpha = 0.5 * (1.0 - np.cos(np.pi * s))

        dt = (now - self.t_prev).nanoseconds * 1e-9
        self.t_prev = now

        msg = LowCmd()
        msg.mode_pr = 0
        msg.mode_machine = self.mode_machine
        for i in range(NUM_MOTORS):
            c = msg.motor_cmd[i]
            c.mode = 0
            c.q = c.dq = c.tau = c.kp = c.kd = 0.0

        for side, st in self.state.items():
            q_des = st['q_start'] + alpha * (st['q_goal'] - st['q_start'])
            q_des = np.clip(q_des, st['q_min'], st['q_max'])

            # Integral con anti-windup por saturación: compensa el par de
            # gravedad que el PD solo puede equilibrar con error.
            if self.use_integral and 0.0 < dt < 0.1:
                st['tau_i'] = np.clip(
                    st['tau_i'] + self.ki * (q_des - st['q_meas']) * dt,
                    -self.tau_i_max, self.tau_i_max)
            tau_ff = st['tau_i'] if self.use_integral else np.zeros(7)

            for k, i in enumerate(st['idx']):
                c = msg.motor_cmd[i]
                c.mode = 1
                c.q = float(q_des[k])
                c.dq = 0.0
                c.tau = float(tau_ff[k])
                c.kp = float(self.kp[k])
                c.kd = float(self.kd[k])
            st['q_cmd'] = q_des

        attach_crc(msg)
        self.pub_cmd.publish(msg)

        if self.pub_js is not None:
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            for side, st in self.state.items():
                js.name.extend(JL.ARM_JOINT_NAMES[side])
                js.position.extend([float(v) for v in st['q_cmd']])
            self.pub_js.publish(js)

    def on_report(self):
        """Informe periódico del error real del efector."""
        if not self.solved or not self._ready():
            return
        for side, st in self.state.items():
            q = st['q_meas']
            x = TF2xyzquat(FK[side](q))
            e = pose_error(self.targets[side], x)
            self.get_logger().info(
                f"[{side}] efector={np.round(x[:3], 4)} "
                f"e_pos={np.linalg.norm(e[:3])*1000:6.1f} mm "
                f"e_ori={np.linalg.norm(e[3:]):.4f} "
                f"|q-q_cmd|max={np.max(np.abs(q - st['q_cmd'])):.4f} rad")


def main(args=None):
    rclpy.init(args=args)
    node = IkLowCmdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            "Interrumpido: se dejan de enviar comandos; el watchdog del "
            "bridge llevará el torque a cero en rampa.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
