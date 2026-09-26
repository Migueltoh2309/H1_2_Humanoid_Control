#!/usr/bin/env python3
"""
fk_right_arm_raise.py — Levanta el brazo DERECHO del H1-2 hacia adelante sobre
el bridge de simulación (h1_2_mujoco_sim_bridge) y registra el movimiento para
poder compararlo después con el robot real.

Qué hace
--------
1. Genera una trayectoria articular suave (perfil quíntico, velocidad y
   aceleración nulas en los extremos) desde la pose de arranque q=0 hasta
   `q_goal`, para los 7 motores del brazo derecho.
2. Publica esa referencia en /joint_cmd (posición + velocidad). El bridge le
   aplica su PD y MuJoCo decide qué pasa de verdad: el brazo NO sigue la
   referencia exactamente, y eso es justamente lo que se quiere medir.
3. Usa la cinemática directa de `fk_functions.fkine_arm_right_unitree` (DH del
   brazo derecho, verificada contra MuJoCo: coincide al milímetro) para
   calcular la pose del efector final tanto de la REFERENCIA como de lo que
   MuJoCo reporta en /joint_states.
4. Guarda todo en un CSV y publica markers en RViz (esfera roja = referencia,
   esfera verde = real, línea verde = camino real del efector).

El CSV que deja es el insumo de `plot_right_arm_raise` (las dos gráficas: los
7 motores y el efector final).

Uso
---
    # Terminal 1 — simulador vacío (solo el robot, sin mesa/faja/mandarina):
    ros2 launch h1_2_mujoco_sim_bridge sim_bridge_empty.launch.py

    # Terminal 2 — el movimiento:
    ros2 run h1_2_algoritms fk_right_arm_raise

    # Gráficas (la ruta del CSV la imprime el nodo al terminar):
    ros2 run h1_2_algoritms plot_right_arm_raise <ruta_del_csv>

Parámetros útiles:
    -p q_goal:="[-1.4, -0.25, 0.3, 0.7, 0.35, 0.2, 0.4]"   # pose final [rad]
    -p t_rise:=4.0        # duración de la subida [s]
    -p t_hold:=2.0        # tiempo sosteniendo la pose al final [s]
    -p return_home:=true  # además de subir, vuelve a q=0 (ciclo completo)
    -p t_home:=3.0        # homing previo a q_home antes de registrar (0 = sin homing)
    -p rate:=100.0        # frecuencia de comando y de registro [Hz]
    -p csv_path:=/ruta/mi_ensayo.csv
"""

# ===== ROS =====
import rclpy
from rclpy.node import Node

# ===== LIBS =====
import csv
import os
from datetime import datetime

import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

# ===== CUSTOM =====
from h1_2_algoritms.fk_functions import fkine_arm_right_unitree, TF2xyzquat
from h1_2_algoritms.markers import (create_line_marker, create_sphere_marker,
                                    set_marker_pose)
from h1_2_algoritms import joint_limits as JL

# Orden de la cadena cinemática del brazo derecho: el MISMO que espera
# fkine_arm_right_unitree(q) y el mismo de joint_limits.ARM_JOINT_NAMES.
RIGHT_ARM_JOINTS = JL.ARM_JOINT_NAMES["right"]

# Pose final por defecto [rad], en el orden de RIGHT_ARM_JOINTS:
#   shoulder_pitch = -1.40  -> es el que LEVANTA el brazo hacia adelante
#                              (negativo = flexión hacia delante/arriba)
#   shoulder_roll  = -0.25  -> separa un poco el brazo del torso
#   shoulder_yaw   = +0.30
#   elbow          = +0.70  -> codo ligeramente flexionado (pose natural)
#   wrist_*        -> pequeño reajuste de la muñeca, para que los 7 motores
#                     tengan algo que mostrar en la gráfica
# Verificado en MuJoCo: todo dentro de los límites articulares y sin ninguna
# autocolisión a lo largo del camino (los únicos contactos son pies-piso).
DEFAULT_Q_GOAL = [-1.40, -0.25, 0.30, 0.70, 0.35, 0.20, 0.40]


def quintic(s):
    """Perfil quíntico 0->1 con velocidad y aceleración nulas en los extremos.

    Devuelve (posición, derivada respecto de s normalizado)."""
    s = min(max(s, 0.0), 1.0)
    pos = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    vel = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
    return pos, vel


class RightArmRaiseNode(Node):

    def __init__(self):
        super().__init__('fk_right_arm_raise')

        # ===== Parámetros =====
        self.declare_parameter('q_goal', DEFAULT_Q_GOAL)
        self.declare_parameter('t_rise', 4.0)
        self.declare_parameter('t_hold', 2.0)
        self.declare_parameter('return_home', False)
        # Fase previa de homing: lleva el brazo desde donde esté HASTA q_home
        # antes de empezar a registrar, para que dos ensayos seguidos (o un
        # ensayo después de otro movimiento) arranquen siempre igual. El PD
        # del bridge sostiene la última posición comandada, así que sin esto
        # la segunda corrida empieza con un tirón desde la pose levantada.
        # 0.0 = sin homing (arranca a registrar de una).
        self.declare_parameter('t_home', 3.0)
        self.declare_parameter('rate', 100.0)
        self.declare_parameter('csv_path', '')
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('shutdown_when_done', True)

        q_goal = np.array(self.get_parameter('q_goal').value, dtype=float)
        if q_goal.size != 7:
            raise ValueError(f"q_goal debe tener 7 elementos, tiene {q_goal.size}")

        self.t_rise = float(self.get_parameter('t_rise').value)
        self.t_hold = float(self.get_parameter('t_hold').value)
        self.return_home = bool(self.get_parameter('return_home').value)
        self.t_home = max(0.0, float(self.get_parameter('t_home').value))
        self.rate = float(self.get_parameter('rate').value)
        self.publish_markers = bool(self.get_parameter('publish_markers').value)
        self.shutdown_when_done = bool(self.get_parameter('shutdown_when_done').value)

        # Saturar la meta a los límites reales del brazo derecho (los mismos
        # que usa el bridge); si hay que recortar, avisar en vez de callarlo.
        q_clamped = JL.clamp(q_goal, "right")
        if not np.allclose(q_clamped, q_goal):
            self.get_logger().warn(
                f"q_goal fuera de límites, saturado: {np.round(q_goal, 3)} -> "
                f"{np.round(q_clamped, 3)}")
        self.q_goal = q_clamped
        self.q_home = np.zeros(7)

        # Duración total: subida + sostenido [+ bajada + sostenido].
        self.t_end = self.t_rise + self.t_hold
        if self.return_home:
            self.t_end += self.t_rise + self.t_hold

        # ===== Publishers =====
        # /joint_cmd = comando al bridge. NO /joint_states: ese lo publica el
        # bridge con lo que realmente pasó en la física.
        self.pub_cmd = self.create_publisher(JointState, '/joint_cmd', 10)
        self.marker_pub = self.create_publisher(Marker, 'ee_marker', 10)

        self.cmd_msg = JointState()
        self.cmd_msg.name = list(RIGHT_ARM_JOINTS)

        # ===== Markers (frame torso_link, igual que el resto del paquete) =====
        self.marker_ref = create_sphere_marker(
            frame="torso_link", ns="right_ee", marker_id=0, scale=0.04,
            color=(1.0, 0.0, 0.0, 0.9))          # rojo  = referencia (FK del comando)
        self.marker_real = create_sphere_marker(
            frame="torso_link", ns="right_ee", marker_id=1, scale=0.05,
            color=(0.0, 1.0, 0.2, 0.9))          # verde = real (FK de /joint_states)
        self.marker_path = create_line_marker(frame="torso_link")
        self.marker_path.id = 2

        # ===== Estado medido (lo llena /joint_states) =====
        self.q_meas = None
        self.dq_meas = np.zeros(7)
        self.tau_meas = np.zeros(7)
        self._idx_meas = None      # índices de los 7 joints dentro del mensaje
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)

        # ===== Registro =====
        self.rows = []
        self._t0 = None
        self._done = False
        self._warned_no_state = False
        self._homing_from = None   # q medido al arrancar la fase de homing
        self._homing_t0 = None
        self._homing_done = (self.t_home <= 0.0)

        self.timer = self.create_timer(1.0 / self.rate, self.update)

        self.get_logger().info(
            f"Levantando el brazo DERECHO hacia adelante: q_goal="
            f"{np.round(self.q_goal, 3).tolist()} rad, subida {self.t_rise:.1f} s, "
            f"sostenido {self.t_hold:.1f} s"
            + (f", vuelta a home {self.t_rise:.1f} s" if self.return_home else "")
            + f" (total {self.t_end:.1f} s a {self.rate:.0f} Hz).")

    # ============================================================ TRAYECTORIA
    def q_ref(self, t):
        """Referencia articular (q_des, dq_des) en el instante t [s]."""
        d = self.q_goal - self.q_home

        if t < self.t_rise:                                   # subida
            s, ds = quintic(t / self.t_rise)
            return self.q_home + s * d, (ds / self.t_rise) * d

        if t < self.t_rise + self.t_hold:                     # sostenido arriba
            return self.q_goal.copy(), np.zeros(7)

        if not self.return_home:
            return self.q_goal.copy(), np.zeros(7)

        t2 = t - (self.t_rise + self.t_hold)
        if t2 < self.t_rise:                                  # bajada
            s, ds = quintic(t2 / self.t_rise)
            return self.q_goal - s * d, -(ds / self.t_rise) * d

        return self.q_home.copy(), np.zeros(7)                # sostenido abajo

    # =============================================================== CALLBACKS
    def _on_joint_states(self, msg: JointState):
        """Extrae por NOMBRE los 7 joints del brazo derecho del estado que
        publica el bridge (trae los 27 motores del robot)."""
        if self._idx_meas is None:
            try:
                self._idx_meas = [msg.name.index(n) for n in RIGHT_ARM_JOINTS]
            except ValueError as e:
                if not self._warned_no_state:
                    self._warned_no_state = True
                    self.get_logger().warn(f"/joint_states no trae el brazo derecho: {e}")
                return

        def take(seq):
            if len(seq) <= max(self._idx_meas):
                return np.zeros(7)
            return np.array([seq[i] for i in self._idx_meas], dtype=float)

        self.q_meas = take(msg.position)
        self.dq_meas = take(msg.velocity)
        self.tau_meas = take(msg.effort)

    # ============================================================ LOOP PRINCIPAL
    def update(self):
        if self._done:
            return

        # Esperar el primer /joint_states: así t=0 coincide con el simulador
        # ya corriendo y la primera fila del CSV tiene medida real.
        if self.q_meas is None:
            if not self._warned_no_state:
                self._warned_no_state = True
                self.get_logger().info("Esperando /joint_states del bridge...")
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if not self._homing_done:
            self._do_homing(now)
            return

        if self._t0 is None:
            self._t0 = now
            self.get_logger().info("Movimiento iniciado.")
        t = now - self._t0

        q_des, dq_des = self.q_ref(t)

        # ===== 1. Comando al bridge =====
        self.cmd_msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_msg.position = q_des.tolist()
        self.cmd_msg.velocity = dq_des.tolist()
        self.pub_cmd.publish(self.cmd_msg)

        # ===== 2. Cinemática directa (la del paquete) =====
        x_des = TF2xyzquat(fkine_arm_right_unitree(q_des))
        x_real = TF2xyzquat(fkine_arm_right_unitree(self.q_meas))

        # ===== 3. Registro =====
        self.rows.append(
            [t]
            + q_des.tolist() + self.q_meas.tolist()
            + dq_des.tolist() + self.dq_meas.tolist() + self.tau_meas.tolist()
            + x_des.tolist() + x_real.tolist()
        )

        # ===== 4. Markers para RViz =====
        if self.publish_markers:
            self._publish_markers(x_des, x_real)

        # ===== 5. ¿Terminó? =====
        if t >= self.t_end:
            self._finish(t)

    def _do_homing(self, now):
        """Rampa suave desde donde esté el brazo hasta q_home, ANTES de empezar
        a registrar. Así el ensayo siempre parte de la misma referencia y es
        comparable entre corridas (y contra el robot real)."""
        if self._homing_t0 is None:
            self._homing_t0 = now
            self._homing_from = self.q_meas.copy()
            if np.allclose(self._homing_from, self.q_home, atol=1e-3):
                self._homing_done = True
                return
            self.get_logger().info(
                f"Homing: llevando el brazo a q_home en {self.t_home:.1f} s "
                f"(estaba en {np.round(self._homing_from, 3).tolist()}).")

        s_, _ = quintic((now - self._homing_t0) / self.t_home)
        q_cmd = self._homing_from + s_ * (self.q_home - self._homing_from)

        self.cmd_msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_msg.position = q_cmd.tolist()
        self.cmd_msg.velocity = [0.0] * 7
        self.pub_cmd.publish(self.cmd_msg)

        if now - self._homing_t0 >= self.t_home:
            self._homing_done = True
            self.get_logger().info("Homing terminado.")

    def _publish_markers(self, x_des, x_real):
        m = set_marker_pose(self.marker_ref, x_des, self)
        if m is not None:
            self.marker_pub.publish(m)

        m = set_marker_pose(self.marker_real, x_real, self)
        if m is not None:
            self.marker_pub.publish(m)

        p = Point()
        p.x, p.y, p.z = float(x_real[0]), float(x_real[1]), float(x_real[2])
        self.marker_path.points.append(p)
        self.marker_path.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(self.marker_path)

    # ==================================================================== FIN
    def _finish(self, t):
        self._done = True
        self.timer.cancel()

        path = self._write_csv()

        data = np.array(self.rows)
        q_des_f, q_meas_f = data[-1, 1:8], data[-1, 8:15]
        err = q_meas_f - q_des_f
        ee_des_f, ee_real_f = data[-1, 36:39], data[-1, 43:46]
        ee_0 = data[0, 43:46]

        self.get_logger().info(f"Movimiento terminado en t={t:.2f} s, {len(self.rows)} muestras.")
        self.get_logger().info(f"  q comandado final : {np.round(q_des_f, 4).tolist()}")
        self.get_logger().info(f"  q real final      : {np.round(q_meas_f, 4).tolist()}")
        self.get_logger().info(
            f"  error articular   : max |e| = {np.abs(err).max():.4f} rad "
            f"({np.degrees(np.abs(err).max()):.2f}°) en "
            f"{RIGHT_ARM_JOINTS[int(np.argmax(np.abs(err)))]}")
        self.get_logger().info(f"  efector (FK del comando) : {np.round(ee_des_f, 4).tolist()} m")
        self.get_logger().info(f"  efector (FK del real)    : {np.round(ee_real_f, 4).tolist()} m")
        self.get_logger().info(
            f"  desvío comando-real del efector: "
            f"{np.linalg.norm(ee_real_f - ee_des_f) * 1000:.1f} mm")
        self.get_logger().info(
            f"  el efector se movió {np.linalg.norm(ee_real_f - ee_0) * 1000:.0f} mm "
            f"desde el inicio (Δ = {np.round(ee_real_f - ee_0, 4).tolist()} m en torso_link)")
        self.get_logger().info(f"CSV -> {path}")
        self.get_logger().info(f"Gráficas: ros2 run h1_2_algoritms plot_right_arm_raise {path}")

        if self.shutdown_when_done:
            # El bridge mantiene la última posición comandada (su PD no tiene
            # timeout), así que salir aquí no "suelta" el brazo.
            raise SystemExit

    def _write_csv(self):
        path = self.get_parameter('csv_path').value
        if not path:
            out_dir = os.path.join(os.path.expanduser("~"), "humanoid_ws",
                                    "resultados_brazo_derecho")
            os.makedirs(out_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(out_dir, f"sim_right_arm_raise_{stamp}.csv")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        short = [n.replace("right_", "").replace("_joint", "") for n in RIGHT_ARM_JOINTS]
        header = (["t"]
                  + [f"q_des_{n}" for n in short]
                  + [f"q_meas_{n}" for n in short]
                  + [f"dq_des_{n}" for n in short]
                  + [f"dq_meas_{n}" for n in short]
                  + [f"tau_meas_{n}" for n in short]
                  + ["ee_des_x", "ee_des_y", "ee_des_z",
                     "ee_des_qw", "ee_des_qx", "ee_des_qy", "ee_des_qz"]
                  + ["ee_meas_x", "ee_meas_y", "ee_meas_z",
                     "ee_meas_qw", "ee_meas_qx", "ee_meas_qy", "ee_meas_qz"])

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(self.rows)
        return path


def main(args=None):
    rclpy.init(args=args)
    node = RightArmRaiseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
