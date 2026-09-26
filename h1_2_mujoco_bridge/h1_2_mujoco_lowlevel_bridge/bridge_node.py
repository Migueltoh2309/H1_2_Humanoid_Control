"""
bridge_node.py — Bridge ROS 2 <-> MuJoCo de bajo nivel para el Unitree H1-2.

Emula la interfaz /lowcmd (unitree_hg/LowCmd) y /lowstate (unitree_hg/LowState)
del robot real, EXCLUSIVAMENTE dentro de la simulación local: no existe ningún
modo hardware, ninguna IP del robot ni SDK de red. Los mensajes unitree_hg se
usan solo como tipos de datos ROS 2.

Ciclo (hilo de simulación, a control_frequency):
  1. Se recibe LowCmd por /lowcmd (callback ROS) -> CRC/saneo/límites
     (lowcmd_handler) -> consigna del MotorController + rearme del watchdog.
  2. En cada tick: watchdog.check(); n_sub sub-pasos de mj_step recalculando
     tau = tau_ff + kp(q_des-q) + kd(dq_des-dq) por sub-paso (motor_controller
     + mujoco_simulator); saturaciones absolutas y de tasa incluidas.
  3. Se lee el estado simulado (q, dq, ddq, tau aplicado, IMU).
  4. Se construye LowState (lowstate_builder) y se publica en /lowstate a
     lowstate_frequency (diezmado del tick de control).
  5. Auxiliar: /joint_states (sensor_msgs/JointState) a joint_state_frequency.
  6. Viewer opcional sincronizado a viewer_frequency; NUNCA marca el ritmo.

QoS:
  * /lowstate  -> qos_profile_sensor_data (BEST_EFFORT/VOLATILE), igual que el
    robot real; compatible con la suscripción de test_leer.py.
  * /lowcmd    -> suscripción BEST_EFFORT: es compatible tanto con publishers
    RELIABLE (test_mandar_modificado usa depth 10 RELIABLE por defecto) como
    BEST_EFFORT, evitando "offering incompatible QoS".
"""
from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import JointState
from unitree_hg.msg import LowCmd, LowState

from .imu_simulator import ImuSimulator
from .joint_mapping import (DEFAULT_UNITREE_TO_MUJOCO, build_joint_mapping,
                            load_mapping_yaml)
from .lowcmd_handler import LowCmdHandler
from .lowstate_builder import fill_lowstate
from .motor_controller import MotorController
from .mujoco_simulator import MujocoSimulator
from .safety_limits import build_limits, load_limits_yaml
from .watchdog import CommandWatchdog


class LowLevelBridgeNode(Node):

    def __init__(self):
        super().__init__("h1_2_mujoco_lowlevel_bridge")
        self._declare_params()
        p = self._read_params()

        log = self.get_logger()

        # ── Modelo + mapeo (validación dura: aborta si el mapeo es inválido) ──
        model_path = self._resolve_model_path(p)
        log.info(f"Cargando MJCF: {model_path}")

        name_map = (load_mapping_yaml(p["joint_mapping_file"])
                    if p["joint_mapping_file"] else DEFAULT_UNITREE_TO_MUJOCO)

        self.sim = MujocoSimulator(
            model_path,
            mapping_builder=lambda m: build_joint_mapping(m, name_map),
            timestep=p["mujoco_timestep"],
            joint_armature=p["joint_armature"],
            joint_damping=p["joint_damping"],
            base_height_offset=p["base_height_offset"],
        )
        mapping = self.sim.mapping
        log.info("Mapeo Unitree<->MuJoCo validado (27/27):\n" + mapping.summary())
        if p["joint_armature"] > 0 or p["joint_damping"] > 0:
            log.info(
                f"Overrides de estabilidad aplicados en memoria a los 27 dof: "
                f"armature={p['joint_armature']}, damping={p['joint_damping']} "
                f"(ver README: evita el chattering de muñecas con Euler)"
            )
        if p["base_height_offset"] != 0.0:
            log.info(
                f"Base (pelvis) elevada {p['base_height_offset']:+.2f} m: modo "
                f"'colgado' tipo pórtico del SDK — los pies no tocan el suelo."
            )

        # ── Límites ───────────────────────────────────────────────────────────
        if p["motor_limits_file"]:
            defaults, per_joint = load_limits_yaml(p["motor_limits_file"])
        else:
            defaults, per_joint = {}, {}
        defaults.setdefault("kp_max", p["max_kp"])
        defaults.setdefault("kd_max", p["max_kd"])
        defaults.setdefault("tau_rate_max", p["max_torque_rate"])
        defaults.setdefault("dq_max", p["default_dq_max"])
        defaults.setdefault("tau_max", p["default_tau_max"])
        limits = build_limits(mapping, defaults, per_joint)

        # ── Frecuencias / diezmados ───────────────────────────────────────────
        self.control_dt = 1.0 / p["control_frequency"]
        n_sub_f = self.control_dt / self.sim.dt
        self.n_sub = max(1, int(round(n_sub_f)))
        if abs(n_sub_f - self.n_sub) > 1e-6:
            log.warn(
                f"control_frequency ({p['control_frequency']} Hz) no es múltiplo "
                f"exacto del timestep de MuJoCo ({self.sim.dt}s); usando "
                f"{self.n_sub} sub-pasos por tick "
                f"(control efectivo {1.0/(self.n_sub*self.sim.dt):.1f} Hz)."
            )
        self.k_state = max(1, int(round(p["control_frequency"] / p["lowstate_frequency"])))
        self.k_js = max(1, int(round(p["control_frequency"] / p["joint_state_frequency"])))
        self.k_view = max(1, int(round(p["control_frequency"] / p["viewer_frequency"])))
        log.info(
            f"Frecuencias: física {1.0/self.sim.dt:.0f} Hz (dt={self.sim.dt}s) | "
            f"control {1.0/self.control_dt:.0f} Hz ({self.n_sub} sub-pasos/tick) | "
            f"lowstate cada {self.k_state} tick(s) | joint_states cada {self.k_js} | "
            f"viewer cada {self.k_view} | realtime={p['realtime']}"
        )

        # ── Componentes ───────────────────────────────────────────────────────
        self.controller = MotorController(limits, dt_substep=self.sim.dt)
        self.handler = LowCmdHandler(
            mapping, limits,
            verify_crc=p["verify_crc"],
            warn=lambda s: self.get_logger().warn(s),
            warn_period_s=p["log_throttle_sec"],
        )
        log.info(f"Verificación de CRC en /lowcmd: "
                 f"{'ACTIVADA' if p['verify_crc'] else 'desactivada'}")

        self.watchdog = CommandWatchdog(
            timeout=p["command_timeout"],
            behavior=p["timeout_behavior"],
            hold_kp=p["hold_kp"],
            hold_kd=p["hold_kd"],
            on_event=lambda s: self.get_logger().warn(s),
        )
        log.info(f"Watchdog: timeout={p['command_timeout']}s, "
                 f"behavior={p['timeout_behavior']}")

        self.imu = ImuSimulator(self.sim.model)
        self.publish_imu = p["publish_imu"]
        log.info(f"IMU: {self.imu.describe()}"
                 + ("" if self.publish_imu else " [publicación desactivada -> ceros]"))

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self.mode_machine = int(p["mode_machine"])
        self._pub_lowstate = self.create_publisher(
            LowState, p["lowstate_topic"], qos_profile_sensor_data
        )
        self._sub_lowcmd = self.create_subscription(
            LowCmd, p["lowcmd_topic"], self._lowcmd_callback,
            qos_profile_sensor_data,   # BEST_EFFORT: compatible con pubs RELIABLE
        )
        self.publish_joint_states = p["publish_joint_states"]
        if self.publish_joint_states:
            self._pub_js = self.create_publisher(JointState, p["joint_state_topic"], 10)
            self._js_msg = JointState()
            self._js_msg.name = list(mapping.joint_names)

        self._lowstate_msg = LowState()   # reutilizado (se rellena cada publicación)

        # ── Viewer ────────────────────────────────────────────────────────────
        self.use_viewer = p["use_viewer"]
        if self.use_viewer:
            if self.sim.launch_viewer():
                log.info("Viewer de MuJoCo lanzado (pasivo).")
            else:
                log.warn("No se pudo abrir el viewer (¿sin display?); "
                         "continuando en modo headless.")
                self.use_viewer = False

        # ── Hilo de simulación ────────────────────────────────────────────────
        self.realtime = p["realtime"]
        self._stop = threading.Event()
        self._tick = 0
        self._lag_warned = 0.0
        self._sim_thread = threading.Thread(
            target=self._sim_loop, name="mujoco_sim", daemon=True
        )
        self._sim_thread.start()
        log.info("Bridge de bajo nivel iniciado: "
                 f"{p['lowcmd_topic']} -> MuJoCo -> {p['lowstate_topic']} ✓")

    # ==================================================================== ROS
    def _lowcmd_callback(self, msg: LowCmd) -> None:
        parsed = self.handler.process(msg)
        if parsed is None:
            return  # rechazado (CRC); la consigna previa sigue hasta el watchdog
        self.controller.set_command(
            parsed["mode"], parsed["q_des"], parsed["dq_des"],
            parsed["tau_ff"], parsed["kp"], parsed["kd"],
        )
        self.watchdog.notify_command()

    # ================================================================ SIM LOOP
    def _sim_loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set() and rclpy.ok():
            # 1) Watchdog (necesita q actual por si engancha hold_position)
            q_now, _ = self.sim.get_q_dq()
            self.watchdog.check(self.controller, q_now)

            # 2) Física: n_sub sub-pasos con PD por sub-paso
            self.sim.step_control_tick(self.controller, self.n_sub)
            self._tick += 1

            # 3) Publicaciones diezmadas
            if self._tick % self.k_state == 0:
                self._publish_lowstate()
            if self.publish_joint_states and self._tick % self.k_js == 0:
                self._publish_joint_states()
            if self.use_viewer and self._tick % self.k_view == 0:
                self.sim.sync_viewer()

            # 4) Pacing de tiempo real
            if self.realtime:
                next_t += self.control_dt
                now = time.monotonic()
                sleep_t = next_t - now
                if sleep_t > 0:
                    time.sleep(sleep_t)
                elif sleep_t < -0.25:
                    if now - self._lag_warned > 5.0:
                        self.get_logger().warn(
                            f"La simulación va {-sleep_t:.2f}s por detrás del "
                            f"tiempo real; re-sincronizando (considere bajar "
                            f"lowstate_frequency o control_frequency)."
                        )
                        self._lag_warned = now
                    next_t = now  # re-sincronizar sin acumular deuda

    def _publish_lowstate(self) -> None:
        snap = self.sim.snapshot(self.controller)
        if self.publish_imu:
            quat, rpy, gyro, accel = self.imu.read(self.sim.data, self.control_dt)
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            rpy = np.zeros(3); gyro = np.zeros(3); accel = np.zeros(3)
        fill_lowstate(
            self._lowstate_msg, snap, quat, rpy, gyro, accel,
            mode_pr=self.handler.last_mode_pr,
            mode_machine=self.mode_machine,
        )
        self._pub_lowstate.publish(self._lowstate_msg)

    def _publish_joint_states(self) -> None:
        q, dq = self.sim.get_q_dq()
        snap_tau = self.sim.data.actuator_force[self.sim.mapping.act_id]
        self._js_msg.header.stamp = self.get_clock().now().to_msg()
        self._js_msg.position = q.tolist()
        self._js_msg.velocity = dq.tolist()
        self._js_msg.effort = snap_tau.tolist()
        self._pub_js.publish(self._js_msg)

    # ================================================================== PARAMS
    def _declare_params(self) -> None:
        dp = self.declare_parameter
        dp("model_path", "")                       # absoluto; si vacío, se resuelve:
        dp("model_package", "h1_2_mujoco_bridge")  #   share(model_package)/model_relpath
        dp("model_relpath", "mjcf/h1_2_scene_qp_reachable.xml")
        dp("joint_mapping_file", "")
        dp("motor_limits_file", "")

        dp("lowcmd_topic", "/lowcmd")
        dp("lowstate_topic", "/lowstate")
        dp("joint_state_topic", "/joint_states")

        dp("mujoco_timestep", 0.001)
        dp("control_frequency", 500.0)
        dp("lowstate_frequency", 500.0)
        dp("joint_state_frequency", 50.0)
        dp("viewer_frequency", 60.0)

        dp("use_viewer", True)
        dp("realtime", True)

        dp("verify_crc", True)

        dp("command_timeout", 0.1)
        dp("timeout_behavior", "zero_torque")
        dp("hold_kp", 60.0)
        dp("hold_kd", 2.0)

        dp("publish_joint_states", True)
        dp("publish_imu", True)

        dp("max_kp", 300.0)
        dp("max_kd", 20.0)
        dp("max_torque_rate", 1000.0)
        dp("default_dq_max", 20.0)
        dp("default_tau_max", 80.0)

        # Overrides de estabilidad (ver mujoco_simulator.py y README)
        dp("joint_armature", 0.01)
        dp("joint_damping", 0.05)
        # Elevar la base soldada en Z [m]: 0.0 = escena tal cual (de pie);
        # 0.3 = robot "colgado" como en el pórtico de los tests del SDK.
        dp("base_height_offset", 0.0)

        dp("mode_machine", 4)      # eco arbitrario en simulación (configurable)
        dp("log_throttle_sec", 2.0)

    def _read_params(self) -> dict:
        g = lambda k: self.get_parameter(k).value
        keys = [
            "model_path", "model_package", "model_relpath",
            "joint_mapping_file", "motor_limits_file",
            "lowcmd_topic", "lowstate_topic", "joint_state_topic",
            "mujoco_timestep", "control_frequency", "lowstate_frequency",
            "joint_state_frequency", "viewer_frequency",
            "use_viewer", "realtime", "verify_crc",
            "command_timeout", "timeout_behavior", "hold_kp", "hold_kd",
            "publish_joint_states", "publish_imu",
            "max_kp", "max_kd", "max_torque_rate",
            "default_dq_max", "default_tau_max",
            "joint_armature", "joint_damping", "base_height_offset",
            "mode_machine", "log_throttle_sec",
        ]
        return {k: g(k) for k in keys}

    def _resolve_model_path(self, p: dict) -> str:
        if p["model_path"]:
            return p["model_path"]
        from ament_index_python.packages import get_package_share_directory
        import os
        share = get_package_share_directory(p["model_package"])
        return os.path.join(share, p["model_relpath"])

    # ================================================================= shutdown
    def shutdown(self) -> None:
        self._stop.set()
        if self._sim_thread.is_alive():
            self._sim_thread.join(timeout=2.0)
        self.sim.close_viewer()


def main(args=None):
    rclpy.init(args=args)
    node = LowLevelBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
