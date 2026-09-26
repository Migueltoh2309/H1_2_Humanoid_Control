"""
sim_bridge_node.py — Bridge ROS 2 <-> MuJoCo de SIMULACIÓN GENERAL para el
Unitree H1-2 (manipulación bimanual, aprendizaje por demostración, picking).

A diferencia de h1_2_mujoco_lowlevel_bridge (que emula /lowcmd+/lowstate del
robot real con mensajes unitree_hg), este bridge NO replica ningún protocolo
de hardware: MuJoCo es el motor de física/colisiones y expone una interfaz
genérica pensada para trabajar cómodo desde ROS 2/RViz2:

  * Comando  /joint_cmd  (sensor_msgs/JointState): q_des/dq_des/tau_ff por
    joint, por NOMBRE, actualización parcial (los joints no incluidos en el
    mensaje mantienen su último objetivo — ver pd_controller.py).
  * Estado   /joint_states (sensor_msgs/JointState): para robot_state_publisher
    (TF) y para registrar demostraciones.
  * Cámara   /camera/color/{image_raw,camera_info}
             /camera/depth/{image_raw,camera_info}
    renderizados offscreen desde la cámara del MJCF (robot_rgbd_camera),
    alineados (mismo punto de vista) porque es una sola cámara virtual.
  * Contactos /contacts (visualization_msgs/MarkerArray): esferas en cada
    punto de contacto activo, color/tamaño según fuerza normal — para ver
    colisiones en RViz durante manipulación.
  * TF estático: <camera_body>_link -> camera_color_optical_frame /
    camera_depth_optical_frame (pose calculada desde el MJCF, no depende de
    que el URDF tenga un link de cámara coincidente).

TF dinámico (joints, mundo->pelvis, etc.) lo publica robot_state_publisher
externo a partir de /joint_states (ver launch/). El robot es de base FIJA
(pelvis soldada, sin floating_base_joint activo en el URDF), así que pelvis
es la raíz real del árbol de TF: no hace falta publicar odom.

Dos hilos:
  * Física (_sim_loop): aplica el último /joint_cmd recibido (ya está en el
    controller), mj_step con el PD recalculado en cada paso, publica
    /joint_states (diezmado a joint_state_frequency) y contactos (diezmado a
    contacts_frequency), sincroniza el viewer opcional (diezmado a
    viewer_frequency, nunca marca el ritmo), pacing a reloj de pared si
    realtime=true.
  * Cámara (_camera_loop): renderizar es 10-100x más lento que un mj_step
    (ver README); si compartiera el hilo de física le robaría presupuesto al
    PD y a /joint_states, así que corre aparte con su propio pacing a
    camera_frequency. mujoco_sim.MujocoSim protege con un lock el único
    punto de carrera real entre los dos hilos (mj_step escribiendo `data`
    mientras Renderer.update_scene lee `data`).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import MarkerArray

from .joint_mapping import load_joint_names_yaml
from .msg_builders import (camera_info_msg, contact_marker_array,
                            depth_image_msg, rgb_image_msg)
from .mujoco_sim import MujocoSim
from .pd_controller import JointPD

DEFAULT_JOINT_NAMES = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# kp, kd, tau_max por joint — tau_max copiado de actuatorfrcrange del MJCF.
DEFAULT_GAINS = {
    "left_hip_yaw_joint": (120.0, 4.0, 200.0), "left_hip_pitch_joint": (120.0, 4.0, 200.0),
    "left_hip_roll_joint": (120.0, 4.0, 200.0), "left_knee_joint": (150.0, 5.0, 300.0),
    "left_ankle_pitch_joint": (40.0, 2.0, 60.0), "left_ankle_roll_joint": (30.0, 1.5, 40.0),
    "right_hip_yaw_joint": (120.0, 4.0, 200.0), "right_hip_pitch_joint": (120.0, 4.0, 200.0),
    "right_hip_roll_joint": (120.0, 4.0, 200.0), "right_knee_joint": (150.0, 5.0, 300.0),
    "right_ankle_pitch_joint": (40.0, 2.0, 60.0), "right_ankle_roll_joint": (30.0, 1.5, 40.0),
    "torso_joint": (150.0, 5.0, 200.0),
    "left_shoulder_pitch_joint": (60.0, 2.0, 40.0), "left_shoulder_roll_joint": (60.0, 2.0, 40.0),
    "left_shoulder_yaw_joint": (40.0, 1.5, 18.0), "left_elbow_joint": (40.0, 1.5, 18.0),
    "left_wrist_roll_joint": (15.0, 0.8, 19.0), "left_wrist_pitch_joint": (15.0, 0.8, 19.0),
    "left_wrist_yaw_joint": (15.0, 0.8, 19.0),
    "right_shoulder_pitch_joint": (60.0, 2.0, 40.0), "right_shoulder_roll_joint": (60.0, 2.0, 40.0),
    "right_shoulder_yaw_joint": (40.0, 1.5, 18.0), "right_elbow_joint": (40.0, 1.5, 18.0),
    "right_wrist_roll_joint": (15.0, 0.8, 19.0), "right_wrist_pitch_joint": (15.0, 0.8, 19.0),
    "right_wrist_yaw_joint": (15.0, 0.8, 19.0),
}


def load_gains_yaml(path: str) -> dict:
    import yaml
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}
    table = raw.get('gains', raw)
    return {k: (float(v['kp']), float(v['kd']), float(v['tau_max'])) for k, v in table.items()}


class SimBridgeNode(Node):

    def __init__(self):
        super().__init__("h1_2_mujoco_sim_bridge")
        self._declare_params()
        p = self._read_params()
        log = self.get_logger()

        model_path = self._resolve_share_path(p["model_package"], p["model_relpath"], p["model_path"])
        log.info(f"Cargando MJCF: {model_path}")

        joint_names = (load_joint_names_yaml(p["joint_names_file"])
                       if p["joint_names_file"] else list(DEFAULT_JOINT_NAMES))

        self.sim = MujocoSim(
            model_path,
            joint_names=joint_names,
            timestep=p["mujoco_timestep"],
            joint_armature=p["joint_armature"],
            joint_damping=p["joint_damping"],
            base_body=p["base_body"],
            camera_name=p["camera_name"] or None,
            camera_width=p["camera_width"],
            camera_height=p["camera_height"],
            initial_keyframe=p["initial_keyframe"] or None,
            gravity_comp=p["gravity_comp"],
        )
        if p["initial_keyframe"]:
            log.info(f"Keyframe inicial cargado: '{p['initial_keyframe']}'")
        mapping = self.sim.mapping
        log.info(f"Mapeo de joints validado ({len(mapping)}):\n" + mapping.summary())
        if self.sim.aux_actuators:
            log.info(f"Actuadores de posición auxiliares (comandables por /joint_cmd, sin PD): "
                     f"{sorted(self.sim.aux_actuators)}")
        if p["grasp_assist"]:
            sides = self.sim.enable_grasp_assist()
            if sides:
                log.info(f"Asistencia de agarre (solo simulación) habilitada en {sides}")
            else:
                log.warn("grasp_assist=true pero el modelo no tiene '<side>_grasp_assist' / mano Inspire")
        log.info(
            "Control de motor: PD + compensación de gravedad g(q) ✓ (como el "
            "H1-2 real)" if p["gravity_comp"] else
            "Control de motor: PD puro, SIN compensación de gravedad "
            "(gravity_comp=false) — habrá error estacionario por el peso del brazo")
        if p["joint_armature"] > 0 or p["joint_damping"] > 0:
            log.info(f"Overrides de estabilidad: armature={p['joint_armature']}, "
                     f"damping={p['joint_damping']} (mismo motivo que h1_2_mujoco_lowlevel_bridge)")

        # ── Ganancias PD ──────────────────────────────────────────────────
        gains = load_gains_yaml(p["joint_gains_file"]) if p["joint_gains_file"] else DEFAULT_GAINS
        kp = np.zeros(len(mapping)); kd = np.zeros(len(mapping)); tau_max = np.zeros(len(mapping))
        missing = []
        for i, name in enumerate(mapping.joint_names):
            if name not in gains:
                missing.append(name)
                kp[i], kd[i], tau_max[i] = 40.0, 1.5, 20.0
                continue
            kp[i], kd[i], tau_max[i] = gains[name]
        if missing:
            log.warn(f"Sin ganancias configuradas para {missing}; usando default (40, 1.5, 20)")

        q_init, _ = self.sim.get_q_dq()
        self.controller = JointPD(mapping, kp, kd, tau_max, q_init,
                                   warn=lambda s: self.get_logger().warn(s))
        log.info("Objetivo inicial = pose de arranque del MJCF (el robot no se mueve "
                 "hasta que llegue el primer /joint_cmd para cada joint).")

        # ── Frecuencias ───────────────────────────────────────────────────
        # La cámara vive en su PROPIO hilo con su propio pacing (ver más
        # abajo): renderizar es 10-100x más lento que un mj_step y si
        # compartiera el hilo de física, un frame lento retrasaría el PD y
        # /joint_states. mujoco_sim.MujocoSim protege el acceso concurrente
        # a self.data con un lock interno (mj_step vs. update_scene).
        self.physics_hz = 1.0 / self.sim.dt
        self.k_state = max(1, round(self.physics_hz / p["joint_state_frequency"]))
        self.k_contacts = max(1, round(self.physics_hz / p["contacts_frequency"]))
        self.k_view = max(1, round(self.physics_hz / p["viewer_frequency"]))
        self.camera_period_s = 1.0 / p["camera_frequency"]
        log.info(f"Física {self.physics_hz:.0f} Hz (dt={self.sim.dt}s) | "
                 f"joint_states cada {self.k_state} paso(s) | "
                 f"contactos cada {self.k_contacts} | viewer cada {self.k_view} | "
                 f"cámara (hilo propio) {p['camera_frequency']:.1f} Hz | "
                 f"realtime={p['realtime']}")

        # ── ROS I/O ───────────────────────────────────────────────────────
        self._pub_js = self.create_publisher(JointState, p["joint_state_topic"], 10)
        self._js_msg = JointState()
        self._js_msg.name = list(mapping.joint_names)

        self._sub_cmd = self.create_subscription(
            JointState, p["joint_cmd_topic"], self._joint_cmd_callback, 10)
        self._got_first_cmd = False

        self.publish_camera = bool(p["publish_camera"]) and self.sim.camera_id >= 0
        if p["publish_camera"] and self.sim.camera_id < 0:
            log.warn("publish_camera=true pero camera_name está vacío o no existe; desactivada.")
        if self.publish_camera:
            self._pub_rgb = self.create_publisher(Image, p["color_image_topic"], 10)
            self._pub_rgb_info = self.create_publisher(CameraInfo, p["color_info_topic"], 10)
            self._pub_depth = self.create_publisher(Image, p["depth_image_topic"], 10)
            self._pub_depth_info = self.create_publisher(CameraInfo, p["depth_info_topic"], 10)
            self._publish_camera_static_tf()
            log.info(f"Cámara '{p['camera_name']}' {self.sim.camera_width}x{self.sim.camera_height} "
                     f"fovy={self.sim.camera_fovy:.1f}° -> {p['color_image_topic']} / {p['depth_image_topic']}")

        self.publish_contacts = bool(p["publish_contacts"])
        if self.publish_contacts:
            self._pub_contacts = self.create_publisher(MarkerArray, p["contacts_topic"], 10)

        # ── Verdad de terreno (evaluación) ────────────────────────────────
        self._gt = []
        for name in [b for b in p["ground_truth_bodies"] if b]:
            bid = self.sim.body_id(name)
            if bid < 0:
                log.warn(f"ground_truth_bodies: '{name}' no existe en el modelo")
                continue
            self._gt.append((bid, self.create_publisher(PoseStamped, f"/sim/ground_truth/{name}", 10)))
        self.k_gt = max(1, round(self.physics_hz / p["ground_truth_frequency"]))
        if self._gt:
            log.info(f"Verdad de terreno (solo evaluación) en /sim/ground_truth/*: "
                     f"{[b for b in p['ground_truth_bodies'] if b]}")

        # ── Viewer ────────────────────────────────────────────────────────
        self.use_viewer = p["use_viewer"]
        if self.use_viewer:
            if self.sim.launch_viewer():
                log.info("Viewer de MuJoCo lanzado (pasivo). RViz sigue siendo la fuente de "
                         "verdad para TF/joint_states; este viewer es solo un espejo visual.")
            else:
                log.warn("No se pudo abrir el viewer (¿sin display?); continuando headless.")
                self.use_viewer = False

        # ── Hilo de simulación ────────────────────────────────────────────
        self.realtime = p["realtime"]
        self.control_dt = self.sim.dt
        self._stop = threading.Event()
        self._tick = 0
        self._lag_warned = 0.0
        self._sim_thread = threading.Thread(target=self._sim_loop, name="mujoco_sim", daemon=True)
        self._sim_thread.start()

        self._camera_thread = None
        if self.publish_camera:
            self._camera_thread = threading.Thread(
                target=self._camera_loop, name="mujoco_camera", daemon=True)
            self._camera_thread.start()

        log.info(f"Sim bridge iniciado: {p['joint_cmd_topic']} -> MuJoCo -> "
                 f"{p['joint_state_topic']} ✓")

    # ==================================================================== ROS
    def _joint_cmd_callback(self, msg: JointState) -> None:
        names = list(msg.name)
        aux = [k for k, nm in enumerate(names) if nm in self.sim.aux_actuators]
        if aux:
            # Actuadores de posición del MJCF (dedos de la mano): no pasan
            # por el PD, se fija su ctrl directamente.
            self.sim.set_aux_ctrl([names[k] for k in aux],
                                  [msg.position[k] for k in aux if k < len(msg.position)])
            keep = [k for k in range(len(names)) if k not in set(aux)]
            pick = lambda arr: [arr[k] for k in keep] if len(arr) == len(names) else list(arr)  # noqa: E731
            names, pos, vel, eff = [names[k] for k in keep], pick(msg.position), pick(msg.velocity), \
                pick(msg.effort)
        else:
            pos, vel, eff = list(msg.position), list(msg.velocity), list(msg.effort)
        n = self.controller.set_targets_by_name(names, pos, vel, eff) if names else len(aux)
        if not self._got_first_cmd and n > 0:
            self._got_first_cmd = True
            self.get_logger().info(f"Primer /joint_cmd recibido ({n} joint(s) actualizados).")

    def _publish_camera_static_tf(self) -> None:
        pos, quat_xyzw = self.sim.camera_optical_pose_in_body()
        broadcaster = StaticTransformBroadcaster(self)
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for child in ("camera_color_optical_frame", "camera_depth_optical_frame"):
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.sim.camera_body_name
            t.child_frame_id = child
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            t.transform.rotation.x = float(quat_xyzw[0])
            t.transform.rotation.y = float(quat_xyzw[1])
            t.transform.rotation.z = float(quat_xyzw[2])
            t.transform.rotation.w = float(quat_xyzw[3])
            transforms.append(t)
        broadcaster.sendTransform(transforms)
        self._static_tf_broadcaster = broadcaster  # mantener vivo
        self.get_logger().info(
            f"TF estático publicado: {self.sim.camera_body_name} -> "
            f"camera_{{color,depth}}_optical_frame (RGB-D alineadas, misma cámara virtual)")

    # ================================================================ SIM LOOP
    def _sim_loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set() and rclpy.ok():
            self.sim.step(self.controller)
            self._tick += 1

            if self._tick % self.k_state == 0:
                self._publish_joint_states()
            if self._gt and self._tick % self.k_gt == 0:
                self._publish_ground_truth()
            if self.publish_contacts and self._tick % self.k_contacts == 0:
                self._publish_contacts()
            if self.use_viewer and self._tick % self.k_view == 0:
                self.sim.sync_viewer()

            if self.realtime:
                next_t += self.control_dt
                now = time.monotonic()
                sleep_t = next_t - now
                if sleep_t > 0:
                    time.sleep(sleep_t)
                elif sleep_t < -0.25:
                    if now - self._lag_warned > 5.0:
                        self.get_logger().warn(
                            f"La simulación va {-sleep_t:.2f}s por detrás del tiempo real "
                            f"(¿joint_state_frequency/contacts_frequency muy altas?); "
                            f"re-sincronizando.")
                        self._lag_warned = now
                    next_t = now

    def _camera_loop(self) -> None:
        """Hilo dedicado: renderizar es lento (~15-30 ms/frame) y correr esto
        en el hilo de física retrasaría el PD y /joint_states. Ver el lock en
        mujoco_sim.MujocoSim para la sección crítica compartida.

        El contexto EGL/GL del Renderer queda ligado al hilo que lo usa
        (eglMakeCurrent falla con EGL_BAD_ACCESS si otro hilo lo toca
        después), así que TODO el renderizado — incluido el primer
        "precalentamiento" que compila shaders (~0.5-1 s la primera vez) —
        pasa aquí, nunca en el hilo constructor."""
        self.get_logger().info("Precalentando renderer offscreen (compilación de shaders, una vez)...")
        self.sim.warmup_renderer()
        self.get_logger().info("Renderer listo.")

        next_t = time.monotonic()
        while not self._stop.is_set() and rclpy.ok():
            self._publish_camera_frame()
            if self.realtime:
                next_t += self.camera_period_s
                now = time.monotonic()
                sleep_t = next_t - now
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_t = now  # el render tardó más que el período; no acumular deuda

    def _publish_joint_states(self) -> None:
        snap = self.sim.snapshot(self.controller)
        self._js_msg.header.stamp = self.get_clock().now().to_msg()
        self._js_msg.position = snap["q"].tolist()
        self._js_msg.velocity = snap["dq"].tolist()
        self._js_msg.effort = snap["tau_est"].tolist()
        self._pub_js.publish(self._js_msg)

    def _publish_ground_truth(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for bid, pub in self._gt:
            pos, quat_wxyz = self.sim.body_pose_in_base(bid)
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self.sim.base_body_name
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, pos)
            (msg.pose.orientation.w, msg.pose.orientation.x,
             msg.pose.orientation.y, msg.pose.orientation.z) = map(float, quat_wxyz)
            pub.publish(msg)

    def _publish_camera_frame(self) -> None:
        stamp = self.get_clock().now().to_msg()
        rgb = self.sim.render_rgb()
        self._pub_rgb.publish(rgb_image_msg(rgb, "camera_color_optical_frame", stamp))
        self._pub_rgb_info.publish(camera_info_msg(
            self.sim.camera_width, self.sim.camera_height, self.sim.camera_fovy,
            "camera_color_optical_frame", stamp))
        depth = self.sim.render_depth()
        self._pub_depth.publish(depth_image_msg(depth, "camera_depth_optical_frame", stamp))
        self._pub_depth_info.publish(camera_info_msg(
            self.sim.camera_width, self.sim.camera_height, self.sim.camera_fovy,
            "camera_depth_optical_frame", stamp))

    def _publish_contacts(self) -> None:
        stamp = self.get_clock().now().to_msg()
        contacts = self.sim.get_contacts()
        self._pub_contacts.publish(
            contact_marker_array(contacts, self.sim.base_body_name, stamp))

    # ================================================================== PARAMS
    def _declare_params(self) -> None:
        dp = self.declare_parameter
        dp("model_path", "")
        dp("model_package", "h1_2_description")
        dp("model_relpath", "mjcf/h1_2_scene_qp_reachable.xml")
        dp("joint_names_file", "")
        dp("joint_gains_file", "")
        # Nombre de un <key> del MJCF para inicializar qpos/qvel/ctrl (p.ej.
        # "mandarina_moving" en h1_2_scene_surgery_table.xml). Vacío = estado
        # por defecto del modelo (todo en 0 / pos de arranque del XML).
        dp("initial_keyframe", "")
        # Weld palma-fruta al cerrar la mano con contacto real (solo escena
        # con manos; ver MujocoSim.enable_grasp_assist).
        dp("grasp_assist", False)
        # Verdad de terreno SOLO PARA EVALUACIÓN (nunca como entrada de un
        # controlador): pose de estos cuerpos en /sim/ground_truth/<cuerpo>,
        # frame base_body, a ground_truth_frequency Hz.
        dp("ground_truth_bodies", [""])
        dp("ground_truth_frequency", 10.0)

        dp("joint_cmd_topic", "/joint_cmd")
        dp("joint_state_topic", "/joint_states")

        dp("base_body", "pelvis")

        dp("mujoco_timestep", 0.002)
        dp("joint_state_frequency", 100.0)
        dp("viewer_frequency", 60.0)

        dp("use_viewer", True)
        dp("realtime", True)

        dp("gravity_comp", True)
        dp("joint_armature", 0.01)
        dp("joint_damping", 0.05)

        dp("publish_camera", True)
        dp("camera_name", "robot_rgbd_camera")
        dp("camera_width", 640)
        dp("camera_height", 480)
        dp("camera_frequency", 15.0)
        dp("color_image_topic", "/camera/color/image_raw")
        dp("color_info_topic", "/camera/color/camera_info")
        dp("depth_image_topic", "/camera/depth/image_raw")
        dp("depth_info_topic", "/camera/depth/camera_info")

        dp("publish_contacts", True)
        dp("contacts_topic", "/contacts")
        dp("contacts_frequency", 15.0)

    def _read_params(self) -> dict:
        g = lambda k: self.get_parameter(k).value
        keys = [
            "model_path", "model_package", "model_relpath",
            "joint_names_file", "joint_gains_file", "initial_keyframe", "grasp_assist",
            "ground_truth_bodies", "ground_truth_frequency",
            "joint_cmd_topic", "joint_state_topic", "base_body",
            "mujoco_timestep", "joint_state_frequency", "viewer_frequency",
            "use_viewer", "realtime", "gravity_comp",
            "joint_armature", "joint_damping",
            "publish_camera", "camera_name", "camera_width", "camera_height",
            "camera_frequency", "color_image_topic", "color_info_topic",
            "depth_image_topic", "depth_info_topic",
            "publish_contacts", "contacts_topic", "contacts_frequency",
        ]
        return {k: g(k) for k in keys}

    def _resolve_share_path(self, package: str, relpath: str, override: str) -> str:
        if override:
            return override
        from ament_index_python.packages import get_package_share_directory
        import os
        return os.path.join(get_package_share_directory(package), relpath)

    # ================================================================= shutdown
    def shutdown(self) -> None:
        self._stop.set()
        if self._sim_thread.is_alive():
            self._sim_thread.join(timeout=2.0)
        if self._camera_thread is not None and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)
        self.sim.close_viewer()


def main(args=None):
    rclpy.init(args=args)
    node = SimBridgeNode()
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
