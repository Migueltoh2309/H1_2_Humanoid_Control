#!/usr/bin/env python3
"""QP bimanual con evasión de colisiones sobre h1_2_mujoco_sim_bridge.

Extiende QP_whole_body.py a bimanualidad segura:
  * un solo QP de 14 GDL para los dos brazos (bimanual_avoidance.py);
  * autocolisión brazo-brazo / brazo-torso desde la cinemática (cápsulas,
    collision_model.py) — no necesita cámara;
  * obstáculos del entorno desde la cámara RGB-D del torso
    (/camera/depth/image_raw): nube -> torso_link -> auto-filtrado del
    robot -> voxelizado -> memoria con borrado por espacio libre
    (depth_obstacles.py).

Entradas:  /joint_states, /camera/depth/image_raw, /camera/depth/camera_info,
           TF torso_link -> camera_depth_optical_frame (lo publica el bridge;
           si no llega se usa la pose del MJCF).
Salidas:   /joint_cmd (posición + velocidad de los 14 joints de brazos),
           /bimanual/markers (cápsulas coloreadas por distancia, objetivos),
           /bimanual/obstacles (nube de obstáculos que ve el QP),
           /bimanual/min_distance (Float32MultiArray: [d_brazos, d_entorno]).

Uso:
    ros2 launch h1_2_mujoco_sim_bridge sim_bridge.launch.py use_viewer:=false
    ros2 run h1_2_algoritms QP_bimanual_avoidance --ros-args -p scenario:=circles
    # comparar: sin evasión (los brazos chocan, se ve en /contacts de RViz)
    ros2 run h1_2_algoritms QP_bimanual_avoidance --ros-args -p scenario:=circles \\
        -p avoid_self:=false -p use_camera:=false

Escenarios (objetivos de la muñeca en torso_link, igual que
demos/evaluate_bimanual_avoidance.py): home, shared, converge, swap, table,
circles. Arranca yendo a 'home' durante `t_home` s y luego ejecuta el
escenario.

Capa 2 (`use_coordinator`, por defecto true): bimanual_coordinator.py
detecta cuando los brazos se bloquean mutuamente y decide cuál pasa
primero; el otro se retira. Se ve con claridad en `scenario:=shared` (las
dos manos quieren el mismo punto). Estado en /bimanual/coordinator.

Capa 3 (`use_planner`, por defecto true): bimanual_planner.py. Si un brazo
no avanza (p.ej. trabado debajo de un obstáculo ancho), planifica un camino
con RRT-Connect en un hilo aparte (el lazo sigue a 20 Hz con los brazos
quietos) y lo guía punto a punto. Para verlo hace falta una escena con un
obstáculo que el modelo no conoce:
    ros2 run h1_2_mujoco_sim_bridge sim_bridge --ros-args \
        -p model_relpath:=mjcf/h1_2_scene_tray_obstacle.xml -p use_viewer:=false
    ros2 run h1_2_algoritms QP_bimanual_avoidance --ros-args -p scenario:=tray

Visual servoing + agarre (`scenario:=grasp`, ver VISUAL_SERVOING_PLAN.md):
la meta ya no es fija sino la mandarina que publica la percepción
(/perception/target_position, PointStamped; se lleva a torso_link por TF).
visual_servoing.FruitTracker (Kalman de velocidad constante) predice dónde
está la fruta ahora y a qué velocidad va; visual_servoing.GraspServoing
elige el brazo, da la pose de pre-agarre/agarre de la muñeca, el
feedforward de velocidad (faja) y el comando de la mano (6 joints por
lado, publicados en /joint_cmd). Es la misma lógica que mide
demos/visual_servoing_grasp.py. La nube de obstáculos excluye la zona de la
fruta (si no, el damper no dejaría acercarse la mano). La capa 3 se
desactiva en este modo: la meta se mueve y un camino RRT planificado hacia
una meta vieja no sirve. Todo junto, en simulación:
    ros2 launch h1_2_algoritms visual_servoing_sim.launch.py
"""
import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from h1_2_algoritms.bimanual_avoidance import BimanualAvoidanceController
from h1_2_algoritms.bimanual_coordinator import BimanualCoordinator, TaskSequence
from h1_2_algoritms.bimanual_planner import GlobalPlannerLayer
from h1_2_algoritms.collision_model import arm_frames, arm_capsules, TORSO_CAPSULE
from h1_2_algoritms.depth_obstacles import (
    obstacles_from_depth, camera_pose_from_mjcf_xyaxes, ObstacleMemory)
from h1_2_algoritms.joint_limits import ARM_JOINT_NAMES
from h1_2_algoritms.visual_servoing import FruitTracker, GraspServoing
from h1_2_algoritms import grasp_geometry as GG
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

FRAME = "torso_link"
Q_FWD = [1.0, 0.0, 0.0, 0.0]
Q_DOWN = [0.70710678, 0.0, 0.70710678, 0.0]

SCENARIOS = {
    "home": {"left": [0.30, 0.20, 0.05, *Q_FWD], "right": [0.30, -0.20, 0.05, *Q_FWD]},
    "converge": {"left": [0.35, -0.03, 0.05, *Q_FWD], "right": [0.35, 0.03, 0.05, *Q_FWD]},
    "swap": {"left": [0.30, -0.12, 0.08, *Q_FWD], "right": [0.30, 0.12, 0.08, *Q_FWD]},
    "table": {"left": [0.30, 0.15, -0.06, *Q_DOWN], "right": [0.30, -0.15, -0.06, *Q_DOWN]},
    # tray: la mano derecha sube por encima de la bandeja de
    # mjcf/h1_2_scene_tray_obstacle.xml (escenario "obstacle" de la evaluación).
    "tray": {"left": [0.30, 0.20, 0.05, *Q_FWD], "right": [0.34, -0.22, 0.35, *Q_FWD]},
}
# shared: las dos manos al mismo punto (1 s de "agarre") y de vuelta a casa.
SHARED_POINT = [0.34, 0.0, 0.15, *Q_FWD]
# circles: centros a ±3 cm, círculo de 8 cm en el plano Y-Z y en contrafase:
# en cada vuelta las dos manos se cruzan por el centro.
CIRCLE_CENTER = {"left": np.array([0.33, 0.03, 0.05]), "right": np.array([0.33, -0.03, 0.05])}
CIRCLE_RADIUS = 0.08
CIRCLE_PHASE = {"left": 0.0, "right": np.pi}
# grasp: reposo con las manos arriba y afuera de la faja (el mismo HOME con
# el que se midió demos/visual_servoing_grasp.py).
GRASP_HOME = {"left": [0.25, 0.25, 0.10, *Q_FWD], "right": [0.25, -0.25, 0.10, *Q_FWD]}


def image_to_depth(msg):
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)


class BimanualAvoidanceNode(Node):

    def __init__(self):
        super().__init__("QP_bimanual_avoidance")

        dp = self.declare_parameter
        dp("scenario", "circles")
        dp("rate", 20.0)
        dp("t_home", 4.0)
        dp("circle_omega", 0.8)
        dp("avoid_self", True)
        dp("use_camera", True)
        dp("use_memory", True)
        dp("d_safe", 0.03)
        dp("d_influence", 0.15)
        dp("d_safe_env", 0.05)
        dp("d_influence_env", 0.20)
        dp("xi", 0.4)
        dp("use_coordinator", True)
        dp("use_planner", True)
        dp("planning_time", 5.0)   # s, tope de RRT-Connect por intento
        dp("w_elbow", 0.03)
        dp("shoulder_roll_inward_max", 0.10)   # rad, ver bimanual_avoidance.py
        dp("voxel", 0.03)
        dp("self_padding", 0.04)
        # ----- scenario:=grasp -----
        dp("target_topic", "/perception/target_position")
        dp("hand_cmd_topic", "/joint_cmd")
        dp("fruit_exclude_radius", 0.12)   # m: la nube no ve obstáculos alrededor de la fruta
        dp("rearm_after_missed", 2.0)      # s; < 0 = no rearmar
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.scenario = gp("scenario")
        self.grasp_mode = self.scenario == "grasp"
        if self.scenario not in SCENARIOS and self.scenario not in ("circles", "shared", "grasp"):
            raise ValueError(f"scenario desconocido: {self.scenario}")
        self.dt = 1.0 / float(gp("rate"))
        self.t_home = float(gp("t_home"))
        self.omega = float(gp("circle_omega"))
        self.use_camera = bool(gp("use_camera"))
        self.voxel = float(gp("voxel"))
        self.self_padding = float(gp("self_padding"))

        self.ctrl = BimanualAvoidanceController(
            dt=self.dt, avoid_self=bool(gp("avoid_self")), avoid_obstacles=self.use_camera,
            d_safe=float(gp("d_safe")), d_influence=float(gp("d_influence")), xi=float(gp("xi")),
            d_safe_env=float(gp("d_safe_env")), d_influence_env=float(gp("d_influence_env")),
            w_elbow=float(gp("w_elbow")),
            shoulder_roll_inward_max=float(gp("shoulder_roll_inward_max")))
        self.coord = BimanualCoordinator(self.dt, d_influence=self.ctrl.d_i) \
            if bool(gp("use_coordinator")) else None
        use_planner = bool(gp("use_planner")) and not self.grasp_mode
        self.planner = GlobalPlannerLayer(
            self.dt, self.ctrl.q_min, self.ctrl.q_max, planning_time=float(gp("planning_time")),
            asynchronous=True) if use_planner else None
        self._n_plan_events = 0
        self.tasks = None
        if self.scenario == "shared":
            self.tasks = {s: TaskSequence([(SHARED_POINT, 1.0), (SCENARIOS["home"][s], 0.0)])
                          for s in ("left", "right")}
        self.d_self = np.inf
        self._n_events = 0
        self.memory = ObstacleMemory(voxel=self.voxel) if bool(gp("use_memory")) else None

        self.jnames = ARM_JOINT_NAMES["left"] + ARM_JOINT_NAMES["right"]
        self.q_meas = None      # último /joint_states (14)
        self.q_ref = None       # referencia integrada que se comanda
        self.t = 0.0

        # ===== Cámara =====
        self.K = None
        self.depth_msg = None
        self.T_torso_cam = None
        self.obstacles = np.zeros((0, 3))
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ===== ROS =====
        self.pub_cmd = self.create_publisher(JointState, "/joint_cmd", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/bimanual/markers", 10)
        self.pub_obst = self.create_publisher(Marker, "/bimanual/obstacles", 10)
        self.pub_dist = self.create_publisher(Float32MultiArray, "/bimanual/min_distance", 10)
        self.pub_coord = self.create_publisher(String, "/bimanual/coordinator", 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        if self.use_camera:
            self.create_subscription(Image, "/camera/depth/image_raw", self._on_depth,
                                     qos_profile_sensor_data)
            self.create_subscription(CameraInfo, "/camera/depth/camera_info", self._on_info,
                                     qos_profile_sensor_data)
        # ===== Visual servoing (scenario:=grasp) =====
        self._ff = None
        if self.grasp_mode:
            self.fruit_exclude_radius = float(gp("fruit_exclude_radius"))
            self.rearm_after_missed = float(gp("rearm_after_missed"))
            self.tracker = FruitTracker()
            self.fsm = GraspServoing(GRASP_HOME)
            self._fsm_phase = self.fsm.phase
            self._t_phase_wall = None
            self.hand_names = {s: GG.hand_actuator_names(s) for s in ("left", "right")}
            self.pub_hand = self.create_publisher(JointState, gp("hand_cmd_topic"), 10)
            self.pub_vs = self.create_publisher(String, "/visual_servoing/state", 10)
            self.create_subscription(PointStamped, gp("target_topic"), self._on_target, 10)
        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info(
            f"QP bimanual iniciado: escenario '{self.scenario}', autocolisión="
            f"{self.ctrl.avoid_self}, cámara={self.use_camera}, memoria={self.memory is not None}, "
            f"capa 2={self.coord is not None}, capa 3={self.planner is not None}"
            f"{' (desactivada en grasp)' if self.grasp_mode else ''}. "
            f"Esperando /joint_states...")

    # ------------------------------------------------------------ callbacks
    def _on_joint_states(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(n in idx for n in self.jnames):
            return
        self.q_meas = np.array([msg.position[idx[n]] for n in self.jnames])

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_target(self, msg):
        """Centro de la fruta de la percepción -> torso_link -> Kalman, con el
        instante de CAPTURA (header.stamp), no el de llegada: así la latencia
        de la percepción queda compensada por la predicción."""
        pt = msg
        if msg.header.frame_id and msg.header.frame_id != FRAME:
            try:
                tf = self.tf_buffer.lookup_transform(FRAME, msg.header.frame_id, Time())
            except tf2_ros.TransformException as ex:
                self.get_logger().warn(f"Sin TF {FRAME} <- {msg.header.frame_id}: {ex}",
                                       throttle_duration_sec=2.0)
                return
            pt = do_transform_point(msg, tf)
        t_cap = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if t_cap <= 0.0:
            t_cap = self._now()
        z = np.array([pt.point.x, pt.point.y, pt.point.z])
        self.fsm.observe(t_cap, z)      # verificación del agarre (retiro)
        if not self.fsm.tracking():
            return      # la fruta ya está en la mano: sus medidas no son la faja
        if not self.tracker.update(z, t_cap):
            self.get_logger().warn(f"Medida de fruta descartada (salto > {self.tracker.gate} m)",
                                   throttle_duration_sec=2.0)

    def _on_info(self, msg):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_depth(self, msg):
        self.depth_msg = msg

    def _camera_pose(self, frame_id):
        if self.T_torso_cam is not None:
            return self.T_torso_cam
        try:
            tf = self.tf_buffer.lookup_transform(FRAME, frame_id, Time())
            tr, rq = tf.transform.translation, tf.transform.rotation
            x, y, z, w = rq.x, rq.y, rq.z, rq.w
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tr.x, tr.y, tr.z]
            self.get_logger().info(f"Pose de cámara tomada de TF {FRAME} -> {frame_id}")
        except tf2_ros.TransformException:
            T = camera_pose_from_mjcf_xyaxes()
            self.get_logger().warn(f"Sin TF {FRAME} -> {frame_id}: uso la pose del MJCF")
        # La cámara va fija al torso: basta con leerla una vez.
        self.T_torso_cam = T
        return T

    def _process_depth(self):
        msg, self.depth_msg = self.depth_msg, None
        if msg is None or self.K is None or msg.encoding != "32FC1":
            return
        depth = image_to_depth(msg)
        T = self._camera_pose(msg.header.frame_id)
        # q de la imagen: la medida más reciente (la cámara va a ~7 Hz; el
        # desfase lo cubre self_padding del auto-filtrado).
        q = self.q_meas
        excl = None
        if self.grasp_mode:
            est, _ = self.tracker.predict(self._now())
            if est is not None:
                excl = [est]
        pts = obstacles_from_depth(
            depth, self.K, T, arm_frames("left", q[:7]), arm_frames("right", q[7:]),
            voxel=self.voxel, self_padding=self.self_padding,
            exclude_centers=excl, exclude_radius=self.fruit_exclude_radius if excl else 0.0)
        if self.memory is not None:
            pts = self.memory.update(pts, depth, self.K, T, self.t)
        if excl is not None and len(pts):
            # También lo que la memoria recuerda de antes (la fruta se mueve).
            pts = pts[np.linalg.norm(pts - excl[0], axis=1) > self.fruit_exclude_radius]
        self.obstacles = pts

    # ------------------------------------------------------------ control
    def _grasp_targets(self):
        now = self._now()
        est, vel = self.tracker.predict(now)
        out = self.fsm.step(now, self.q_meas, est, vel, self.tracker.n_updates)
        self._ff = out["x_dot_ff"]
        if out["phase"] != self._fsm_phase:
            self._on_phase_change(out, now)
        if self.fsm.phase in ("MISSED", "FAILED") and self.rearm_after_missed >= 0 and \
                now - self.fsm.t_phase > self.rearm_after_missed:
            self.get_logger().info("[grasp] rearmado: esperando la siguiente fruta")
            self.fsm.reset()
            self.tracker.reset()
            self._fsm_phase = self.fsm.phase
        hand = JointState()
        hand.header.stamp = self.get_clock().now().to_msg()
        hand.name = self.hand_names["left"] + self.hand_names["right"]
        hand.position = [float(v) for s in ("left", "right") for v in out["hand"][s]]
        self.pub_hand.publish(hand)
        self.pub_vs.publish(String(data=f"{out['phase']}" + (f" ({out['side']})" if out["side"] else "")))
        self._fruit_est = est
        return out["targets"]

    def _on_phase_change(self, out, now):
        ev = self.fsm.events
        msg = f"[grasp] {self._fsm_phase} -> {out['phase']}"
        if out["phase"] == "APPROACH":
            msg += f": brazo {out['side']}, fruta en y={self.tracker.predict(now)[0][1]:+.3f} m"
        elif out["phase"] == "CLOSE":
            msg += (f" ({ev['t_close'] - ev['t_start']:.1f} s desde el inicio, punto de agarre a "
                    f"{1e3 * ev['err_close']:.0f} mm)")
        elif out["phase"] == "DONE":
            msg += (f": agarre verificado (la cámara ve la fruta en la mano), "
                    f"ciclo {ev['t_done'] - ev['t_start']:.1f} s")
        elif out["phase"] == "FAILED":
            msg += f": {ev.get('why', '')} -> se abre la mano"
        elif out["phase"] == "MISSED":
            msg += f": {ev.get('why', 'la fruta salió de la ventana del brazo antes de cerrar')}"
        self.get_logger().info(msg)
        self._fsm_phase = out["phase"]

    def _targets(self):
        if self.t < self.t_home:
            return GRASP_HOME if self.grasp_mode else SCENARIOS["home"]
        if self.grasp_mode:
            return self._grasp_targets()
        if self.tasks is not None:
            return {s: self.tasks[s].goal() for s in self.tasks}
        if self.scenario != "circles":
            return SCENARIOS[self.scenario]
        tt = self.t - self.t_home
        out = {}
        for side in ("left", "right"):
            c = CIRCLE_CENTER[side]
            a = self.omega * tt + CIRCLE_PHASE[side]
            out[side] = [c[0], c[1] + CIRCLE_RADIUS * np.cos(a), c[2] + CIRCLE_RADIUS * np.sin(a), *Q_FWD]
        return out

    def update(self):
        if self.q_meas is None:
            return
        if self.q_ref is None:
            self.q_ref = self.q_meas.copy()
            self.get_logger().info("Estado inicial recibido: yendo a 'home'.")
        self.t += self.dt

        if self.use_camera:
            self._process_depth()

        goals = self._targets()
        plan_busy = self.planner is not None and self.planner.active
        if self.coord is not None and self.t >= self.t_home and not plan_busy:
            targets, task_err = self.coord.step(self.t, self.q_ref, goals, self.d_self)
            for t_ev, msg in self.coord.events[self._n_events:]:
                self.get_logger().info(f"[capa 2] {msg}")
            self._n_events = len(self.coord.events)
            self.pub_coord.publish(String(data=self.coord.state if self.coord.master is None else
                                          f"{self.coord.state}: pasa {self.coord.master}"))
        else:
            targets, task_err = goals, BimanualCoordinator.task_error(self.q_ref, goals)
        if self.tasks is not None and self.t >= self.t_home:
            for s in self.tasks:
                self.tasks[s].update(self.t, task_err[s])
            if all(x.done for x in self.tasks.values()) and not getattr(self, "_logged_done", False):
                self._logged_done = True
                self.get_logger().info(
                    f"Tarea compartida completada por ambos brazos en {self.t - self.t_home:.1f} s")
        joint_targets = None
        if self.planner is not None and self.t >= self.t_home:
            l2_busy = self.coord is not None and self.coord.state == "yield"
            joint_targets = self.planner.step(
                self.t, self.q_ref, goals, task_err,
                self.obstacles if self.use_camera else None, l2_busy=l2_busy)
            for t_ev, msg in self.planner.events[self._n_plan_events:]:
                self.get_logger().info(f"[capa 3] {msg}")
            self._n_plan_events = len(self.planner.events)
            if self.planner.active:
                if self.coord is not None:
                    self.coord.reset()     # la capa 3 manda; la 2 recomienza después
                self.pub_coord.publish(String(data="capa 3: " + (
                    "planificando" if self.planner.plan is None else
                    f"siguiendo camino ({self.planner.plan['side']})")))
        dq, diag = self.ctrl.step(self.q_ref, targets,
                                  self.obstacles if self.use_camera else None,
                                  logger=self.get_logger(), joint_targets=joint_targets,
                                  x_dot_ff=self._ff if self.grasp_mode and self.t >= self.t_home else None)
        self.q_ref = self.q_ref + dq * self.dt
        self.d_self = diag["d_arms_min"]

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.jnames
        cmd.position = self.q_ref.tolist()
        cmd.velocity = dq.tolist()
        self.pub_cmd.publish(cmd)

        dist = Float32MultiArray()
        dist.data = [float(diag["d_arms_min"]), float(min(diag["d_env_min"], 9.9))]
        self.pub_dist.publish(dist)
        self._publish_markers(targets, diag)
        self.get_logger().info(
            f"t={self.t:5.1f}s  d_brazos={1000 * diag['d_arms_min']:5.0f}mm  "
            f"d_entorno={1000 * min(diag['d_env_min'], 9.9):5.0f}mm  "
            f"restricciones={diag['n_constraints']:2d}  obst={len(self.obstacles)}  "
            f"err=({1000 * diag['err']['left'][0]:.0f},{1000 * diag['err']['right'][0]:.0f})mm",
            throttle_duration_sec=1.0)

    # ------------------------------------------------------------ RViz
    def _capsule_markers(self, stamp, diag):
        """Cápsulas en torso_link: verde lejos, amarillo dentro de d_i,
        rojo por debajo de d_s (de la distancia mínima de cada cápsula)."""
        dmin = {}
        for name, d in diag["self"]:
            for part in name.split("-"):
                dmin[part] = min(dmin.get(part, np.inf), d)
        for name, d, _ in diag["env"]:
            part = name.split("-")[0]
            dmin[part] = min(dmin.get(part, np.inf), d)

        caps = [(f"{s[0].upper()}.{c['name']}", c) for s, q in
                (("left", self.q_ref[:7]), ("right", self.q_ref[7:])) for c in arm_capsules(s, q)]
        _, ta, tb, tr = TORSO_CAPSULE
        caps.append(("torso", {"a": ta, "b": tb, "r": tr}))

        out = []
        for i, (name, c) in enumerate(caps):
            d = dmin.get(name, np.inf)
            color = (0.9, 0.1, 0.1) if d < self.ctrl.d_s else \
                    (1.0, 0.8, 0.0) if d < self.ctrl.d_i else (0.1, 0.8, 0.2)
            m = Marker()
            m.header.frame_id = FRAME
            m.header.stamp = stamp
            m.ns = "capsules"
            m.id = i
            m.type = Marker.LINE_LIST
            m.scale.x = 2 * c["r"]   # grosor de línea ~ diámetro de la cápsula
            m.color.r, m.color.g, m.color.b, m.color.a = *color, 0.35
            m.pose.orientation.w = 1.0
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in (c["a"], c["b"])]
            out.append(m)
        return out

    def _publish_markers(self, targets, diag):
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        arr.markers = self._capsule_markers(stamp, diag)
        for i, side in enumerate(("left", "right")):
            m = Marker()
            m.header.frame_id = FRAME
            m.header.stamp = stamp
            m.ns = "targets"
            m.id = i
            m.type = Marker.SPHERE
            m.scale.x = m.scale.y = m.scale.z = 0.04
            m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.0, 0.0, 0.8) if i == 0 else (0.0, 0.0, 1.0, 0.8)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, targets[side][:3])
            m.pose.orientation.w = 1.0
            arr.markers.append(m)
        est = getattr(self, "_fruit_est", None)
        if est is not None:
            m = Marker()
            m.header.frame_id = FRAME
            m.header.stamp = stamp
            m.ns = "fruit"
            m.id = 0
            m.type = Marker.SPHERE
            m.scale.x = m.scale.y = m.scale.z = 0.08
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.5
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, est)
            m.pose.orientation.w = 1.0
            arr.markers.append(m)
        self.pub_markers.publish(arr)

        if self.use_camera:
            m = Marker()
            m.header.frame_id = FRAME
            m.header.stamp = stamp
            m.ns = "obstacles"
            m.type = Marker.CUBE_LIST
            m.scale.x = m.scale.y = m.scale.z = self.voxel
            m.color.r, m.color.g, m.color.b, m.color.a = 0.6, 0.2, 0.9, 0.6
            m.pose.orientation.w = 1.0
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.obstacles]
            self.pub_obst.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = BimanualAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
