"""
mujoco_sim.py — Envoltura de la simulación MuJoCo para el bridge de
simulación general (física + colisiones + cámara RGB-D + contactos).

A diferencia de h1_2_mujoco_lowlevel_bridge/mujoco_simulator.py (que separa
"tick de control" en n sub-pasos de física para emular la cadencia de
/lowcmd del robot real), aquí no hay protocolo de hardware que preservar: la
física avanza un `mj_step` por iteración del hilo de simulación y el PD se
recalcula en cada paso con el estado más reciente. Se reutiliza la MISMA
razón de estabilidad numérica documentada en el bridge de bajo nivel
(overrides de armature/damping — ver ese README) porque es el mismo modelo.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import mujoco
import numpy as np

from .joint_mapping import JointMapping
from .pd_controller import JointPD


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Producto de Hamilton, convención (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


# 180° respecto al eje X local: convierte la convención de cámara de MuJoCo
# (X derecha, Y arriba, mira hacia -Z) a la convención "optical frame" de ROS
# / REP-103 (X derecha, Y abajo, Z adelante). Verificado numéricamente contra
# data.cam_xmat/data.xmat (ver notas de diseño): error < 1e-15.
_ROS_OPTICAL_FLIP_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])


class Contact:
    __slots__ = ("pos", "normal_force", "body1", "body2")

    def __init__(self, pos: np.ndarray, normal_force: float, body1: str, body2: str):
        self.pos = pos
        self.normal_force = normal_force
        self.body1 = body1
        self.body2 = body2


class MujocoSim:

    def __init__(
        self,
        model_path: str,
        joint_names: List[str],
        timestep: Optional[float] = None,
        joint_armature: float = 0.0,
        joint_damping: float = 0.0,
        base_body: str = "pelvis",
        camera_name: Optional[str] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        initial_keyframe: Optional[str] = None,
        gravity_comp: bool = True,
    ):
        from .joint_mapping import build_joint_mapping

        # Compensación de gravedad en el lazo del motor, como el H1-2 real.
        self.gravity_comp = bool(gravity_comp)

        self.model = mujoco.MjModel.from_xml_path(model_path)

        if timestep is not None and timestep > 0.0:
            self.model.opt.timestep = float(timestep)

        self.mapping: JointMapping = build_joint_mapping(self.model, joint_names)

        self.joint_armature = float(joint_armature)
        self.joint_damping = float(joint_damping)
        if self.joint_armature > 0.0 or self.joint_damping > 0.0:
            for dof in self.mapping.qvel_adr:
                if self.joint_armature > 0.0:
                    self.model.dof_armature[dof] = self.joint_armature
                if self.joint_damping > 0.0:
                    self.model.dof_damping[dof] = self.joint_damping

        self.data = mujoco.MjData(self.model)
        # MjData auxiliar reutilizado por gravity_torque() (no se asigna por
        # paso: MjData es caro de construir) y buffer de salida del RNE.
        self._grav_data = mujoco.MjData(self.model)
        self._grav_out = np.zeros(self.model.nv)
        if initial_keyframe:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, initial_keyframe)
            if key_id < 0:
                raise RuntimeError(f"initial_keyframe '{initial_keyframe}' no existe en el modelo")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)
        self.dt = float(self.model.opt.timestep)

        # Protege self.data contra la única combinación insegura: mj_step
        # (hilo de física) escribiendo mientras Renderer.update_scene (hilo
        # de cámara) lee. Renderer.render() en sí no toca self.data (trabaja
        # sobre la MjvScene ya construida) y es la parte lenta (~15-30 ms),
        # así que corre SIN el lock — no bloquea la física mientras renderiza.
        self._data_lock = threading.Lock()

        # Origen del frame TF raíz (base_body, p.ej. "pelvis") en coordenadas
        # de mundo de MuJoCo: como robot_state_publisher no publica un frame
        # "world" (el URDF no tiene floating_base_joint activo), todo lo que
        # se exprese en coordenadas de mundo de MuJoCo (p.ej. contactos) hay
        # que trasladarlo a este origen antes de publicarlo con frame_id
        # = base_body.
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        if self.base_body_id < 0:
            raise RuntimeError(f"base_body '{base_body}' no existe en el modelo")
        self.base_body_name = base_body
        self.world_offset = self.data.xpos[self.base_body_id].copy()

        # ── Cámara (opcional) ────────────────────────────────────────────
        self.camera_name = camera_name
        self.camera_id = -1
        self.camera_body_name = None
        self._renderer: Optional["mujoco.Renderer"] = None
        self._renderer_depth_mode = False
        self.camera_width = camera_width
        self.camera_height = camera_height
        if camera_name:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cid < 0:
                raise RuntimeError(f"camera_name '{camera_name}' no existe en el modelo")
            self.camera_id = cid
            body_id = int(self.model.cam_bodyid[cid])
            self.camera_body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            res = getattr(self.model, "cam_resolution", None)
            if res is not None and int(res[cid][0]) > 0 and int(res[cid][1]) > 0:
                self.camera_width, self.camera_height = int(res[cid][0]), int(res[cid][1])
            self.camera_fovy = float(self.model.cam_fovy[cid])

        # Viewer pasivo (opcional)
        self._viewer = None
        self._viewer_lock = threading.Lock()

        # ── Actuadores de posición auxiliares (p.ej. los 6 de cada mano
        # Inspire en h1_2_scene_surgery_table_hands.xml) ──────────────────
        # No pasan por el PD del bridge: son servos de posición del propio
        # MJCF (como el control de posición de la mano real), así que un
        # /joint_cmd con su nombre fija directamente su ctrl.
        mapped = set(int(a) for a in self.mapping.act_id)
        self.aux_actuators = {}
        for a in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if a in mapped or not name:
                continue
            if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and \
                    self.model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE:
                self.aux_actuators[name] = a

        # ── Asistencia de agarre (solo simulación, ver enable_grasp_assist) ──
        self._assist = {}

    # ------------------------------------------------------ verdad de terreno
    def body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def body_pose_in_base(self, bid: int):
        """(pos, quat wxyz) del cuerpo, relativo al origen de base_body
        (orientación de mundo: la base está soldada sin rotación)."""
        with self._data_lock:
            pos = self.data.xpos[bid] - self.world_offset
            quat = self.data.xquat[bid].copy()
        return pos, quat

    # --------------------------------------------------- actuadores auxiliares
    def set_aux_ctrl(self, names, values) -> int:
        """Fija el ctrl de los actuadores auxiliares nombrados (saturado a su
        ctrlrange). Devuelve cuántos se actualizaron."""
        n = 0
        for name, v in zip(names, values):
            a = self.aux_actuators.get(name)
            if a is None:
                continue
            lo, hi = self.model.actuator_ctrlrange[a]
            if self.model.actuator_ctrllimited[a]:
                v = min(max(float(v), lo), hi)
            self.data.ctrl[a] = v
            n += 1
        return n

    def enable_grasp_assist(self, object_body: str = "mandarina",
                            close_joint: str = "index_proximal_joint",
                            close_on: float = 0.6, release_below: float = 0.3) -> list:
        """Asistencia de agarre, como el "assistive grasping" de iGibson 2.0
        (ver h1_2_algoritms/REFERENCIAS_VISUAL_SERVOING.md [30]): un weld
        palma-fruta (`{side}_grasp_assist` en el MJCF, inactivo por defecto)
        que se ACTIVA solo cuando la mano está cerrando (ctrl del índice >
        close_on) y la fruta ya toca >= 2 grupos distintos de esa mano
        (pulgar, índice, medio, anular, meñique, palma). Se suelta al abrir
        (ctrl < release_below). Motivo: el contacto rígido esfera-malla no
        sostiene la fruta al levantarla (VISUAL_SERVOING_PLAN.md §4); con el
        requisito de contacto real, una mano mal posicionada sigue fallando.
        Devuelve los lados habilitados."""
        m = self.model
        obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if obj < 0:
            return []
        out = []
        for side, pre in (("left", "L"), ("right", "R")):
            eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{side}_grasp_assist")
            act = self.aux_actuators.get(f"{pre}_{close_joint}")
            wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
            if eq < 0 or act is None or wrist < 0:
                continue
            group = {}
            for b in range(m.nbody):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if b == wrist:
                    group[b] = "palm"
                elif name.startswith(pre + "_"):
                    for g in ("thumb", "index", "middle", "ring", "pinky"):
                        if g in name:
                            group[b] = g
            self._assist[side] = {"eq": eq, "act": act, "wrist": wrist, "obj": obj,
                                  "group": group, "on": close_on, "off": release_below}
            out.append(side)
        return out

    def _update_grasp_assist(self) -> None:
        m, d = self.model, self.data
        for side, a in self._assist.items():
            closing = d.ctrl[a["act"]]
            if d.eq_active[a["eq"]]:
                if closing < a["off"]:
                    d.eq_active[a["eq"]] = 0
                continue
            if closing < a["on"]:
                continue
            touched = set()
            for i in range(d.ncon):
                c = d.contact[i]
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if a["obj"] == b1 and b2 in a["group"]:
                    touched.add(a["group"][b2])
                elif a["obj"] == b2 and b1 in a["group"]:
                    touched.add(a["group"][b1])
            if len(touched) < 2:
                continue
            # Weld con la pose relativa ACTUAL (no la de qpos0).
            w, o = a["wrist"], a["obj"]
            R1 = d.xmat[w].reshape(3, 3)
            q1i, rq = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(q1i, d.xquat[w])
            mujoco.mju_mulQuat(rq, q1i, d.xquat[o])
            m.eq_data[a["eq"]][0:3] = 0.0
            m.eq_data[a["eq"]][3:6] = R1.T @ (d.xpos[o] - d.xpos[w])
            m.eq_data[a["eq"]][6:10] = rq
            m.eq_data[a["eq"]][10] = 1.0
            d.eq_active[a["eq"]] = 1

    # ------------------------------------------------------------------ estado
    def get_q_dq(self):
        q = self.data.qpos[self.mapping.qpos_adr].copy()
        dq = self.data.qvel[self.mapping.qvel_adr].copy()
        return q, dq

    def gravity_torque(self) -> np.ndarray:
        """Par de gravedad g(q) en los joints controlados, en el estado actual.

        Es la compensación de gravedad que llevan los motores del H1-2 real:
        el par que hay que inyectar para que el brazo se sostenga solo y el PD
        no tenga que generar error para hacerlo (un PD puro solo equilibra la
        gravedad con kp·Δq ≠ 0 — ver README).

        NO se usa `data.qfrc_bias` aunque mj_step ya lo tenga calculado: ese
        término es gravedad + Coriolis/centrífuga, y lo que compensan los
        motores es gravedad. Se recalcula con RNE sobre un MjData auxiliar con
        qvel = 0, que es justo g(q). Medido en la escena vacía: la diferencia
        entre ambos es 0.75% de g durante el movimiento, y el RNE cuesta 12 us
        contra los 174 us de un mj_step (~7% extra, sobre un presupuesto de
        2000 us por paso a 500 Hz).
        """
        self._grav_data.qpos[:] = self.data.qpos
        self._grav_data.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, self._grav_data)
        mujoco.mj_comPos(self.model, self._grav_data)
        mujoco.mj_rne(self.model, self._grav_data, 0, self._grav_out)
        return self._grav_out[self.mapping.qvel_adr].copy()

    def step(self, controller: JointPD) -> None:
        q, dq = self.get_q_dq()
        tau_ext = self.gravity_torque() if self.gravity_comp else None
        tau = controller.compute(q, dq, tau_ext)
        self.data.ctrl[self.mapping.act_id] = tau
        with self._data_lock:
            mujoco.mj_step(self.model, self.data)
            if self._assist:
                self._update_grasp_assist()

    def snapshot(self, controller: JointPD) -> dict:
        q, dq = self.get_q_dq()
        return {
            "q": q,
            "dq": dq,
            "tau_est": self.data.actuator_force[self.mapping.act_id].copy(),
            "sim_time": float(self.data.time),
        }

    # ---------------------------------------------------------------- contactos
    def get_contacts(self, min_force: float = 1e-3) -> List[Contact]:
        out: List[Contact] = []
        force6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            mujoco.mj_contactForce(self.model, self.data, i, force6)
            normal_force = float(abs(force6[0]))
            if normal_force < min_force:
                continue
            pos = np.array(c.pos) - self.world_offset
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    int(self.model.geom_bodyid[c.geom1]))
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    int(self.model.geom_bodyid[c.geom2]))
            out.append(Contact(pos, normal_force, b1 or "?", b2 or "?"))
        return out

    # ------------------------------------------------------------------ cámara
    def camera_optical_pose_in_body(self):
        """(pos_xyz, quat_xyzw) del frame óptico ROS de la cámara, relativo a
        camera_body_name (p.ej. torso_link) — para un tf_static."""
        cid = self.camera_id
        pos = self.model.cam_pos[cid].copy()
        quat_wxyz = _quat_mul(self.model.cam_quat[cid], _ROS_OPTICAL_FLIP_WXYZ)
        w, x, y, z = quat_wxyz
        return pos, np.array([x, y, z, w])

    def _ensure_renderer(self) -> "mujoco.Renderer":
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.camera_height,
                                              width=self.camera_width)
        return self._renderer

    def render_rgb(self) -> np.ndarray:
        r = self._ensure_renderer()
        if self._renderer_depth_mode:  # volver a modo color si quedó en profundidad
            r.disable_depth_rendering()
            self._renderer_depth_mode = False
        with self._data_lock:
            r.update_scene(self.data, camera=self.camera_id)
        return r.render()

    def render_depth(self) -> np.ndarray:
        r = self._ensure_renderer()
        if not self._renderer_depth_mode:
            r.enable_depth_rendering()
            self._renderer_depth_mode = True
        with self._data_lock:
            r.update_scene(self.data, camera=self.camera_id)
        return r.render()

    def warmup_renderer(self) -> None:
        """Fuerza la compilación de shaders / setup de contexto GL de una vez
        (puede tardar >1s la primera vez) para que no le pase esa demora al
        primer frame publicado desde el hilo de cámara."""
        self.render_rgb()
        self.render_depth()

    # ------------------------------------------------------------------ viewer
    def launch_viewer(self) -> bool:
        try:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            return True
        except Exception:
            self._viewer = None
            return False

    def sync_viewer(self) -> None:
        v = self._viewer
        if v is None:
            return
        with self._viewer_lock:
            if v.is_running():
                v.sync()
            else:
                self._viewer = None

    def close_viewer(self) -> None:
        v = self._viewer
        if v is not None:
            try:
                v.close()
            except Exception:
                pass
            self._viewer = None
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
