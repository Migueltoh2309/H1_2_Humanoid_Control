"""
mujoco_simulator.py — Envoltura de la simulación MuJoCo del H1-2.

Responsabilidades:
  * Cargar el MJCF y aplicar overrides EN MEMORIA (el XML no se toca):
      - timestep del integrador (mujoco_timestep del YAML),
      - armature/damping en los 27 dof mapeados (ver nota de estabilidad).
  * Avanzar la simulación por "ticks de control": cada tick ejecuta n_sub
    sub-pasos de mj_step; en CADA sub-paso se recalcula el PD con el estado
    actual (emula el servo del motor a mayor frecuencia que /lowcmd).
  * Producir el snapshot del estado (q, dq, ddq, tau aplicado, IMU, tiempo)
    en orden de índice Unitree, para construir LowState.
  * Viewer pasivo opcional (nunca marca el ritmo de la simulación).

NOTA DE ESTABILIDAD (por qué armature/damping por defecto)
----------------------------------------------------------
Los joints del MJCF vienen con armature=0 y damping=0, y las muñecas tienen
inercia efectiva minúscula (~6e-4 kg·m²). Con integración Euler explícita el
término kd del PD solo es numéricamente estable si  kd·dt/M ≲ 1; con kd=1 de
los scripts de referencia y dt=1 ms se obtiene kd·dt/M≈1.6 en wrist_roll →
chattering a frecuencia de Nyquist ("movimiento raro" de muñecas). Añadir la
inercia reflejada del rotor (armature≈0.01, presente en los modelos oficiales
de Unitree pero ausente en este XML) sube M a ~0.0106 y deja kd·dt/M≈0.09,
estable y bien amortiguado. Por eso joint_armature=0.01 y joint_damping=0.05
son el default (configurable / desactivable con 0.0).

tau_est reportado = data.actuator_force (torque realmente aplicado por el
actuador tras las saturaciones del bridge; con <motor> gear=1 coincide con
el ctrl escrito).
"""
from __future__ import annotations

import threading
from typing import Optional

import mujoco
import numpy as np

from .joint_mapping import JointMapping
from .motor_controller import MotorController


class MujocoSimulator:

    def __init__(
        self,
        model_path: str,
        mapping_builder,                 # callable(model) -> JointMapping
        timestep: Optional[float] = None,
        joint_armature: float = 0.0,
        joint_damping: float = 0.0,
        base_height_offset: float = 0.0,
        base_body: str = "pelvis",
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)

        if timestep is not None and timestep > 0.0:
            self.model.opt.timestep = float(timestep)

        # Elevación opcional de la base soldada (en memoria). Con la escena tal
        # cual el robot está DE PIE (pies en contacto con el suelo, ~460 N de
        # normal media medida); con +0.30 m queda "colgado" como en el pórtico
        # del que Unitree suspende el robot en los tests de bajo nivel del SDK
        # (tobillos libres). Default 0.0 = escena original.
        self.base_height_offset = float(base_height_offset)
        if self.base_height_offset != 0.0:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body)
            if bid < 0:
                raise RuntimeError(
                    f"base_height_offset: el body '{base_body}' no existe en el modelo"
                )
            self.model.body_pos[bid][2] += self.base_height_offset

        # El mapeo se construye sobre el modelo YA cargado (validación dura).
        self.mapping: JointMapping = mapping_builder(self.model)

        # Overrides de estabilidad en los dof mapeados (en memoria, no en XML)
        self.joint_armature = float(joint_armature)
        self.joint_damping = float(joint_damping)
        if self.joint_armature > 0.0 or self.joint_damping > 0.0:
            for dof in self.mapping.qvel_adr:
                if self.joint_armature > 0.0:
                    self.model.dof_armature[dof] = self.joint_armature
                if self.joint_damping > 0.0:
                    self.model.dof_damping[dof] = self.joint_damping

        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.dt = float(self.model.opt.timestep)

        # Estado auxiliar para ddq (diferencia finita por tick de control)
        self._prev_dq = np.zeros(len(self.mapping))
        self._ddq = np.zeros(len(self.mapping))

        # Viewer pasivo (opcional)
        self._viewer = None
        self._viewer_lock = threading.Lock()

    # ------------------------------------------------------------------ estado
    def get_q_dq(self):
        q = self.data.qpos[self.mapping.qpos_adr].copy()
        dq = self.data.qvel[self.mapping.qvel_adr].copy()
        return q, dq

    # ------------------------------------------------------------------ paso
    def step_control_tick(self, controller: MotorController, n_sub: int) -> None:
        """
        Un tick de control = n_sub sub-pasos de física. El PD se recalcula en
        cada sub-paso con el estado más reciente; la consigna del controller
        permanece constante durante el tick.
        """
        m, d = self.model, self.data
        qadr, dadr, aid = (self.mapping.qpos_adr, self.mapping.qvel_adr,
                           self.mapping.act_id)

        dq_start = d.qvel[dadr].copy()

        for _ in range(n_sub):
            q = d.qpos[qadr]
            dq = d.qvel[dadr]
            tau = controller.compute(q, dq)
            d.ctrl[aid] = tau
            mujoco.mj_step(m, d)

        dq_end = d.qvel[dadr].copy()
        tick_dt = self.dt * n_sub
        self._ddq = (dq_end - self._prev_dq) / max(tick_dt, 1e-9)
        self._prev_dq = dq_end

    # ------------------------------------------------------------------ snapshot
    def snapshot(self, controller: MotorController) -> dict:
        """Estado en orden de índice Unitree para construir LowState."""
        q, dq = self.get_q_dq()
        return {
            "q": q,
            "dq": dq,
            "ddq": self._ddq.copy(),
            # Torque aplicado por el actuador (== ctrl saturado con gear=1)
            "tau_est": self.data.actuator_force[self.mapping.act_id].copy(),
            "mode": controller.get_mode(),
            "sim_time": float(self.data.time),
        }

    # ------------------------------------------------------------------ viewer
    def launch_viewer(self) -> bool:
        """Viewer pasivo. Devuelve False si no hay display disponible."""
        try:
            import mujoco.viewer  # import perezoso: falla limpio sin GUI
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
