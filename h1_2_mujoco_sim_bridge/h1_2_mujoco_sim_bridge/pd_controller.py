"""
pd_controller.py — PD por joint para el bridge de simulación.

tau_i = g_i(q) + tau_ff_i + kp_i·(q_des_i − q_i) + kd_i·(dq_des_i − dq_i),
saturado a ±tau_max_i. El término g(q) (compensación de gravedad, igual que
los motores del H1-2 real) lo inyecta el bridge por `tau_ext`; sin él un PD
puro deja error estacionario, porque el par de sostén ES kp·Δq. Los objetivos (q_des/dq_des/tau_ff) se actualizan por NOMBRE desde
un sensor_msgs/JointState de comando: los joints que no aparecen en el
mensaje simplemente conservan su objetivo anterior (no hay "timeout" que
lleve a torque cero — a diferencia del robot real, aquí no hay riesgo de
hardware, así que "mantener la última posición pedida" es ya el
comportamiento seguro por defecto, sin lógica extra de watchdog).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from .joint_mapping import JointMapping


class JointPD:

    def __init__(
        self,
        mapping: JointMapping,
        kp: np.ndarray,
        kd: np.ndarray,
        tau_max: np.ndarray,
        q_init: np.ndarray,
        warn: Optional[Callable[[str], None]] = None,
    ):
        n = len(mapping)
        assert kp.shape == (n,) and kd.shape == (n,) and tau_max.shape == (n,)
        self.mapping = mapping
        self.kp = kp.astype(float)
        self.kd = kd.astype(float)
        self.tau_max = np.abs(tau_max.astype(float))
        self.q_des = np.clip(q_init.astype(float), mapping.q_min, mapping.q_max)
        self.dq_des = np.zeros(n)
        self.tau_ff = np.zeros(n)
        self._warn = warn or (lambda s: None)

    def set_targets_by_name(
        self,
        names: Sequence[str],
        position: Sequence[float] = (),
        velocity: Sequence[float] = (),
        effort: Sequence[float] = (),
    ) -> int:
        """Actualiza objetivos para los joints presentes en `names` que este
        bridge controla. `velocity`/`effort` son opcionales (0 si no se dan o
        no coinciden en longitud). Devuelve cuántos joints se actualizaron."""
        has_vel = len(velocity) == len(names)
        has_eff = len(effort) == len(names)
        updated = 0
        for k, name in enumerate(names):
            i = self.mapping.index.get(name)
            if i is None:
                self._warn(f"/joint_cmd: joint '{name}' no está en joint_names "
                            f"de este bridge; ignorado")
                continue
            if k < len(position):
                q = float(position[k])
                lo, hi = self.mapping.q_min[i], self.mapping.q_max[i]
                if q < lo or q > hi:
                    self._warn(f"/joint_cmd: '{name}' q_des={q:.3f} fuera de "
                                f"rango [{lo:.3f},{hi:.3f}]; saturado")
                self.q_des[i] = min(max(q, lo), hi)
            self.dq_des[i] = float(velocity[k]) if has_vel else 0.0
            self.tau_ff[i] = float(effort[k]) if has_eff else 0.0
            updated += 1
        return updated

    def compute(self, q: np.ndarray, dq: np.ndarray,
                tau_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """tau = tau_ext + tau_ff + kp·(q_des−q) + kd·(dq_des−dq), saturado.

        `tau_ext` es un par extra que pone el propio bridge, no el usuario: hoy
        la compensación de gravedad g(q) (ver MujocoSim.gravity_torque). Se suma
        ANTES de saturar, a propósito: si se sumara después, el resultado podría
        pasarse de tau_max y el motor entregaría un par que el robot real no
        puede dar. `tau_ff` (el campo `effort` de /joint_cmd) sigue siendo del
        usuario y se suma igual, así que ambos conviven.
        """
        tau = self.tau_ff + self.kp * (self.q_des - q) + self.kd * (self.dq_des - dq)
        if tau_ext is not None:
            tau = tau + tau_ext
        return np.clip(tau, -self.tau_max, self.tau_max)
