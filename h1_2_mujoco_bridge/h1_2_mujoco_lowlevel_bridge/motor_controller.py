"""
motor_controller.py — Ley de control de motor idéntica a la del H1-2 real.

Para cada articulación i:

    tau_i = tau_ff_i + kp_i * (q_des_i - q_i) + kd_i * (dq_des_i - dq_i)

con las saturaciones de safety_limits (tau_max y limitación de delta_tau por
sub-paso). El torque resultante se escribe en data.ctrl del actuador <motor>
correspondiente (torque puro, gear=1), por lo que el PD se aplica UNA sola vez
(aquí) y MuJoCo no añade ningún control adicional.

Modos (motor_cmd[i].mode):
    0 -> motor deshabilitado: el torque objetivo es 0. La transición al cero es
         SUAVE porque pasa por el rate-limiter de delta_tau (no hay saltos).
    1 -> PD + feedforward habilitado.
    otros -> el lowcmd_handler los reporta con warning y los trata como 0.

El compute() se ejecuta en CADA sub-paso de MuJoCo (p.ej. 1 kHz) releyendo
q/dq actuales, emulando el servo interno del motor que corre más rápido que
la frecuencia de comandos (500 Hz). La consigna (q_des, kp, ...) se mantiene
constante entre mensajes /lowcmd.

Thread-safety: set_command()/disable_all()/set_hold() llegan desde el hilo de
ROS y compute() desde el hilo de simulación; un Lock protege el estado.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from .safety_limits import SafetyLimits


class MotorController:

    def __init__(self, limits: SafetyLimits, dt_substep: float):
        self.limits = limits
        self.n = limits.n
        self.dt = float(dt_substep)

        self._lock = threading.Lock()

        # Consigna vigente (ya saneada/saturada por lowcmd_handler)
        self.mode = np.zeros(self.n, dtype=int)     # arranque: todo deshabilitado
        self.q_des = np.zeros(self.n)
        self.dq_des = np.zeros(self.n)
        self.tau_ff = np.zeros(self.n)
        self.kp = np.zeros(self.n)
        self.kd = np.zeros(self.n)

        # Estado del rate-limiter (último torque realmente aplicado)
        self._tau_applied = np.zeros(self.n)

    # ------------------------------------------------------------------ setters
    def set_command(self, mode, q_des, dq_des, tau_ff, kp, kd) -> None:
        """Instala una consigna nueva (arrays de tamaño n, ya saturados)."""
        with self._lock:
            self.mode = np.asarray(mode, dtype=int).copy()
            self.q_des = np.asarray(q_des, dtype=float).copy()
            self.dq_des = np.asarray(dq_des, dtype=float).copy()
            self.tau_ff = np.asarray(tau_ff, dtype=float).copy()
            self.kp = np.asarray(kp, dtype=float).copy()
            self.kd = np.asarray(kd, dtype=float).copy()

    def disable_all(self) -> None:
        """Watchdog zero_torque: torque objetivo 0 en todos los motores.
        La bajada es suave gracias al limitador de delta_tau."""
        with self._lock:
            self.mode[:] = 0

    def set_hold(self, q_hold: np.ndarray, hold_kp: float, hold_kd: float) -> None:
        """Watchdog hold_position: mantiene la posición capturada con ganancias
        de mantenimiento (saturadas a los límites por-joint)."""
        with self._lock:
            self.mode[:] = 1
            self.q_des = self.limits.clamp_q(np.asarray(q_hold, dtype=float).copy())
            self.dq_des[:] = 0.0
            self.tau_ff[:] = 0.0
            self.kp = self.limits.clamp_kp(np.full(self.n, float(hold_kp)))
            self.kd = self.limits.clamp_kd(np.full(self.n, float(hold_kd)))

    def get_mode(self) -> np.ndarray:
        with self._lock:
            return self.mode.copy()

    # ------------------------------------------------------------------ compute
    def compute(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Calcula el torque a aplicar en ESTE sub-paso.
        q, dq: estado actual de MuJoCo (orden = índice Unitree).
        Devuelve tau [n] tras saturación absoluta y de tasa de cambio.
        """
        with self._lock:
            tau = self.tau_ff + self.kp * (self.q_des - q) + self.kd * (self.dq_des - dq)
            tau = np.where(self.mode == 1, tau, 0.0)

        # Rechazo defensivo de no-finitos (no debería llegar: el handler sanea)
        tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)

        # Saturación absoluta
        tau = self.limits.clamp_tau(tau)

        # Limitación de variación entre ciclos: |delta_tau| <= tau_rate_max * dt
        max_step = self.limits.tau_rate_max * self.dt
        dtau = np.clip(tau - self._tau_applied, -max_step, max_step)
        self._tau_applied = self._tau_applied + dtau

        return self._tau_applied.copy()

    def reset_rate_limiter(self, tau: Optional[np.ndarray] = None) -> None:
        self._tau_applied = (np.zeros(self.n) if tau is None
                             else np.asarray(tau, dtype=float).copy())
