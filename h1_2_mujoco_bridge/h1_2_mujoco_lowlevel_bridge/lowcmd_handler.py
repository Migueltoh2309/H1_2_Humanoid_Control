"""
lowcmd_handler.py — Validación y conversión de unitree_hg/msg/LowCmd.

Pipeline por mensaje:
  1. (opcional) Verificación del CRC con la misma serialización del SDK.
     Si falla -> el mensaje COMPLETO se rechaza (la consigna anterior sigue
     vigente hasta que actúe el watchdog) y se emite un warning limitado
     por frecuencia.
  2. mode_pr: solo se soporta PR (0). Si llega AB (1) se avisa (throttled) y
     los valores se interpretan igualmente como espacio articular PR
     (simplificación documentada: este MJCF modela los tobillos como joints
     serie pitch/roll, no como el paralelogramo A/B real).
  3. Por motor (índices Unitree 0..26):
     - Valores no finitos (NaN/Inf) en q/dq/tau/kp/kd -> ese motor se
       deshabilita en este ciclo (mode 0) + warning throttled.
     - mode desconocido (ni 0 ni 1) -> warning throttled + tratar como 0.
       Nunca se ignora silenciosamente.
     - Saturaciones: q a [q_min,q_max], dq a ±dq_max, tau_ff a ±tau_max,
       kp a [0,kp_max], kd a [0,kd_max].
  4. Devuelve los arrays listos para MotorController.set_command().

Los índices 27..34 del mensaje (motores inexistentes en el H1-2) se ignoran.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional

import numpy as np

from . import crc as crc_mod
from .joint_mapping import JointMapping, NUM_CMD_MOTORS
from .safety_limits import SafetyLimits

MODE_DISABLED = 0
MODE_PD_FF = 1

MODE_PR = 0
MODE_AB = 1


class _Throttle:
    """Warning limitado por frecuencia (por clave)."""

    def __init__(self, period_s: float, warn: Callable[[str], None]):
        self.period = float(period_s)
        self.warn = warn
        self._last: Dict[str, float] = {}
        self._suppressed: Dict[str, int] = {}

    def __call__(self, key: str, text: str) -> None:
        now = time.monotonic()
        last = self._last.get(key, -1e9)
        if now - last >= self.period:
            n = self._suppressed.pop(key, 0)
            suffix = f" (+{n} similares suprimidos)" if n else ""
            self.warn(text + suffix)
            self._last[key] = now
        else:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1


class LowCmdHandler:

    def __init__(
        self,
        mapping: JointMapping,
        limits: SafetyLimits,
        verify_crc: bool = True,
        warn: Optional[Callable[[str], None]] = None,
        warn_period_s: float = 2.0,
    ):
        self.mapping = mapping
        self.limits = limits
        self.verify_crc = bool(verify_crc)
        self._throttle = _Throttle(warn_period_s, warn or (lambda s: None))

        # Estadísticas (útiles para tests y diagnóstico)
        self.n_accepted = 0
        self.n_rejected_crc = 0
        self.last_mode_pr = MODE_PR
        self.last_mode_machine = 0

    # ------------------------------------------------------------------
    def process(self, msg) -> Optional[Dict[str, np.ndarray]]:
        """
        Valida y convierte un LowCmd. Devuelve dict con arrays
        {mode,q_des,dq_des,tau_ff,kp,kd} en orden de índice Unitree,
        o None si el mensaje fue rechazado (CRC inválido).
        """
        # 1) CRC ---------------------------------------------------------------
        if self.verify_crc:
            if not crc_mod.verify_crc(msg):
                self.n_rejected_crc += 1
                self._throttle(
                    "crc",
                    f"LowCmd rechazado: CRC inválido (recibido {int(msg.crc)}, "
                    f"esperado {crc_mod.compute_crc(msg)}). "
                    f"Total rechazados: {self.n_rejected_crc}",
                )
                return None

        # 2) mode_pr -----------------------------------------------------------
        mode_pr = int(msg.mode_pr)
        if mode_pr == MODE_AB:
            self._throttle(
                "mode_ab",
                "mode_pr=AB recibido: la simulación solo modela tobillos serie "
                "PR (pitch/roll); los comandos se interpretan como PR.",
            )
        self.last_mode_pr = mode_pr
        self.last_mode_machine = int(msg.mode_machine)

        # 3) Extracción por motor ----------------------------------------------
        n = NUM_CMD_MOTORS
        mode = np.zeros(n, dtype=int)
        q = np.zeros(n)
        dq = np.zeros(n)
        tau = np.zeros(n)
        kp = np.zeros(n)
        kd = np.zeros(n)

        motor_cmd = msg.motor_cmd
        for i in range(n):
            mc = motor_cmd[i]
            vals = (float(mc.q), float(mc.dq), float(mc.tau),
                    float(mc.kp), float(mc.kd))

            if not all(math.isfinite(v) for v in vals):
                self._throttle(
                    f"nan_{i}",
                    f"Motor {i} ({self.mapping.joint_names[i]}): valores no "
                    f"finitos en el comando; motor deshabilitado este ciclo.",
                )
                mode[i] = MODE_DISABLED
                continue

            m = int(mc.mode)
            if m == MODE_PD_FF:
                mode[i] = MODE_PD_FF
            elif m == MODE_DISABLED:
                mode[i] = MODE_DISABLED
            else:
                self._throttle(
                    f"mode_{i}",
                    f"Motor {i} ({self.mapping.joint_names[i]}): mode={m} no "
                    f"soportado (solo 0/1); tratado como deshabilitado.",
                )
                mode[i] = MODE_DISABLED

            q[i], dq[i], tau[i], kp[i], kd[i] = vals

        # 4) Saturaciones (previas a aplicar cualquier comando) -----------------
        q = self.limits.clamp_q(q)
        dq = self.limits.clamp_dq(dq)
        tau = self.limits.clamp_tau(tau)
        kp = self.limits.clamp_kp(kp)
        kd = self.limits.clamp_kd(kd)

        self.n_accepted += 1
        return {
            "mode": mode, "q_des": q, "dq_des": dq,
            "tau_ff": tau, "kp": kp, "kd": kd,
        }
