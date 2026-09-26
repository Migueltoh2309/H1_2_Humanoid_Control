"""
watchdog.py — Vigilancia de la llegada de /lowcmd.

Si transcurren más de `timeout` segundos sin recibir un LowCmd válido:

  * behavior = "zero_torque":
      El torque objetivo de TODOS los motores pasa a 0 (mode=0 en el
      controlador). La bajada es SUAVE porque el MotorController limita
      |delta_tau| <= tau_rate_max·dt en cada sub-paso: el torque desciende en
      rampa, nunca de golpe, y jamás se mantiene indefinidamente el último
      torque aplicado.

  * behavior = "hold_position":
      Se captura la posición actual q en el instante del timeout y se aplica
      un PD de mantenimiento con ganancias hold_kp/hold_kd (saturadas a los
      límites por-joint).

Al volver a recibir un LowCmd válido el watchdog se rearma automáticamente y
la consigna del controlador vuelve a ser la del mensaje.

El watchdog pertenece exclusivamente a la simulación: permite estudiar cómo
reaccionaría el controlador ante una pérdida de comunicación, sin hardware.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .motor_controller import MotorController

BEHAVIOR_ZERO_TORQUE = "zero_torque"
BEHAVIOR_HOLD_POSITION = "hold_position"
_VALID = (BEHAVIOR_ZERO_TORQUE, BEHAVIOR_HOLD_POSITION)


class CommandWatchdog:

    def __init__(
        self,
        timeout: float,
        behavior: str = BEHAVIOR_ZERO_TORQUE,
        hold_kp: float = 60.0,
        hold_kd: float = 2.0,
        on_event: Optional[Callable[[str], None]] = None,
    ):
        if behavior not in _VALID:
            raise ValueError(
                f"timeout_behavior='{behavior}' inválido; opciones: {_VALID}"
            )
        self.timeout = float(timeout)
        self.behavior = behavior
        self.hold_kp = float(hold_kp)
        self.hold_kd = float(hold_kd)
        self._on_event = on_event or (lambda s: None)

        self._last_cmd_t: Optional[float] = None   # None = aún sin comandos
        self.engaged = False
        self.n_timeouts = 0

    # ------------------------------------------------------------------
    def notify_command(self, t: Optional[float] = None) -> None:
        """Llamar con cada LowCmd VÁLIDO aceptado."""
        self._last_cmd_t = time.monotonic() if t is None else t
        if self.engaged:
            self.engaged = False
            self._on_event("Watchdog rearmado: /lowcmd recibido de nuevo.")

    def check(
        self,
        controller: MotorController,
        current_q: np.ndarray,
        t: Optional[float] = None,
    ) -> bool:
        """
        Llamar en cada tick de control. Devuelve True si el watchdog está
        activado (timeout vigente).
        Antes del primer comando no se activa: el controlador arranca ya en
        modo deshabilitado (torque 0), que es el estado seguro.
        """
        if self._last_cmd_t is None or self.engaged:
            return self.engaged

        now = time.monotonic() if t is None else t
        if now - self._last_cmd_t > self.timeout:
            self.engaged = True
            self.n_timeouts += 1
            if self.behavior == BEHAVIOR_ZERO_TORQUE:
                controller.disable_all()
                self._on_event(
                    f"Watchdog: sin /lowcmd por más de {self.timeout:.3f}s -> "
                    f"torque a cero (rampa suave). Timeouts: {self.n_timeouts}"
                )
            else:  # hold_position
                controller.set_hold(current_q, self.hold_kp, self.hold_kd)
                self._on_event(
                    f"Watchdog: sin /lowcmd por más de {self.timeout:.3f}s -> "
                    f"manteniendo posición capturada "
                    f"(kp={self.hold_kp}, kd={self.hold_kd}). "
                    f"Timeouts: {self.n_timeouts}"
                )
        return self.engaged
