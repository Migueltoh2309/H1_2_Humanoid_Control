"""
safety_limits.py — Límites y saturaciones por articulación.

Fuentes de límites, en orden:
  1. XML (MJCF): rangos de posición q_min/q_max (jnt_range). Los actuadores de
     este modelo son <motor> SIN ctrlrange/forcerange, por lo que el torque
     máximo NO puede leerse del XML y debe venir del YAML.
  2. motor_limits.yaml: defaults globales + overrides por joint para
     dq_max, tau_max, kp_max, kd_max y tau_rate_max.

Saturaciones aplicadas (ver también motor_controller.py):
  q_des      -> clamp a [q_min, q_max]
  dq_des     -> clamp a [-dq_max, +dq_max]
  tau_ff     -> clamp a [-tau_max, +tau_max]
  kp, kd     -> clamp a [0, kp_max] / [0, kd_max]
  tau_total  -> clamp a [-tau_max, +tau_max]
  delta_tau  -> |tau_k - tau_{k-1}| <= tau_rate_max * dt   (por sub-paso)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .joint_mapping import JointMapping


@dataclass
class SafetyLimits:
    q_min: np.ndarray          # [n] rad
    q_max: np.ndarray          # [n] rad
    dq_max: np.ndarray         # [n] rad/s (>0)
    tau_max: np.ndarray        # [n] N·m   (>0)
    kp_max: np.ndarray         # [n]
    kd_max: np.ndarray         # [n]
    tau_rate_max: np.ndarray   # [n] N·m/s (>0)

    @property
    def n(self) -> int:
        return len(self.q_min)

    # ---- clamps vectorizados -------------------------------------------------
    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.q_min, self.q_max)

    def clamp_dq(self, dq: np.ndarray) -> np.ndarray:
        return np.clip(dq, -self.dq_max, self.dq_max)

    def clamp_tau(self, tau: np.ndarray) -> np.ndarray:
        return np.clip(tau, -self.tau_max, self.tau_max)

    def clamp_kp(self, kp: np.ndarray) -> np.ndarray:
        return np.clip(kp, 0.0, self.kp_max)

    def clamp_kd(self, kd: np.ndarray) -> np.ndarray:
        return np.clip(kd, 0.0, self.kd_max)


def build_limits(
    mapping: JointMapping,
    defaults: Optional[Dict] = None,
    per_joint: Optional[Dict[str, Dict]] = None,
) -> SafetyLimits:
    """
    mapping   : JointMapping validado (aporta q_min/q_max del XML).
    defaults  : dict con dq_max, tau_max, kp_max, kd_max, tau_rate_max globales.
    per_joint : dict {nombre_joint: {clave: valor}} con overrides individuales.
    """
    d = {
        'dq_max': 20.0,
        'tau_max': 80.0,
        'kp_max': 300.0,
        'kd_max': 20.0,
        'tau_rate_max': 1000.0,
    }
    d.update(defaults or {})
    pj = per_joint or {}

    n = len(mapping)

    def col(key: str) -> np.ndarray:
        out = np.full(n, float(d[key]))
        for i, name in enumerate(mapping.joint_names):
            if name in pj and key in pj[name]:
                out[i] = float(pj[name][key])
        return out

    # q_min/q_max: base = XML; el YAML solo puede ESTRECHAR el rango, nunca ampliarlo
    q_min = mapping.q_min.copy()
    q_max = mapping.q_max.copy()
    for i, name in enumerate(mapping.joint_names):
        if name in pj:
            if 'q_min' in pj[name]:
                q_min[i] = max(q_min[i], float(pj[name]['q_min']))
            if 'q_max' in pj[name]:
                q_max[i] = min(q_max[i], float(pj[name]['q_max']))

    lim = SafetyLimits(
        q_min=q_min,
        q_max=q_max,
        dq_max=np.abs(col('dq_max')),
        tau_max=np.abs(col('tau_max')),
        kp_max=np.abs(col('kp_max')),
        kd_max=np.abs(col('kd_max')),
        tau_rate_max=np.abs(col('tau_rate_max')),
    )
    return lim


def load_limits_yaml(path: str):
    """Carga motor_limits.yaml -> (defaults: dict, per_joint: dict)."""
    import yaml
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}
    return dict(raw.get('defaults', {})), dict(raw.get('joints', {}))
