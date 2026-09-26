"""
joint_mapping.py — Mapeo índice Unitree <-> joint/actuador de MuJoCo.

El orden de índices reproduce H12JointIndex del SDK (ver test_mandar_modificado):

  0-5   pierna izquierda  (hip_yaw, hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll)
  6-11  pierna derecha    (ídem)
  12    cintura            (WAIST_YAW -> 'torso_joint' en este MJCF)
  13-19 brazo izquierdo   (sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw)
  20-26 brazo derecho     (ídem)

NUNCA se asume que este orden coincide con qpos/qvel/ctrl de MuJoCo: para cada
articulación se resuelven explícitamente por NOMBRE el joint id, qpos_adr,
qvel_adr (dof), el actuador y se verifica que el actuador realmente transmite
a ese joint (actuator_trnid). Si algo falta o es inconsistente, se lanza
RuntimeError con la lista completa de problemas y el bridge NO arranca.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import mujoco
import numpy as np

# Mapeo por defecto índice Unitree -> nombre de joint en el MJCF.
# El actuador debe llamarse igual que el joint (así está definido en el XML).
DEFAULT_UNITREE_TO_MUJOCO: Dict[int, str] = {
    0:  "left_hip_yaw_joint",
    1:  "left_hip_pitch_joint",
    2:  "left_hip_roll_joint",
    3:  "left_knee_joint",
    4:  "left_ankle_pitch_joint",
    5:  "left_ankle_roll_joint",
    6:  "right_hip_yaw_joint",
    7:  "right_hip_pitch_joint",
    8:  "right_hip_roll_joint",
    9:  "right_knee_joint",
    10: "right_ankle_pitch_joint",
    11: "right_ankle_roll_joint",
    12: "torso_joint",              # WAIST_YAW
    13: "left_shoulder_pitch_joint",
    14: "left_shoulder_roll_joint",
    15: "left_shoulder_yaw_joint",
    16: "left_elbow_joint",
    17: "left_wrist_roll_joint",
    18: "left_wrist_pitch_joint",
    19: "left_wrist_yaw_joint",
    20: "right_shoulder_pitch_joint",
    21: "right_shoulder_roll_joint",
    22: "right_shoulder_yaw_joint",
    23: "right_elbow_joint",
    24: "right_wrist_roll_joint",
    25: "right_wrist_pitch_joint",
    26: "right_wrist_yaw_joint",
}

NUM_CMD_MOTORS = 27   # motores controlables del H1-2 (índices 0..26)


@dataclass(frozen=True)
class JointMapEntry:
    unitree_index: int
    joint_name: str        # nombre del joint en el XML
    actuator_name: str     # nombre del actuador en el XML
    joint_id: int          # id del joint
    actuator_id: int       # id del actuador
    qpos_adr: int          # dirección en data.qpos
    qvel_adr: int          # dirección en data.qvel (dof)
    q_min: float           # límite inferior de posición (del XML)
    q_max: float           # límite superior de posición (del XML)


class JointMapping:
    """Contenedor del mapeo validado + arrays de índices para acceso vectorizado."""

    def __init__(self, entries: List[JointMapEntry]):
        if len(entries) != NUM_CMD_MOTORS:
            raise RuntimeError(
                f"El mapeo debe tener {NUM_CMD_MOTORS} entradas, tiene {len(entries)}"
            )
        entries = sorted(entries, key=lambda e: e.unitree_index)
        for i, e in enumerate(entries):
            if e.unitree_index != i:
                raise RuntimeError(
                    f"Índices Unitree no contiguos: falta el índice {i}"
                )
        self.entries = entries
        # Arrays para gather/scatter vectorizado (orden = índice Unitree)
        self.qpos_adr = np.array([e.qpos_adr for e in entries], dtype=int)
        self.qvel_adr = np.array([e.qvel_adr for e in entries], dtype=int)
        self.act_id = np.array([e.actuator_id for e in entries], dtype=int)
        self.q_min = np.array([e.q_min for e in entries], dtype=float)
        self.q_max = np.array([e.q_max for e in entries], dtype=float)
        self.joint_names = [e.joint_name for e in entries]

    def __len__(self) -> int:
        return len(self.entries)

    def summary(self) -> str:
        lines = [f"{'idx':>3s}  {'joint (XML)':30s} {'act':>3s} {'qpos':>4s} {'qvel':>4s}  rango [rad]"]
        for e in self.entries:
            lines.append(
                f"{e.unitree_index:3d}  {e.joint_name:30s} {e.actuator_id:3d} "
                f"{e.qpos_adr:4d} {e.qvel_adr:4d}  [{e.q_min:+.2f}, {e.q_max:+.2f}]"
            )
        return "\n".join(lines)


def build_joint_mapping(
    model: mujoco.MjModel,
    unitree_to_mujoco: Optional[Dict[int, str]] = None,
) -> JointMapping:
    """
    Construye y VALIDA el mapeo. Lanza RuntimeError con todos los problemas
    encontrados si algo falta o es inconsistente (el bridge no debe arrancar
    con un mapeo parcial).
    """
    name_map = dict(unitree_to_mujoco or DEFAULT_UNITREE_TO_MUJOCO)
    errors: List[str] = []
    entries: List[JointMapEntry] = []

    for idx in range(NUM_CMD_MOTORS):
        jname = name_map.get(idx)
        if jname is None:
            errors.append(f"[idx {idx}] sin nombre de joint en el mapeo")
            continue

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            errors.append(f"[idx {idx}] joint '{jname}' NO existe en el modelo")
            continue

        jtype = model.jnt_type[jid]
        if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            errors.append(
                f"[idx {idx}] joint '{jname}' no es hinge/slide (type={int(jtype)})"
            )
            continue

        # Actuador con el mismo nombre que el joint (convención del XML)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if aid < 0:
            errors.append(f"[idx {idx}] actuador '{jname}' NO existe en el modelo")
            continue

        # Verificar que el actuador realmente transmite a ESTE joint
        trn_jid = int(model.actuator_trnid[aid, 0])
        if trn_jid != jid:
            trn_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trn_jid)
            errors.append(
                f"[idx {idx}] el actuador '{jname}' transmite al joint "
                f"'{trn_name}' (id {trn_jid}), no a '{jname}' (id {jid})"
            )
            continue

        if bool(model.jnt_limited[jid]):
            q_min, q_max = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        else:
            q_min, q_max = -np.inf, np.inf

        entries.append(JointMapEntry(
            unitree_index=idx,
            joint_name=jname,
            actuator_name=jname,
            joint_id=jid,
            actuator_id=aid,
            qpos_adr=int(model.jnt_qposadr[jid]),
            qvel_adr=int(model.jnt_dofadr[jid]),
            q_min=q_min,
            q_max=q_max,
        ))

    if errors:
        raise RuntimeError(
            "Mapeo Unitree<->MuJoCo INVÁLIDO — el bridge no puede arrancar:\n  - "
            + "\n  - ".join(errors)
        )

    return JointMapping(entries)


def load_mapping_yaml(path: str) -> Dict[int, str]:
    """Carga config/joint_mapping.yaml -> dict {índice Unitree: nombre de joint}."""
    import yaml
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}
    table = raw.get('unitree_to_mujoco', raw)
    return {int(k): str(v) for k, v in table.items()}
