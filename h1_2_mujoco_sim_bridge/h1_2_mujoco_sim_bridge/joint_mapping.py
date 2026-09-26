"""
joint_mapping.py — Mapeo POR NOMBRE joint/actuador de MuJoCo para el bridge de
simulación.

A diferencia de h1_2_mujoco_lowlevel_bridge (que fija un orden de 27 índices
para reproducir el protocolo unitree_hg), aquí el orden es arbitrario: es
simplemente la lista de nombres de joint que este bridge va a controlar y
publicar. Cada nombre se resuelve y valida contra el modelo cargado (joint
hinge/slide, actuador con el mismo nombre, y que ese actuador transmita
efectivamente a ese joint). Si algo no cuadra, se lanza RuntimeError con
todos los problemas encontrados y el bridge no arranca — mismo criterio que
el bridge de bajo nivel: preferir fallar temprano a simular con un mapeo
parcialmente inválido.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import mujoco
import numpy as np


@dataclass(frozen=True)
class JointMapEntry:
    joint_name: str
    joint_id: int
    actuator_id: int
    qpos_adr: int
    qvel_adr: int
    q_min: float
    q_max: float


class JointMapping:
    """Mapeo validado + arrays para acceso vectorizado, en el orden de `names`."""

    def __init__(self, entries: List[JointMapEntry]):
        self.entries = entries
        self.joint_names = [e.joint_name for e in entries]
        self.index = {name: i for i, name in enumerate(self.joint_names)}
        self.qpos_adr = np.array([e.qpos_adr for e in entries], dtype=int)
        self.qvel_adr = np.array([e.qvel_adr for e in entries], dtype=int)
        self.act_id = np.array([e.actuator_id for e in entries], dtype=int)
        self.q_min = np.array([e.q_min for e in entries], dtype=float)
        self.q_max = np.array([e.q_max for e in entries], dtype=float)

    def __len__(self) -> int:
        return len(self.entries)

    def summary(self) -> str:
        lines = [f"{'joint (XML)':30s} {'act':>4s} {'qpos':>5s} {'qvel':>5s}  rango [rad]"]
        for e in self.entries:
            lines.append(
                f"{e.joint_name:30s} {e.actuator_id:4d} {e.qpos_adr:5d} "
                f"{e.qvel_adr:5d}  [{e.q_min:+.2f}, {e.q_max:+.2f}]"
            )
        return "\n".join(lines)


def build_joint_mapping(model: mujoco.MjModel, joint_names: List[str]) -> JointMapping:
    if len(set(joint_names)) != len(joint_names):
        dupes = [n for n in set(joint_names) if joint_names.count(n) > 1]
        raise RuntimeError(f"joint_names tiene nombres repetidos: {dupes}")

    errors: List[str] = []
    entries: List[JointMapEntry] = []

    for jname in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            errors.append(f"joint '{jname}' NO existe en el modelo")
            continue

        jtype = model.jnt_type[jid]
        if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            errors.append(f"joint '{jname}' no es hinge/slide (type={int(jtype)})")
            continue

        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if aid < 0:
            errors.append(f"actuador '{jname}' NO existe en el modelo "
                           f"(se espera un actuador con el mismo nombre que el joint)")
            continue

        trn_jid = int(model.actuator_trnid[aid, 0])
        if trn_jid != jid:
            trn_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trn_jid)
            errors.append(
                f"el actuador '{jname}' transmite al joint '{trn_name}' "
                f"(id {trn_jid}), no a '{jname}' (id {jid})"
            )
            continue

        if bool(model.jnt_limited[jid]):
            q_min, q_max = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        else:
            q_min, q_max = -np.inf, np.inf

        entries.append(JointMapEntry(
            joint_name=jname,
            joint_id=jid,
            actuator_id=aid,
            qpos_adr=int(model.jnt_qposadr[jid]),
            qvel_adr=int(model.jnt_dofadr[jid]),
            q_min=q_min,
            q_max=q_max,
        ))

    if errors:
        raise RuntimeError(
            "Mapeo de joints INVÁLIDO — el bridge no puede arrancar:\n  - "
            + "\n  - ".join(errors)
        )

    return JointMapping(entries)


def load_joint_names_yaml(path: str) -> List[str]:
    """Carga config/joint_names.yaml -> lista ordenada de nombres de joint."""
    import yaml
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}
    names = raw.get('joint_names', raw)
    return [str(n) for n in names]
