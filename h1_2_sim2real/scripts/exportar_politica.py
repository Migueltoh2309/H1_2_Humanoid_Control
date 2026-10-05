#!/usr/bin/env python3
"""Regenera src/h1_2_sim2real/politicas/h1_2_marcha.npz desde unitree_rl_gym y lo verifica.

  1. Clona unitreerobotics/unitree_rl_gym en terceros/ (si no esta).
  2. Lee deploy/pre_train/h1_2/motion.pt (TorchScript: LSTM 47->64 + actor 64->32->12)
     y las constantes de despliegue (deploy_mujoco/configs/h1_2.yaml y
     deploy_real/configs/h1_2.yaml) y las guarda en el .npz.
  3. Compara la red en numpy (politica_marcha.py) con el .pt en 300 pasos seguidos
     (con el estado de la LSTM): tiene que dar < 1e-4.

Necesita PyTorch (solo para esto; el simulador no lo usa):
    source entorno.sh && python3 scripts/exportar_politica.py [--commit <hash>]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

S2R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REPO = os.path.join(S2R, "terceros", "unitree_rl_gym")
SALIDA = os.path.join(S2R, "src", "h1_2_sim2real", "politicas", "h1_2_marcha.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default=None, help="commit de unitree_rl_gym (por defecto el ultimo)")
    a = ap.parse_args()
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "-q", "https://github.com/unitreerobotics/unitree_rl_gym.git", REPO], check=True)
    if a.commit:
        subprocess.run(["git", "-C", REPO, "checkout", "-q", a.commit], check=True)
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    import torch
    import yaml
    pt = torch.jit.load(os.path.join(REPO, "deploy", "pre_train", "h1_2", "motion.pt"))
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in pt.state_dict().items()}
    cfg = yaml.safe_load(open(os.path.join(REPO, "deploy", "deploy_mujoco", "configs", "h1_2.yaml")))
    real = yaml.safe_load(open(os.path.join(REPO, "deploy", "deploy_real", "configs", "h1_2.yaml")))
    meta = {k: cfg[k] for k in ("kps", "kds", "default_angles", "ang_vel_scale", "dof_pos_scale", "dof_vel_scale",
                                "action_scale", "cmd_scale", "num_actions", "num_obs", "control_decimation",
                                "simulation_dt")}
    meta.update(max_cmd=real["max_cmd"], arm_waist_kps=real["arm_waist_kps"], arm_waist_kds=real["arm_waist_kds"],
                periodo_fase=0.8,
                origen=f"unitreerobotics/unitree_rl_gym deploy/pre_train/h1_2/motion.pt (commit {commit})")
    np.savez(SALIDA, W_ih=sd["memory.weight_ih_l0"], W_hh=sd["memory.weight_hh_l0"], b_ih=sd["memory.bias_ih_l0"],
             b_hh=sd["memory.bias_hh_l0"], A0=sd["actor.0.weight"], a0=sd["actor.0.bias"], A2=sd["actor.2.weight"],
             a2=sd["actor.2.bias"], meta=json.dumps(meta))
    print(f"exportado: {SALIDA} (unitree_rl_gym {commit})")

    sys.path.insert(0, os.path.join(S2R, "src", "h1_2_sim2real"))
    from h1_2_sim2real.politica_marcha import PoliticaMarcha
    pn = PoliticaMarcha(SALIDA)
    pt = torch.jit.load(os.path.join(REPO, "deploy", "pre_train", "h1_2", "motion.pt"))   # estado LSTM a cero
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(300):
        obs = rng.normal(0, 0.5, 47).astype(np.float32)
        a_t = pt(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
        err = max(err, float(np.abs(a_t - pn.red(obs.astype(np.float64))).max()))
    print(f"verificacion: 300 pasos, diferencia maxima numpy vs torch = {err:.2e}")
    if err > 1e-4:
        sys.exit("NO coincide: no uses este .npz")


if __name__ == "__main__":
    main()
