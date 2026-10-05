"""Politica de marcha del H1-2 de Unitree (unitree_rl_gym, deploy/pre_train/h1_2/motion.pt)
reimplementada en numpy: sin PyTorch en el proceso de fisica (y portable al PC2).

Red (TorchScript del repositorio): LSTM de 1 capa (47 -> 64, estado h, c guardado
entre llamadas) + actor Linear(64, 32) -> ELU -> Linear(32, 12). Pesos y constantes de
despliegue en politicas/h1_2_marcha.npz (exportados y verificados con scripts/exportar_politica.py; la
salida coincide con la del .pt a 1e-6).

Observacion (47), igual que deploy/deploy_mujoco/deploy_mujoco.py:
   [0:3]   velocidad angular de la PELVIS en su marco      * 0.25
   [3:6]   gravedad proyectada en la pelvis (del cuaternion)
   [6:9]   comando (vx, vy, vyaw)                          * (2, 2, 0.25)
   [9:21]  q piernas - postura por defecto
   [21:33] dq piernas                                      * 0.05
   [33:45] accion anterior
   [45:47] sin, cos de la fase de la marcha (periodo 0.8 s)
Accion (12) -> objetivo de las piernas = accion * 0.25 + postura por defecto, con PD
kp (200, 200, 200, 300, 40, 40), kd (2.5, 2.5, 2.5, 4, 2, 2), a 50 Hz.
Orden de las 12 juntas: el de h1_2_juntas (cadera yaw, pitch, roll, rodilla, tobillo
pitch, roll; izquierda y luego derecha), el mismo del MJCF de unitree_rl_gym.
"""
import json
import math
import os

import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def gravedad_proyectada(qw, qx, qy, qz):
    """La de deploy_mujoco.get_gravity_orientation (gravedad en el marco de la pelvis)."""
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])


class PoliticaMarcha:
    def __init__(self, ruta=None):
        if ruta is None:
            from .comun import recurso
            ruta = recurso("politicas", "h1_2_marcha.npz")
        z = np.load(ruta)
        self.W_ih, self.W_hh = z["W_ih"].astype(np.float64), z["W_hh"].astype(np.float64)
        self.b = (z["b_ih"] + z["b_hh"]).astype(np.float64)
        self.A0, self.a0 = z["A0"].astype(np.float64), z["a0"].astype(np.float64)
        self.A2, self.a2 = z["A2"].astype(np.float64), z["a2"].astype(np.float64)
        self.meta = json.loads(str(z["meta"]))
        m = self.meta
        self.kp = np.array(m["kps"], float)
        self.kd = np.array(m["kds"], float)
        self.q0 = np.array(m["default_angles"], float)
        self.cmd_escala = np.array(m["cmd_scale"], float)
        self.max_cmd = np.array(m["max_cmd"], float)
        self.periodo = m["periodo_fase"]
        self.n_h = self.W_hh.shape[1]
        self.reiniciar()

    def reiniciar(self):
        self.h = np.zeros(self.n_h)
        self.c = np.zeros(self.n_h)
        self.accion = np.zeros(12)
        self.t = 0.0

    def red(self, obs):
        """Un paso de LSTM + actor (mantiene el estado)."""
        g = self.W_ih @ obs + self.W_hh @ self.h + self.b
        n = self.n_h
        i, f, gg, o = _sig(g[:n]), _sig(g[n:2 * n]), np.tanh(g[2 * n:3 * n]), _sig(g[3 * n:])
        self.c = f * self.c + i * gg
        self.h = o * np.tanh(self.c)
        x = self.A0 @ self.h + self.a0
        x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))          # ELU
        return self.A2 @ x + self.a2

    def observacion(self, quat_wxyz, omega, q, dq, cmd):
        m = self.meta
        fase = (self.t % self.periodo) / self.periodo
        obs = np.empty(m["num_obs"])
        obs[0:3] = np.asarray(omega) * m["ang_vel_scale"]
        obs[3:6] = gravedad_proyectada(*quat_wxyz)
        obs[6:9] = np.asarray(cmd) * self.cmd_escala
        obs[9:21] = (np.asarray(q) - self.q0) * m["dof_pos_scale"]
        obs[21:33] = np.asarray(dq) * m["dof_vel_scale"]
        obs[33:45] = self.accion
        obs[45:47] = (math.sin(2 * math.pi * fase), math.cos(2 * math.pi * fase))
        return obs

    def paso(self, quat_wxyz, omega, q, dq, cmd, dt):
        """Una decision (a 50 Hz): devuelve los objetivos de las 12 juntas de las piernas."""
        cmd = np.clip(np.asarray(cmd, float), -self.max_cmd, self.max_cmd)
        self.t += dt
        obs = self.observacion(quat_wxyz, omega, q, dq, cmd)
        self.accion = self.red(obs)
        return self.accion * self.meta["action_scale"] + self.q0
