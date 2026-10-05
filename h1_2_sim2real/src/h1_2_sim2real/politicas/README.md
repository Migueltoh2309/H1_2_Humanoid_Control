# Políticas

`h1_2_marcha.npz`: pesos de la política de marcha del H1-2 de Unitree,
`unitreerobotics/unitree_rl_gym`, `deploy/pre_train/h1_2/motion.pt` (commit 276801e),
exportados a numpy con las constantes de despliegue de `deploy/deploy_mujoco/configs/h1_2.yaml`
y `deploy/deploy_real/configs/h1_2.yaml`. Licencia BSD-3 de Unitree: `LICENSE_unitree_rl_gym.txt`.
La red la ejecuta `h1_2_sim2real/politica_marcha.py` (verificada contra el .pt: diferencia < 3e-6).
