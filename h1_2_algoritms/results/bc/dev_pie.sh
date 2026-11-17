#!/bin/bash
# Robot DE PIE (arranque con arnés + equilibrio, H12_CUERPO=pie) vs pelvis soldada, dev 20000-20039, ancho,
# RH56, dx 3 cm, red dx3. Reanudable. 2 workers: de pie cada uno ocupa ~2.8 GB.
cd "$(dirname "$0")/../../demos"
source ~/venvs/h12/bin/activate; source /opt/ros/humble/setup.bash; source ~/humanoid_ws/install/setup.bash
export MUJOCO_GL=egl H12_FAJA_DX=0.03
R="--robot --vel-max 2 --tau-max 15 --kd-robot 4 --lost-timeout 1 --workers 2"
T="--timing reach_extra=0.37,close_lift=1.3,deposit=1.15,home=0.55"
F="evaluar --episodios 40 --seed 20000 --escenario ancho --perfil suave_DR --red politica_bc_ancho_suave_DR_dx3.pt"
O=../results/bc/mesa_u/rh56/pie
[ -f $O/dev_pie_marco.csv ] || H12_CUERPO=pie python bc_faja.py $F --politicas experto bc $R $T --out mesa_u/rh56/pie/dev_pie_marco.csv 2>&1 | grep -v warn | tail -4
[ -f $O/dev_soldado_bc_dx3.csv ] || python bc_faja.py $F --politicas bc $R $T --out mesa_u/rh56/pie/dev_soldado_bc_dx3.csv 2>&1 | grep -v warn | tail -4
# Perfil de equilibrio rígido (cuerpo_pie.cfg_perfil, 2026-11-17): piernas x2, IMU x2, cintura de arm_sdk x3
[ -f $O/dev_pie_rigido.csv ] || H12_CUERPO=pie H12_EQUILIBRIO=rigido python bc_faja.py $F --politicas experto bc $R $T --out mesa_u/rh56/pie/dev_pie_rigido.csv 2>&1 | grep -v warn | tail -4
