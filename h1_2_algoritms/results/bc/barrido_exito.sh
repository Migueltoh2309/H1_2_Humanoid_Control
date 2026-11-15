#!/bin/bash
# Barrido dev (semillas 21000-21029) del predictor de éxito: docs/MAPA_LFD.md §14
cd "$(dirname "$0")/../../demos"
source ~/venvs/h12/bin/activate; source /opt/ros/humble/setup.bash; source ~/humanoid_ws/install/setup.bash
export MUJOCO_GL=egl H12_FAJA_DX=0.03
R="--robot --vel-max 2 --tau-max 15 --kd-robot 4 --lost-timeout 1 --workers 4"
P="--escenario ancho --perfil suave_DR $R"
T="--timing reach_extra=0.37,close_lift=1.3,deposit=1.15,home=0.55"
E="--episodios 30 --seed 21000 --predictor ../results/bc/predictor_exito_ancho_suave_DR_dx3.pt"
[ -f ../results/bc/mesa_u/rh56/exito/base.csv ] || python bc_faja.py evaluar $P $T $E --politicas experto bc --out mesa_u/rh56/exito/base.csv | tail -4
for cfg in "1 0" "3 0" "1 0.4"; do set -- $cfg
  [ -f ../results/bc/mesa_u/rh56/exito/bc_a$1_u$2.csv ] || python bc_faja.py evaluar $P $T $E --politicas bc_exito --alpha-exito $1 --umbral-exito $2 --out mesa_u/rh56/exito/bc_a$1_u$2.csv | tail -3
done
[ -f ../results/bc/mesa_u/rh56/exito/experto_u0.4.csv ] || python bc_faja.py evaluar $P $T $E --politicas experto_exito --umbral-exito 0.4 --out mesa_u/rh56/exito/experto_u0.4.csv | tail -3
echo FIN_BARRIDO
