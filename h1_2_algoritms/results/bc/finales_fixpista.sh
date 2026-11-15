#!/bin/bash
# Finales (10000-10089) con la corrección del tracker: exigente y rápido, RH56, dx 3 cm
cd "$(dirname "$0")/../../demos"
source ~/venvs/h12/bin/activate; source /opt/ros/humble/setup.bash; source ~/humanoid_ws/install/setup.bash
export MUJOCO_GL=egl H12_FAJA_DX=0.03
R="--robot --vel-max 2 --tau-max 15 --kd-robot 4 --lost-timeout 1 --workers 4"
T="--timing reach_extra=0.37,close_lift=1.3,deposit=1.15,home=0.55"
F="evaluar --politicas experto bc --episodios 90 --seed 10000 --red politica_bc_ancho_suave_DR_dx3.pt"
for e in exigente rapido; do
  o=../results/bc/mesa_u/rh56/final_dx3_${e}_fixpista.csv
  [ -f $o ] || python bc_faja.py $F --escenario $e --perfil suave_DR $R $T --out mesa_u/rh56/final_dx3_${e}_fixpista.csv | tail -3
done
echo FIN_FINALES
