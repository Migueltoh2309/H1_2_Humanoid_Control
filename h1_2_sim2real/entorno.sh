# Entorno de h1_2_sim2real. En CADA terminal:   source ~/humanoid_ws/src/h1_2_sim2real/entorno.sh
# venv del workspace (mujoco, numpy...) + ROS 2 Humble + SDK de Unitree (terceros/, sin tocar el venv)
# + humanoid_ws/install (para usar tus nodos de h1_2_algoritms con esto) + este workspace.
S2R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HOME/venvs/h12/bin/activate" ]; then source "$HOME/venvs/h12/bin/activate"; fi
source /opt/ros/humble/setup.bash
export PYTHONPATH="$S2R/terceros/pydeps:$S2R/terceros/unitree_sdk2_python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export H12_S2R_COMUN="$S2R/comun"
unset CYCLONEDDS_URI          # el SDK monta su propia config (como en Codigos/comun.sh)
# workspace principal (humanoid_ws): esta carpeta vive en humanoid_ws/src/h1_2_sim2real
# (esta carpeta vive en el repo H1_2_Humanoid_Control y se enlaza desde humanoid_ws/src: si se
# carga por la ruta del repo, ../.. no es el workspace -> se usa ~/humanoid_ws)
H12_WS="${H12_WS:-$(cd "$S2R/../.." && pwd)}"
if [ ! -f "$H12_WS/install/setup.bash" ] && [ -f "$HOME/humanoid_ws/install/setup.bash" ]; then
    H12_WS="$HOME/humanoid_ws"
fi
if [ -f "$H12_WS/install/setup.bash" ]; then source "$H12_WS/install/setup.bash"; fi
if [ -f "$S2R/install/setup.bash" ]; then source "$S2R/install/setup.bash"; fi
