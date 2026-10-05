#!/usr/bin/env bash
# El H1-2 simulado en una pista del reto. Por defecto CAMINA con la politica de marcha de
# unitree_rl_gym (LocoClient.Move -> politica -> piernas, fisica completa: puede caerse).
#   ./scripts/sim_pista.sh 1                       # nivel 1, con el viewer de MuJoCo
#   ./scripts/sim_pista.sh 3 use_viewer:=false     # argumentos extra de sim.launch.py
#   MARCHA=ninguna ./scripts/sim_pista.sh 2         # robot quieto: solo vision
#   ./scripts/sim_pista.sh 1 emisor_ir:=false      # sin el patron del emisor IR
# Las pistas se generan con:  python3 sim/construir_pistas.py
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
N=${1:?nivel 1..4}; shift
XML="$AQUI/sim/pistas/pista_nivel$N.xml"
[ -f "$XML" ] || { echo "no existe $XML: python3 sim/construir_pistas.py"; exit 1; }
# cd (ruta logica) y no "$AQUI/../.." a pelo: tareas/ puede ser un enlace simbolico (al repo
# de GitHub) y el kernel resolveria ".." sobre la ruta fisica, fuera de humanoid_ws/src
source "$(cd "$AQUI/../.." && pwd)/h1_2_sim2real/entorno.sh"
# marcha por defecto: la POLITICA de unitree_rl_gym (camina de verdad, fisica completa).
# MARCHA=cinematica ./scripts/sim_pista.sh 1   -> base rigida que sigue Move (no camina)
exec ros2 launch h1_2_sim2real sim.launch.py modelo:="$XML" marcha:=${MARCHA:-politica} "$@"
