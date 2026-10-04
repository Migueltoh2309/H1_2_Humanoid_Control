#!/usr/bin/env bash
# El H1-2 simulado en una pista del reto, con la marcha cinematica (la base sigue
# LocoClient.Move como solido rigido: NO camina, no se cae).
#   ./scripts/sim_pista.sh 1                       # nivel 1, con el viewer de MuJoCo
#   ./scripts/sim_pista.sh 3 use_viewer:=false     # argumentos extra de sim.launch.py
#   ./scripts/sim_pista.sh 2 marcha:=ninguna       # robot quieto: solo vision
#   ./scripts/sim_pista.sh 1 emisor_ir:=false      # sin el patron del emisor IR
# Las pistas se generan con:  python3 sim/construir_pistas.py
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
N=${1:?nivel 1..4}; shift
XML="$AQUI/sim/pistas/pista_nivel$N.xml"
[ -f "$XML" ] || { echo "no existe $XML: python3 sim/construir_pistas.py"; exit 1; }
source "$AQUI/../../h1_2_sim2real/entorno.sh"
exec ros2 launch h1_2_sim2real sim.launch.py modelo:="$XML" marcha:=cinematica "$@"
