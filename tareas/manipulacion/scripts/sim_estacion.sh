#!/usr/bin/env bash
# El H1-2 simulado delante de la mesa del reto (R40-RT-H1_2-0002), de pie y QUIETO.
#   ./scripts/sim_estacion.sh 3            # nivel 3, semilla 0, con el viewer de MuJoCo
#   ./scripts/sim_estacion.sh 1 2          # nivel 1, semilla 2
#   ./scripts/sim_estacion.sh 0 0 use_viewer:=false     # argumentos extra de sim.launch.py
# Las escenas se generan con:  python3 sim/construir_estacion.py --todos
#
# MARCHA (por defecto "ninguna"): de pie con las piernas rigidas sobre base flotante (fisica
# completa: contactos, gravedad, IMU; el brazo empuja al cuerpo de verdad). La politica de
# marcha de unitree_rl_gym (MARCHA=politica) NO sirve aqui: con orden cero pisa en el sitio y
# deriva (medido: 13 cm y 11 grados en 10 s), y el reto cuenta pasos. Ver README.
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
N=${1:?nivel 0..4}; shift
S=0; if [[ "${1:-}" =~ ^[0-9]+$ ]]; then S=$1; shift; fi
XML="$AQUI/sim/estaciones/estacion_n${N}_s${S}.xml"
[ -f "$XML" ] || { echo "no existe $XML: python3 sim/construir_estacion.py --nivel $N --semilla $S"; exit 1; }
source "$(cd "$AQUI/../.." && pwd)/h1_2_sim2real/entorno.sh"
exec ros2 launch h1_2_sim2real sim.launch.py modelo:="$XML" marcha:=${MARCHA:-ninguna} "$@"
