#!/usr/bin/env bash
# Una tirada completa y AUTOMATICA en el simulador: lanza la pista (sin ventanas),
# corre el seguidor sin pedir la palabra, cierra el simulador y saca las metricas
# con verdad de terreno.
#   ./scripts/tirada_sim.sh 2                 # nivel 2
#   ./scripts/tirada_sim.sh 3 --escala 0.5    # argumentos extra para seguidor.py
#   SEMILLA=4 ./scripts/tirada_sim.sh 1       # signo de la deriva de rumbo reproducible
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
N=${1:?nivel 1..4}; shift
source "$AQUI/scripts/entorno.sh"
LOG=$(mktemp /tmp/sim_pista_XXXX.log)
"$AQUI/scripts/sim_pista.sh" "$N" use_viewer:=false marcha_semilla:=${SEMILLA:--1} > "$LOG" 2>&1 &
SIM=$!
# SIGINT (como Ctrl+C): con SIGTERM, ros2 launch muere sin cerrar el simulador y quedan
# procesos huerfanos ocupando los puertos Modbus de las manos
# (bash ignora SIGINT en los procesos en segundo plano de un script: se manda al hijo de
# ros2 launch, el nodo del simulador; launch termina solo cuando el nodo acaba)
trap 'pkill -INT -P $SIM 2>/dev/null; wait $SIM 2>/dev/null' EXIT
until grep -q "D435 simulada" "$LOG"; do sleep 0.5; kill -0 $SIM 2>/dev/null || { cat "$LOG"; exit 1; }; done
sleep 2
EXTRA=(); [ "${MARCHA:-politica}" = cinematica ] && EXTRA=(--config-extra "$AQUI/config/planta_cinematica.yaml")
python3 "$AQUI/seguidor.py" --sim "${EXTRA[@]}" --sin-confirmacion --nivel "$N" "$@" 2>&1 | grep -v "multicast"
CSV=$(ls -t "$AQUI"/datos/tiradas/nivel"$N"_*.csv | head -1)
python3 "$AQUI/herramientas/metricas.py" "$CSV" --png
