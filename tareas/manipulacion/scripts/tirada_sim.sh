#!/usr/bin/env bash
# Una tirada en el simulador de principio a fin: arranca la estación (sin ventanas), lanza
# manipulacion.py y para el simulador al acabar (SIGTERM: lanzado en segundo plano ignora SIGINT).
#   ./scripts/tirada_sim.sh NIVEL SEMILLA [argumentos de manipulacion.py]
#   ./scripts/tirada_sim.sh 3 0 --instructor
#   ./scripts/tirada_sim.sh 4 1 --objeto prisma
# Registro en datos/tiradas/; métricas: python3 herramientas/metricas.py datos/tiradas/nivel*
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AQUI"
N=${1:?nivel}; SEM=${2:?semilla}; shift 2
mkdir -p datos/tiradas
LOG="datos/tiradas/sim_n${N}_s${SEM}.log"
./scripts/sim_estacion.sh "$N" "$SEM" use_viewer:=false > "$LOG" 2>&1 &
L=$!
parar() {
    for P in $(ps -eo pid,cmd | grep -E "ros2 launch h1_2_sim2real|h1_2_sim2real/sim --ros-args" | grep -v grep | awk '{print $1}'); do
        kill -TERM "$P" 2>/dev/null
    done
    sleep 3
}
trap parar EXIT
until grep -q "D435 simulada" "$LOG"; do sleep 1; kill -0 $L 2>/dev/null || { cat "$LOG"; exit 1; }; done
sleep 2
source scripts/entorno.sh > /dev/null 2>&1
timeout 2400 python3 -u manipulacion.py --sim --nivel "$N" --semilla "$SEM" "$@" < /dev/null 2>&1 | grep --line-buffered -v multicast
