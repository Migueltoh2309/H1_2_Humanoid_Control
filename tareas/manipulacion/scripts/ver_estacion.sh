#!/usr/bin/env bash
# Ver la estacion: MuJoCo (viewer) + RViz (robot, mesa, zona, bandejas, objetos donde estan
# de verdad, imagen de color y profundidad nativa). Todo lo grafico en la GPU NVIDIA.
#   ./scripts/ver_estacion.sh 3            # nivel 3, semilla 0
#   ./scripts/ver_estacion.sh 1 2          # nivel 1, semilla 2
# Enter (o Ctrl+C) cierra todo.
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
N=${1:?nivel 0..4}; S=${2:-0}
source "$AQUI/scripts/entorno.sh"
# Viewer de MuJoCo y RViz en la NVIDIA (PRIME offload); las camaras del simulador ya renderizan
# por EGL en la NVIDIA. Con la Intel integrada el simulador iba al 28 % del tiempo real.
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
fi
LOG=$(mktemp /tmp/ver_estacion_XXXX.log)
# menos carga para mirar: estado a 200 Hz, camaras a 10 Hz
"$AQUI/scripts/sim_estacion.sh" "$N" "$S" visor:=true rviz_config:="$AQUI/rviz/manipulacion.rviz" \
    hz_lowstate:=200.0 hz_camara:=10.0 contactos:=false > "$LOG" 2>&1 &
SIM=$!
python3 "$AQUI/herramientas/rviz_estacion.py" "$AQUI/sim/estaciones/estacion_n${N}_s${S}.json" &
MARC=$!
trap 'kill $MARC 2>/dev/null; pkill -INT -P $SIM 2>/dev/null; wait $SIM 2>/dev/null' EXIT
echo "abriendo MuJoCo y RViz (log: $LOG)..."
until grep -q "D435 simulada" "$LOG"; do sleep 0.5; kill -0 $SIM 2>/dev/null || { cat "$LOG"; exit 1; }; done
read -r -p "Enter para cerrar el simulador y las ventanas " _
