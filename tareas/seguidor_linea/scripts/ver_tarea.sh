#!/usr/bin/env bash
# Ver una tirada en MuJoCo y en RViz: abre la pista con el viewer de MuJoCo (sigue al
# robot) y RViz (pista, linea detectada en verde, recordada en azul, mira en rojo,
# imagen IR), y lanza el seguidor. Pide la palabra SEGUIR como en el robot.
#   ./scripts/ver_tarea.sh 1                 # nivel 1
#   ./scripts/ver_tarea.sh 3 --escala 0.5    # argumentos extra para seguidor.py
# Ctrl+C para el seguidor (y el robot); al salir se cierran el simulador y las ventanas.
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
N=${1:?nivel 1..4}; shift
source "$AQUI/scripts/entorno.sh"
# Viewer de MuJoCo y RViz en la GPU NVIDIA (PRIME offload): con la Intel integrada el
# simulador iba al 28 % del tiempo real con las dos ventanas abiertas; asi, al ~90 %.
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
fi
LOG=$(mktemp /tmp/ver_tarea_XXXX.log)
# menos carga para poder mirar (viewer de MuJoCo + RViz a la vez): estado a 200 Hz, camaras a 10 Hz
"$AQUI/scripts/sim_pista.sh" "$N" visor:=true rviz_config:="$AQUI/rviz/seguidor.rviz" hz_lowstate:=200.0 hz_camara:=10.0 contactos:=false > "$LOG" 2>&1 &
SIM=$!
trap 'pkill -INT -P $SIM 2>/dev/null; wait $SIM 2>/dev/null' EXIT
echo "abriendo MuJoCo y RViz (log: $LOG)..."
until grep -q "D435 simulada" "$LOG"; do sleep 0.5; kill -0 $SIM 2>/dev/null || { cat "$LOG"; exit 1; }; done
sleep 3
EXTRA=(); [ "${MARCHA:-politica}" = cinematica ] && EXTRA=(--config-extra "$AQUI/config/planta_cinematica.yaml")
python3 "$AQUI/seguidor.py" --sim "${EXTRA[@]}" --rviz --nivel "$N" "$@" 2>&1 | grep -v multicast
read -r -p "Enter para cerrar el simulador y las ventanas " _
