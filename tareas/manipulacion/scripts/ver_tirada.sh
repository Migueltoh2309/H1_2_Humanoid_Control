#!/usr/bin/env bash
# Ver una tirada EN VIVO: el simulador con el visor de MuJoCo y RViz (GPU NVIDIA) y el programa
# principal; al acabar la tirada se cierra todo.
#   ./scripts/ver_tirada.sh 3 1 --instructor
#   ./scripts/ver_tirada.sh 4 0 --objeto prisma
# En el visor de MuJoCo: ratón izquierdo gira, derecho desplaza, rueda acerca; Tab oculta los
# paneles; Espacio pausa (¡la tirada sigue esperando: mejor no pausar!).
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AQUI"
N=${1:?nivel}; SEM=${2:?semilla}; shift 2
source scripts/entorno.sh > /dev/null 2>&1
if nvidia-smi > /dev/null 2>&1; then
    export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
fi
mkdir -p datos/tiradas
LOG="datos/tiradas/sim_n${N}_s${SEM}_visor.log"
./scripts/sim_estacion.sh "$N" "$SEM" visor:=true rviz_config:="$AQUI/rviz/manipulacion.rviz" \
    contactos:=false > "$LOG" 2>&1 &
L=$!
python3 herramientas/rviz_estacion.py "sim/estaciones/estacion_n${N}_s${SEM}.json" > /dev/null 2>&1 &
MARC=$!
parar() {
    kill $MARC 2>/dev/null
    for P in $(ps -eo pid,cmd | grep -E "ros2 launch h1_2_sim2real|h1_2_sim2real/sim --ros-args|rviz2" | grep -v grep | awk '{print $1}'); do
        kill -TERM "$P" 2>/dev/null
    done
    sleep 3
}
trap parar EXIT
until grep -q "D435 simulada" "$LOG"; do sleep 1; kill -0 $L 2>/dev/null || { cat "$LOG"; exit 1; }; done
echo "simulador listo; la tirada empieza en 5 s (mira la ventana de MuJoCo)"
sleep 5
timeout 2400 python3 -u manipulacion.py --sim --nivel "$N" --semilla "$SEM" "$@" < /dev/null 2>&1 | grep --line-buffered -v multicast
echo "tirada terminada; cierro en 10 s"
sleep 10
