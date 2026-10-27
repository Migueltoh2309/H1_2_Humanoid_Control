#!/usr/bin/env bash
# Hito 2 en el simulador, de principio a fin y sin ventanas: estacion vacia (nivel 0) con el
# emisor IR apagado (como se apaga el proyector de la D435 para calibrar), calibracion
# camara-torso con el ArUco del dorso, curva de pasos y fuerza de la mano. Log en datos/.
#   ./scripts/hito2_sim.sh            # todo (~8 min)
#   ./scripts/hito2_sim.sh calibrar   # solo una parte: calibrar | curva | fuerza
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AQUI"
source scripts/entorno.sh
PARTES=${*:-calibrar curva fuerza}
mkdir -p datos
LOG="$AQUI/datos/hito2_sim_sim.log"
if [[ " $PARTES " == *" calibrar "* || " $PARTES " == *" curva "* ]]; then
    ./scripts/sim_estacion.sh 0 use_viewer:=false emisor_ir:=false > "$LOG" 2>&1 &
    SIM=$!
    trap 'pkill -INT -P $SIM 2>/dev/null; kill -INT $SIM 2>/dev/null; wait $SIM 2>/dev/null' EXIT
    until grep -q "D435 simulada" "$LOG"; do sleep 0.5; kill -0 $SIM 2>/dev/null || { cat "$LOG"; exit 1; }; done
    sleep 3
fi
for p in $PARTES; do
    case $p in
        calibrar) python3 herramientas/calibrar.py --sim --escena sim/estaciones/estacion_n0_s0.xml ;;
        curva)    python3 herramientas/curva_pasos.py --sim --lado izq ;;
        fuerza)   python3 herramientas/fuerza_mano.py --sim --dedo indice ;;
    esac 2>&1 | grep -v multicast
done
