#!/usr/bin/env bash
# Atajos de simulacion (cada uno en su terminal):
#   ./scripts/sim.sh robot [args de sim.launch.py]      el H1-2 simulado (= encender el robot), escena de la faja
#   ./scripts/sim.sh suelo [args]                       el H1-2 de pie en el suelo, fisica completa (base flotante)
#   ./scripts/sim.sh puente [args de puente.launch.py]  agente local + puente + RViz
#   ./scripts/sim.sh codigos [args]                     EN VEZ de "robot": sim + RViz con su verdad de terreno,
#                                                       para scripts del SDK (Codigos) sin puente
S2R="$(cd "$(dirname "$0")/.." && pwd)"
source "$S2R/entorno.sh"
QUE=${1:-}; shift || true
case "$QUE" in
    robot)   exec ros2 launch h1_2_sim2real sim.launch.py "$@" ;;
    suelo)   exec ros2 launch h1_2_sim2real sim.launch.py escena:=suelo "$@" ;;
    puente)  exec ros2 launch h1_2_sim2real puente.launch.py objetivo:=sim "$@" ;;
    codigos) exec ros2 launch h1_2_sim2real sim.launch.py visor:=true "$@" ;;
    *) sed -n '2,7p' "$0"; exit 1 ;;
esac
