#!/usr/bin/env bash
# Lanza el seguidor EN EL ROBOT (PC2) por ssh -t: el lazo corre alli (por la WiFi hay
# picos de medio segundo: sec. 7 del reto). La camara pide sudo.
#   ./scripts/seguidor_robot.sh --simulacro               # hito 3: no manda Move
#   ./scripts/seguidor_robot.sh --nivel 1 --escala 0.5    # primeras tiradas, a la mitad
# Antes: FSM 201 con el mando, nadie en la pista, L2 + B a mano.
ROBOT_SSH=${ROBOT_SSH:-unitree@192.168.0.143}
PY_ROBOT=${PY_ROBOT:-/home/unitree/teleop_venv/bin/python}
ARGS=""; for a in "$@"; do ARGS+=" $(printf '%q' "$a")"; done
exec ssh -t "$ROBOT_SSH" "cd ~/seguidor_linea && unset CYCLONEDDS_URI && sudo -E PYTHONNOUSERSITE=1 $PY_ROBOT seguidor.py --iface eth0 --config config/seguidor.json$ARGS"
