#!/usr/bin/env bash
# Copia el agente SDK (robot/agente_sdk.py + comun/*.py) al robot, en ~/robotics40/sim2real,
# y comprueba que importa con el python del robot. NO mueve nada.
#   ./scripts/instalar_en_robot.sh
#   ROBOT_SSH=unitree@192.168.123.164 ./scripts/instalar_en_robot.sh
set -e
source "$(dirname "$0")/comun_robot.sh"
echo "-> $ROBOT_SSH:~/$DESTINO"
ssh -o ConnectTimeout=10 "$ROBOT_SSH" "mkdir -p ~/$DESTINO"
scp -q "$S2R/robot/agente_sdk.py" "$S2R"/comun/*.py "$ROBOT_SSH:$DESTINO/"
ssh "$ROBOT_SSH" "cd ~/$DESTINO && PYTHONNOUSERSITE=1 $PY_ROBOT -c '
import agente_sdk, protocolo_sim2real as P
print(\"agente importa bien con\", __import__(\"sys\").executable, \"· protocolo v\", P.VERSION)'"
