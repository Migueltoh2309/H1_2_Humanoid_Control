#!/usr/bin/env bash
# Copia el seguidor al robot, en SU PROPIA carpeta (~/seguidor_linea): el reto prohibe
# modificar ~/robotics40 (sec. 10). Sin datos/ ni sim/ (no hacen falta en el robot).
# Genera config/seguidor.json por si el python del robot no tiene PyYAML.
#   ./scripts/instalar_en_robot.sh
#   ROBOT_SSH=unitree@192.168.123.164 ./scripts/instalar_en_robot.sh
set -e
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
ROBOT_SSH=${ROBOT_SSH:-unitree@192.168.0.143}
PY_ROBOT=${PY_ROBOT:-/home/unitree/teleop_venv/bin/python}     # unitree_sdk2py.h1 + pyrealsense2 + cv2
python3 -c "import json, yaml; json.dump(yaml.safe_load(open('$AQUI/config/seguidor.yaml')), open('$AQUI/config/seguidor.json', 'w'), indent=1)"
rsync -a --delete --exclude datos --exclude sim --exclude __pycache__ --exclude .pytest_cache \
      "$AQUI/" "$ROBOT_SSH:seguidor_linea/"
ssh "$ROBOT_SSH" "cd ~/seguidor_linea && PYTHONNOUSERSITE=1 $PY_ROBOT -c '
import cv2, numpy, pyrealsense2
from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient
from seguidor import percepcion, estimacion, control, supervisor
print(\"OK en el robot: cv2\", cv2.__version__, \"numpy\", numpy.__version__)'"
echo "Instalado en $ROBOT_SSH:~/seguidor_linea"
