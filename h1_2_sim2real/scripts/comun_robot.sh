# Variables del robot (mismas que Codigos/comun.sh). Se carga con: source "$(dirname "$0")/comun_robot.sh"
ROBOT_SSH=${ROBOT_SSH:-unitree@192.168.0.143}         # WiFi UTEC_H1_2; por cable: unitree@192.168.123.164
ROBOT_IP=${ROBOT_SSH#*@}
DESTINO=${DESTINO:-robotics40/sim2real}                # en el robot, junto al resto de robotics40
PY_ROBOT=${PY_ROBOT:-/home/unitree/teleop_venv/bin/python}   # tiene unitree_sdk2py (el que usa caja_cuadrado.py)
S2R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
