#!/usr/bin/env bash
# Hito 3 en el simulador (sec. 8): los registros que se entregan.
#   1. simulacro: cámara y planes en vivo, trayectorias registradas y reproducidas en MuJoCo, sin
#      publicar arm_sdk; la mano abre y cierra en el aire;
#   2. un único movimiento real a la pose de observación y vuelta, sin objeto;
#   3. aborto con salida segura al TAPAR LA CÁMARA (a los 60 s de una tirada del nivel 2);
#   4. aborto con salida segura al SIMULAR UN GIRO DEL CUERPO (6 grados en la IMU a los 60 s).
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AQUI"
./scripts/tirada_sim.sh 2 0 --simulacro --t-max 240
D=$(ls -td datos/tiradas/nivel2_* | head -1)
source scripts/entorno.sh > /dev/null 2>&1
MUJOCO_GL=egl python3 herramientas/reproducir.py "$D"
./scripts/tirada_sim.sh 2 0 --solo-observacion
./scripts/tirada_sim.sh 2 0 --tapar-camara 60
./scripts/tirada_sim.sh 2 0 --simular-giro 60
