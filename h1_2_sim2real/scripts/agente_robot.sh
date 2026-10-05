#!/usr/bin/env bash
# Arranca el agente SDK EN EL ROBOT REAL, por ssh, en primer plano.
#   ./scripts/agente_robot.sh                 # arm_sdk: robot DE PIE en FSM 201 (L2+UP)
#   ./scripts/agente_robot.sh lowcmd          # cuerpo entero: robot COLGADO y en Debug de verdad
#   ./scripts/agente_robot.sh arm_sdk --vel-max 0.5 --t-rampa 15
# Luego, en este PC:  ros2 launch h1_2_sim2real puente.launch.py objetivo:=real
# Ctrl-C aqui = rampa de salida. L2+B en el mando = parada de emergencia.
source "$(dirname "$0")/comun_robot.sh"
MODO=${1:-arm_sdk}; shift || true
case "$MODO" in arm_sdk|lowcmd) ;; *) echo "modo: arm_sdk | lowcmd"; exit 1 ;; esac
PC_IP=$(ip route get "$ROBOT_IP" 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p')
[ -n "$PC_IP" ] || { echo "no encuentro la ruta a $ROBOT_IP"; exit 1; }
echo "=================================================================="
echo "  ESTO MUEVE EL ROBOT: $([ "$MODO" = arm_sdk ] && echo 'torso y brazos (rt/arm_sdk), robot de pie en FSM 201' \
                                                   || echo 'TODAS las juntas (rt/lowcmd), robot COLGADO y en Debug')"
echo "  manos Inspire por Modbus. Estado -> este PC ($PC_IP)"
echo "  L2 + B en el mando es la parada de emergencia. Tenlo en la mano."
echo "=================================================================="
read -r -p "Enter para arrancar el agente en $ROBOT_SSH (Ctrl-C para cancelar) " _
exec ssh -t -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "$ROBOT_SSH" \
    "cd ~/$DESTINO && unset CYCLONEDDS_URI RMW_IMPLEMENTATION && PYTHONNOUSERSITE=1 exec $PY_ROBOT -u agente_sdk.py \
     --iface eth0 --modo $MODO --pc $PC_IP $* 2>&1 | tee -a agente_\$(date +%Y%m%d).log"
