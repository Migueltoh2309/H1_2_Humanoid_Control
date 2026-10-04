# Seguidor de línea del H1-2 — reto R40-RT-H1_2-0001

Enunciado: `src/Reto_Seguidor_Linea_H1_2_Robotics40.pdf`. El robot sigue una cinta oscura de
5 cm usando solo la D435i de la cabeza y se para solo en la barra de fin, moviéndose con la
marcha de Unitree (`LocoClient.Move`).

**El mismo código corre en el simulador y en el robot.** Solo cambian la fuente de imagen
(ROS 2 del simulador o `pyrealsense2` en el PC2) y el dominio DDS. La IMU (`rt/lowstate`) y el
`LocoClient` son los del SDK de Unitree en los dos casos: el simulador de `h1_2_sim2real` atiende
el servicio `loco` igual que el robot.

```text
seguidor_linea/
├── seguidor.py              programa principal (los 5 bloques; --sim | --iface eth0; --simulacro)
├── seguidor/                percepcion · estimacion · control · supervisor · registro · geometria · fuentes · robot_io
├── config/seguidor.yaml     TODAS las ganancias, límites y medidas
├── INTERFACES.md            interfaces entre bloques (sección 5 del reto)
├── RESPUESTAS.md            respuestas a la sección 6 con números del simulador
├── herramientas/            geometria_camara · calibrar_camara · escalon_vyaw · teleop · grabar_dataset ·
│                            evaluar_percepcion · metricas
├── sim/construir_pistas.py  genera sim/pistas/pista_nivel{1..4}.xml + .json (verdad de terreno)
├── scripts/                 sim_pista · tirada_sim · bateria_sim · instalar_en_robot · seguidor_robot
├── tests/                   19 pruebas automáticas (sin robot, sin MuJoCo, sin ROS)
└── datos/                   tiradas (CSV + JSON + PNG), conjuntos de datos, calibración (no se versiona)
```

## Simulación

El entorno es el de `h1_2_sim2real` (venv + ROS 2 + SDK de Unitree):
`source scripts/entorno.sh` en cada terminal.

```bash
python3 sim/construir_pistas.py            # una vez (o tras cambiar la inclinación: --inclinacion 45)

# Terminal 1: el robot en la pista (MuJoCo con su viewer)
./scripts/sim_pista.sh 3                   # nivel 3; emisor_ir:=false, marcha:=ninguna (solo visión)
# Terminal 2: el seguidor (pide la palabra SEGUIR, como el robot)
python3 seguidor.py --sim --nivel 3
python3 seguidor.py --sim --simulacro      # hito 3: calcula y registra, NO manda Move

# VERLO: MuJoCo (sigue al robot) + RViz (pista, línea detectada en verde, recordada en azul,
# punto de mira en rojo, estado del supervisor, imagen IR). Pide SEGUIR; Enter al final cierra todo.
./scripts/ver_tarea.sh 1                   # 2, 3, 4; extra: --escala 0.5

# Todo automático: lanza la pista, sigue, cierra y saca métricas con verdad de terreno
./scripts/tirada_sim.sh 2
./scripts/bateria_sim.sh                   # 4 niveles × 3 semillas de deriva -> tabla (12/12 con éxito)
```

**Qué es el robot en esta simulación.** Es la escena del suelo de `h1_2_sim2real` (base flotante,
manos Inspire, D435 con RGB, profundidad y 2 IR) con la **marcha cinemática**: la base sigue a
`LocoClient.Move` como un sólido rígido, con el retardo, la respuesta de primer orden, la deriva
de ~2°/s y el balanceo a 1.43 Hz que describe el reto. **No camina** (las piernas quedan quietas
en la postura de pie) y **no puede caerse**: hace de arnés. Sirve para cerrar el lazo y probar
percepción, estimación, control y supervisor. La política de marcha queda para más adelante.

Las herramientas, con el simulador corriendo:

```bash
python3 herramientas/geometria_camara.py           # 6.1.1 y 6.1.3: FOV, zona ciega, alcance
python3 herramientas/calibrar_camara.py --sim      # 6.1.2: altura e inclinación por el plano del suelo
python3 herramientas/escalon_vyaw.py --sim         # 6.3.1: retardo y constante de tiempo de la marcha
python3 herramientas/teleop.py --sim               # llevar el robot con el teclado (w s a d q e, espacio, x)
python3 herramientas/grabar_dataset.py --sim --nivel 3
python3 herramientas/evaluar_percepcion.py datos/dataset_n3_XXXX --video   # fijo vs adaptativo, verdad
python3 herramientas/metricas.py datos/tiradas/nivel3_XXXX.csv --png       # sección 9 + verdad de terreno
python3 -m pytest tests -q                         # 19 pruebas
```

## En el robot real (siguiendo el orden de hitos del reto)

**L2 + B es la parada de emergencia. FSM 201 se pone con el mando. Nada autónomo antes del hito 3.**

```bash
./scripts/instalar_en_robot.sh                     # copia a ~/seguidor_linea del robot (NO a ~/robotics40)
ssh -t unitree@192.168.0.143
cd ~/seguidor_linea
# Hito 1 — medir (robot de pie en FSM 201, suelo vacío delante)
sudo -E PYTHONNOUSERSITE=1 ~/teleop_venv/bin/python herramientas/calibrar_camara.py
#   -> copiar altura / inclinacion a config/seguidor.yaml (y volver a instalar_en_robot.sh)
# Conjunto de datos: un operador con wasd.py en otra terminal y, a la vez:
sudo -E PYTHONNOUSERSITE=1 ~/teleop_venv/bin/python herramientas/grabar_dataset.py --iface eth0 --nivel 1
~/teleop_venv/bin/python herramientas/escalon_vyaw.py --iface eth0      # 6.3.1 -> retardo_marcha, tau_marcha
# Hito 2 — percepción sobre los datos REALES, en el PC (traer datos/ con rsync)
# Hito 3 — simulacro: cámara en vivo, el robot lo lleva el operador, Move NO se manda
./scripts/seguidor_robot.sh --simulacro            # desde el PC
# Tiradas: a mitad de velocidad la primera de cada nivel
./scripts/seguidor_robot.sh --nivel 1 --escala 0.5
```

Lo que hay que medir en el robot antes de fiarse de la config (todo marcado en `seguidor.yaml`):
altura e inclinación de la cámara, `retardo_marcha` y `tau_marcha`, `factor_vx` (velocidad
real/pedida, con cinta), la máscara de pies y manos, y el tiempo por fotograma en el PC2.

**Sin probar en el robot.** Todo está verificado solo en simulación (ver la sección final de
`RESPUESTAS.md`: lo que el simulador no reproduce).
