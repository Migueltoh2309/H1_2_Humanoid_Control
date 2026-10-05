# h1_2_sim2real — el mismo comando para MuJoCo y para el H1-2 real

Vive en `~/humanoid_ws/src/h1_2_sim2real` (junto a `src/Codigos`), pero es un **workspace aparte**: el
`COLCON_IGNORE` de su raíz hace que el `colcon build` de `~/humanoid_ws` no lo vea (ni a `terceros/`), y se
compila con `./scripts/compilar.sh`, que deja su propio `build/` e `install/` aquí dentro.

Carpeta de pruebas que junta lo mejor de las dos partes, **sin tocar ninguna de las que ya funcionan**:

| Viene de | Qué se toma |
|---|---|
| `Codigos/` (lo del robot) | el protocolo real: SDK de Unitree por DDS (`rt/lowcmd`, `rt/arm_sdk`, `rt/lowstate`), manos Inspire por Modbus TCP, ganancias, límites y rampas ya probadas en el robot |
| `src/` de humanoid_ws (tu sim) | tu escena con **manos Inspire** y la **D435 (RGB, depth y 2 IR)**, faja + mandarina, asistencia de agarre y la interfaz `/joint_cmd` de tus nodos |

```text
             tus nodos ROS 2 (QP, visual servoing, LfD...)  /  ros2 topic pub
                   │ /joint_cmd (JointState, por nombre, parcial; dedos incluidos)
                   ▼                                  ▲ /joint_states, /h1_2/imu, /h1_2/agente
            ┌─────────────── puente (ROS 2) ─────────────┐
            │  UDP 47930 (órdenes)   UDP 47931 (estado)  │   ← idéntico en sim y real
            └───────────┬────────────────────────▲───────┘
   SIMULADOR: 127.0.0.1 │       ROBOT REAL: WiFi │ 192.168.0.143
                        ▼                        │
            ┌──────────── agente_sdk.py (sin ROS) ──────────┐  ← MISMO programa
            │ rampas de arm_sdk, límites, vel. máx, watchdog │     --sim       → dominio 1, lo, manos 127.0.1.21x
            └───┬──────────────── DDS ──────────────▲───┬────┘     --iface eth0 → dominio 0, manos 192.168.124.21x
     rt/arm_sdk │ rt/lowcmd               rt/lowstate │   │ Modbus TCP :6000 (manos)
                ▼                                     │   ▼
   ┌── sim: MuJoCo con tu escena ──┐      o      el H1-2 real (control de Unitree + Inspire)
   │ + D435 → /camera/* (ROS 2)    │
   └───────────────────────────────┘
```

La clave es que **el simulador habla exactamente el protocolo del robot**: publica `rt/lowstate`, obedece
`rt/lowcmd` y también `rt/arm_sdk` (el `unitree_mujoco` de Codigos no lo implementa; por eso `caja_cuadrado.Brazo`
tenía un modo `sim` aparte), y se hace pasar por las dos manos Inspire en Modbus TCP. Así:

* lo que mandes por ROS 2 (`/joint_cmd`) llega al simulador o al robot por el **mismo camino** (puente → agente);
* los scripts del SDK de `Codigos/` corren **sin cambios** contra este simulador (probado:
  `pc/simulacion_mujoco_h1_2/scripts/h1_2_arms_example.py --pose ../poses/1_saludo_derecha.json`).

## 0. Una sola vez

```bash
cd ~/humanoid_ws/src/h1_2_sim2real
./scripts/instalar.sh      # SDK de Unitree + cyclonedds 0.10.2 en terceros/ (NO toca ~/venvs/h12) y compila
```

(Ya está hecho en este PC. Para recompilar tras cambiar algo: `./scripts/compilar.sh`. Compila solo este
workspace, con su propio `build/` e `install/`; el `COLCON_IGNORE` de la raíz hace que el `colcon build` de
`~/humanoid_ws` no lo vea. Si mueves la carpeta, borra `build/ install/ log/` y recompila: guardan rutas absolutas.)

En **cada terminal**: `source ~/humanoid_ws/src/h1_2_sim2real/entorno.sh` (activa el venv, ROS 2, el SDK de
`terceros/`, `~/humanoid_ws/install` —para tener tus nodos de `h1_2_algoritms`— y este workspace).

## 1. Simulación

```bash
# Terminal 1: el robot simulado (equivale a encender el H1-2)
./scripts/sim.sh robot                          # con el viewer de MuJoCo
./scripts/sim.sh robot use_viewer:=false keyframe:=mandarina_moving grasp_assist:=true

# Terminal 2: agente (local, contra el sim) + puente + robot_state_publisher + RViz
./scripts/sim.sh puente                          # modo arm_sdk (torso + brazos)
./scripts/sim.sh puente modo:=lowcmd             # cuerpo entero

# Terminal 3: mover
ros2 topic pub -r 20 /joint_cmd sensor_msgs/msg/JointState \
  "{name: [left_shoulder_pitch_joint, left_elbow_joint, L_index_proximal_joint], position: [-0.6, 1.0, 1.2]}"
ros2 topic echo /h1_2/agente                     # modo, peso de arm_sdk, banderas, latencia
ros2 topic pub -w 1 --once /h1_2/soltar std_msgs/msg/Empty '{}'   # el agente devuelve el control y termina
# (-w 1: espera a que el puente este suscrito; sin el, --once puede publicar antes y perderse)
```

Para usar los **scripts de Codigos** (SDK puro) en vez del puente: `./scripts/sim.sh codigos` (sim + RViz con la
verdad de terreno) y, en otra terminal con `entorno.sh`, el script tal cual (interfaz `lo` o sin argumentos).

## 1b. Escena del suelo: el robot solo, de pie, con la física completa

```bash
./scripts/sim.sh suelo                      # de pie en el suelo (base flotante, gravedad, contactos, piernas)
./scripts/sim.sh suelo banda:=true          # colgado de la cinta elástica, pies ~16 cm sobre el suelo
```

`mjcf/h1_2_escena_suelo.xml` lo **genera** `scripts/construir_escena_suelo.py` a partir de la escena con manos
(no editarlo a mano). Respecto a la escena de la faja:

* **base flotante** (`floating_base_joint`): el robot se sostiene sobre sus pies o se cae; nada soldado;
* **colisiones** de todo el cuerpo (mallas) y, en los pies, una **caja con las medidas de la suela** (26 × 8.6 cm,
  sacadas de los vértices de la malla). Con la malla del pie los puntos de contacto saltaban entre 4 y 5 y la IMU
  veía picos de ±3 m/s² con el robot quieto; con la caja, acc_z = 9.81 constante;
* **IMU completa** en el sitio `imu` de `torso_link`, donde la pone el `h1_2.xml` de Unitree: cuaternión, giróscopo
  y acelerómetro van a `rt/lowstate.imu_state`. Ojo: al girar la cintura, gira la IMU (comprobar en el robot real
  si su IMU va en el torso o en la pelvis);
* **postura `de_pie`**: cadera −0.16, rodilla 0.36, tobillo −0.20 (pie plano) y la pelvis a 1.0151 m, calculada para
  que la suela quede exactamente en z = 0;
* TF `world → pelvis` (verdad de terreno) y `/contacts` en `world`.

**Quién lo sostiene.** El "control interno" del simulador (el que emula a FSM 201) usa en esta escena piernas
rígidas (kp tobillo 800): con los pies apoyados el cuerpo es un péndulo invertido sobre los tobillos y hace falta
más rigidez que m·g·h ≈ 590 N·m/rad. Con las ganancias de `h1_2_juntas` (tobillo 80) se cae de bruces en 1.6 s.
Medido: quieto aguanta indefinidamente (pitch −1°), con los brazos adelante y la cintura girada también, aguanta
un empujón de 50 N × 0.2 s y se cae con 100 N (no da pasos: en el robot real eso lo hace la política de Unitree).
En `lowcmd` mandan **tus** ganancias: como en el robot real en Debug, cuélgalo antes.

**Cinta elástica** (como `unitree_mujoco`, teclas en el viewer): **9** engancha/suelta, **7** sube, **8** baja.
Para dejarlo de pie desde colgado: baja con 8 hasta que apoye y luego suelta con 9 (soltarlo desde arriba = cae
10-15 cm y se vuelca, igual que en `unitree_mujoco`).

## 1c. Servicio `loco` y marcha cinemática (para tareas de alto nivel)

El simulador atiende el **servicio RPC `loco`** del SDK (el de `LocoClient`): `Move`/`StopMove`
(SetVelocity 8105), `SetFsmId`, `SetStandHeight` y los `Get*`. Así, código de alto nivel como
`wasd.py` o `cuadrado.py` (con dominio 1) o `src/tareas/seguidor_linea` corre **sin cambios**.

Con `marcha:=cinematica` (solo con base flotante), la base sigue esas velocidades **como un
sólido rígido**: retardo 0.3 s, primer orden con τ 0.35 s, deriva de rumbo de 2°/s con signo
aleatorio por tirada y balanceo de la cámara a 1.43 Hz (valores del reto, supuestos hasta
medirlos). **No camina** (piernas quietas, base a 4 mm del suelo) y no puede caerse. Sirve para
cerrar lazos de percepción y control mientras no haya política de marcha. El acelerómetro se
calcula a partir del movimiento impuesto. Solo FSM 201/204 anda (`fsm_inicial`, 201 por defecto).

```bash
ros2 launch h1_2_sim2real sim.launch.py escena:=suelo marcha:=cinematica
ros2 launch h1_2_sim2real sim.launch.py modelo:=/ruta/a/otra_escena.xml marcha:=cinematica marcha_semilla:=3
```

Otros parámetros: `emisor_ir:=true|false` (patrón de puntos del proyector IR en las dos IR),
`marcha_retardo`, `marcha_tau`, `marcha_deriva_deg`, `marcha_cadencia`, `marcha_roll_deg`,
`marcha_pitch_deg`, `marcha_z_mm` (en `sim_node.py`).

### `marcha:=politica`: el robot CAMINA (política de unitree_rl_gym)

```bash
ros2 launch h1_2_sim2real sim.launch.py escena:=suelo marcha:=politica
```

La política de marcha del H1-2 que publica Unitree (`unitreerobotics/unitree_rl_gym`,
`deploy/pre_train/h1_2/motion.pt`, BSD-3) hace de **control interno en FSM 201**:
`LocoClient.Move(vx, vy, vyaw)` → servicio `loco` → política (LSTM 64 + MLP, 47 entradas, 50 Hz)
→ objetivos de las 12 juntas de las piernas con su PD (kp 200/200/200/300/40/40). Torso y brazos
se sostienen con las ganancias de `deploy_real` de Unitree, y `rt/arm_sdk` se mezcla encima como
en el robot. **Física completa**: da pasos de verdad, puede tropezar y caerse, y la deriva y el
balanceo salen solos. La red está reimplementada en numpy (`politica_marcha.py`, pesos en
`politicas/h1_2_marcha.npz`), sin PyTorch en el simulador, y verificada contra el `.pt` (< 3e-6).

Medido con el `LocoClient` del SDK sin cambios (escena del suelo, tiempo real):

| Orden | Resultado |
|---|---|
| sin Move | marca el paso; deriva 2 cm/s y 1°/s |
| `Move(0.3, 0, 0)` | 0.26 m/s (factor 0.886 en 6 s) |
| `Move(0, 0, 0.4)` | +18°/s (~0.72–0.8 del giro pedido; τ ≈ 0.12 s) |
| `Move(0, 0.2, 0)` | 0.15 m/s lateral |

Es la marcha de la política pública de Unitree, **no** necesariamente la que lleva el firmware
del robot. Las constantes de la planta (factor de velocidad, ganancia y respuesta del giro) hay
que medirlas en el robot real.

## 2. Robot real (mismo puente, mismos comandos)

```bash
./scripts/instalar_en_robot.sh                   # copia agente + comun/ a ~/robotics40/sim2real y prueba el import
./scripts/agente_robot.sh                        # arm_sdk: robot DE PIE en FSM 201 (L2+UP). Pide Enter.
./scripts/agente_robot.sh lowcmd                 # lowcmd: robot COLGADO y en Debug de verdad (CheckMode '')
# en este PC:
ros2 launch h1_2_sim2real puente.launch.py objetivo:=real robot_ip:=192.168.0.143
```

A partir de ahí, **el mismo** `/joint_cmd` que movía el simulador mueve el robot.
`ROBOT_SSH=unitree@192.168.123.164` para ir por cable; `PY_ROBOT=` para otro python (por defecto
`teleop_venv`, el que ya usa `caja_cuadrado.py`).

> **Sin probar todavía en el robot real.** Todo lo de abajo está verificado en simulación; en el robot falta la
> primera prueba (robot colgado, `--vel-max 0.3`, una junta). L2 + B es la parada de emergencia.

## 3. Interfaz ROS 2

| Tópico | Tipo | Quién | Qué |
|---|---|---|---|
| `/joint_cmd` | `sensor_msgs/JointState` | tú → puente | por **nombre** y parcial, como `h1_2_mujoco_sim_bridge`. `position` = q; `velocity` = dq y `effort` = par feedforward (opcionales, caducan a los 0.2 s). Dedos: `L_/R_` + `thumb_proximal_yaw/pitch`, `index/middle/ring/pinky_proximal_joint` en rad (se convierten a 0-1000 de la Inspire) |
| `/joint_states` | `sensor_msgs/JointState` | puente | 27 juntas (de `rt/lowstate`) + 12 dedos (Modbus; en `effort`, la fuerza en gramos-fuerza) |
| `/h1_2/imu` | `sensor_msgs/Imu` | puente | IMU de la pelvis |
| `/h1_2/agente` | `std_msgs/String` (JSON) | puente | conectado, modo, peso de arm_sdk, banderas, latencia |
| `/h1_2/soltar` | `std_msgs/Empty` | tú → puente | rampa de salida y fin del agente |
| `/camera/color/image_raw` + `camera_info` | `Image rgb8` | sim | D435 RGB 640x480 |
| `/camera/depth/image_raw` + `camera_info` | `Image 32FC1` (m) | sim | alineada al RGB (mismos tópicos que tus nodos de percepción) |
| `/camera/infra1/image_rect_raw`, `/camera/infra2/...` + `camera_info` | `Image mono8` | sim | par IR 848x480, línea de base 50 mm (`P[3] = -fx·0.05` en infra2), nombres de `realsense2_camera` |
| `/sim/joint_states`, `/sim/fuente`, `/contacts`, `/sim/ground_truth/mandarina` | | sim | solo simulación: verdad de terreno, quién manda, contactos |

TF estático `torso_link → camera_{color,depth,infra1,infra2}_optical_frame` desde el MJCF; el resto, de
`robot_state_publisher` con `urdf/h1_2_manos.urdf` (el `h1_2.urdf` con manos, dedos *mimic* incluidos).

## 4. El agente: la seguridad vive en el lado del robot

`robot/agente_sdk.py` (sin ROS, sin numpy; solo `unitree_sdk2py`):

* **arm_sdk** (por defecto): torso + brazos (12-26), peso en el slot 27. Comprueba que el control interno publica
  `rt/lowcmd` (si no, `arm_sdk` no hace nada: mismo control que `prueba_brazos_armsdk.py`), rampa de entrada
  0→1 en 12 s siguiendo la postura medida con un tope de 2° (`caja_cuadrado.Brazo.rampa_peso`) y de salida en 5 s.
* **lowcmd**: las 27 juntas; las no comandadas se sostienen en la postura medida al arrancar. Al salir,
  amortiguación (kp 0, kd 2) 2 s.
* Consignas **recortadas** a los límites (`comun/h1_2_juntas.py`: los de `caja_cuadrado.py` + URDF, margen
  0.05 rad) y con **velocidad máxima** (`--vel-max`, 1 rad/s): un salto en `/joint_cmd` no da un latigazo.
* **Watchdog**: sin órdenes 0.5 s → sostiene; 3 s → en arm_sdk suelta (en lowcmd sigue sosteniendo: está colgado).
  El puente manda un latido a 50 Hz, así que esto salta si cae el puente o la WiFi, no si tu nodo deja de publicar
  (entonces se sostiene la última consigna, como en `h1_2_mujoco_sim_bridge`).
* Motor en fallo o inclinación > 10° → suelta. Ctrl-C, SIGTERM o `/h1_2/soltar` → rampa de salida.
  Si se corta el ssh, el agente **no muere** (ignora SIGHUP) y su watchdog decide.
* El puente avisa si una junta comandada se queda > 0.2 rad lejos de la consigna más de 2 s (p.ej. el robot no
  está en el modo correcto: así falló el selector el 2026-09-25).

## 5. El simulador

`ros2 launch h1_2_sim2real sim.launch.py` — argumentos: `use_viewer`, `visor`, `modo_robot`, `keyframe`,
`grasp_assist`, `camaras`, `ir`, `hz_camara`.

* **modo_robot**: `auto` (por defecto; si llega `rt/lowcmd` externo manda él, si no, control interno +
  `arm_sdk`), `ai` (ignora `rt/lowcmd`, como el robot con CheckMode `ai`), `debug` (solo `rt/lowcmd`; sin él,
  motores sin par).
* El "control interno" sostiene la postura de arranque con las ganancias de `h1_2_juntas` y publica `rt/lowcmd`
  (marcado en `reserve[0]`, el propio simulador lo ignora) para que los controles de "¿hay control interno?"
  funcionen igual que en el robot. Si el publicador de `arm_sdk` se calla, el sim devuelve el control en 1 s.
* Motores: **PD puro** `tau = tau_ff + kp·Δq + kd·Δdq`, saturado al par del URDF, como `unitree_mujoco` y los
  motores reales. Sin compensación de gravedad (a diferencia de tu `h1_2_mujoco_sim_bridge`): con kp 100 el
  hombro se queda ~5° corto con el brazo adelante, del orden de lo medido en el robot (prueba 043621).
* Manos: Modbus TCP en `127.0.1.211:6000` (izq) y `127.0.1.210:6000` (der) con ANGLE_SET/ACT, FORCE_ACT,
  SPEED_SET (recorrido completo en 0.6 s a velocidad 1000)...
* Escenas: `escena:=faja` (base soldada: equivale al robot colgado o de pie sin dar pasos; IMU constante) y
  `escena:=suelo` (base flotante, ver 1b).
* **Tres procesos** unidos por memoria compartida (`compartido.py`): física a 500 Hz, DDS (`rt/lowstate` a
  ~500 Hz) y cámaras. En un solo proceso no llegaba: serializar un `LowState_` en Python cuesta 0.7 ms (más que
  un `mj_step`) y el GIL lo pisaba todo.

### Rendimiento medido (portátil, GTX 1650)

| | con cargador | con batería |
|---|---|---|
| física | tiempo real (~490 pasos/s de 500) | igual |
| `rt/lowstate` | 476-493 Hz | 310-480 Hz |
| cámaras (RGB + depth + 2 IR) | 15 Hz por defecto; **29.9 Hz** con `hz_camara:=30.0` (los 30 fps de la D435 real), GPU en P3 al ~37 % | ~10 Hz (la GTX 1650 se queda en P8 a 300 MHz) |
| latencia puente ↔ agente | 3-12 ms (local) | |

Trampas resueltas por el camino:
* **Interbloqueo del proceso DDS** (se quedaba mudo, `rt/lowstate` a 0, al arrancar un script de Codigos con
  `rt/lowcmd`). Traza con `faulthandler`: atascado en el `write()` del `rt/lowcmd` del control interno. Ese proceso
  escribía `rt/lowcmd` y tenía un lector con *listener* Python en el mismo tópico; cyclone entrega lo local de forma
  síncrona en el hilo que escribe (con el GIL) y, si a la vez llegaba el `rt/lowcmd` externo, su hilo de recepción
  tomaba el cerrojo del lector y esperaba el GIL. Ahora el proceso DDS no tiene listeners (sondea con `take()`),
  ignora lo propio (`IgnoreLocal`) y escribe best-effort. 3 repeticiones de la secuencia que fallaba, 0 cuelgues.
  Por si acaso, un vigilante reinicia el proceso DDS si `rt/lowstate` se para 2 s (y deja la traza en el log).
  *Los scripts de Codigos y el agente no tienen ese patrón (no escriben y escuchan con listener el mismo tópico).*
* El render offscreen usa **EGL** (`MUJOCO_GL=egl` en el proceso de cámaras): con el GLFW por defecto cae en la
  Intel integrada (65 ms por imagen).
* `Image.data = bytes` en rclpy (Humble) valida byte a byte: **120 ms** por imagen 640x480. Con
  `array.array('B', ...)`, 0.2 ms. Arreglado también en `h1_2_mujoco_sim_bridge/msg_builders.py` (2026-10-04):
  su cámara pasó a cumplir los 7.5 Hz configurados y llega a ~19 Hz si se le piden 30.

## 6. Ficheros

```text
h1_2_sim2real/
├── entorno.sh                  source en cada terminal
├── comun/                      FICHEROS MAESTROS (los usan sim, puente y agente; al robot van copiados)
│   ├── h1_2_juntas.py          nombres, índices DDS, límites, kp/kd, mapeo de las manos
│   ├── protocolo_sim2real.py   datagramas UDP orden/estado (VERSION)
│   └── modbus_mini.py          cliente y servidor Modbus TCP mínimos (sin pymodbus)
├── robot/agente_sdk.py         el agente SDK (sim y real)
├── scripts/                    instalar, compilar, sim.sh, instalar_en_robot.sh, agente_robot.sh
├── terceros/                   unitree_sdk2_python (git) + pydeps/ (cyclonedds 0.10.2)
└── src/h1_2_sim2real/          paquete ROS 2
    ├── h1_2_sim2real/          sim_node, robot_simulado, procesos, compartido, puente_node, mensajes
    ├── mjcf/h1_2_escena_faja_manos.xml   copia de h1_2_scenes/.../h1_2_scene_surgery_table_hands.xml
    ├── mjcf/h1_2_escena_suelo.xml        GENERADA por scripts/construir_escena_suelo.py (de pie, base flotante)
    ├── meshes/                 las 56 mallas que usan el MJCF y el URDF
    ├── urdf/h1_2_manos.urdf    h1_2.urdf con package://h1_2_sim2real/meshes
    └── launch/  rviz/
```

El MJCF, las mallas y el URDF son **copias**: si cambias la escena en `h1_2_scenes` (regenerando con
`build_scene_hands.py`), vuelve a copiarla aquí.

## 7. Pendiente

* **Primera prueba en el robot real** (colgado, lowcmd, `--vel-max 0.3`, una junta; luego arm_sdk en FSM 201).
* La **D435 real** en ROS 2: hoy el robot la sirve por HTTP (`Codigos/camara.py`). Para tener los mismos
  `/camera/*` con el robot real hace falta un emisor en el robot (RGB comprimido + depth) y su receptor, al estilo
  de `rviz_h1_2`.
* Compensación de gravedad opcional en el puente (como `effort` de `/joint_cmd`), igual en sim y real.
* Una política de equilibrio/marcha para la escena del suelo (hoy la sostiene un PD rígido: no da pasos).
