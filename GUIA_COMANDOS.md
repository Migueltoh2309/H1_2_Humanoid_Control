# Guía de comandos

Qué ejecutar, **en qué carpeta** y en qué orden, para el simulador `h1_2_sim2real` y la tarea
del seguidor de línea. Para el bridge de la tesis (`h1_2_mujoco_sim_bridge`) y los algoritmos,
ver el [README](README.md).

## ¿`ros2 run`, `ros2 launch` o scripts?

Las tres cosas, según la pieza:

| Pieza | Cómo se lanza | Por qué |
|---|---|---|
| Simulador (`h1_2_sim2real`) | `ros2 launch h1_2_sim2real sim.launch.py ...` | Es un paquete ROS 2: abre MuJoCo, publica cámaras y TF, y abre RViz2 si se pide |
| Puente ROS 2 ↔ robot | `ros2 launch h1_2_sim2real puente.launch.py ...` | Paquete ROS 2 |
| Seguidor de línea (`tareas/`) | `python3 seguidor.py ...` | **No es un paquete ROS**: tiene que correr también en el PC2 del robot, que lo ejecuta sin ROS (SDK de Unitree + cámara RealSense). En el simulador usa ROS solo para leer la cámara |
| Scripts `.sh` | `./scripts/...` | Son **atajos**: cargan el entorno y llaman a los `ros2 launch` y a `seguidor.py` con los argumentos correctos. No hacen nada que no se pueda hacer a mano (abajo están los equivalentes) |

`ros2 run h1_2_sim2real sim` también funciona, pero entonces los parámetros van con
`--ros-args -p nombre:=valor` y no abre RViz2 ni `robot_state_publisher`: para el día a día,
`ros2 launch` o los scripts.

## Carpetas

```text
~/humanoid_ws/                      workspace ROS 2 (colcon build se hace aquí)
└── src/
    ├── h1_2_sim2real  ─┐           enlaces simbólicos a este repo
    ├── tareas         ─┤           (~/Github/H1_2_Humanoid_Control/...)
    └── ...            ─┘
```

Ejecuta siempre desde `~/humanoid_ws/src/...` (la ruta del enlace), no desde `~/Github/...`.
Así funcionan todas las rutas relativas.

## 0. Una sola vez (instalación)

```bash
# Carpeta: ~/humanoid_ws
source ~/venvs/h12/bin/activate && source /opt/ros/humble/setup.bash
colcon build                                       # bridge, algoritmos, escenas...

# Carpeta: ~/humanoid_ws/src/h1_2_sim2real
./scripts/instalar.sh                              # SDK de Unitree + cyclonedds en terceros/ y compila

# Carpeta: ~/humanoid_ws/src/tareas/seguidor_linea   (solo en una máquina nueva o si cambias la cámara)
source scripts/entorno.sh && python3 sim/construir_pistas.py
```

Si no existe el venv `~/venvs/h12` (otra máquina): sigue `~/humanoid_ws/RECONSTRUIR.md`
(sección "Entorno Python"). Recompilar el simulador tras cambiar su código:
`~/humanoid_ws/src/h1_2_sim2real/scripts/compilar.sh`.

## 1. En CADA terminal nueva

```bash
source ~/humanoid_ws/src/h1_2_sim2real/entorno.sh
```

Carga, en este orden: el venv (que además envía MuJoCo y RViz a la GPU NVIDIA), ROS 2 Humble, el
SDK de Unitree, `~/humanoid_ws/install` y el simulador. Después funcionan tanto los scripts como
`ros2 launch ...`. **Cargador conectado**: con batería la GPU baja de rendimiento.

## 2. Ver el seguidor de línea (MuJoCo + RViz2, el robot caminando)

```bash
# Carpeta: ~/humanoid_ws/src/tareas/seguidor_linea
./scripts/ver_tarea.sh 1                     # nivel 1..4. Escribe SEGUIR; Enter al final cierra todo
./scripts/ver_tarea.sh 3 --escala 0.5        # a la mitad de velocidad
MARCHA=cinematica ./scripts/ver_tarea.sh 2   # base rígida que no camina
```

**Lo mismo a mano** (tres terminales, todas con `entorno.sh` cargado):

```bash
# Terminal 1 — el robot en la pista, con viewer de MuJoCo y RViz2 (cualquier carpeta)
ros2 launch h1_2_sim2real sim.launch.py \
    modelo:=$HOME/humanoid_ws/src/tareas/seguidor_linea/sim/pistas/pista_nivel1.xml \
    marcha:=politica visor:=true \
    rviz_config:=$HOME/humanoid_ws/src/tareas/seguidor_linea/rviz/seguidor.rviz \
    hz_lowstate:=200.0 hz_camara:=10.0 contactos:=false

# Terminal 2 — el seguidor (carpeta: ~/humanoid_ws/src/tareas/seguidor_linea)
python3 seguidor.py --sim --rviz --nivel 1             # pide la palabra SEGUIR
python3 seguidor.py --sim --rviz --nivel 1 --simulacro # calcula y registra, pero no manda Move

# Terminal 3 — opcional, mirar desde ROS
ros2 topic echo /sim/fuente                  # quién manda al robot
ros2 topic hz /camera/infra1/image_rect_raw  # cámara IR
```

## 3. Probar y medir sin ventanas

```bash
# Carpeta: ~/humanoid_ws/src/tareas/seguidor_linea
./scripts/tirada_sim.sh 2         # una tirada automática + métricas con la posición real
./scripts/bateria_sim.sh          # 4 niveles × 3 tiradas -> tabla
python3 -m pytest tests -q        # 21 pruebas (sin simulador)
python3 herramientas/metricas.py datos/tiradas/nivel2_XXXX.csv --png   # métricas de una tirada
```

Los resultados quedan en `datos/tiradas/` (CSV, JSON, PNG).

### Videos de cada hito (para entregar o presentar)

```bash
# Carpeta: ~/humanoid_ws/src/tareas/seguidor_linea   (sin ventanas, ~20 min; no hace falta T1)
./scripts/videos_hitos.sh                 # todos -> videos/*.mp4 (1280x720, H.264)
./scripts/videos_hitos.sh hito1 hito3     # solo algunos: hito1 datos escalon hito2 hito3 nivel1..4 cierre
```

Cada uno arranca y cierra su propio simulador: **no lo lances con otro simulador abierto**.

## 4. Medidas del reto (sección 6) en el simulador

```bash
# Terminal 1 (carpeta: ~/humanoid_ws/src/tareas/seguidor_linea) — el simulador, sin ventanas
./scripts/sim_pista.sh 1 use_viewer:=false

# Terminal 2 (misma carpeta)
python3 herramientas/geometria_camara.py            # FOV, zona ciega, alcance (no necesita simulador)
python3 herramientas/calibrar_camara.py --sim       # altura e inclinación de la cámara
python3 herramientas/escalon_vyaw.py --sim          # respuesta de la marcha al giro
python3 herramientas/calibrar_vx.py --sim           # velocidad real / pedida
python3 herramientas/teleop.py --sim                # manejar con el teclado: w s a d q e, espacio, x
python3 herramientas/grabar_dataset.py --sim --nivel 1   # graba mientras manejas; Ctrl+C para
python3 herramientas/evaluar_percepcion.py datos/dataset_n1_XXXX --video
```

## 5. El simulador por su cuenta

```bash
# Carpeta: ~/humanoid_ws/src/h1_2_sim2real   (o el ros2 launch equivalente desde cualquier carpeta)
./scripts/sim.sh suelo marcha:=politica      # = ros2 launch h1_2_sim2real sim.launch.py escena:=suelo marcha:=politica
./scripts/sim.sh robot                       # = ros2 launch h1_2_sim2real sim.launch.py   (mesa + faja)
./scripts/sim.sh suelo banda:=true           # colgado de la cinta elástica (teclas 7 / 8 / 9 en el viewer)
./scripts/sim.sh puente                      # = ros2 launch h1_2_sim2real puente.launch.py objetivo:=sim
```

Con el simulador y el puente en marcha, cualquier nodo mueve el robot por ROS 2:

```bash
ros2 topic pub -r 20 /joint_cmd sensor_msgs/msg/JointState \
  "{name: [left_shoulder_pitch_joint, left_elbow_joint], position: [-0.6, 1.0]}"
ros2 topic pub -w 1 --once /h1_2/soltar std_msgs/msg/Empty '{}'
```

Argumentos de `sim.launch.py`: `escena` (faja | suelo), `modelo` (ruta a un MJCF), `marcha`
(ninguna | cinematica | politica), `visor`, `use_viewer`, `rviz_config`, `banda`, `emisor_ir`,
`hz_camara`, `hz_lowstate`, `contactos`, `keyframe`, `grasp_assist`, `modo_robot`.
Lista completa: `ros2 launch h1_2_sim2real sim.launch.py --show-args`.

## 6. Robot real (no probado aún: L2 + B a mano, FSM 201 con el mando)

```bash
# Carpeta: ~/humanoid_ws/src/tareas/seguidor_linea  (en el PC; los scripts entran al robot por ssh)
./scripts/instalar_en_robot.sh                     # copia a ~/seguidor_linea del robot
./scripts/seguidor_robot.sh --simulacro            # hito 3: calcula órdenes pero NO mueve
./scripts/seguidor_robot.sh --nivel 1 --escala 0.5 # primeras tiradas, a la mitad
```

Antes de andar, mide en el robot (dentro del PC2, carpeta `~/seguidor_linea`) con
`calibrar_camara.py`, `escalon_vyaw.py` y `calibrar_vx.py` (`--iface eth0`), y copia los valores a
`config/seguidor.yaml`: los que hay ahora salen del simulador.

## Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `Package 'h1_2_sim2real' not found` | falta `source ~/humanoid_ws/src/h1_2_sim2real/entorno.sh` en esa terminal |
| `No llega rt/lowstate` | el simulador no está arrancado (o el robot, con `--iface eth0`) |
| `Address already in use` al lanzar el simulador | quedó otro abierto: `pkill -INT -f "[l]ib/h1_2_sim2real/sim"` |
| El robot va a tirones o las cámaras van lentas | cargador; ¿se activó el venv (GPU NVIDIA)? `echo $__NV_PRIME_RENDER_OFFLOAD` debe dar `1` |
| Las pistas no cargan (mallas) | `python3 sim/construir_pistas.py` (guardan la ruta absoluta de las mallas) |
