# H1-2 Humanoid Control

Workspace ROS 2 (Python) para el humanoide **Unitree H1-2** de base fija:
movimiento de cuerpo completo, manipulación bimanual con evasión de colisiones
y visión RGB-D para agarrar fruta en movimiento, todo validado en **MuJoCo**
con visualización en **RViz2** antes de pasar al robot real.

Es la parte robótica de mi tesis de maestría en UTEC: *picking bimanual de
mandarinas sobre faja transportadora*. La captura de demostraciones humanas
está en [Multiview_Demo_Capture](https://github.com/Migueltoh2309/Multiview_Demo_Capture).

<p align="center">
  <a href="h1_2_algoritms/videos"><b>▶ Ver los videos del control bimanual</b></a>
</p>

## Qué incluye

| Área | Qué hace | Dónde |
|---|---|---|
| **Movimiento** | Cinemática directa e inversa de cuerpo completo con límites articulares, control cinemático, control en espacio nulo y QP | [`h1_2_algoritms/.../movimiento`](h1_2_algoritms/h1_2_algoritms/movimiento) |
| **Bimanualidad** | Evasión de colisiones en 3 capas: QP con *velocity dampers* entre brazos y entorno, coordinación de quién cede y planificación global RRT-Connect | [`h1_2_algoritms/.../bimanual`](h1_2_algoritms/h1_2_algoritms/bimanual) |
| **Visión** | Detección de la mandarina (color o YOLO + profundidad), localización 3D por ajuste de esfera, filtro de Kalman y *visual servoing* para el agarre | [`h1_2_algoritms/.../vision`](h1_2_algoritms/h1_2_algoritms/vision) |
| **Simulación** | Bridge ROS 2 ↔ MuJoCo: física y colisiones en MuJoCo, cámara RGB-D simulada, contactos y visualización en RViz2 | [`h1_2_mujoco_sim_bridge`](h1_2_mujoco_sim_bridge) |
| **Robot real** | Lectura de `/lowstate` y envío de comandos de bajo nivel `/lowcmd` al H1-2 | [`h1_2_real`](h1_2_real) |

```text
                ┌──────────── h1_2_algoritms ────────────┐
 cámara RGB-D ─►│ visión ──► bimanual (QP + 3 capas) ──► │──► /joint_cmd
 /joint_states ►│              ▲                          │
                │          movimiento (FK, IK, QP)        │
                └─────────────────────────────────────────┘
                                   │
          ┌────────────────────────┴────────────────────────┐
          ▼                                                 ▼
 h1_2_mujoco_sim_bridge                                h1_2_real
 MuJoCo: física, colisiones, cámara RGB-D              Unitree H1-2 (/lowcmd, /lowstate)
          │
          ▼
        RViz2
```

## Resultados destacados

Todos medidos en simulación con MuJoCo como verdad de terreno:

- **IK con límites articulares:** 97 % de soluciones usables, frente a 13 % de la IK original (DLS).
- **Evasión bimanual:** con la capa 1, los brazos que antes chocaban en cada vuelta no bajan de ~55 mm entre sí. Con la capa 2, la tarea de zona compartida pasa de 0/2 a 2/2 tareas completas. Con la capa 3, la mano esquiva una bandeja que solo ve la cámara.
- **Localización de la fruta:** el ajuste de esfera reduce el error de ~34 mm (mediana de profundidad) a ~2.4 mm con ruido tipo RealSense D435.
- **Visual servoing + agarre:** 5/6 agarres con fruta quieta y 3/3 sobre la faja en movimiento (hasta 0.07 m/s).

## Estructura

```text
├── h1_2_algoritms/              # Paquete ROS 2 con los algoritmos
│   ├── h1_2_algoritms/
│   │   ├── movimiento/          #   FK, IK, control cinemático, espacio nulo, QP, límites
│   │   ├── bimanual/            #   modelo de colisión, QP bimanual, coordinador, planificador
│   │   ├── vision/              #   detección, localización, geometría de agarre, servoing
│   │   ├── nodos/               #   nodos ROS 2 ejecutables (ros2 run)
│   │   └── markers.py
│   ├── launch/                  #   visual_servoing_sim.launch.py
│   ├── demos/                   #   evaluaciones reproducibles (sin ROS)
│   ├── docs/                    #   teoría y planes de bimanualidad, percepción y servoing
│   ├── results/                 #   CSV y gráficas de las evaluaciones
│   └── videos/                  #   videos comparativos del control bimanual
├── h1_2_mujoco_sim_bridge/      # Bridge ROS 2 ↔ MuJoCo + RViz2
└── h1_2_real/                   # Nodos para el robot físico
```

## Requisitos

- Ubuntu 22.04 + **ROS 2 Humble** y `colcon`
- Python: `pip install mujoco numpy scipy osqp opencv-python matplotlib` (y `ultralytics` para el detector YOLO)
- En el mismo workspace:
  - [`h1_2_utec`](https://github.com/oscar-ramos/h1_2_utec): modelo del robot (`h1_2_description`, MJCF y URDF)
  - [`unitree_ros2`](https://github.com/unitreerobotics/unitree_ros2): mensajes `unitree_hg` (solo para `h1_2_real` y `/lowcmd`)

> **Nota:** las escenas de la tesis (mesa quirúrgica, faja, mandarina y
> bandeja: `h1_2_scene_surgery_table*.xml`, `h1_2_scene_tray_obstacle.xml` y
> sus mallas) todavía no están publicadas. Sin ellas, los demos y la
> simulación de la faja no corren.

## Compilar y ejecutar

```bash
mkdir -p ~/humanoid_ws/src && cd ~/humanoid_ws/src
git clone https://github.com/Migueltoh2309/H1_2_Humanoid_Control.git
ln -s H1_2_Humanoid_Control/h1_2_algoritms H1_2_Humanoid_Control/h1_2_mujoco_sim_bridge \
      H1_2_Humanoid_Control/h1_2_real .
git clone https://github.com/oscar-ramos/h1_2_utec.git
cd ~/humanoid_ws
colcon build --symlink-install
source install/setup.bash
```

```bash
# Simulación con RViz2 (escena con faja estática o en movimiento)
ros2 launch h1_2_mujoco_sim_bridge sim_bridge.launch.py

# Visual servoing completo: bridge + detector + QP bimanual + agarre
ros2 launch h1_2_algoritms visual_servoing_sim.launch.py use_rviz:=true

# Evaluaciones sin ROS (MuJoCo directo)
cd ~/humanoid_ws/src/h1_2_algoritms
python3 demos/benchmark_ik.py 15
python3 demos/evaluate_bimanual_avoidance.py
```

Los nodos disponibles y la lista completa de demos están en
[`h1_2_algoritms/README.md`](h1_2_algoritms/README.md) y
[`h1_2_algoritms/demos/README.md`](h1_2_algoritms/demos/README.md).

## Documentación

| Tema | Documento |
|---|---|
| Teoría del control bimanual (capas 1-3) | [`BIMANUAL_TEORIA.md`](h1_2_algoritms/docs/BIMANUAL_TEORIA.md), [`GUIA_ESTUDIO_BIMANUAL.md`](h1_2_algoritms/docs/GUIA_ESTUDIO_BIMANUAL.md) |
| Percepción: color vs. YOLO | [`PERCEPTION_PLAN.md`](h1_2_algoritms/docs/PERCEPTION_PLAN.md) |
| Visual servoing y agarre | [`VISUAL_SERVOING_PLAN.md`](h1_2_algoritms/docs/VISUAL_SERVOING_PLAN.md) |
| Referencias | [`REFERENCIAS_BIMANUAL.md`](h1_2_algoritms/docs/REFERENCIAS_BIMANUAL.md), [`REFERENCIAS_VISUAL_SERVOING.md`](h1_2_algoritms/docs/REFERENCIAS_VISUAL_SERVOING.md) |
| Bridge de simulación | [`h1_2_mujoco_sim_bridge/README.md`](h1_2_mujoco_sim_bridge/README.md) |

## Autor

Miguel Olortegui — Maestría, UTEC
