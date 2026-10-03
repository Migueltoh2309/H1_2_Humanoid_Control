# h1_2_algoritms

Paquete ROS 2 con los algoritmos de movimiento, manipulación bimanual y visión
para el Unitree H1-2. Los módulos de `movimiento/`, `bimanual/` y `vision/` no
dependen de ROS: los usan igual los nodos en vivo y los demos de evaluación,
así que lo que se mide en los demos es exactamente lo que corre en el robot.

## Módulos

### `movimiento/`: cinemática y control de cuerpo completo

| Módulo | Contenido |
|---|---|
| `fk_functions.py` | Cinemática directa de ambos brazos (marco `torso_link`), utilidades de pose y cuaterniones |
| `ik_functions.py` | Cinemática inversa con límites articulares: proyección en espacio nulo, bloqueo de articulaciones en tope y saturación por iteración |
| `kine_control_functions.py` | Jacobiano y ley de control cinemático diferencial |
| `null_control_functions.py` | Control con tarea secundaria en el espacio nulo |
| `QP_functions.py` | Control cinemático como QP (OSQP) |
| `joint_limits.py` | Límites articulares de los brazos, tomados del MJCF |

### `bimanual/`: evasión de colisiones en 3 capas

| Módulo | Capa | Contenido |
|---|---|---|
| `collision_model.py` | — | Eslabones aproximados como cápsulas; distancias brazo-brazo, brazo-torso y brazo-entorno |
| `bimanual_avoidance.py` | 1 | QP único de 14 GDL para los dos brazos, con *velocity dampers* como restricciones lineales |
| `depth_obstacles.py` | 1 | Nube de la cámara de profundidad → vóxeles de obstáculo, con memoria frente a oclusiones |
| `bimanual_coordinator.py` | 2 | Coordinación por prioridad: detecta bloqueos y decide qué brazo pasa y cuál cede |
| `bimanual_planner.py` | 3 | Planificador global RRT-Connect cuando el camino directo queda tapado |

### `vision/`: percepción y agarre

| Módulo | Contenido |
|---|---|
| `perception_common.py` | Detección compartida: método A (color HSV + profundidad) y método B (YOLO + profundidad) |
| `fruit_localization.py` | Cuatro estimadores del centro 3D de la fruta: mediana de profundidad, mediana + radio, ajuste de esfera y estéreo IR |
| `grasp_geometry.py` | Pose de la muñeca para el pre-agarre y el agarre con la mano Inspire |
| `visual_servoing.py` | `FruitTracker` (Kalman de velocidad constante) y máquina de estados del agarre: APPROACH → DESCEND → CLOSE → LIFT |

## Nodos

| Ejecutable | Qué hace | Necesita |
|---|---|---|
| `fk_whole_body_h1_2` | Cinemática directa: publica `joint_states` y la pose de los efectores | RViz2 |
| `ik_whole_body_h1_2` | Cinemática inversa de ambos brazos hacia poses objetivo | RViz2 |
| `kine_control_whole_body` | Control cinemático diferencial de ambos brazos | RViz2 |
| `null_control_whole_body` | Control cinemático con tarea secundaria en el espacio nulo | RViz2 |
| `QP_whole_body` | Control cinemático resuelto como QP | RViz2 |
| `telekeyop_whole_body` | Teleoperación de los efectores con el teclado (`w`/`s`/`a`/`d`…) | RViz2 |
| `ik_whole_body_bridge_demo` | IK sobre la simulación con física: manda `/joint_cmd` y detiene el movimiento si un brazo choca | `h1_2_mujoco_sim_bridge` |
| `fk_right_arm_raise` / `plot_right_arm_raise` | Ensayo de levantar el brazo derecho, registro en CSV y gráficas para comparar con el robot real | `h1_2_mujoco_sim_bridge` |
| `QP_bimanual_avoidance` | QP bimanual con las 3 capas de evasión; con `scenario:=grasp` hace el visual servoing completo | `h1_2_mujoco_sim_bridge` |
| `color_depth_detector_node` | Detector de la mandarina por color + profundidad (método A) | cámara RGB-D |
| `yolo_color_depth_detector_node` | Detector de la mandarina con YOLO + profundidad (método B) | cámara RGB-D, `ultralytics` |
| `ik_lowcmd_node` | IK → `/lowcmd` con PD + término integral | consumidor de `/lowcmd` (robot real) |

```bash
ros2 run h1_2_algoritms QP_bimanual_avoidance
ros2 launch h1_2_algoritms visual_servoing_sim.launch.py use_rviz:=true belt:=moving
```

## Carpetas

| Carpeta | Contenido |
|---|---|
| [`demos/`](demos) | Evaluaciones reproducibles contra MuJoCo, sin ROS |
| [`docs/`](docs) | Teoría y planes: bimanualidad, percepción, visual servoing y referencias |
| [`results/`](results) | CSV y gráficas que generan los demos (`bimanual/`, `percepcion/`, `servoing/`, `trayectorias/`) |
| [`videos/`](videos) | Videos comparativos del control bimanual |
