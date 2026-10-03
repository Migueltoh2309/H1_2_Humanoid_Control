# H1-2 — Simulación, control y percepción para el humanoide Unitree H1-2

Workspace ROS 2 (Python) para el humanoide **Unitree H1-2** de base fija:
cinemática y control de cuerpo completo, manipulación bimanual con evitación de
colisiones, percepción RGB-D y *visual servoing*, todo validable en **MuJoCo**
antes de pasar al robot real. Es la parte robótica de un proyecto de picking
bimanual de mandarinas sobre faja transportadora.

## Paquetes

| Paquete | Qué hace |
|---|---|
| [`h1_2_algoritms`](h1_2_algoritms) | Algoritmos: FK/IK de cuerpo completo, control cinemático, espacio nulo, QP, planificación y coordinación bimanual, detección de fruta (color + profundidad / YOLO), localización 3D y *visual servoing*. Incluye `demos/` de evaluación y sus resultados. |
| [`h1_2_mujoco_lowlevel_bridge`](h1_2_mujoco_bridge) | Bridge de **bajo nivel** ROS 2 ↔ MuJoCo: emula `/lowcmd` y `/lowstate` (`unitree_hg`) para que un controlador escrito para el robot real corra en simulación sin cambios. |
| [`h1_2_mujoco_sim_bridge`](h1_2_mujoco_sim_bridge) | Bridge de **simulación general**: interfaz `sensor_msgs/JointState`, cámara RGB-D simulada, contactos y visualización en RViz2. Escena con mesa, faja y mandarina. |
| [`h1_2_real`](h1_2_real) | Nodos para el robot físico: lectura de estado y envío de comandos de bajo nivel. |

```text
            h1_2_algoritms  (IK, QP, bimanual, percepción, servoing)
               │                          │
     /lowcmd · /lowstate          /joint_cmd · /joint_states · /camera/*
               ▼                          ▼
   h1_2_mujoco_bridge          h1_2_mujoco_sim_bridge  ──► RViz2
   (igual que el robot real)   (prototipado, cámara, contactos)
               │
               └── mismo controlador ──► h1_2_real ──► Unitree H1-2
```

## Requisitos

- Ubuntu 22.04 + **ROS 2 Humble** y `colcon`
- `pip install mujoco numpy scipy pyyaml opencv-python`
- En el mismo workspace:
  - `unitree_hg` (definiciones de mensajes de [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2))
  - los paquetes de modelo `h1_2_description` (URDF) y `h1_2_mujoco_bridge` (MJCF del H1-2 y escenas; no incluidos en este repo)

## Compilar y ejecutar

```bash
mkdir -p ~/humanoid_ws/src && cd ~/humanoid_ws/src
git clone https://github.com/Migueltoh2309/H1_2_MiTo.git
cd ~/humanoid_ws
colcon build --symlink-install
source install/setup.bash

# Simulación con RViz2 + cámara RGB-D (también: _static, _moving, _empty, _headless)
ros2 launch h1_2_mujoco_sim_bridge sim_bridge.launch.py

# Bridge de bajo nivel (/lowcmd, /lowstate)
ros2 launch h1_2_mujoco_lowlevel_bridge mujoco_lowlevel_bridge.launch.py

# Demo de visual servoing en simulación
ros2 launch h1_2_algoritms visual_servoing_sim.launch.py
```

Cada paquete tiene su propio README con la interfaz detallada (tópicos,
parámetros y configuración).

## Documentación

Notas técnicas en [`h1_2_algoritms/`](h1_2_algoritms):

- `BIMANUAL_TEORIA.md`, `GUIA_ESTUDIO_BIMANUAL.md`, `REFERENCIAS_BIMANUAL.md` — manipulación bimanual
- `VISUAL_SERVOING_PLAN.md`, `REFERENCIAS_VISUAL_SERVOING.md` — visual servoing
- `PERCEPTION_PLAN.md` — percepción

## Autor

Miguel Olortegui — UTEC
