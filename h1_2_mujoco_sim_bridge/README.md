# h1_2_mujoco_sim_bridge

Bridge **ROS 2 ↔ MuJoCo de simulación general** para el Unitree H1-2, pensado
para manipulación bimanual / aprendizaje por demostración (picking): MuJoCo
corre la física y las colisiones, RViz2 visualiza TF/joint_states/cámara/
contactos.

A diferencia de [`h1_2_mujoco_lowlevel_bridge`](../h1_2_mujoco_bridge/README.md)
(que emula `/lowcmd`+`/lowstate` con mensajes `unitree_hg` para que un
controlador escrito para el robot real corra sin cambios), este paquete **no
replica ningún protocolo de hardware**: la interfaz es genérica
(`sensor_msgs/JointState`), pensada para prototipar/experimentar rápido desde
ROS 2, con MuJoCo aportando lo que el robot real y RViz solo no dan:
colisiones físicas reales y una cámara RGB-D renderizada desde dentro de la
simulación.

```text
tu nodo (planner, teleop, política de LfD, ...)
   │  publica /joint_cmd (sensor_msgs/JointState, parcial por nombre)
   │  recibe  /joint_states, /camera/color/*, /camera/depth/*, /contacts
   ▼
h1_2_mujoco_sim_bridge (este paquete)         robot_state_publisher
   │  PD por joint -> data.ctrl -> mj_step         │  usa /joint_states + URDF
   ▼                                                ▼
MuJoCo (h1_2_scene_surgery_table.xml:             TF completo
 mesa quirúrgica + faja + mandarina + cámara)        │
                                                       ▼
                                                     RViz2
```

## Por qué existe (vs. el bridge de bajo nivel)

| | `h1_2_mujoco_lowlevel_bridge` | `h1_2_mujoco_sim_bridge` (este) |
|---|---|---|
| Propósito | Validar un controlador de bajo nivel **tal cual correría en el robot real** | Simular/experimentar: manipulación, colisiones, cámara, ambientes propios |
| Mensajes | `unitree_hg/LowCmd`, `unitree_hg/LowState` | `sensor_msgs/JointState` genérico |
| Visualizador | Viewer nativo de MuJoCo | RViz2 (TF vía `robot_state_publisher`) + viewer nativo opcional |
| Cámara / contactos | No | Sí: RGB-D + `visualization_msgs/MarkerArray` |
| Base | Fija (pelvis soldada) | Fija (mismo MJCF) |

Ambos pueden convivir en el workspace; no comparten nodos ni tópicos.

## Interfaz

* **Comando** `/joint_cmd` (`sensor_msgs/JointState`, sub): `q_des` en
  `position[]`, `dq_des` opcional en `velocity[]` (si se omite o no coincide
  en longitud, se asume 0), `tau_ff` opcional en `effort[]`. Los joints que
  **no** aparecen en el mensaje conservan su último objetivo — para mover
  solo el brazo izquierdo basta con publicar esos 7 nombres, el resto no se
  toca. No hay "timeout a torque cero": a diferencia del robot real, aquí no
  hay riesgo de hardware, así que mantener la última posición pedida ya es
  el comportamiento seguro por defecto (sin watchdog).
* **Estado** `/joint_states` (`sensor_msgs/JointState`, pub): posición/
  velocidad/torque aplicado de los 27 motores, en el orden de
  `joint_names` — lo consume `robot_state_publisher` para TF y sirve para
  registrar demostraciones.
* **Cámara** `robot_rgbd_camera` del MJCF (torso), renderizada offscreen:
  * `/camera/color/image_raw` (`rgb8`) + `/camera/color/camera_info`
  * `/camera/depth/image_raw` (`32FC1`, metros) + `/camera/depth/camera_info`
  * RGB y depth vienen de la **misma cámara virtual**: ya están alineados/
    registrados, sin necesidad de calibración extrínseca extra.
  * TF estático `torso_link -> camera_color_optical_frame` /
    `camera_depth_optical_frame`, calculado directamente desde la pose de la
    cámara en el MJCF (`cam_pos`/`cam_quat`) más la conversión fija a
    convención óptica de ROS (REP-103); no depende de que el URDF tenga un
    link de cámara con la misma orientación.
* **Contactos** `/contacts` (`visualization_msgs/MarkerArray`, pub,
  `publish_contacts:=true` por defecto): una esfera por punto de contacto
  activo, color/tamaño según la fuerza normal — para ver en RViz cuándo el
  robot toca la mesa/el objeto durante manipulación.
* **TF dinámico** (joints, cadena `pelvis -> ... -> manos`): lo publica
  `robot_state_publisher` (nodo externo, en el launch) a partir de
  `/joint_states` + el URDF `h1_2_handless_bright.urdf`. El robot es de
  **base fija** (ese URDF no tiene `floating_base_joint` activo), así que
  `pelvis` es la raíz real del árbol de TF — no hace falta odometría.

## Modelo

Por defecto usa `h1_2_description/mjcf/h1_2_scene_surgery_table.xml`: mesa
quirúrgica en forma de "U" (`surgery_table`, marrón/mostaza, mesh
`Extra/surgery_table.stl` de la mesa real), faja transportadora encima
(`conveyor_belt`, blanca, mesh `h1_2_mujoco_sim_bridge/Extra/faja_L201.stl`
decimado a 60k caras porque el decoder STL de MuJoCo rechaza >200k
triángulos) y una mandarina (`mandarina`, esfera naranja) apoyada sobre la
faja, además de la cámara del torso. El resto del bridge no depende de esa
geometría en particular, solo de los 27 joints/actuadores nombrados igual y
de que exista una cámara llamada `robot_rgbd_camera` (si
`publish_camera:=true`) — para cambiar de escena basta con apuntar
`model_relpath` (parámetro) a otro MJCF, p.ej. de vuelta a
`h1_2_scene_qp_reachable.xml` (mesa simple `table_front` + `mandarina` con
`freejoint`, usado por `h1_2_mujoco_lowlevel_bridge`).

Geometría verificada con `mj_ray` (raycast) y cinemática directa/inversa —
no solo a ojo — porque el bounding box de una malla no siempre corresponde
a su superficie real de apoyo (ver notas en el propio XML):

* **Pelvis en z=1.0282** (no 1.1022, el valor que traía el archivo): con
  las piernas en pose de arranque (q=0), esa es la altura que deja la
  suela del pie exactamente en el piso (z=0) — verificada con los vértices
  reales del mesh de `left/right_ankle_roll_link`, no con el comentario
  previo del archivo (que decía tocar el piso y en realidad dejaba los
  pies flotando ~7 cm).
* **Faja en `pos="0.38 0 0.794"`**, rotada 90° en X (su eje largo de 1.2 m
  queda de pie en Z en el mesh original). x=0.38 la deja sobre la única
  franja de la mesa confirmada sólida y ancha en todo Y (`mj_ray` barriendo
  x,y — la mesa NO es sólida en todo su footprint como sugería el
  comentario original de `surgery_table`), cerca del borde para que los
  brazos alcancen a agarrar de ella sin quedar al ras.
* **Mandarina en `pos="0.34 -0.03 0.864"`**: el bbox del mesh de la faja
  sube hasta z_local=+0.174 pero eso es solo un saliente puntual (bloque
  motor en una punta), no la superficie plana; la altura real (0.824,
  +radio 0.04 = 0.864) se sacó con `mj_ray` apuntando hacia abajo sobre
  `conveyor_belt_geom`. La posición X/Y se verificó con IK real (7 joints
  del brazo + ±23° de cintura): ambos brazos la alcanzan con error ~0 cm.

## Faja activa: mandarina estática vs. en movimiento

La mandarina tiene un único joint `slide` (1 GDL, a lo largo del eje Y —
el eje largo de la faja) con un actuador de velocidad
(`conveyor_motor`). Por diseño, **sin ningún estado inicial especial se
queda quieta** en el centro de la faja (la `pos` del body ya es ese punto,
y el actuador en `ctrl=0` la mantiene ahí) — es el comportamiento por
defecto del modelo, sin necesidad de keyframe.

Para simular la faja realmente en marcha, el MJCF trae un keyframe
`mandarina_moving` que reposiciona la mandarina en una punta de la faja y
le da velocidad constante (~0.07 m/s; recorre los ~1.05 m hasta la otra
punta en ~15 s y se frena sola al límite mecánico del joint, sin rebotar).
Se activa con el parámetro `initial_keyframe` (ver `mujoco_sim.py` ->
`mj_resetDataKeyframe`), y hay dos launch files que solo difieren en eso:

```bash
# Mandarina quieta — para algoritmos de picking sobre objeto estático
ros2 launch h1_2_mujoco_sim_bridge sim_bridge_static.launch.py

# Mandarina en movimiento — para algoritmos de tracking/picking dinámico
ros2 launch h1_2_mujoco_sim_bridge sim_bridge_moving.launch.py
```

Ambos son wrappers de una línea sobre `sim_bridge.launch.py
initial_keyframe:=<nombre o vacío>` (aceptan los mismos argumentos:
`use_viewer`, `use_rviz`, `joint_names_file`, `rviz_config`).

**Limitación conocida:** al ser 1 solo GDL (desliza en línea recta, sin
gravedad/contacto reales — el joint fija X/Z), la mandarina sirve para
probar detección/planificación de trayectoria, pero el robot no puede
levantarla físicamente de la faja con este modelo. Para agarre real haría
falta un `freejoint` (6 GDL) y otro mecanismo para el arrastre (p.ej. un
`weld`/`connect` de equality mientras está "sobre" la faja).

## Instalación

```bash
pip install mujoco numpy pyyaml   # si no están ya
cd ~/humanoid_ws
colcon build --packages-select h1_2_description h1_2_mujoco_sim_bridge
source install/setup.bash
```

> **Nota:** `h1_2_description/CMakeLists.txt` no instalaba la carpeta
> `mjcf/` (solo `urdf/meshes/launch/rviz`) — se corrigió como parte de este
> cambio; sin eso, `colcon build` limpio dejaba el MJCF fuera de
> `install/.../share/h1_2_description/`.

## Ejecución

```bash
# Con RViz2 + viewer nativo de MuJoCo (mandarina quieta, ver arriba):
ros2 launch h1_2_mujoco_sim_bridge sim_bridge_static.launch.py
# ... o la variante con la faja "activa" (mandarina en movimiento):
ros2 launch h1_2_mujoco_sim_bridge sim_bridge_moving.launch.py

# Solo RViz2 (sin viewer nativo):
ros2 launch h1_2_mujoco_sim_bridge sim_bridge_static.launch.py use_viewer:=false

# Headless (sin RViz ni viewer; para grabar bags, RL, CI):
ros2 launch h1_2_mujoco_sim_bridge sim_bridge_headless.launch.py

# Controlar solo cintura+brazos (dejar las piernas fuera del bridge):
ros2 launch h1_2_mujoco_sim_bridge sim_bridge_static.launch.py \
    joint_names_file:=$(ros2 pkg prefix h1_2_mujoco_sim_bridge)/share/h1_2_mujoco_sim_bridge/config/joint_names_arms_only.yaml
```

`sim_bridge.launch.py` (el launch "base", con todos los argumentos:
`use_viewer`, `use_rviz`, `joint_names_file`, `rviz_config`,
`initial_keyframe`) se puede seguir usando directo si no importa la
mandarina — `sim_bridge_static.launch.py`/`sim_bridge_moving.launch.py`
son wrappers de una línea que solo fijan `initial_keyframe`.

Mover un joint a mano:

```bash
ros2 topic pub /joint_cmd sensor_msgs/msg/JointState \
    "{name: ['left_elbow_joint'], position: [1.0]}" --once
```

Con MUJOCO_GL sin configurar, el render offscreen de la cámara usa el
backend por defecto de tu instalación de `mujoco` (normalmente EGL en
Linux). En un servidor sin GPU/X, exporta `MUJOCO_GL=egl` (con Mesa
llvmpipe funciona en software, más lento pero correcto) antes de lanzar.

Verificado en esta laptop (NVIDIA GTX 1650, `prime-select` en modo
`on-demand`) que EGL usa la GPU dedicada por defecto sin necesidad de
`__NV_PRIME_RENDER_OFFLOAD`/`__GLX_VENDOR_LIBRARY_NAME` (`GL_RENDERER:
NVIDIA GeForce GTX 1650`, no llvmpipe) — si en otra máquina el render sale
lento, lo primero a chequear es que no esté cayendo a la GPU integrada o a
software rendering.

## Frecuencias y por qué el hilo de cámara es aparte

| Parámetro | Default | Motivo |
|---|---|---|
| `mujoco_timestep` | 0.002 s (500 Hz) | Un `mj_step` tarda ~0.2 ms (mucho margen), pero el **loop en Python** que lo llama necesita presupuesto real por iteración para no perder el paso por contención del GIL con el hilo de cámara/DDS. A 1 kHz el presupuesto es 1 ms — demasiado ajustado en este entorno; a 500 Hz hay el doble de margen y con las ganancias PD por defecto (más suaves que las del bridge de bajo nivel) `kd·dt/M` sigue muy por debajo del límite de estabilidad documentado en `h1_2_mujoco_lowlevel_bridge`. |
| `joint_state_frequency` | 100 Hz | Estado para TF/registro de demos |
| `camera_frequency` | 7.5 Hz | Renderizar (~15-30 ms/frame en este entorno) es 10-100x más lento que un `mj_step`; publicarla desde el mismo hilo que la física le comería presupuesto al PD y a `/joint_states`. Bajado de 15 a 7.5 Hz para aliviar GPUs modestas (laptop) que además comparten GPU con RViz y el viewer nativo. |
| `contacts_frequency` | 5 Hz | Barato (µs-ms), pero no hace falta más rápido que la cámara para depurar visualmente — bajado de 15 Hz por lo mismo |
| `viewer_frequency` | 30 Hz | Refresco del viewer nativo; nunca marca el ritmo. Bajado de 60 Hz por lo mismo |

(Los tres valores bajados son ajustables en `config/sim_bridge.yaml` — si
corrés en una máquina con más GPU, subirlos de nuevo no tiene contra.)

**La cámara corre en su propio hilo**, con su propio pacing a
`camera_frequency`, independiente del hilo de física. `mujoco_sim.MujocoSim`
protege el único punto de carrera real (`mj_step` escribiendo `data`
mientras `Renderer.update_scene` lee `data`) con un lock interno; el
render en sí (`Renderer.render()`, la parte lenta) NO necesita el lock
porque ya no toca `data`, así que corre en paralelo sin bloquear la física.
El contexto EGL/GL queda ligado al hilo que lo usa (moverlo de hilo tira
`EGL_BAD_ACCESS`), así que el renderer se crea y se "precalienta"
(compilación de shaders, ~0.5-1 s la primera vez) **dentro** del hilo de
cámara, nunca en el hilo que construye el nodo.

Si ves warnings de `"la simulación va X s por detrás del tiempo real"`
ocasionales justo al arrancar, es normal (arranque de ROS/discovery); si son
persistentes, baja `joint_state_frequency`/`contacts_frequency` o pon
`realtime:=false` para correr tan rápido como se pueda (útil para
recolectar datos de demostración fuera de tiempo real).

## Ganancias PD y qué joints controla el bridge

`config/joint_gains.yaml` trae `kp`/`kd`/`tau_max` por joint (`tau_max`
copiado de `actuatorfrcrange` del MJCF). Por defecto el bridge mapea los 27
motores del H1-2 (mismo MJCF que el bridge de bajo nivel); las piernas
mantienen la pose de arranque salvo que se les mande `/joint_cmd` explícito
(la base está soldada, no cargan peso). Para excluir joints del todo (no
solo "no comandarlos") usa `joint_names_file` — ver
`config/joint_names_arms_only.yaml` como ejemplo.

## RViz2

`rviz/h1_2_sim.rviz`: `RobotModel` (tópico `/robot_description`),
`MarkerArray` de `/contacts`, `Image` de `/camera/color/image_raw`
(activado) y `/camera/depth/image_raw` (desactivado por defecto, actívalo
para verla), `TF` (desactivado, actívalo para depurar frames). `Fixed
Frame: pelvis`.

## Tests (sin ROS, corren contra el MJCF real)

```bash
cd h1_2_mujoco_sim_bridge
H12_MJCF_PATH=<ruta>/h1_2_description/mjcf/h1_2_scene_qp_reachable.xml \
    python3 -m pytest test/ -v
```

`test_msg_builders.py` sí necesita ROS 2 fuenteado (usa tipos de
`sensor_msgs`/`visualization_msgs`); se salta solo (`importorskip`) si no
está disponible.

## Percepción (siguiente paso)

El plan para el nodo de percepción que consume `/camera/color/*` y
`/camera/depth/*` de arriba (detección de color + distancia, pensado para
visual servoing sobre la mandarina) está en
[`h1_2_algoritms/PERCEPTION_PLAN.md`](../h1_2_algoritms/PERCEPTION_PLAN.md)
— todavía en fase de planeamiento, sin nodos implementados.

## Simplificaciones conocidas

* Sin compensación de gravedad: el PD puro deja error en régimen permanente
  contra la gravedad (normal, visible sobre todo en hombro/codo con brazos
  extendidos); si hace falta, es el punto natural para añadir
  `qfrc_bias`/`mj_inverse` como feedforward en `pd_controller.py`.
* Una sola cámara (`robot_rgbd_camera` del torso); para más cámaras (p.ej.
  una fija en la mesa) el patrón de `mujoco_sim.py`/`msg_builders.py` se
  replica igual, publicando con otro `camera_name`/tópicos.
* Los contactos se listan sin filtrar por par de cuerpos; si con muchos
  objetos en la mesa `/contacts` se vuelve ruidoso, filtrar en
  `MujocoSim.get_contacts()` por `body1`/`body2` es directo.
* La mandarina de `h1_2_scene_surgery_table.xml` es 1 GDL (slide sobre la
  faja, ver arriba) — no se puede levantar físicamente todavía; para eso
  hace falta pasarla a `freejoint` + algún mecanismo de "pegado" a la faja
  mientras no está agarrada.
