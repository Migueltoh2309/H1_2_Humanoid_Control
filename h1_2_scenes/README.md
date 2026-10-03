# h1_2_scenes

Escenas MuJoCo de la tesis para el Unitree H1-2 de base fija: la mesa
quirúrgica con la faja transportadora y la mandarina, la versión con manos
Inspire para el agarre, la bandeja-obstáculo y una escena vacía.

| Escena | Contenido | La usa |
|---|---|---|
| `h1_2_scene_surgery_table.xml` | Mesa en "U", faja con motor virtual, mandarina (1 GDL sobre la faja), cámara RGB-D del torso. Keyframe `mandarina_moving` | `sim_bridge.launch.py` (por defecto), evaluación bimanual, percepción |
| `h1_2_scene_surgery_table_hands.xml` | La anterior + dedos de las manos Inspire (12 joints por mano, acoplados como la mano real), mandarina libre sobre un carro de la faja y par IR estéreo. Keyframes `mandarina_moving` y `mandarina_moving_fast` | `visual_servoing_sim.launch.py`, `visual_servoing_grasp.py`, `evaluate_ir_stereo.py` |
| `h1_2_scene_tray_obstacle.xml` | La escena de la mesa + una bandeja de 24×24 cm que el modelo del robot no conoce (solo la ve la cámara) | escenario `obstacle` de la capa 3 bimanual |
| `h1_2_scene_empty.xml` | El H1-2 sobre el piso, sin entorno | `sim_bridge_empty.launch.py` |

`meshes/` trae las dos mallas propias: `faja_L201.stl` (faja transportadora,
decimada) y `surgery_table.stl` (mesa).

## Cómo encuentra las mallas

Las escenas usan las mallas del robot de `h1_2_description` y las de este
paquete, pero MuJoCo solo admite un `meshdir`. Por eso:

- **En el código fuente**, las escenas quedan con `meshdir="../meshes/"`, igual
  que en `h1_2_description`. Así se editan sin cambios respecto al original.
- **Al compilar**, el `CMakeLists.txt` instala una copia con rutas absolutas:
  el `meshdir` apunta a `share/h1_2_description/meshes/` y la faja y la mesa a
  `share/h1_2_scenes/meshes/`.

Las escenas se cargan siempre desde el share instalado:

```python
from ament_index_python.packages import get_package_share_directory
path = os.path.join(get_package_share_directory("h1_2_scenes"), "mjcf", "h1_2_scene_surgery_table.xml")
```

```bash
# en el bridge de simulación (ya es la configuración por defecto)
ros2 launch h1_2_mujoco_sim_bridge sim_bridge.launch.py
ros2 run h1_2_mujoco_sim_bridge sim_bridge --ros-args \
    -p model_package:=h1_2_scenes -p model_relpath:=mjcf/h1_2_scene_tray_obstacle.xml
```

Después de editar una escena hay que volver a ejecutar `colcon build` para
que se instale.

## Regenerar la escena con manos

`h1_2_scene_surgery_table_hands.xml` **se genera** a partir de la escena de la
mesa y de las manos de `h1_2_description/mjcf/h1_2_scene.xml`; no se edita a
mano. Si cambias la mesa, la faja, la cámara o la postura en
`h1_2_scene_surgery_table.xml`, regenera la escena con manos:

```bash
source ~/humanoid_ws/install/setup.bash
cd ~/humanoid_ws/src/h1_2_scenes
python3 scripts/build_scene_hands.py        # busca h1_2_description solo
python3 scripts/build_scene_hands.py --description-dir ~/humanoid_ws/src/h1_2_utec/h1_2_description
```

## Requisitos

- `h1_2_description` del repositorio [h1_2_utec](https://github.com/oscar-ramos/h1_2_utec)
  en el mismo workspace (mallas y modelo del robot).
