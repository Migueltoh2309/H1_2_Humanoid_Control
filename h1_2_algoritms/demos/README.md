# Demos — H1-2: bridge MuJoCo y cinemática inversa

Secuencia para mostrar los resultados. Cada bloque es una terminal.
En **todas** las terminales, primero:

```bash
cd ~/humanoid_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

---

## 0. Compilar (una sola vez)

```bash
cd ~/humanoid_ws
colcon build --packages-select h1_2_mujoco_lowlevel_bridge h1_2_algoritms --symlink-install
source install/setup.bash
```

---

## 1. El bridge es correcto — 31 pruebas unitarias

```bash
export H12_MJCF_PATH=~/humanoid_ws/install/h1_2_mujoco_bridge/share/h1_2_mujoco_bridge/mjcf/h1_2_scene_qp_reachable.xml
cd ~/humanoid_ws/src/h1_2_mujoco_bridge
python3 -m pytest test/ -v
```

`31 passed`. Cubren CRC, mapeo de articulaciones, construcción de LowState,
control PD y saturaciones.

> Sin `H12_MJCF_PATH` se saltan 12 pruebas: el `conftest.py` busca el modelo
> en una ruta que no existe en esta máquina.

---

## 2. Lazo cerrado sin ROS, con gráfica

```bash
cd ~/humanoid_ws/src/h1_2_mujoco_bridge
python3 -m h1_2_mujoco_lowlevel_bridge.offline_demo \
    --mjcf $H12_MJCF_PATH --out ~/demo_bridge.png
```

Imprime métricas (RMSE muñeca ~43 mrad, tobillo ~13 mrad, 0 CRC rechazados)
y guarda `~/demo_bridge.png` con el seguimiento de consigna. **Buena lámina.**

---

## 3. IK con límites articulares — el resultado principal

No necesita ROS ni el bridge.

```bash
cd ~/humanoid_ws/src/h1_2_algoritms
python3 demos/benchmark_ik.py 15
```

Compara la IK original contra la nueva sobre objetivos que **sí** tienen
solución válida. Resultado típico con arranque en frío:

| IK | soluciones usables |
|---|---|
| original (DLS) | ~13 % |
| **nueva con límites** | **~97 %** |

Sube el número (`30`, `60`) para más muestras; tarda ~0.5 s por objetivo
con la IK original.

---

## 4. El bridge en vivo, con visor de MuJoCo

**Terminal A** — dejar corriendo:

```bash
ros2 launch h1_2_mujoco_lowlevel_bridge mujoco_lowlevel_bridge.launch.py
```

Se abre la ventana de MuJoCo con el H1-2. El log muestra el mapeo 27/27
validado y las frecuencias. Los brazos cuelgan: sin comandos, los motores
están deshabilitados (comportamiento seguro).

**Terminal B** — comprobar la interfaz:

```bash
ros2 topic list
ros2 topic info /lowcmd
ros2 topic info /lowstate
ros2 topic hz /lowstate        # ~500 Hz
ros2 topic echo /lowstate --once | head -30
```

---

## 5. El robot se mueve: IK → /lowcmd → MuJoCo

Con el visor de la terminal A abierto, en la **terminal B**:

```bash
ros2 run h1_2_algoritms ik_lowcmd_node
```

Los dos brazos van a su pose objetivo **en el visor**. El log reporta cada
segundo el error real del efector, que baja a 0.0 mm.

Cambiar el objetivo en vivo:

```bash
ros2 run h1_2_algoritms ik_lowcmd_node --ros-args \
    -p left_target:="[0.30, 0.28, 0.15, 1.0, 0.0, 0.0, 0.0]" \
    -p right_target:="[0.30, -0.28, 0.15, 1.0, 0.0, 0.0, 0.0]"
```

El formato es `[x, y, z, qw, qx, qy, qz]` en el marco `torso_link`.

Al cortar con Ctrl-C, el watchdog del bridge baja el torque en rampa
(se ve en el log de la terminal A).

---

## 6. Medición del lazo cerrado — PD puro vs PD + integral

Con el bridge corriendo (terminal A), en la **terminal B**:

```bash
cd ~/humanoid_ws/src/h1_2_algoritms
python3 demos/validacion_lazo_cerrado.py 4
```

Para cada objetivo mide el error del efector con PD puro y con PD +
integral. Resultado típico:

| | error medio del efector |
|---|---|
| PD puro | ~75 mm (caída por gravedad) |
| **PD + integral** | **~0.07 mm** |

Tarda ~14 s por objetivo (rampa + asentamiento).

---

## Qué se arregló

| Problema | Estado |
|---|---|
| La IK ignoraba los límites articulares (13 % de soluciones usables) | Corregido: proyección en espacio nulo + bloqueo de articulaciones en tope + saturación por iteración → ~97 % |
| Cuaternión objetivo `[0,0,0,1]` era un giro de 180° en Z, no la identidad | Corregido a `[1,0,0,0]` |
| Nada conectaba la IK con el bridge | Nuevo nodo `ik_lowcmd_node` |
| PD puro deja error de ~75 mm por gravedad | Término integral con anti-windup sobre `tau_ff` |

## Limitaciones conocidas (conviene mencionarlas)

- **No hay planificación de trayectoria.** El nodo interpola en línea recta
  en el espacio articular. La escena tiene una mesa (`table_front`): si el
  camino la atraviesa, el brazo choca, el par satura y no llega. El objetivo
  puede ser alcanzable aunque el camino recto no lo sea.
- **La IK no conoce las colisiones**, solo los límites articulares.
- Si el objetivo cae fuera del espacio de trabajo admisible, la IK lo
  informa (`ok=False` y el residuo en mm) en vez de devolver una pose falsa.
- El bridge tiene la pelvis soldada al mundo; la IMU es casi estática.
