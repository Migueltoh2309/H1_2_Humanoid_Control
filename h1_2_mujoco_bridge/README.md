# h1_2_mujoco_lowlevel_bridge

Bridge **ROS 2 ↔ MuJoCo de bajo nivel** para el humanoide **Unitree H1-2**.
Emula la interfaz `/lowcmd` (`unitree_hg/msg/LowCmd`) y `/lowstate`
(`unitree_hg/msg/LowState`) del robot real **exclusivamente dentro de la
simulación local**. Un controlador escrito para la interfaz de bajo nivel del
H1-2 (como `test_mandar_modificado.py`) corre sobre MuJoCo **sin ninguna
modificación**.

```text
Controlador ROS 2 (p. ej. test_mandar_modificado)
        │  publica /lowcmd (unitree_hg/LowCmd)
        │  recibe  /lowstate (unitree_hg/LowState)
        ▼
h1_2_mujoco_lowlevel_bridge  (este paquete)
        │  tau = tau_ff + kp·(q_des − q) + kd·(dq_des − dq)  →  data.ctrl
        ▼
Simulación MuJoCo del H1-2 (MJCF del paquete h1_2_mujoco_bridge)
```

## Aislamiento del hardware (garantías)

Este paquete **no contiene**: direcciones IP del robot, interfaces de red
Unitree, configuración de CycloneDDS orientada al robot, SDK de Ethernet,
modos `real`/`hardware`, launch files para el robot físico ni ningún selector
sim/real. Los mensajes `unitree_hg` se usan **solo como tipos de datos ROS 2**
en el DDS local. Todo corre en tu computadora.

## Estructura

```text
h1_2_mujoco_lowlevel_bridge/
├── h1_2_mujoco_lowlevel_bridge/
│   ├── bridge_node.py        # Nodo ROS 2: parámetros, QoS, hilo de simulación
│   ├── mujoco_simulator.py   # Carga MJCF, overrides, sub-pasos, snapshot, viewer
│   ├── lowcmd_handler.py     # CRC, NaN, modos, saturaciones de LowCmd
│   ├── lowstate_builder.py   # Construcción de LowState (35 motores + IMU)
│   ├── joint_mapping.py      # Mapeo por NOMBRES Unitree<->MuJoCo, validado
│   ├── motor_controller.py   # PD+ff, tau_max y rate-limit de delta_tau
│   ├── imu_simulator.py      # IMU desde los sensores del MJCF (site 'imu')
│   ├── crc.py                # Serialización + CRC de Unitree (tabla, rápido)
│   ├── watchdog.py           # zero_torque / hold_position ante timeout
│   ├── safety_limits.py      # Límites XML+YAML y clamps
│   └── offline_demo.py       # Validación de lazo cerrado SIN ROS (Etapa 0)
├── config/{bridge,joint_mapping,motor_limits}.yaml
├── launch/mujoco_lowlevel_bridge[_headless].launch.py
├── test/  (pytest, corren sin ROS)
├── package.xml / setup.py / setup.cfg / README.md
```

## Requisitos e instalación

1. ROS 2 (Humble o similar) y `colcon`.
2. MuJoCo por pip: `pip install mujoco numpy pyyaml`
3. Los **tipos de mensaje** `unitree_hg`: compilar en el workspace el paquete
   `unitree_hg` de [unitreerobotics/unitree_ros2] (solo definiciones `.msg`;
   no se usa nada más de ese repositorio).
4. El **modelo**: este paquete lee el MJCF del paquete existente
   `h1_2_mujoco_bridge` (`mjcf/h1_2_scene_qp_reachable.xml`), que debe estar
   en el mismo workspace. También puede apuntarse a un XML absoluto con el
   parámetro `model_path`.

```bash
cd ~/ws/src        # junto a h1_2_mujoco_bridge y unitree_hg
# (copiar aquí este paquete)
cd ~/ws
colcon build --packages-select unitree_hg h1_2_mujoco_bridge h1_2_mujoco_lowlevel_bridge
source install/setup.bash
```

## Ejecución

```bash
# Con viewer de MuJoCo:
ros2 launch h1_2_mujoco_lowlevel_bridge mujoco_lowlevel_bridge.launch.py

# Headless:
ros2 launch h1_2_mujoco_lowlevel_bridge mujoco_lowlevel_bridge_headless.launch.py

# Desactivar la verificación de CRC para pruebas iniciales:
ros2 launch h1_2_mujoco_lowlevel_bridge mujoco_lowlevel_bridge.launch.py verify_crc:=false
```

Con el bridge activo, el controlador del usuario corre tal cual:

```bash
ros2 run <paquete_controlador> test_mandar_modificado
python3 test_leer.py     # o como nodo del paquete del usuario
```

## Frecuencias (quién marca cada ritmo)

| Parámetro | Default | Significado |
|---|---|---|
| `mujoco_timestep` | 0.001 s | dt interno del integrador (física a 1000 Hz) |
| `control_frequency` | 500 Hz | tick de control: 2 sub-pasos/tick; el PD se **recalcula en cada sub-paso** con el estado actual (emula el servo del motor, que en el robot corre más rápido que /lowcmd) |
| `lowstate_frequency` | 500 Hz | publicación de `/lowstate` (diezmado del tick) |
| `joint_state_frequency` | 50 Hz | publicación auxiliar `/joint_states` |
| `viewer_frequency` | 60 Hz | refresco del viewer; **nunca** marca el ritmo de la física |
| `realtime` | true | pacing a reloj de pared; `false` = tan rápido como se pueda |

La recepción de `/lowcmd` es asíncrona (callback DDS): la última consigna
válida queda vigente entre mensajes.

## QoS (verificado contra los scripts)

| Tópico | Perfil del bridge | Compatibilidad |
|---|---|---|
| `/lowstate` (pub) | `qos_profile_sensor_data` (BEST_EFFORT, VOLATILE, depth 5) | igual que el robot real; `test_leer.py` se suscribe con el mismo perfil ✓ |
| `/lowcmd` (sub) | `qos_profile_sensor_data` (BEST_EFFORT) | una suscripción BEST_EFFORT acepta publishers RELIABLE **y** BEST_EFFORT; `test_mandar_modificado.py` publica RELIABLE depth 10 ✓. Así no aparece `offering incompatible QoS`. |
| `/joint_states` (pub) | RELIABLE depth 10 | estándar para RViz/herramientas |

## Conversión LowCmd → MuJoCo

Los actuadores del XML son `<motor>` puros (torque, `gear=1`, sin PD interno),
así que **el PD se aplica una sola vez, en el bridge**:

```text
tau_i = tau_ff_i + kp_i·(q_des_i − q_i) + kd_i·(dq_des_i − dq_i)
```

con este pipeline por mensaje (lowcmd_handler + motor_controller):

1. `verify_crc` (opcional, default on): CRC recalculado con la MISMA
   serialización del SDK; si no coincide, **el mensaje completo se rechaza**
   (warning throttled) y la consigna previa sigue hasta que actúe el watchdog.
2. Valores no finitos (NaN/Inf) en un motor → ese motor se deshabilita este
   ciclo (warning throttled). Nunca se propagan a la física.
3. `motor_cmd[i].mode`: `0` = deshabilitado (torque objetivo 0, bajada en
   rampa), `1` = PD+ff. Otros valores → warning throttled + tratados como 0
   (nunca ignorados en silencio).
4. Saturaciones **antes** de aplicar: `q_des→[q_min,q_max]` (del XML),
   `dq_des→±dq_max`, `tau_ff→±tau_max`, `kp→[0,kp_max]`, `kd→[0,kd_max]`.
5. En cada sub-paso: `tau→±tau_max` y `|Δtau| ≤ tau_rate_max·dt`
   (rate-limiter; también produce la rampa suave a cero del watchdog).
6. Escritura en `data.ctrl[actuador]` y `mj_step`.

`mode_pr`: solo se soporta **PR** (0). Este MJCF modela los tobillos como
joints serie pitch/roll; si llega AB (1) se avisa (throttled) y los valores se
interpretan como PR (simplificación documentada).

## LowState desde MuJoCo

| Campo | Fuente |
|---|---|
| `motor_state[i].q/dq` (i<27) | `data.qpos/qvel` vía el mapeo por nombres |
| `motor_state[i].ddq` | diferencia finita de dq entre ticks de control |
| `motor_state[i].tau_est` | **`data.actuator_force`** = torque realmente aplicado por el actuador (con `<motor>` gear=1 coincide con el torque de control tras saturaciones). Documentado aquí y en el código. |
| `motor_state[i].mode` | modo efectivo aplicado (0/1) |
| `motor_state[27..34]` | **ceros y mode=0** (el mensaje exige 35 elementos; el H1-2 solo tiene 27 motores) |
| `imu_state` | ver abajo |
| `mode_pr` | eco del último LowCmd |
| `mode_machine` | parámetro `mode_machine` (default 4; arbitrario en simulación, los controladores solo lo repiten) |
| `tick` | milisegundos de simulación (mod 2³²) |
| `temperature/vol/sensor/motorstate/wireless_remote/reserve` | valores seguros (0) — no existen en simulación |
| `crc` | **0**: los consumidores no verifican el CRC de LowState (solo el robot verifica el de LowCmd); se documenta y se evita el coste a 500 Hz. |

## IMU simulada (convenciones verificadas)

El MJCF ya trae una IMU: sensores `gyro` + `accelerometer` en el **site `imu`
del body `torso_link`** (misma ubicación que el H1-2 real). Se leen de
`data.sensordata`:

* **Cuaternión**: orientación del site, orden **(w, x, y, z)** — MuJoCo y
  `IMUState.quaternion` usan el mismo orden, se copia sin reordenar.
* **rpy**: (roll, pitch, yaw) en **radianes**, convención ZYX intrínseca
  (la del SDK).
* **gyroscope**: velocidad angular en el **marco local** del site, rad/s
  (así la entrega el sensor gyro de MuJoCo).
* **accelerometer**: aceleración **propia** en el marco local, m/s²,
  **incluye la reacción a la gravedad** (en reposo ≈ (0,0,+9.81)), igual que
  una IMU MEMS real y que la de Unitree.
* Ejes: X adelante, Y izquierda, Z arriba en la pose home.

Nota: en esta escena la **pelvis está soldada al mundo** (no hay freejoint),
así que la IMU es casi estática (quat≈identidad, gyro≈0, accel≈(0,0,9.81) más
vibraciones del torso). El código es genérico: con base libre funcionaría
igual, e incluye un fallback documentado (cinemática del body `pelvis`) por si
se usa un MJCF sin sensores.

## CRC

* Serialización idéntica al layout C++ (1000 bytes: header 4 + 35×28 + 16).
* El algoritmo bit a bit de Unitree se conserva como **referencia** y el
  camino de producción usa una **tabla de 256 entradas** (el algoritmo es
  algebraicamente un CRC-32/MPEG-2 procesando cada palabra uint32 en orden
  big-endian): **0.15 ms/mensaje** vs 1.3 ms del bit a bit — verificado
  idéntico con mensajes aleatorios en `test_crc.py`. A 500 Hz esto importa
  (7% de un core vs 64%).
* `verify_crc: true` → mensajes con CRC inválido se **rechazan** con warning
  limitado por frecuencia y contador. `false` para pruebas iniciales.

## Watchdog (`command_timeout`, default 0.1 s)

* `timeout_behavior: zero_torque` → el torque objetivo pasa a 0 y **baja en
  rampa** vía el rate-limiter (nunca se mantiene indefinidamente el último
  torque, nunca hay saltos).
* `timeout_behavior: hold_position` → captura la q del instante del timeout y
  aplica PD de mantenimiento con `hold_kp`/`hold_kd` (saturados por-joint).
* Se rearma automáticamente al volver `/lowcmd`. Antes del **primer** comando
  el bridge ya está en el estado seguro (motores deshabilitados), por lo que
  los brazos cuelgan por gravedad hasta que el controlador arranca — la etapa
  1 de `test_mandar_modificado` parte de la q medida, así que el arranque es
  suave, igual que en el robot.

## Modificaciones al modelo — explicación detallada

**El XML NO se modifica.** Se aplican dos overrides **en memoria** al cargar:

1. `mujoco_timestep: 0.001` (el XML trae 0.002) — necesario para que 500 Hz de
   control sea un múltiplo exacto (2 sub-pasos/tick) y para dar margen de
   estabilidad al PD explícito.
2. `joint_armature: 0.01` y `joint_damping: 0.05` en los 27 dof mapeados.
   **Por qué**: el XML trae armature=0/damping=0 y las muñecas tienen inercia
   efectiva ~6·10⁻⁴ kg·m². Con integración Euler explícita el término kd del
   PD solo es estable si `kd·dt/M ≲ 2`. Con kd=5 (permitido por los límites)
   y sin armature: `kd·dt/M ≈ 8` → chattering (medido: |dq|max = 25.5 rad/s
   con la muñeca ordenada a quedarse QUIETA). Con armature=0.01 (inercia
   reflejada del rotor, presente en los modelos oficiales de Unitree y ausente
   en este XML): M≈0.0106, `kd·dt/M ≈ 0.47` → inmóvil (<5·10⁻⁴ rad/s).
   Desactivable con `joint_armature: 0.0` (en ese caso bajar `kd_max` de
   muñecas a ≤1 en `motor_limits.yaml`).

3. `base_height_offset` (opcional, default **0.0**): eleva la pelvis soldada
   en Z. Con la escena tal cual el robot está DE PIE y los pies apoyan en el
   suelo (fuerza normal media **medida: ~460 N**): los senos de tobillo del
   ejemplo del SDK quedan bloqueados por el contacto (RMSE 132 mrad — física
   correcta, no un bug). Con `0.3` el robot queda "colgado" como en el pórtico
   del que Unitree suspende el robot en ese mismo test: contacto = 0 N y RMSE
   de tobillo **13 mrad**. El demo offline usa 0.30 por defecto (`--standing`
   para la escena original).

Los `<motor>` del XML ya son actuadores de torque puro sin PD interno, así que
**no hubo que cambiar actuadores** y no hay riesgo de PD duplicado.

## Guía de validación progresiva

**Etapa 0 (sin ROS)** — pipeline completo salvo el transporte DDS:
```bash
python3 -m pytest test/ -v                      # 27 tests
python3 -m h1_2_mujoco_lowlevel_bridge.offline_demo \
    --mjcf <ruta>/h1_2_scene_qp_reachable.xml --out demo.png
```

**Etapa 1 — solo MuJoCo**: `ros2 launch h1_2_mujoco_lowlevel_bridge
mujoco_lowlevel_bridge.launch.py`. El log debe mostrar el mapeo 27/27
validado, los overrides y las frecuencias. El viewer muestra el H1-2 con los
brazos cayendo lentamente (motores deshabilitados = comportamiento seguro).

**Etapa 2 — tópicos**:
```bash
ros2 topic list
ros2 topic info /lowcmd ; ros2 topic info /lowstate
ros2 interface show unitree_hg/msg/LowCmd
ros2 interface show unitree_hg/msg/LowState
ros2 topic hz /lowstate        # ≈ lowstate_frequency
```

**Etapa 3 — lectura**: correr `test_leer.py`; debe imprimir
`Motor 19 | q: …` con la q simulada de `left_wrist_yaw_joint`.

**Etapa 4 — envío**: correr `test_mandar_modificado.py`; el log del bridge
mostrará el conteo de comandos aceptados; con `verify_crc:=true` ningún
mensaje válido se rechaza. Al detener el sender, el watchdog dispara
(mensaje en el log) y el robot baja el torque en rampa.

**Etapa 5 — una articulación**: enviar movimiento pequeño a un solo joint del
brazo y registrar q_des/q/dq/tau_ff/tau_PD/tau_total (todo visible en
`/lowstate` + `motor_state[i].tau_est`; el offline_demo ya registra y grafica
exactamente esas señales para la muñeca).

**Etapa 6 — brazo completo**: referencias en las 7 articulaciones de un brazo;
verificar límites (probar q_des fuera de rango → se satura al rango del XML),
estabilidad y frecuencia (`ros2 topic hz /lowstate`).

En ninguna etapa existe ni se necesita el robot físico.

## Simplificaciones documentadas (vs. el robot real)

* Solo modo **PR** de tobillos (AB → warning + interpretación PR).
* Base **soldada**: la IMU es casi estática. Con la escena tal cual los pies
  apoyan en el suelo; para tests de bajo nivel tipo SDK (senos de tobillo) usar
  `base_height_offset: 0.3` — "colgar" el robot como en el pórtico real.
* `motor_state[27..34]`, temperaturas, voltaje, mando inalámbrico: ceros.
* `LowState.crc = 0` (no verificado por los consumidores).
* `mode_machine` es un parámetro (default 4), sin significado en simulación.
