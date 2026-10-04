# Interfaces entre bloques (reto R40-RT-H1_2-0001, sección 5)

Cinco bloques separados. Esta separación es la que permite probar la percepción sin robot
(sobre el conjunto de datos) y el control sin cámara (pruebas automáticas). Los tipos están en
`seguidor/tipos.py`; todo en SI (m, rad, s) y en el **marco del robot**: origen en el suelo bajo
la pelvis, **x adelante, y a la izquierda**, z arriba (gira con el rumbo, no con el balanceo).

```text
 Fuente (RealSense | ROS 2 del sim | dataset) ──Fotograma──▶ PERCEPCIÓN ──MedidaLinea──┐
 rt/lowstate (IMU) ──Imu──────────────────────────────────────┬──────────────────────▶ ESTIMACIÓN
                                                               │                         │ EstadoLinea
                                                               │                         ▼
                                                               │                      CONTROL ──Orden──┐
                                                               └──── salud ──────────────────────────▶ SUPERVISOR ──Move/StopMove──▶ LocoClient
                                     REGISTRO ◀── todo lo anterior, cada ciclo (CSV) + resumen (JSON)
```

| Bloque | Entrada | Salida | Requisito del reto |
|---|---|---|---|
| **Percepción** `seguidor/percepcion.py` | `Fotograma` (t, IR uint8 640×480, K = fx, fy, cx, cy) + roll/pitch de la IMU relativos a la calibración | `MedidaLinea`: `desplazamiento` (m, y de la línea en x = 0), `angulo` (rad), `curvatura` (1/m), `objetivo` (punto de la línea a la distancia de mira), `confianza` 0..1, `barra` (m o None), `esquina` ((m, ±1) o None), `puntos` (x, y) detectados, t, ms | Igual sobre el dataset que en vivo, sin cambiar código: solo recibe un `Fotograma` |
| **Estimación** `seguidor/estimacion.py` | `MedidaLinea` + yaw y giróscopo de la IMU + la `Orden` aplicada | `EstadoLinea`: lo mismo que la medida pero reconstruido con la **memoria** de la línea (cubre zona ciega y huecos), `sin_linea_m/s`, `barra`/`esquina` por estima, `recorrido`, `yaw`, `sesgo_yaw` (deriva) | Balanceo: compensado por fotograma con roll/pitch (sin retardo). Confianza baja: la medida no entra en la memoria y se sigue con lo recordado |
| **Control** `seguidor/control.py` | `EstadoLinea` + punto de mira | `Orden` (vx, vy, vyaw) saturada y con límite de aceleración | Ganancias y límites en `config/seguidor.yaml` |
| **Supervisor** `seguidor/supervisor.py` | `EstadoLinea` + `Orden` del control + salud (edad de cámara e IMU, roll/pitch, motores, excepción) | `Orden` final o parada | Máquina de estados explícita; **único** bloque que decide Move/StopMove |
| **Registro** `seguidor/registro.py` | todo lo anterior | `datos/tiradas/nivelN_<fecha>.csv` (un ciclo por fila) + `.json` (métricas sec. 9 y eventos) | Suficiente para reconstruir la tirada |

## Frecuencias y tiempos

* Lazo principal y envío de `Move`: 20 Hz (`supervisor.hz`), `duration = 1 s` en cada `Move`
  (si el programa muere, el robot para solo).
* Percepción: cada fotograma nuevo (15 Hz en el simulador por defecto, 30 Hz en la D435i).
  Si el lazo va más rápido que la cámara, la estimación propaga con la IMU y el modelo de la
  marcha entre fotogramas (6.3.6).
* Todos los tiempos son `time.monotonic()` del proceso (cámara, IMU y lazo con el mismo reloj).

## Estados del supervisor

`ESPERA → SEGUIMIENTO ⇄ LINEA_PERDIDA`, `SEGUIMIENTO → GIRO_ESQUINA → SEGUIMIENTO`,
`SEGUIMIENTO → LLEGADA → FIN`, `LINEA_PERDIDA → FIN`, cualquiera `→ PARADA` (vigilante).
Detalle en el docstring de `seguidor/supervisor.py`.
