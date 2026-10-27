# Interfaces entre bloques (reto R40-RT-H1_2-0002, sección 5)

Siete bloques separados, con las interfaces escritas **antes** de programarlos (entregable del
hito 1). Esta separación es la que permite probar la percepción sin brazos (sobre datos
grabados), la planificación sin robot (contra el modelo) y la mano sin cámara. Los tipos están
en `manip/tipos.py`.

**Convenciones.** Unidades SI (m, rad, s); las fuerzas de la mano en gf, como sus registros.
Tiempos de `time.monotonic()`. Todo en el **marco del torso** (`torso_link`): x adelante, y
izquierda, z arriba, origen en la cintura. Es el marco de la cinemática de la tesis
(`fkine_arm_*_unitree`), verificada contra el MJCF del simulador con error < 1e-6 m. La
estación se describe en el marco *suelo* (`config/estacion.yaml`), que es el mismo desplazado
`torso_z` en z. Lados `izq` / `der`. Brazos con 7 juntas en el orden de la tesis.

```text
                       config/estacion.yaml · config/objetos.yaml · config/agarres.yaml (hito 2)
                                                 │
 D435 (color + prof. nativa) ──Fotograma──▶ PERCEPCIÓN ──ListaObjetos──▶ PLANIFICACIÓN ──Plan──┐
        ▲                                   ▲     (torso_link)               ▲  (Tramos articulares)
        │              CALIBRACIÓN ─────────┘ (T_torso_cam, fichero)         │                 │
        │                  ▲                                                 │                 ▼
        │   MuestraCalibracion (juntas + ArUco)          EstadoBrazos ───────┤          SUPERVISOR ──▶ BRAZO ──▶ rt/arm_sdk
        │                                                                    │   (único que       │  (rampas de peso,
 rt/lowstate (juntas, IMU) ──────────────────────────────────────────────────┴── cambia el peso)  │   mínima sacudida)
 Manos Inspire (Modbus) ◀── OrdenMano ── MANO ◀───────────────────────────────────────────────────┘
                              │ EstadoMano (ángulos, fuerzas, táctil, status)
 REGISTRO ◀── todo lo anterior: CSV ≥ 50 Hz + PNG de cada agarre + JSON por tirada
```

| Bloque | Entrada | Salida | Requisito del reto |
|---|---|---|---|
| **Calibración** `manip/calibracion.py` (hito 2) | `MuestraCalibracion`: q **medidas** (y consignas) de un brazo quieto + esquinas del ArUco del dorso en la imagen (+ profundidad) | `Calibracion`: `T_torso_cam` (4x4), `T_muneca_marcador` por mano, residuo medio y máximo en mm, nº de poses, intrínsecos, fecha | Fichero con fecha en `datos/calibracion/`. Ningún otro bloque lleva una transformación escrita en el código: la pose de la cámara sale **solo** de ese fichero |
| **Percepción** `manip/percepcion.py` | `Fotograma` (color, profundidad nativa en m, K de cada una, q de los brazos para el autofiltro, cintura) + `Calibracion` | `ListaObjetos`: por objeto clase, centro (torso_link), guiñada en planta y su simetría, medidas, confianza, fracción visible; plano de la mesa; ms | Igual sobre datos grabados que en vivo, sin cambiar el código: solo recibe un `Fotograma`. Fuentes intercambiables: ROS del simulador, `pyrealsense2` en el PC2, carpeta grabada |
| **Planificación** `manip/planificacion.py` | `ObjetoDetectado` + `EstadoBrazos` + estación + tabla de agarres | `Plan`: mano, `Agarre`, bandeja y `Tramo`s articulares (preagarre, descenso, elevación, transporte, suelta, retirada), dentro de los límites del URDF; `ok` y `motivo` | Comprobada **sin robot** contra el modelo: ningún punto de paso por debajo de la altura de seguridad sobre la mesa salvo el descenso final; holgura mínima con la mesa en el `Plan` |
| **Brazo** `manip/brazo.py` | `Tramo`s + peso pedido por el supervisor | `rt/arm_sdk` a frecuencia fija; `EstadoBrazos` (q medida, consigna, error de seguimiento, peso) | Rampa de peso de entrada de 12 s siguiendo la postura medida con tope de 2 grados; salida ≥ 5 s hasta **0.00**; duración mínima y velocidad máxima (≤ 0.5 rad/s) en `config/brazo.yaml`, de la medida de 6.3; kp ≤ 150; la cintura (índice 12) recibe su postura medida |
| **Mano** `manip/mano.py` | `OrdenMano` (preforma, cerrar, sujetar, soltar, abrir) | Modbus TCP (registros de uno en uno por grupo); `EstadoMano` (ángulos, fuerzas, status, error, táctil resumido, responde) | `FORCE_SET` escrito **siempre** antes de cerrar y nunca con el valor de fábrica (2000); consigna de sujeción **cerrada** con la fuerza limitada (≤ 800 gf; corte a 1200 medidos) |
| **Supervisor** `manip/supervisor.py` | Todo lo anterior + IMU + FSM | Estado siguiente o `SALIDA_SEGURA` | Máquina de estados explícita; **único** bloque que cambia el peso de `arm_sdk` y único que da por terminado un objeto. Salida segura: congelar, abrir la mano, subir a la altura de seguridad y solo entonces peso a 0 |
| **Registro** `manip/registro.py` | Todo lo anterior | `datos/tiradas/nivelN_<fecha>.csv` (≥ 50 Hz), PNG de cada agarre, `.json` con métricas de la sec. 9 y eventos | Suficiente para reconstruir cada objeto después (como `datos_caja/`) |

## Estados del supervisor (borrador; se fija en el hito 3)

`ESPERA` (peso 0, brazos en reposo; el instructor coloca) → palabra de confirmación →
`ENTRADA` (rampa de peso 12 s) → `OBSERVAR` (brazos en la pose de observación, fuera de la
vista de la mesa; percepción) → `PLANIFICAR` → `APROXIMAR` (preagarre a la altura de
seguridad) → `DESCENDER` (hasta el agarre, contacto por la mano) → `CERRAR` →
`VERIFICAR` (dedos parados lejos de 0 con fuerza) → `ELEVAR` → `TRANSPORTAR` →
`SOLTAR` (abrir, luego retirar) → `RETIRAR` → `OBSERVAR` (¿quedó en la bandeja? ¿se movió
otro objeto?) … → `SALIDA` (rampa de peso ≥ 5 s a 0.00) → `ESPERA`.
Desde cualquier estado → `SALIDA_SEGURA` por: paso detectado, |yaw| > 5 grados, roll o pitch
> 10 grados, error de seguimiento por encima del umbral (choque), fuerza > 1200 gf, cámara
sin fotogramas o tapada, mano sin respuesta, FSM distinta de 201, excepción o Ctrl+C.

## Frecuencias (provisionales)

* `rt/arm_sdk`: 50 Hz fijos (como `caja_cuadrado.py`); lazo del supervisor al mismo ritmo.
* Percepción: bajo demanda en `OBSERVAR` (varios fotogramas promediados con el robot quieto),
  no en el lazo del brazo: ningún brazo se mueve con la cámara en el lazo antes del hito 3.
* Mano: estado a 10-20 Hz por Modbus; táctil solo cuando hace falta (bajar a la mesa, verificar).
* Registro: cada ciclo del lazo (50 Hz).
