# Respuestas a la sección 6 del reto, con datos

Todo lo que sigue se obtuvo en el **simulador** (h1_2_sim2real + pistas de `sim/`), con el
comando que lo reproduce. Cada respuesta indica qué valida el simulador (el **método**, contra
una verdad conocida) y qué hay que **medir en el robot** (los **números** reales). Los números
del simulador no son los del robot: varios parámetros de la planta están supuestos.

## 6.1 Geometría de la cámara — `herramientas/geometria_camara.py`, `herramientas/calibrar_camara.py`

| Pregunta | Respuesta |
|---|---|
| 1. FOV vertical | Color: 2·atan(240/603.6) = **43.4°** (horizontal 55.8°). IR: 2·atan(240/378.8) = **64.7°** (horizontal 80.4°). **La IR ve más suelo**: con 50.8° de inclinación, en el eje de la imagen ve de **0.30 a 5.10 m**; el color, de 0.64 a 3.14 m. |
| 2. Altura e inclinación | Método: RANSAC del plano del suelo en la profundidad → altura = distancia al plano, inclinación = asin(−n·z). **En el simulador recupera 1.7057 m y 50.79°** frente a los reales 1.707 m y 50.76° (error 1.3 mm y 0.03°, 8 fotogramas). **En el robot: medir** (`calibrar_camara.py` en el PC2, robot de pie en FSM 201, suelo vacío delante). Los valores de partida (1.703 m, 50.8°) salen del URDF de Unitree (`camera_joint`), no de una medida. |
| 3. Zona ciega y alcance útil | Geométrica: el suelo empieza a 0.30 m del centro del robot. **Real: mayor**, porque los pies tapan hasta ~0.36 m en el centro y las manos, al ser objetos elevados, ocupan la vista cenital hasta ~0.9 m a los lados (máscara en `percepcion.mascara_robot`). Alcance útil: **0.35–3.0 m** (`percepcion.bev_x`). Más lejos un píxel cubre ~3 cm del suelo y la cinta de 5 cm queda en ~1.5 px. Ese intervalo es el horizonte del controlador. |
| 4. ¿Cambia la inclinación al caminar? | **Medir en el robot** con la IMU a ≥ 100 Hz (`grabar_dataset.py` graba `imu.csv` a 100 Hz). El simulador **supone** ±1.5° de roll y ±0.8° de pitch a la cadencia de 1.43 Hz (`marcha_roll_deg`, `marcha_pitch_deg`). |
| 5. Píxel → suelo | Proyección inversa sobre el plano z = 0: rayo = R·K⁻¹·(u, v, 1), cortado con el suelo desde el centro óptico. En la otra dirección, la homografía **H = K [r₁ r₂ −Rᵀ C]**, con la que se genera una **vista cenital a 1 cm/píxel** (`warpPerspective`). R y C se corrigen en cada fotograma con el roll/pitch de la IMU (`seguidor/geometria.py`). |

## 6.2 Percepción — `herramientas/evaluar_percepcion.py` sobre `datos/dataset_n3_*`

| Pregunta | Respuesta |
|---|---|
| 1. ¿Color, IR o profundidad? Emisor | **IR**: más FOV (6.1.1) y no depende de la luz visible. La **profundidad no sirve**: la cinta es plana y no tiene relieve. El emisor proyecta puntos que, en el suelo claro, parecen manchas: un **filtro de mediana de 5 px** los quita. En el simulador, con el emisor encendido y apagado el resultado es igual (89.2 % frente a 89.4 % de detección; 1.54 frente a 1.58 cm de error). **Limitación del simulador**: el emisor real también *ilumina* (las sombras se notan menos con él encendido) y la IR real depende de la luz IR ambiente; eso no está modelado. **Medirlo en el robot**: grabar con `--emisor on` y `--emisor off` y comparar. |
| 2. Umbral en la sombra | Nivel 3, 660 fotogramas con verdad de terreno: con **umbral fijo**, en la zona en sombra salen **15 puntos, 67 % falsos**; con el **adaptativo**, **2 897 puntos, error medio 1.0 cm, 0.03 % falsos**. Fuera de la sombra los dos rinden igual (~1.55 cm). El adaptativo usa **contraste relativo** (fondo − I)/fondo, con el fondo estimado por un cierre morfológico de 15 cm: una sombra multiplica fondo y cinta por lo mismo, así que el contraste no cambia. |
| 3. Región procesada | La vista cenital de 0.35–3.0 m × ±1.2 m, recorrida en **franjas de 5 cm** de cerca a lejos. La asociación empieza en la franja más cercana (cerca de la última posición) y avanza con una puerta de 15 cm sobre lo extrapolado. Varias franjas permiten ver la curvatura, la barra y la esquina. Un ajuste local ponderado hacia lo cercano da el desplazamiento y el ángulo. |
| 4. Línea, barra, junta, sombra | Por **anchura en metros** dentro de cada franja: línea de 2.5–16 cm (oblicua se ve más ancha); **transversal ≥ 30 cm**: barra si cruza la línea por los dos lados (±15 cm) donde la línea acaba; esquina si sale hacia un solo lado y la línea llega **recta** (< 12° de cambio en los últimos 50 cm: una curva de R 1.2 m que se pone horizontal cambia ~24°). **Junta del suelo**: < 2.5 cm y poco contraste, se descarta. **Sombra de una persona**: más ancha que el núcleo de 15 cm, así que el cierre no la convierte en "cinta". Todo esto está en las pruebas (`tests/test_bloques.py`). |
| 5. Confianza y tiempo | Confianza = continuidad (franjas con línea / franjas visibles entre el primer y el último punto: que la línea **acabe** no la baja, que tenga huecos sí) × longitud vista (/0.8 m) × calidad del ajuste (residuo) × contraste. Tiempo por fotograma **en este PC**: 7.7 ms de media (p95 9.3 ms). **Medir en el PC2**: es otro procesador; el registro de cada tirada guarda `med_ms`. |

## 6.3 Dinámica y control — `herramientas/escalon_vyaw.py`

| Pregunta | Respuesta |
|---|---|
| 1. Respuesta a un escalón de vyaw | Método: escalón de ±0.3 rad/s, giróscopo a ~200 Hz y ajuste de primer orden con retardo. **En el simulador recupera retardo 0.325 s y τ 0.29 s** (los reales del simulador son 0.30 s y 0.35 s). **Medir en el robot** y poner los números en `estimacion.retardo_marcha` y `tau_marcha`. |
| 2. Latencia total | Cámara (1 fotograma: 33 ms a 30 fps) + percepción (8–20 ms) + RPC (unos ms) + retardo de la marcha (0.3 s supuestos) ≈ **0.35–0.4 s**. Lo que domina es la marcha: hay que medirla (6.3.1). |
| 3. Ley de control | **Pure pursuit** sobre la línea recordada, κ = 2y/(x²+y²) y vyaw = vx·κ, con mira **L = 0.55 + 0.8·vx** (~0.8 m a 0.3 m/s). A 0.3 m/s, los 0.3 s de retardo son 9 cm, muy por debajo de L. No deriva el error (el balanceo a 1.43 Hz metería ruido en un término D). **Con L = 1.15 m el robot recortaba las curvas de R 1.2 m y el pie se salía 0.33 m (nivel 3 fallido); con L ≈ 0.75 m, 0.20 m.** Además **se compensa la deriva de rumbo**: el sesgo (giróscopo − giro esperado por el modelo de la marcha) se resta de vyaw. Sin esa compensación, el nivel 1 serpenteaba ±8 cm y se abría 17 cm al frenar en la llegada. |
| 4. ¿vy, vyaw o ambos? | **Solo vyaw** (`usar_vy: false`); en el simulador basta. **Sin probar**: en el simulador vy es un desplazamiento lateral perfecto, y en la marcha real no se sabe cómo responde. Probarlo en el robot con un escalón de vy (como 6.3.1) antes de activarlo. |
| 5. vx con curvatura y confianza | vx = vx_max/(1 + 1.2·κ) × (0.4 + 0.6·confianza), con un mínimo de 0.10 m/s. Al llegar a la barra, vx ≤ 0.6·(distancia restante). |
| 6. Frecuencia de Move | **20 Hz**, cada `Move` con duration = 1 s. La cámara va a 15/30 Hz: entre fotogramas la estimación propaga la pose con la IMU y el modelo de la marcha, y la línea recordada no cambia. Mandar más rápido que la cámara no añade información, pero suaviza (límite de aceleración) y reacciona antes al vigilante. |

## 6.4 Estimación, pérdida y fin

| Pregunta | Respuesta |
|---|---|
| 1. Balanceo | **Compensación por fotograma** con el roll/pitch de la IMU (sin retardo, a diferencia de un filtro paso bajo, que a 1.43 Hz añadiría ~0.1–0.2 s). Prueba: con 2° de pitch sin compensar, el final de una línea a 1.5 m sale desplazado más de 6 cm (`test_compensacion_imu`). La memoria de la línea promedia lo que quede. |
| 2. Interrupción de 40 cm | La **memoria** de la línea (puntos vistos en un marco fijo, con la pose por yaw de la IMU + modelo de la marcha) sigue dando la línea durante el hueco. Además, la asociación por franjas salta el hueco porque extrapola. Resultado: el nivel 3 se completa en 6 de 6 tiradas. |
| 3. Tiempo sin línea | **0.8 m o 4 s** (`perdida_max_m`, `perdida_max_s`). Depende de la longitud del hueco (40 cm: 2× de margen) y de cuánto deriva la estima sin medidas. Con una esquina confirmada cerca, quedarse sin línea **no** cuenta como pérdida. |
| 4. Parar en la barra sin odometría | La barra se ve antes de entrar en la zona ciega. Su posición se guarda en el marco fijo (promediando varias detecciones) y se sigue por estima (vx · factor_vx). Se para con la **punta del pie** sobre ella. En el simulador queda **de −7 a +1 cm** (12 tiradas). **En el robot hay que calibrar `factor_vx`** (velocidad real/pedida, con cinta, como el `--factor` de `cuadrado.py`): con un error del 10 % en 1 m son 10 cm. |
| 5. Esquina del nivel 4 | La línea acaba en un tramo transversal hacia un lado y llega recta. Hacen falta **3 detecciones coherentes**; las que están a menos de 0.45 m no cuentan (el tramo sale recortado y el lado puede salir al revés). Cuando la esquina llega al centro del robot: StopMove, pausa de 0.8 s y **giro sobre el eje de ±90° cerrado con el yaw de la IMU** (constantes de `cuadrado.py`). Luego se olvida la línea vieja y se sigue. |

## Resultados en el simulador — `./scripts/bateria_sim.sh` (4 niveles × 3 semillas de deriva)

**12 de 12 tiradas con éxito** (criterio de la sección 9: llega a la barra, el pie no sale más
de 30 cm de la línea y no se pasa más de 30 cm), vx máx 0.30 m/s, emisor IR encendido:

| Nivel | Error lateral real medio | Máximo | Pie fuera (máx) | Parada vs. barra | Tiempo |
|---|---|---|---|---|---|
| 1 | 1.2–1.5 cm | ≤ 5.0 cm | ≤ 0.17 m | −1.8 a +0.9 cm | 18.5 s |
| 2 | 1.3–1.6 cm | ≤ 7.2 cm | ≤ 0.19 m | −4.2 a −5.7 cm | 34.5 s |
| 3 | 1.9–2.4 cm | ≤ 8.1 cm | ≤ 0.20 m | −5.9 a −7.2 cm | 38.8 s |
| 4 | 3.4–4.9 cm | ≤ 9.9 cm | ≤ 0.22 m | +0.2 a +1.3 cm | 30.6 s |

Percepción frente a la verdad (puntos detectados, nivel 3): 1.5 cm de error medio, p95 3.4 cm.
Estimación (desplazamiento en x = 0): RMS 0.4–3.2 cm según el nivel.

## Lo que el simulador NO reproduce (y por qué los números del robot serán peores)

* **No camina**: la base se desplaza como un sólido rígido con retardo, primer orden, deriva y
  balanceo supuestos. La marcha real tiene pasos, golpes al apoyar y un balanceo que no es una
  sinusoide.
* **La IR es la luminancia del render**, sin ruido de sensor real, sin desenfoque de movimiento
  ni luz IR ambiente. El emisor solo se modela como puntos que multiplican la reflectancia.
* **Intrínsecos perfectos** y sin distorsión. **Inclinación de la cámara**: la del URDF de
  Unitree, que nadie ha medido en este robot.
* Por eso el orden del reto sigue valiendo: medir (6.1, 6.3) → grabar el conjunto de datos →
  percepción sobre los datos REALES → simulacro → tiradas a mitad de velocidad.
