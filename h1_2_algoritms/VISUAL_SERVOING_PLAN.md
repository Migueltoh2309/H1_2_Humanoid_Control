# Plan de visual servoing + agarre (y cómo sirve al LfD de la tesis)

> Las citas [n] remiten a [REFERENCIAS_VISUAL_SERVOING.md](REFERENCIAS_VISUAL_SERVOING.md)
> (y [B-n] a [REFERENCIAS_BIMANUAL.md](REFERENCIAS_BIMANUAL.md)).

Continuación de [PERCEPTION_PLAN.md](PERCEPTION_PLAN.md) (dónde está la
fruta) y de [BIMANUAL_TEORIA.md](BIMANUAL_TEORIA.md) (cómo moverse sin
chocar). Aquí se cierra el lazo cámara → brazo → mano, y la sección 7 evalúa
si todo esto sirve para el núcleo de la tesis: **aprendizaje por
demostración (LfD)** del picking bimanual sobre la faja.

---

## 1. Decisiones de diseño

| Decisión | Elección | Por qué |
|---|---|---|
| Tipo de servoing | **PBVS** (error en 3D) [1], [2] | Cámara fija en el torso (eye-to-hand [4]). La tarea del QP ya es una pose cartesiana, así que el error "muñeca deseada (función de la fruta vista) − muñeca actual (cinemática)" entra sin tocar las capas 1-3. IBVS queda como comparación opcional. |
| Brazo | Uno por fruta, elegido por dónde **estará** la fruta en ~1.5 s (y < 0 → derecho) | Es la regla voraz (FIFO por lado) que el manuscrito usa como línea base (§3.3) y que no es óptima [24]; la política de asignación aprendida se compara contra ella. |
| Agarre | Desde arriba, dedos 20° bajo la horizontal y girados 40° hacia el centro, palma sobre la fruta | Única familia de orientaciones que es a la vez alcanzable en toda la faja y libre de choque con ella (ver §4). |
| Mano | Inspire con 6 actuadores (acoples `mimic` del URDF) y cierre suave (~6 N por dedo, ~12 N el pulgar) | Se ve y se comporta como la mano real; la fuerza queda por debajo de la máxima de la hoja de datos (RH56DFX: 10 N por dedo, 15 N el pulgar) [25], y la mano real admite umbrales de fuerza configurables. |
| Cierre del agarre en simulación | Física de contacto real + **asistencia** (un weld que se activa solo si ya hay contacto real de ≥2 dedos o palma y la fruta está a < 2 cm del punto esperado) | Un contacto rígido esfera-malla no sostiene la fruta (medido en §4). Con la asistencia se sigue midiendo la precisión del servoing: si la mano queda mal puesta, no hay contacto y el agarre falla. Es la misma idea que el agarre asistido de iGibson 2.0 [30]. |
| Percepción | Máscara HSV (Método A) + **ajuste de esfera** sobre la profundidad, con pérdida de Huber [7], [8] | Unas 14 veces más exacto que la mediana actual y robusto a la oclusión de la mano (§3). |
| Objeto en movimiento | Kalman de velocidad constante [20], [21] + feedforward de velocidad [5] | Estima la velocidad de la faja sin conocerla. `predict(t)` es la meta g = p(t_grasp) del manuscrito. |

## 2. Qué se construyó

| Pieza | Archivo |
|---|---|
| Escena con manos Inspire, mandarina libre, faja con carro y par IR de la D435i | `h1_2_utec/h1_2_description/scripts/build_scene_hands.py` → `mjcf/h1_2_scene_surgery_table_hands.xml` (generada, no editar a mano) |
| Cuatro estimadores del centro de la fruta y modelo de ruido D435 | `h1_2_algoritms/fruit_localization.py` |
| Geometría del agarre (fruta → pose de la muñeca, pre-agarre, IK) | `h1_2_algoritms/grasp_geometry.py` |
| Autofiltrado de dedos y pulgar en la nube | `collision_model.py` (`FINGER_CAPSULE`, `THUMB_CAPSULE`) |
| Evaluación depth vs. estéreo IR | `demos/evaluate_ir_stereo.py` → `resultados_percepcion/ir_stereo.csv` |
| Lazo completo de servoing + agarre, con física | `demos/visual_servoing_grasp.py` → `resultados_servoing/` |
| Lógica compartida demo/nodo: Kalman + máquina de estados | `h1_2_algoritms/visual_servoing.py` (`FruitTracker`, `GraspServoing`) |
| Nodo ROS | `QP_bimanual_avoidance.py`, `scenario:=grasp` (§5.1) |
| Bridge: dedos por `/joint_cmd`, asistencia, verdad de terreno | `sim_bridge_node.py` (`grasp_assist`, `ground_truth_bodies`), `mujoco_sim.py` |
| Todo junto en simulación | `ros2 launch h1_2_algoritms visual_servoing_sim.launch.py belt:=static\|moving\|fast` |

La escena nueva corrige además un problema que la vieja no dejaba ver. La
colisión de la malla de la faja es su **envolvente convexa**, que queda
~2.5 cm por encima de la superficie real por el bloque motor de una punta.
Con la mandarina en un slide no se notaba, pero con una fruta libre la
fruta salía expulsada. Por eso se reemplazó por una caja invisible con la
cara superior en la superficie real, medida con `mj_ray`.

## 3. Percepción: profundidad vs. estéreo IR

Pregunta: ¿el par IR de la RealSense localiza la fruta mejor que la
profundidad?

Se midieron 27 posiciones (3 valores de x × 9 de y sobre la faja), con 3
semillas de ruido. Los errores son del **centro** de la fruta contra la
verdad del simulador, en mm.

| Condición | Método | Recall | Error 3D medio | Mediana | p95 | \|ez\| |
|---|---|---|---|---|---|---|
| ideal | mediana de profundidad (lo actual) | 78% | 32.7 | 33.4 | 34.1 | 30.3 |
| ideal | mediana + 0.71·R | 78% | 3.7 | 3.5 | 5.3 | 2.2 |
| ideal | **ajuste de esfera** | 78% | **1.8** | 1.8 | 1.8 | 0.8 |
| ideal | estéreo IR (silueta) | 41% | 12.3 | 10.2 | 24.2 | 10.6 |
| ruido D435 | mediana | 78% | 33.7 | 33.7 | 36.9 | 31.2 |
| ruido D435 | mediana + 0.71·R | 78% | 4.3 | 4.0 | 7.5 | 2.5 |
| ruido D435 | **ajuste de esfera** | 78% | **2.4** | 2.3 | 4.2 | 1.2 |
| ruido D435 | estéreo IR | 42% | 12.6 | 10.1 | 24.9 | 10.6 |
| mano a 10 cm (oclusión parcial) | mediana | 100% | 37.5 | 37.2 | 40.5 | 29.3 |
| mano a 10 cm | mediana + 0.71·R | 100% | 21.2 | 22.3 | 30.0 | 3.5 |
| mano a 10 cm | **ajuste de esfera** | 100% | **6.2** | 6.1 | 11.4 | 3.5 |
| mano a 10 cm | estéreo IR | 0% | — | — | — | — |

(El recall de 78% en las condiciones sin mano viene de 6 posiciones que la
mano derecha, en la pose de reposo q = 0, tapa por completo. No es un fallo
del detector.)

**Respuesta: en simulación, no, el par IR no detecta mejor.** La causa es
física, no de implementación:

- **La resolución del par:** 50 mm de línea de base a ~0.8 m y f ≈ 433 px
  (imagen de 848 px) dan **~28 mm de profundidad por píxel de disparidad**
  (el error crece con z² [13], [15], [16]).
  Triangular dos centros de silueta, con 0.1–0.3 px de error cada uno (la
  sombra de la fruta sobre la faja, perspectiva), deja 5–10 mm.
- **Lo que hace el ASIC:** la profundidad de la D435 sale de esas mismas dos
  IR, pero con correspondencia densa subpíxel (~0.08 px) [12], [13]. El ajuste de
  esfera promedia ~1000 píxeles de la máscara.
- **Oclusión:** el estéreo IR no tolera que la mano tape parte de la fruta.
  El ajuste de esfera sí, porque ajusta la parte visible del casquete.

**El problema real de la percepción actual era otro:** la mediana mide la
**superficie** visible, no el centro. Eso da un sesgo de ~0.7·R, unos 30 mm
en z, independiente del algoritmo de detección, como ya lo había visto
PERCEPTION_PLAN.md §8. El ajuste de esfera de radio conocido lo elimina
(33 → 2.4 mm).

**¿Vale la pena el IR en el robot real?** Como experimento secundario sí,
por razones que MuJoCo no modela:

- Las IR de la D435 son de **obturador global** y la RGB de obturador
  rodante [14]. Sobre la faja en movimiento eso importa [19].
- Con el emisor apagado dan imágenes sin el patrón de puntos.
- Pueden ir a 1280×800, 1.5 veces más resolución.

Lo que no hay que hacer es cambiar el pipeline por el IR. **Recomendación:**
ajuste de esfera como método principal y, con la cámara real, medir IR vs.
profundidad con la fruta en posiciones conocidas (ChArUco). El código ya
existe (`locate_stereo_ir`).

## 4. Agarre con dedos: lo que se midió

- **Alcanzabilidad** (barrido con la IK sobre la faja, x = 0.34, y ∈
  [−0.25, 0.25]):
  - Hay muchas orientaciones "desde arriba" alcanzables.
  - El agarre lateral (palma vertical) solo es alcanzable con los dedos
    **65° hacia abajo**, y ahí las puntas entran en la faja.
  - La faja (z = −0.20 en `torso_link`) está cerca del límite inferior de
    alcance del H1-2.
- **Física pura (sin asistencia)** para mandarinas de 8 y 6 cm: se probaron
  3 orientaciones, varios offsets de la fruta en la palma y 3 órdenes de
  cierre. **Ninguna levantó la fruta.** Las causas medidas:
  1. **El pulgar:** en la preforma con el pulgar ya opuesto (yaw 1.1), el
     pulgar cuelga bajo la palma y **empuja la fruta durante la
     aproximación**. Se arregló aproximando con el pulgar abierto y
     oponiéndolo recién al cerrar.
  2. **La fuerza de los dedos:** con el `actuatorfrcrange` del modelo
     (1 N·m, unos 17-20 N en la punta) los dedos **lanzan** la fruta
     rígida. Se acotó a 0.35 / 0.6 N·m (~6 / ~12 N), por debajo de la
     fuerza máxima de la hoja de datos (RH56DFX: 10 / 15 N; la RH56DFTP del
     H1-2 declara hasta 30 N) [25]: es un cierre suave configurable, no
     una mano más débil que la real.
  3. **La geometría:** con la palma arriba, los dedos completamente
     cerrados llegan a y ≈ 0.04 m del eje de la muñeca, y el ecuador de una
     esfera de 8 cm apoyada en la palma está en y ≈ 0.06. **Los dedos no
     pasan por debajo del ecuador**, así que no hay cierre de forma [27],
     y cualquier empuje hace rodar la esfera [28].
- **Con asistencia**, el agarre funciona: la fruta sube ~100 mm y queda
  envuelta por los dedos.

**Qué implica para el robot real:** los modelos de contacto de los
simuladores difieren entre sí y de la realidad [29], y además una
mandarina real no es una esfera rígida: se deforma, tiene más fricción y no rueda igual. Aun así, el punto
3 sugiere que para frutas grandes convendrá una aproximación que la
**acorrale contra algo** (la palma y la faja) antes de cerrar, o un agarre
de pinza pulgar-índice-medio. Hay que validarlo con la mano real **antes**
de fijar la estrategia de agarre en el LfD.

## 5. Resultados del lazo cerrado

`demos/visual_servoing_grasp.py`: física real (PD + compensación de
gravedad, igual que el bridge), cámara a 15 Hz con latencia (procesamiento
+ 1 cuadro), QP a 20 Hz, ruido D435 en la profundidad.

Batería: 6 posiciones estáticas (y ∈ ±0.05, ±0.15, ±0.25 m) + faja a
0.03 / 0.05 / 0.07 m/s, 5 métodos de percepción, 45 ensayos
(`resultados_servoing/resumen.csv`; trazas por ensayo en `trazas/`).
"gt" = percepción perfecta con la misma latencia: separa lo que es error de
percepción de lo que es error de control.

Protocolo (el mismo del nodo ROS, §5.1): **no se cierra a ciegas** (si la
bajada no llega a < 2 cm del punto de agarre, MISSED) y el éxito se
**verifica con la cámara** tras retirar la mano (DONE solo si la cámara ve
la fruta en la mano). "Éxito" en la tabla = DONE y la fruta, según la
verdad de terreno, efectivamente levantada.

| Escenario | Percepción | Éxito | Error mano-fruta al cerrar | Error de estimación | Inicio → cierre |
|---|---|---|---|---|---|
| estática | gt (perfecta) | **6/6** | 7.3 mm | 0 | 3.6 s |
| estática | mediana (lo actual) | 0/6 | 60 mm | 41 mm | — |
| estática | mediana + 0.71·R | 2/6 | 49 mm | 30 mm | — |
| estática | **ajuste de esfera** | **5/6** | 15.0 mm | 4.8 mm | 3.6 s |
| estática | estéreo IR | 0/6 | 12.5 mm | 19 mm | — |
| faja | gt | **3/3** | 7.6 mm | 0 | 3.4 s |
| faja | mediana | 0/3 | 52 mm | 39 mm | — |
| faja | mediana + 0.71·R | 1/3 | 49 mm | 34 mm | — |
| faja | **ajuste de esfera** | **3/3** (0.03, 0.05 y 0.07 m/s) | 9.7 mm | 4.4 mm | 3.5 s |
| faja | estéreo IR | 0/3 | 14 mm | 18 mm | — |

Lo que dicen los números:

- **La percepción decide el éxito.** Con el mismo control, la mediana
  actual no agarra nunca (sesgo de ~40 mm con el brazo en escena y
  latencia), y el ajuste de esfera agarra 8 de 9 veces. La única falla con
  esfera fue y = +0.05 con la mano izquierda: la fruta quedaba cerca del
  centro, el error de estimación fue de 9 mm y la mano quedó a 24.5 mm del
  punto de agarre, por encima del umbral de 20 mm.
- **El estéreo IR falla por verificación, no solo por servoing.** En 2
  casos agarró la fruta, pero como el estéreo no detecta la fruta
  parcialmente tapada (0% en oclusión, §3), la verificación no pudo
  confirmar el agarre y abrió la mano (falso negativo). El resto terminó en
  MISSED (antes cerraba a ciegas y a veces acertaba: 1/6 y 1/3).
- **Piso de control: ~7 mm** (con percepción perfecta). Viene del criterio
  de cierre (8 mm o 3 s de espera) y del seguimiento del PD.
- **Objeto en movimiento: hizo falta un feedforward de velocidad.** La
  primera batería, sin él, falló a 0.05 y 0.07 m/s **incluso con
  percepción perfecta**: el lazo P del QP (ẋ = Kp·e) quedaba 21 mm detrás
  de la fruta, constante, y nunca bajaba de la tolerancia. Con
  ẋ = ẋ_ff + Kp·e (ẋ_ff = velocidad de la faja estimada por el Kalman, un
  parámetro nuevo `x_dot_ff` de `BimanualAvoidanceController.step`),
  3/3 con gt y 3/3 con esfera.
- **Ciclo de un brazo:** ~3.5 s del inicio al cierre, + 0.8 s de cierre +
  ~2 s de levantar ≈ **6.3 s**. La ventana del brazo derecho (y ∈ [−0.30,
  0.08], 0.38 m) dura 12.7 / 7.6 / 5.4 s a 0.03 / 0.05 / 0.07 m/s. **A 0.07
  m/s el ciclo ya no cabe en la ventana de un brazo:** un brazo solo no
  sostiene el flujo, que es la condición T_cycle > Δt que el manuscrito
  usa para justificar la alternancia bimanual (§3.3). Queda medido en
  simulación.
- Distancia mínima entre brazos ≥ 204 mm en todos los ensayos (solo actúa
  un brazo por fruta).

Videos: `resultados_servoing/moving_sphere_0.05.mp4` y
`static_sphere_-0.15.mp4` (vista externa + la imagen de la cámara del torso
en la esquina).

### 5.1 En vivo: nodo ROS (`scenario:=grasp`)

```bash
ros2 launch h1_2_algoritms visual_servoing_sim.launch.py belt:=moving   # static | moving | fast
```

Qué corre:

- **Bridge:** escena con manos, faja por keyframe, `grasp_assist`, y
  `/sim/ground_truth/mandarina` solo para medir (el control no lo lee).
- **Percepción:** `color_depth_detector_node`, con ajuste de esfera y el
  filtro de visibilidad.
- **Control:** `QP_bimanual_avoidance` con `scenario:=grasp`. Usa la misma
  `GraspServoing`/`FruitTracker` que la batería, las capas 1-2 activas, la
  capa 3 desactivada (la meta se mueve) y la nube sin la zona de la fruta.
- **Mano:** se comanda con los 6 joints por lado en `/joint_cmd`.

Salidas del nodo: `/visual_servoing/state` y la fruta estimada en
`/bimanual/markers`.

Resultado: 3 repeticiones por condición + una corrida final a 0.05 m/s
tras el último cambio, con el éxito juzgado por la verdad de terreno.

| Condición | Éxito real | Punto de agarre al cerrar | Ciclo (inicio → verificado) |
|---|---|---|---|
| fruta quieta | 2/3 | 4-5 mm | 7.8-8.5 s |
| faja 0.05 m/s | 3/4 | 5-8 mm | 7.5-8.8 s |
| faja 0.07 m/s | 3/3 | 4-7 mm | 7.6-8.6 s |

**Ningún falso positivo:** todos los DONE coinciden con la fruta en la
mano según la verdad de terreno.

Problemas que aparecieron **solo en vivo** y que se corrigieron en la lógica
compartida (así la batería de arriba ya los incluye):

1. **Percepción con la mano delante.** El nodo publicaba siluetas
   parciales, con errores de hasta 12 cm, y arrastraban la meta. Se agregó
   a los nodos de percepción `min_visible_fraction` = 0.35: el mismo
   criterio que la batería.
2. **Pista perdida tras una oclusión.** El brazo, al ir al reposo, tapa la
   punta de la faja unos 4 s. Con compuerta fija, el Kalman rechazaba para
   siempre la fruta que reaparecía 20 cm más allá. Ahora la compuerta crece
   con la incertidumbre de la predicción, y la pista se reinicia tras 3
   rechazos seguidos.
3. **Éxitos falsos.**
   - **"DONE por tiempo":** con la mano cerrada al aire, el nodo igual
     reportaba éxito.
   - **"No se ve en la faja":** una fruta golpeada al piso tampoco se ve en
     la faja, y pasaba como éxito.

   Ahora se retira la mano y se exige evidencia positiva: la fruta vista a
   < 10 cm de la mano. La tolerancia es del tamaño de la mano, porque una
   fruta bien sujeta quedó a ~8 cm del punto nominal.
4. **Cierre a ciegas.** Si la bajada no converge, cerrar golpea la fruta.
   Ahora se aborta (MISSED) y se rearma.
5. **Fruta fuera de la faja.** Tras un MISSED en que la mano empujó la
   fruta a la mesa, el nodo la perseguía, porque la ventana solo miraba y.
   Ahora se exige que esté sobre la faja (x, z).

**Falla con faja (0.05 m/s, 1 de 4):** con la mano sobre la fruta, la
cámara del torso no la vio durante **4 s**. La mano cerró donde el Kalman
la predecía (a 5 mm de ese punto), pero un error chico en la velocidad
estimada, acumulado en 4 s sin medidas, la dejó varios centímetros al
lado. La verificación lo detectó (FAILED, no un falso positivo). Es la
brecha 4 del §7 en su forma más directa. Hay dos mitigaciones posibles:

- **Acotar el tiempo ciego:** no cerrar si la última medida tiene más de
  ~1.5 s.
- **Otra vista en el último tramo:** una cámara en la muñeca (eye-in-hand
  [4]) o una segunda cámara con vista complementaria, como en Beldek *et al.* 2025 (ref. [19] del manuscrito).

**La falla (quieta, 1 de 3):** en la bajada, la fruta parcialmente tapada
se estimó ~2 cm corrida, la mano la rozó y se abortó bien (MISSED), pero
el roce ya la había sacado de la faja. Es el límite de la cámara del torso
en el último tramo (§7, brecha 4).

**Relevo entre brazos:** en una corrida a 0.07 m/s (antes de las
repeticiones), el brazo derecho no convergió a tiempo (136 mm) y abortó.
Al rearmarse, **el brazo izquierdo tomó la fruta más adelante** en la faja.
Es la alternancia bimanual que motiva el manuscrito (§3.3), saliendo de la
regla de ventanas por brazo.

## 6. Lo que falta para el robot real (en orden)

1. ~~**Ajuste de esfera en los nodos ROS**~~ **hecho**: los dos nodos lo
   usan por defecto (`localization:=sphere`), 1.7 mm medido en vivo (ver
   PERCEPTION_PLAN.md §9).
2. ~~**Nodo de servoing**~~ **hecho** (§5.1). Para el robot real falta:
   - **La mano:** comandarla por su interfaz real. Hoy los 6 joints van por
     `/joint_cmd`, que solo entiende el bridge; el parámetro
     `hand_cmd_topic` ya permite redirigirlos.
   - **Sin asistencia:** correr sin `grasp_assist` (en el robot no existe).
3. **Validar el agarre con la mano Inspire real** sobre mandarinas reales
   (§4). Es la mayor incertidumbre que queda.
4. **Calibración cámara-torso** (roadmap_vision V7): en simulación era
   exacta. El ajuste de esfera no corrige un error extrínseco, y PBVS es
   sensible a él [3].
5. Comparar IR vs. profundidad en la D435 real (§3).

## 7. ¿Sirve todo esto para el LfD de la tesis?

El manuscrito (§3.3 y los objetivos) plantea tres decisiones acopladas:

1. La **estimación predictiva** del punto de agarre g_i = p_i(t_grasp).
2. La **habilidad aprendida (DMP/ProMP) re-parametrizada** en línea hacia g
   con duración τ.
3. La **asignación** de frutas a los brazos bajo ventanas temporales.

Veamos cada pieza construida contra eso.

### Lo que sirve directamente

| Pieza construida | Papel en la arquitectura LfD |
|---|---|
| `fruit_localization.locate_sphere` + `FruitTracker` | **Es** el módulo 1 (objetivo específico 2): entrega g = p(t_grasp) con 2–6 mm de error y estima la velocidad de la faja sin conocerla. La DMP recibe g y τ de aquí. |
| QP bimanual (capas 1-3) | **La capa de ejecución** de la DMP [31], [35]: la primitiva genera x_ref(t) y ẋ_ref(t) de cada muñeca, y el QP los sigue respetando los límites y sin chocar brazo-brazo ni con el entorno. Una DMP sola no da esas garantías, y la tarea débilmente acoplada [39] las necesita en W∩. |
| Regla de selección de brazo | **La línea base voraz (FIFO)** contra la que se mide la política de asignación aprendida (las secuencias optimizadas superan a FIFO entre 10% y 40% [24]). |
| Trazas de los ensayos (`resultados_servoing/trazas/*.csv`) | Trayectorias de muñeca en el tiempo con la meta y la fase. Sirven para **probar el pipeline de DMP antes de tener las demostraciones humanas**, y para generar demostraciones sintéticas al medir la eficiencia muestral (objetivo 4). |
| Escena de simulación con verdad de terreno | El banco para las métricas de generalización (posiciones, separaciones y velocidades no demostradas), repetible y sin robot. |

### Qué hay que cambiar para enchufar la DMP

- **Término feedforward en el QP: ya está.** Para seguir una trayectoria
  aprendida la tarea tiene que ser ẋ = ẋ_ref + Kp·(x_ref − x). Se agregó
  `x_dot_ff` a `BimanualAvoidanceController.step` para la faja (§5: sin él
  quedaban 21 mm de retraso), y es exactamente la entrada que usará la DMP.
  Falta el equivalente angular si la DMP también genera orientación.
- **La máquina de estados APPROACH → DESCEND → CLOSE → LIFT es lo que el LfD
  debe reemplazar.** Hoy es a mano, y la tesis dice precisamente que eso no
  generaliza. La DMP aprendida cubre el tramo de alcance y retiro, con meta
  g, como en el pick-and-place con DMP de [32], y con meta móvil [34]. La máquina queda solo como supervisor de seguridad (abortar si se
  pierde la fruta, si la ventana se cierra, etc.).

### Brechas que hay que cerrar (importantes)

1. **Orientación de la muñeca.**
   - **Qué falta:** el roadmap usa de MediaPipe Pose solo hombro, codo y
     muñeca. Los otros tres puntos de la mano que da (pulgar, índice,
     meñique) permiten, como mucho, una orientación gruesa [37].
   - **Por qué importa:** el agarre del H1-2 depende de la orientación (§4:
     solo funciona una familia estrecha).
   - **Propuesta (el LfD como traslación + tiempo, y el agarre como modelo
     del robot):** la DMP aprende la **trayectoria de posición** y el
     **tiempo** de la muñeca (la forma del alcance, la coordinación entre
     brazos, la velocidad). La orientación y los últimos centímetros los
     pone `grasp_geometry` (el modelo del efector del robot), mezclados
     cerca de la meta. Es defendible por la correspondencia: la mano
     humana y la Inspire no agarran igual una fruta de 6-8 cm [36].
2. **Estado de agarre en las demostraciones.**
   - **Qué falta:** el manuscrito lo promete (contribución 1: "estados de
     agarre"), pero el roadmap de demostraciones lo excluye hoy ("detección
     del cierre de la mano: NO").
   - **Por qué importa:** sin él no se sabe **cuándo** cerrar ni se puede
     segmentar por fases (alcance, agarre, retiro).
   - **Propuesta:** agregar MediaPipe Hands [38] o, más simple, detectar
     el cierre por la distancia pulgar-índice.
3. **Retargeting del espacio de trabajo.**
   - **Qué pasa:** la faja está cerca del borde inferior del alcance del
     H1-2. Un humano de pie alcanza mucho más, así que las posiciones
     demostradas no se pueden copiar tal cual.
   - **Qué hacer:**
     - **La meta:** que la ponga siempre la percepción del robot (g), no la
       demostración. Esa es justamente la re-parametrización.
     - **La escala de la forma:** hay que ajustarla (DMP con escalado
       espacial).
     - **La factibilidad:** hay que verificarla antes de ejecutar (la capa
       3 ya diagnostica si la postura objetivo existe).
   - **Frames:** falta fijar T_torso←W, la pose del robot anclado respecto
     del {W} de ChArUco.
4. **La última parte del alcance es en lazo abierto.** En el pre-agarre (3
   cm sobre la fruta) la mano **tapa la fruta por completo** para la cámara
   del torso (medido: 0% de detección). El Kalman sigue prediciendo, así
   que no es un bloqueo, pero la meta g queda congelada en los últimos
   ~0.3 s.
   - Sobre la faja, el error de velocidad del Kalman se multiplica por ese
     tiempo.
   - El manuscrito ya prevé actualizar la meta "mientras el brazo ya está
     en movimiento" [23]. Esta medida dice hasta cuándo se puede.
5. **El agarre simulado es optimista** (asistencia). Las métricas de LfD en
   simulación deben centrarse en lo que la simulación mide bien:
   - error de alcance a la meta;
   - tiempo de ciclo;
   - éxito de la ventana temporal;
   - distancia mínima entre brazos.

   El éxito del agarre físico se mide en el robot real.

### Lo que no aporta al núcleo

- **IBVS:** no es necesario. La DMP produce trayectorias cartesianas, y
  PBVS es la interfaz natural.
- **El estéreo IR:** en simulación no mejora. Queda como experimento
  opcional en hardware (§3).
- **El agarre lateral:** no es alcanzable con esta altura de faja.

### Conclusión

Sí sirve, y como **infraestructura del objetivo específico 2**:
percepción predictiva, ejecución segura y línea base de asignación. El LfD
(objetivos 3-4) se monta encima **reemplazando la generación de
trayectorias a mano (la máquina de estados + `grasp_geometry` como meta
fija) por la DMP re-parametrizada**, sin tocar la percepción ni la
seguridad.

Lo más urgente para no bloquear el LfD más adelante:

- **Grasp state y orientación:** agregar el estado de agarre a las
  demostraciones y decidir cómo se maneja la orientación (brechas 1 y 2).
  Son decisiones del lado de la captura humana, y cuanto antes se tomen,
  menos demostraciones habrá que volver a grabar.
- **Agarre real:** validar el agarre con la mano Inspire real (§4).
