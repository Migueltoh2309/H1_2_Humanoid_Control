# Mapa completo: LfD de la faja (DMP + BC) en el H1-2

Este documento junta en un solo lugar todo el sistema de la tesis:

- qué hace cada pieza, con sus ecuaciones;
- la línea de entrenamiento, con los comandos exactos;
- la historia de las mejoras, con el número que movió cada una;
- lo que se probó y no funcionó.

El detalle de cada parte está en [LFD_DMP.md](LFD_DMP.md) (fase A),
[LFD_BC.md](LFD_BC.md) (fase B), [LFD_ROBOT.md](LFD_ROBOT.md) (comandos del
robot real) y [BIMANUAL_TEORIA.md](BIMANUAL_TEORIA.md) (QP). Para la intuición
sin ecuaciones, ver [LFD_CONCEPTOS.md](LFD_CONCEPTOS.md).

Estado al 2026-11-12: manos RH56DFTP oficiales y faja 3 cm más cerca. En todo
el ancho de la faja, el experto llega a 96.0% y la BC a 94.8% (§11).

---

## 0. El sistema de un vistazo

```
            ┌──────────────── simulador MuJoCo (escena u_table + faja + manos RH56) ────────────────┐
            │  faja cinemática: frutas a v ∈ [0.05, 0.08] m/s, llegadas Poisson (media 2 s)         │
            └───────┬───────────────────────────────────────────────────────────────▲────────────┘
                    │ RGB-D 15 Hz (cámara de la cabeza)                             │ q_des, kp, kd, tau_ff (100 Hz)
                    ▼                                                               │ ANGLE_SET manos (30 Hz)
   PERCEPCIÓN  segmentar naranja → ajuste de esfera (Gauss-Newton + Huber)          │
                    │ centros 3D                                                    │
                    ▼                                                     AGENTE arm_sdk EMULADO
   TRACKER     multi-fruta, velocidad de faja COMPARTIDA (mín. cuadrados)  (límites, rampa vel, PD puro)
                    │ (p, v) de cada fruta                                          ▲
                    ▼                                                               │ q_ref (20 Hz)
   ALTO NIVEL  política: FIFO | EXPERTO (planificador) | BC (red)                   │
               decide cada 0.25 s: ¿qué fruta agarra este brazo, o espera?         │
                    │ habilidad + fruta                                             │
                    ▼                                                               │
   HABILIDADES agarrar (DMP alcance + FSM cerrar/levantar/verificar táctil)        │
               depositar (servo articular) → volver (servo articular)              │
                    │ meta cartesiana / articular de cada brazo                    │
                    ▼                                                               │
   QP BIMANUAL 14 variables (OSQP): tarea + límites + distancias brazo-brazo ──────┘
               (velocity damper = CBF) + obstáculos de la nube de profundidad
```

Hay dos niveles de aprendizaje:

- **Fase A, cómo moverse:** una DMP aprendida de demostraciones genera el
  alcance a la fruta en movimiento.
- **Fase B, qué hacer:** una red (behavior cloning + DAgger) imita a un
  planificador experto que decide qué fruta toma cada brazo y cuándo.

---

## 1. El problema y cómo se mide

Dos brazos de 7 GDL toman frutas de una faja que pasa por delante y las
depositan a un costado. Llegan más frutas de las que los dos brazos pueden
atender, así que **decidir cuáles intentar** importa tanto como moverse bien.

**Escenarios** (`bc_faja.ESCENARIOS`). Todos tienen 10 frutas por episodio.

| Escenario | v faja (m/s) | Separación media entre llegadas | x de la fruta (m, torso) | Para qué |
|---|---|---|---|---|
| `normal` | 0.03–0.07 | 3 s | 0.30–0.40 | primera evaluación |
| **`exigente`** | 0.05–0.08 | 2 s | 0.30–0.40 | **el de la tesis** |
| `alcanzable` | 0.05–0.08 | 2 s | 0.30–0.37 | solo frutas alcanzables |
| `rapido` / `muy_rapido` / `extremo` | 0.08–0.10 / 0.10–0.12 / 0.12–0.15 | 2 s | 0.30–0.40 | barrido de velocidad |
| **`ancho`** | 0.05–0.08 | 2 s | **0.30–0.45** | rango de agarre: todo el ancho útil de la faja |

**Métrica:** frutas depositadas sobre frutas que llegaron. También se cuentan:

- los intentos;
- los abortos;
- las caídas (fruta que cae de la mano);
- las pérdidas (fruta que se va de la faja sin que nadie la tome);
- la distancia mínima entre brazos.

**Semillas** (cada episodio es reproducible: un hilo BLAS por proceso):

| Uso | Semillas |
|---|---|
| demos del experto | 0–39 |
| DAgger, ronda r | 500·r … 500·r + 39 |
| desarrollo y ajuste | 5000–5047 (y 3000–3015 al principio) |
| **evaluación final** | **10000–10089** (nunca usadas para entrenar ni ajustar) |
| calibración de tiempos | 900–915 |

**Estadística.** Las políticas y variantes se comparan con las **mismas
semillas**, de forma pareada: d_i = depositadas_B(i) − depositadas_A(i).
Se reportan:

- la media de d;
- un IC 95% por bootstrap (5000 remuestreos);
- la prueba de rangos con signo de Wilcoxon.

Pareado, porque la dificultad varía mucho entre episodios (velocidad,
llegadas); así se compara cada episodio consigo mismo.

---

## 2. Simulación

- **Escena.** `h1_2_scene_surgery_table_hands.xml`, la del laboratorio:
  - mesa `u_table` con el tablero a 0.77 m y la superficie de la faja a
    0.9645 m;
  - H1-2 con base fija y torso en z = 1.0282;
  - cámara RGB-D en la cabeza.

  `escena_faja.build_xml` la toma y agrega n frutas libres, estacionadas
  fuera de la vista hasta que "llegan".
- **Faja cinemática.** En cada paso, una fruta apoyada en la faja (y no
  sujeta) recibe la velocidad de la faja.
- **Asistencia de agarre (weld).** Cuando la mano cerró con ≥ 2 grupos de
  dedos o palma en contacto con una fruta que está a < 2 cm del punto de
  agarre esperado, se activa un weld mano-fruta (`FajaMulti.try_weld`).
  Emula la fricción y deformación de una fruta real, que un contacto rígido
  esfera-malla no captura. **Es el mayor riesgo para el robot real** (§12).
- **Manos RH56DFTP oficiales** (desde 2026-11-12;
  `h1_2_scenes/scripts/build_manos_rh56.py`, desde
  `h1_2_utec/h1_2_inspire_description`):
  - 12 juntas por mano, de las cuales 6 son actuadas: pulgar rotación y
    flexión, y los 4 dedos.
  - Montaje a x = 0.161 de la muñeca, con anillo de 10 mm y brida de 16 mm.
    La mano completa pesa 0.84 kg, medido en balanza; la mano vieja tenía
    unos 50 g de dedos.
  - Acople de las falanges: es el de la mano real, medido sobre fotos, no el
    `<mimic>` lineal. Cada falange acoplada es un polinomio de la actuada,
    ajustado a las curvas de `hand_coupling.yaml`:

    ```
    q_acoplada = c1·q + c2·q² + c3·q³ + c4·q⁴      (equality joint polycoef; error ≤ 8 mrad)
    ```

  - Actuadores de posición: kp = 2, con la fuerza acotada a ~6 N por dedo y
    ~12 N el pulgar. Con más fuerza, el cierre contra una esfera rígida la
    lanzaba.
  - Las mallas se simplificaron a ≤ 3000 triángulos cada una. Con las
    originales (~1 M de triángulos) cada proceso llegaba a ~1.9 GB al
    renderizar y la memoria se agotaba.
- **Contadores.** Una fruta cuenta como depositada cuando la mano abre en la
  habilidad depositar. Cuenta como caída si estaba sujeta y queda 10 cm
  bajo la faja. Cuenta como perdida si sale por el extremo de la faja.

---

## 3. Percepción

1. **Segmentación** por color: máscara naranja en RGB.
2. **Centro 3D por ajuste de esfera** (`fruit_localization.locate_sphere`).
   Se usan los puntos P_j de la máscara erosionada, retroproyectados con la
   profundidad, y se busca el centro c de una esfera de radio r = 4 cm:

   ```
   min_c  Σ_j ρ( ‖P_j − c‖ − r )         ρ = Huber (umbral 5 mm)
   ```

   Se resuelve por Gauss-Newton con pesos IRLS:

   ```
   e_j = ‖P_j − c‖ − r,   J_j = −(P_j − c)ᵀ / ‖P_j − c‖
   w_j = 1 si |e_j| < 5 mm, 5 mm/|e_j| si no
   Δc = −(JᵀWJ)⁻¹ JᵀW e
   ```

   El punto de partida es el centroide empujado un radio a lo largo del
   rayo. Hay un sesgo de ~1 mm medido. Con el par IR estéreo no mejora.
3. **Tracker multi-fruta** (`multi_tracking.MultiFruitTracker`).
   - Asocia las detecciones a pistas.
   - Como la faja lleva a todas las frutas a la misma velocidad, se estima
     **una sola velocidad compartida** por mínimos cuadrados sobre la
     historia de todas las pistas.
   - La posición de cada pista es el promedio de sus medidas del último
     segundo, llevadas al instante pedido con esa velocidad.
   - El filtro de cada pista es un Kalman de velocidad constante con
     compuerta adaptativa:

     ```
     x = [p; v],  F = [I  Δt·I; 0  I],
     Q = σ_a² [Δt⁴/4·I  Δt³/2·I; Δt³/2·I  Δt²·I],   R = σ_m² I
     compuerta: ‖innovación‖ < gate + 3σ_pred (σ crece sin medidas)
     ```

   - *Por qué compartida:* las oclusiones parciales hacían que el Kalman por
     pista subestimara la velocidad (−10 a −13 mm/s en los agarres
     fallidos). Con la velocidad compartida, el error al cerrar bajó de
     5–16 mm a ~1 mm, y los cierres fallidos de 9–22 a 0 de 88 (LFD_BC
     §6.1).
4. **Caja de la faja.** Se descartan las detecciones fuera de
   x ∈ [0.26, 0.47] − FAJA_DX y z = z_faja ± 4 cm. Esa z sale de
   `GG.Z_FRUTA_FAJA`, no está escrita a mano: con la mesa nueva, el valor
   escrito a mano tiraba todas las detecciones (0%).
5. **`lost_timeout` (1 s).** Si durante el alcance no se ve la fruta por 1 s,
   el alcance aborta. Sin esto, una pista borrada dejaba a la máquina de
   estados esperando para siempre (episodios de 74 s).

---

## 4. Control de movimiento (capa 1: QP bimanual)

Es un solo QP de 14 variables (los dos brazos), a 20 Hz, resuelto con OSQP
(`bimanual_avoidance.BimanualAvoidanceController`):

```
ẋ_des = [ K_p (p_d − p) + ẋ_ff ;  K_o · vec(q_d ⊗ q*) ]          (servo cinemático)

min_q̇   ½‖J q̇ − ẋ_des‖²_W + ½ w_c ‖J_c q̇ − ẏ_c‖² + ½ w_r ‖q̇‖²
s.a.    q̇_min ≤ q̇ ≤ q̇_max                       (en modo robot: |q̇| ≤ vel_max del agente)
        (q_min − q)/Δt ≤ q̇ ≤ (q_max − q)/Δt      (límites del AGENTE, no del MJCF)
        nᵀ(J_a q̇_L − J_b q̇_R) ≥ −ξ (d − d_s)/(d_i − d_s)    (velocity damper por par de cápsulas)
```

- **Velocity damper = CBF lineal.** Con h = d − d_s, la restricción es
  ḣ ≥ −γh, con γ = ξ/(d_i − d_s). Así d nunca baja de d_s:
  - entre brazos: d_s = 30 mm, d_i = 150 mm, ξ = 0.4 m/s;
  - con el entorno (nube de profundidad voxelizada): d_s = 50 mm,
    d_i = 200 mm.
- **Cápsulas.** Brazo superior, antebrazo, mano, dedos y pulgar, en el frame
  de cada eslabón. Se recalibraron para la RH56: la mano hasta x = 0.18 y los
  dedos hasta 0.27. Las de dedos y pulgar solo se usan para quitar la propia
  mano de la nube de puntos; si no, el damper congela el brazo.
- **Codo:** prioridad blanda, w_c = 0.03 (1/300 del peso de la tarea). Con
  w_c = 3 el error de la mano era de 38 mm; con 0.03 es de 0.7 mm.
- **Depositar y volver** van en **espacio articular**: servo con Kp = 6 hacia
  posturas precalculadas por IK en la rama de home. Con metas cartesianas,
  el QP local saltaba a otra rama y el brazo quedaba trabado.
- **Perfil `suave`:**
  - kd ×4 en los brazos: el PD dejaba la muñeca oscilando 4–5 cm por ~1 s;
    con kd ×4, el recorrido residual baja de 33 a 10 cm;
  - la referencia se interpola en cada paso de física;
  - volver termina con el brazo asentado (|q̇| < 0.5 rad/s).

  Fue la mejora más grande de la tesis (§11): el experto pasó de 68% a 84%.

---

## 5. Fase A: DMP de alcance

**Relativa a la meta (meta móvil).** La DMP aprende la posición de la muñeca
vista desde la fruta, y = x − g:

```
x_ref(t) = g(t) + y(t),     ẋ_ref(t) = ġ(t) + ẏ(t)        (ġ = velocidad de la faja del tracker)
```

Así la forma del acercamiento (pasar por encima y bajar por la normal de la
palma) se conserva aunque la fruta avance. Además, g se re-estima en cada
ciclo sin reiniciar la primitiva.

**Sistema de transformación** (Pastor 2009). Por eje, con K = α²/4,
D = α y α = 25 (amortiguamiento crítico):

```
τ ż = K(0 − y) − D z − K(0 − y₀) s + K f(s)
τ ẏ = z
s = 1 − t/τ  (fase LINEAL)          f(s) = s · Σᵢ ψᵢ(s) wᵢ / Σᵢ ψᵢ(s)    (20 gaussianas)
```

- f va en metros y **no se escala por (g − y₀)**. Con el escalado clásico de
  Ijspeert, un eje con desplazamiento casi nulo amplifica o invierte la forma.
- La fase es lineal porque con la exponencial la última parte del movimiento
  llega tarde.

**Paramétrica.** Los pesos dependen del punto de partida:

```
w(θ) = W₀ + W₁ θ,     θ = y₀ normalizado con la media/desviación de las demos
```

Se ajusta por mínimos cuadrados lineales. El error de validación cruzada
dejando una demo fuera (LOO) es de **7.7 mm**; con un solo w para todas las
demos (agrupada), 67.5 mm.

**Demos.** Son 20 alcances del oráculo (la máquina de estados con verdad de
terreno), con la fruta quieta y sobre la faja. Se reflejan en y para el
brazo izquierdo, se suavizan con Savitzky-Golay y se interpolan a 100 Hz.

**Ejecución.** Se usa τ = 0.5·τ_demo en el perfil `suave`.

**Intercepción** (g = p(t_fin), la meta predicha al final del alcance): es
un **resultado negativo**. La fruta entra de lado mientras la mano baja:
8–9 de 18 agarres. Se volvió a probar en el perfil A de 2026-11-09 y sigue
empeorando (§11).

---

## 6. Habilidades y modelo de tiempos

`habilidades.ArmExecutor`, una por brazo. Las metas de los dos brazos van al
mismo QP.

| Habilidad | Qué hace | Precondición → efecto |
|---|---|---|
| **agarrar(fruta)** | `GraspServoing`: alcance con la DMP → bajar → cerrar con la mano asentada (\|v_rel\| < 5 cm/s) → cierre táctil adaptativo (levanta apenas los dedos sienten la fruta) → subir 6 cm → **verificación táctil** (3 lecturas ≥ 1 N; en el robot, FORCE_ACT de la Inspire) | brazo libre → fruta en la mano |
| **depositar** | servo articular a la postura de depósito y abrir | fruta verificada → fruta depositada |
| **volver** | servo articular a la **pose de espera** (pre-agarre a 12 cm sobre la faja, ya orientada) | → brazo libre |

**Modelo de tiempos** (`habilidades.TIMING`). Son medianas medidas con
`bc_faja.py calibrar`; el experto y las características de la BC las usan
para predecir:

```
T_alcance = τ + reach_extra,     ciclo = τ + reach_extra + close_lift + show_verify + deposit + home
```

| Perfil | Ciclo por brazo |
|---|---|
| `base` | 8.55 s |
| `espera` | 5.0 s |
| `suave` | ~5.6 s |
| `suave_DR` | ~5.1 s |

**Postura de depósito.** Es delante de la faja (x = 0.20 − FAJA_DX):

- D: y = 0.30, dedos a −30°;
- `DEPOSIT_SHORT`: y = 0.40, dedos a −45°.

Con la mesa nueva, −60° no tiene IK dentro de los límites del robot. La IK
caía en otra rama (hombro yaw −2.6) y el brazo barría la faja. Ahora
`ArmExecutor` avisa si el depósito no tiene IK.

---

## 7. Fase B: políticas de alto nivel

Cada 0.25 s, por cada brazo libre: ¿qué fruta toma, o espera?

### 7.1 Factibilidad temporal (`alto_nivel.start_interval`)

Para un brazo con ventana en y [lo, hi], con margen m = 2 cm, y una fruta en
p con velocidad v_y (la faja va en +y), el alcance puede empezar en [s_min,
s_max]:

```
s_min = max( t + (lo + m − p_y)/v_y − T_alcance ,  t + (lo − b − p_y)/v_y )
s_max = t + (hi − m − p_y)/v_y − T_alcance
```

- La primera condición asegura que **al agarrar** la fruta esté dentro de la
  ventana.
- La segunda asegura que **al empezar** la fruta ya esté a menos de b = 8 cm
  de la ventana. Como la DMP sigue a la fruta, empezar antes llevaba al brazo
  a sus límites articulares.

Ventanas en y: derecho (−0.30, 0.05) e izquierdo (−0.05, 0.30).

### 7.2 FIFO (línea base)

Toma la fruta más antigua cuya posición predicha a 1.5 s cae en la ventana.

### 7.3 Experto (el "profesor")

1. Toma las K ≤ 6 frutas más urgentes y enumera todas las asignaciones
   fruta → {izquierdo, derecho, ninguno}: 3^K ≤ 729.
2. En cada asignación, cada brazo atiende sus frutas por fecha límite (EDF).
   Cada fruta empieza en max(brazo libre, s_min) y tiene que empezar antes de
   s_max; luego el brazo queda ocupado un ciclo.
3. Descarta las asignaciones con agarres de brazos distintos a < 15 cm en y
   y < 1 s en el tiempo, porque los dampers frenarían a los dos brazos.
4. Elige por (más frutas, empezar antes) lexicográfico.

El brazo libre ejecuta su primera fruta del mejor plan si puede empezar ya;
si no, espera.

### 7.4 BC: la red que imita al experto

**Características** (`alto_nivel.features`). Están en la convención del
brazo derecho: el izquierdo se refleja en y, así que la misma red sirve para
los dos.

- φ_k, 12 por fruta candidata:
  - x − 0.35, y, 10·v_y;
  - y al agarrar, y + v_y·T_alcance;
  - [s_min − t, s_max − t] de *este* brazo;
  - si es factible ya;
  - [s_min − t, s_max − t] del *otro* brazo;
  - si el otro la alcanza cuando quede libre;
  - distancia a la meta del otro brazo;
  - rango de urgencia.
- φ_ctx, 7:
  - número de frutas;
  - si el otro brazo está ocupado y cuánto le falta;
  - cuántas frutas son factibles para mí y para el otro;
  - la mínima holgura;
  - si yo estoy ocupado.

**Red** (`build_net`). Es invariante al orden y a la cantidad de frutas:

```
s_k      = MLP_c([φ_k ; φ_ctx])         (19 → 64 → 64 → 1, ReLU)
s_espera = MLP_w(φ_ctx)                 (7 → 64 → 1)
π(a | estado) = softmax([s_1 … s_n, s_espera])
acción = argmax
```

**Pérdida** (`bc_faja.fit`). Es una entropía cruzada ponderada: "esperar"
domina las muestras (se decide cada 0.25 s), así que se repondera para que
las decisiones de agarre pesen igual:

```
L = (1/N) Σ_i w_i · ( −log π(a_i* | s_i) ),     w_i = min(1, N_agarrar/N_esperar) si a_i* = esperar; 1 si no
```

Se entrena con Adam (lr 2·10⁻³, weight decay 10⁻⁴), batch de 128 y 200
épocas. El 15% de las muestras se separa para validación y se guarda la
época con mejor acierto en validación.

**DAgger** (Ross, Gordon & Bagnell 2011). Corrige el desvío de distribución:
el clon comete un error y llega a estados que el experto nunca visitó.

```
D ← demos del experto (40 episodios);  π₀ ← fit(D)
para r = 1 … R:
    correr π_{r−1} (la RED conduce) en 40 episodios nuevos
    el EXPERTO etiqueta cada estado visitado:  D_r = {(s, π*(s))}
    D ← D ∪ D_r ;   π_r ← fit(D)
```

Por defecto R = 2. Con la faja en todo su ancho (`ancho`) se probó R = 4
(§11). DAgger acerca la BC al experto, pero no la lleva más allá: el techo es
el experto, salvo donde el experto planifica con un modelo de tiempos
inexacto (§11, faja rápida).

---

## 8. Modo robot: los comandos del H1-2 real

Con `--robot`, el QP no mueve el simulador directamente. Pasa por una
emulación del camino real (`comandos_robot.py`):

```
PC (QP 20 Hz) ──UDP──► agente_sdk.py ──rt/arm_sdk 100 Hz──► motores (PD puro)
                                    └─Modbus ~30 Hz──► Inspire ANGLE_SET 0..1000
```

- **Agente emulado:**
  - límites articulares del agente (`aplicar_limites_robot`; muñeca pitch
    [−0.41, 0.30], codo [−0.90, 1.55]);
  - rampa de velocidad vel_max = 2 rad/s;
  - tau_ff saturado a 15 N·m.

  El QP y la IK usan **los mismos límites**. Con los del MJCF, el agente
  recortaba y el QP se desbocaba (el codo llegó a pedir 2.85 rad con tope
  1.55).
- **Motor = PD puro + feedforward de gravedad desde el PC:**

  ```
  τ = kp (q_des − q) + kd (q̇_des − q̇) + τ_ff,     τ_ff = g(q) (calculado en el PC)
  ```

  Con kd ×4 (`--kd-robot 4`, `agente_sdk --kd-escala 4`). Hay que calibrar kd
  con el robot colgado.
- **Manos:** ANGLE_SET **no es lineal** en las juntas. Se usa la curva medida
  de la mano real, con s = 1 − ANGLE_SET/1000 interpolado en la tabla de
  `hand_coupling.yaml`. Las manos avanzan a SPEED_SET y reciben consignas a
  ~30 Hz.
- **Grabar y reproducir:** `--registro DIR` guarda el flujo de órdenes, y
  `reproducir_comandos.py` lo manda por UDP al agente real.

---

## 9. Línea de entrenamiento (comandos)

Hay que preparar el entorno, regenerar las manos si cambió el repo oficial y
elegir el montaje:

```bash
source ~/venvs/h12/bin/activate; source /opt/ros/humble/setup.bash; source ~/humanoid_ws/install/setup.bash
export MUJOCO_GL=egl
# manos oficiales (si cambió h1_2_utec): regenera h1_2_scenes/meshes/inspire_rh56dftp/
python3 h1_2_scenes/scripts/build_manos_rh56.py --src <h1_2_utec>/ros_h1_2_ws/src/h1_2_inspire_description
colcon build --packages-select h1_2_scenes
# montaje: faja (y mesa) 3 cm más cerca del robot
export H12_FAJA_DX=0.03            # H12_MANOS=vieja para las manos anteriores
cd h1_2_algoritms/demos
```

Después, el pipeline:

```bash
R="--robot --vel-max 2 --tau-max 15 --kd-robot 4 --lost-timeout 1 --workers 5"
P="--escenario ancho --perfil suave_DR $R"

# 1) medir los tiempos de las habilidades (mediana de 12 episodios) -> --timing
python bc_faja.py calibrar $P --episodios 12
T="--timing reach_extra=0.37,close_lift=1.3,deposit=1.15,home=0.55"

# 2) demos del experto (semillas 0-39)        -> results/bc/datos_ancho_suave_DR_dx3/experto_s*.pkl
python bc_faja.py generar  $P $T --episodios 40
# 3) BC + DAgger (rondas de 40 episodios)      -> results/bc/politica_bc_ancho_suave_DR_dx3.pt
python bc_faja.py entrenar $P $T --episodios 40 --dagger 2
#    seguir con más rondas sin repetir las ya recolectadas:
python bc_faja.py entrenar $P $T --episodios 40 --dagger 4 --dagger-desde 2
# 4) evaluación final (semillas 10000-10089), pareada experto vs BC
python bc_faja.py evaluar  $P $T --politicas experto bc --episodios 90 --seed 10000 --out mesa_u/rh56/final.csv
# video de un episodio
python bc_faja.py video    $P $T --politica bc --seed 10032
```

Los datos y la red llevan los sufijos del escenario, el perfil y el montaje
(`_ancho_suave_DR_dx3`). `--red` evalúa una red de otro escenario.

---

## 10. Variantes de perfil (combinables)

Se seleccionan con `bc_faja.PERFILES` y `--perfil`:

| Perfil | Qué agrega |
|---|---|
| `base` | lo original: mostrar la fruta a la cámara para verificar, depósito al costado (−90°) |
| `tactil` | verificación por fuerza de los dedos, sin pose de mostrar |
| `corto` | táctil + depósito corto al frente + cierre con la mano asentada |
| `espera` | corto + pose de espera en pre-agarre + τ = 0.5 τ_demo + cierre táctil adaptativo + regla anti-interferencia 1 s / 15 cm |
| `suave` | espera + kd ×4 + referencia interpolada + volver asentado |
| `suave_D` | depósito 10 cm más cerca (y = 0.30, −30°) y abrir en movimiento (a 0.30 rad, en 0.3 s) |
| `suave_R` | volver sin asentarse: libre a 0.25 rad, \|q̇\| < 1.5 |
| `suave_A` | alcance con intercepción + empezar hasta 25 cm antes (`TRACK_BAND`) |
| **`suave_DR`** | **D + R (el actual)** |

---

## 11. Historia de las mejoras

Todas las filas son pareadas (mismas semillas), salvo que se indique otra
cosa.

| # | Cambio | Por qué | Efecto medido |
|---|---|---|---|
| 1 | Correcciones del simulador: servo articular para depositar/volver, 1 hilo BLAS, velocidad de faja compartida | brazo trabado en otra rama de IK; semillas no reproducibles; Kalman subestimaba v | FIFO 60 → 88 / 128 (dev); cierres fallidos 9–22 → 0 |
| 2 | **Experto + BC + DAgger** vs FIFO (exigente, final 30 ep) | decidir qué fruta intentar | BC 186 vs FIFO 131 / 300: **+42%**, p = 0.0002; BC = experto (p = 0.80) |
| 3 | Ciclo corto (`espera`): depósito corto, cierre asentado y táctil, pose de espera | ciclo de 8.55 → 5.0 s | experto 60 → 68% (p = 0.011), BC 62 → 67% (90 ep) |
| 4 | **Brazos amortiguados (`suave`, kd ×4)** | la muñeca rebotaba 4–5 cm por ~1 s al volver | **experto 68 → 84%, BC 67 → 84%** (+1.67 fr/ep, p ≈ 10⁻⁷) |
| 5 | Modo robot + `lost_timeout` | comandos reales; pistas borradas colgaban la máquina de estados | tesis 84% → +lost 88% (p < 0.002) |
| 6 | Robot con kd ×4 (vel 2, tau 15) | PD puro + rampa del agente | **93%** (vs tesis+lost: +0.5 fr/ep, p ≈ 10⁻⁶) |
| 7 | Mesa nueva `u_table` (+14 cm) + percepción relativa a la faja + poses que suben con la mesa + reentreno | la caja de la faja escrita a mano (0%), muñeca corta (57%) | tesis 88 → 92.7% (p = 2·10⁻⁴) |
| 8 | Depósito −60° → −45° | sin IK en los límites del robot: el brazo barría la faja (35 caídas) | **robot 97.6%, 0 caídas** (+1.1 fr/ep vs −60°, p ≈ 10⁻¹⁰) |
| 9 | Barrido de velocidad de faja | ¿subir la faja? | no: 94 → 55% de 0.065 a 0.135 m/s; las frutas/min quedan en ~13–14.6 (límite = ciclo del brazo) |
| 10 | **Ciclo DR** (depósito corto + retorno directo), BC reentrenada | ciclo más corto | faja rápida (0.08–0.10): **BC 83 → 91%** (p = 1.5·10⁻⁹); BC > experto (+0.51 fr/ep); faja actual: techo (98%) |
| 11 | Manos RH56DFTP oficiales | las del robot real (0.84 kg, otra geometría, ANGLE_SET no lineal) | franja de siempre: 98 → 96% (−0.2 fr/ep, p = 0.002): más realista |
| 12 | **Faja 3 cm más cerca** (+ mesa, percepción y poses que se corren con ella) | aumentar el rango de agarre | todo el ancho de la faja: **experto 92.0 → 96.0%** (+0.40 fr/ep, p = 0.001); pérdidas 68 → 35 |
| 13 | BC en `ancho`, DAgger 2 → 4 rondas | BC −11 frutas vs experto (p = 0.02) | **no cierra la brecha**: BC 853 (2 rondas) → 850 (4 rondas), Δ −0.03 fr/ep, p = 0.26; frente al experto (864), −0.16 fr/ep, p = 0.003. Queda la red de 2 rondas |
| 14 | **Tracker: posición con la velocidad compartida para asociar y dar de baja + 3 cm de margen en x** (`multi_tracking.SHARED_PRUNE`, `PRUNE_DX`; margen en el aborto de `GraspServoing`) | la mano tapa la fruta, la velocidad del Kalman de la pista se contamina y la predicción deriva ~5 cm/s hacia el robot: la pista se borraba "fuera de la faja" a mitad del alcance (frutas cercanas a 4 cm del borde de la caja: 21% de fracaso vs 5%) | final `ancho` dx 3: **experto 864 → 878 (97.6%, p = 0.04), BC 853 → 880 (97.8%, +0.30 fr/ep, p = 0.0009)**; abortos 85 → 23–25; BC − experto +0.02 fr/ep (p = 0.48): empate |

### Lo que se probó y NO funcionó

| Idea | Resultado |
|---|---|
| Adelantar el punto de agarre (p + v·lead) | 60 → 45 → 33 frutas (peor) |
| DMP con meta predictiva (intercepción) | 8–9/18 agarres; en el ciclo (perfil A) empeora todas las combinaciones (fallos 6 → 17–22) |
| Mapa de alcance dependiente de x | sin mejora |
| Verificación táctil sola (sin ciclo corto) | igual rendimiento; más fallos de agarre |
| kp ×2 además de kd ×4 | recupera 0.35 s de ciclo, pero vuelve la vibración |
| Subir la velocidad de la faja | menos % y las mismas frutas/min |
| Faja 9 cm más cerca | 94 → 68%: las frutas cercanas no se alcanzan o no se ven |
| Más rondas de DAgger (4 en vez de 2) en `ancho` | 853 → 850 / 900, sin cambio (p = 0.26). La brecha con el experto no es falta de datos: la BC tiene más agarres fallidos (25–32 vs 18) |
| Predictor de éxito del agarre en la BC (`bc_exito`: logit + α·log P(ok), y umbral) | Dev 21000–21029: experto 286, BC 283, α=1 284, α=1 + umbral 0.4 285, α=3 280 / 300; experto + umbral 285. Todo dentro del ruido. El predictor sí separa (AUC 0.82), pero el alto nivel casi nunca tiene una alternativa: una fruta saltada se pierde. Ver §14 |
| Faja 5 cm más cerca (2026-10, mesa vieja) | peor, pero la caja de percepción NO se corría con la faja (error corregido en #12) |

---

## 12. Riesgos para el robot real

1. **Asistencia de agarre (weld).** El robot no la tiene; es el mayor riesgo.
2. **Tacto.** En el sim la fuerza está en N; en el robot, FORCE_ACT viene en
   gf por dedo a ~30 Hz. Hay que fijar la conversión y el umbral con la mano
   real.
3. **kd ×4.** Calibrar con el robot colgado.
4. **Soltar en movimiento (D).** Verificar que la fruta caiga en el
   contenedor.
5. **Cámara real, latencia, faja real y deriva del torso.**

Escalera hacia el robot: [LFD_ROBOT.md §4](LFD_ROBOT.md).

## 13. Dónde está cada cosa

| Qué | Archivo |
|---|---|
| episodios, perfiles, escenarios, entrenamiento y evaluación | `demos/bc_faja.py` |
| diagnóstico de intentos y predictor de éxito (§14) | `demos/predictor_exito.py` |
| escena multi-fruta, faja, weld, manos RH56 | `h1_2_algoritms/aprendizaje/escena_faja.py` |
| habilidades y modelo de tiempos | `h1_2_algoritms/aprendizaje/habilidades.py` |
| factibilidad, experto, características y red BC | `h1_2_algoritms/aprendizaje/alto_nivel.py` |
| DMP | `h1_2_algoritms/aprendizaje/dmp.py`, `demos/dmp_alcance.py` |
| máquina de estados del agarre y Kalman | `h1_2_algoritms/vision/visual_servoing.py` |
| tracker multi-fruta | `h1_2_algoritms/vision/multi_tracking.py` |
| ajuste de esfera | `h1_2_algoritms/vision/fruit_localization.py` |
| geometría del agarre, manos, FAJA_DX | `h1_2_algoritms/vision/grasp_geometry.py` |
| QP bimanual y cápsulas | `h1_2_algoritms/bimanual/` |
| modo robot (agente, motor, manos) | `h1_2_algoritms/aprendizaje/comandos_robot.py`, `demos/reproducir_comandos.py` |
| manos oficiales para MuJoCo | `h1_2_scenes/scripts/build_manos_rh56.py` → `h1_2_scenes/meshes/inspire_rh56dftp/` |
| resultados | `results/bc/{robot,mesa_u,mesa_u/vel,mesa_u/rh56}/*.csv` |

---

## 14. Diagnóstico de agarres y predictor de éxito (2026-11-13)

Registro por intento (`BC_INTENTOS=<dir> python bc_faja.py evaluar ...`):
estado al decidir, fruta elegida y resultado (`ok | fallo | perdida | caida`).
120 episodios por política en `ancho`, dx = 3 cm, semillas 20000–20119:

| política | depositadas / 1200 | ok al 1.er intento | abortos | no siente | caídas |
|---|---|---|---|---|---|
| experto | 1135 | 87.4% | 114 | 34 | 16 |
| BC | 1143 | 88.1% | 109 | 29 | 17 |

* La BC NO falla distinto del experto: misma tasa y mismo patrón. La
  diferencia de la evaluación final (853 vs 864) era mayormente semillas.
* El fracaso depende sobre todo de x: 20–23% en la mitad de la faja
  cercana al robot, 3–6% en la lejana; también holgura < 0.5 s (≈ 26%) y
  agarre a < 0.15 m del otro brazo (≈ 24%). La velocidad casi no importa.
* Predictor P(ok | φ_fruta, φ_ctx) (`demos/predictor_exito.py`, MLP 16,
  mismas características de la BC): AUC 0.82 en validación cruzada por
  episodio. Integrado como `bc_exito` (logit_k += α·log P_k, y P_k < umbral
  → descartada) y `experto_exito` (control). Sin efecto (tabla §11).

Conclusión: la política de alto nivel ya está en su techo en este
escenario; lo que faltaba estaba en el bajo nivel.

### 14.1 La causa: el tracker borraba la pista en pleno alcance

Traza de una pista abortada (semilla 20000, fruta a x = 0.30): mientras la
mano baja, las medidas parcialmente tapadas corren la esfera hacia el
robot; el Kalman de la pista aprende una velocidad en x de ~−5 cm/s. Con la
fruta ya tapada del todo, `prune` predecía con ESE Kalman: en 0.9 s la
pista salía de la caja de la faja (borde a 4 cm de las frutas cercanas) y
se borraba -> "pista perdida 1.0 s" -> aborto. Corrección:

* asociación y baja con la misma posición que `estimates()`:
  p̂ = media de las medidas del último `pos_window` + v_faja·(t − tᵢ), con
  la velocidad compartida de la faja (no se contamina con una oclusión);
* 3 cm de margen en x para dar de baja una pista y para abortar un
  alcance en curso por "fuera de la faja" (no para empezar uno).

| dev 20000–20039 (400 frutas) | antes | velocidad compartida | + margen |
|---|---|---|---|
| experto: depositadas / abortos / fracaso en cercanas | 381 / 36 / 26 de 125 | 389 / 16 / 14 de 115 | 390 / 11 / 9 de 110 |
| BC | 388 / 38 / 25 de 128 | 391 / 17 / 13 de 115 | 392 / 13 / 12 de 112 |

Final (10000–10089, 900 frutas, pareado):

| política | antes | con la corrección | Δ |
|---|---|---|---|
| experto | 864 | **878 (97.6%)** | +0.16 fr/ep, IC95 [+0.01, +0.30], p = 0.04 |
| BC | 853 | **880 (97.8%)** | +0.30 fr/ep, IC95 [+0.14, +0.47], p = 0.0009 |
| BC − experto | −0.12 (p = 0.02) | +0.02, IC95 [−0.04, +0.09], p = 0.48 | empate |

Abortos 85 -> 25 (experto) y 23 (BC); frutas que se van de largo 35 -> 20
y 45 -> 19. La corrección es de la percepción: vale igual en el robot.

Los tres escenarios con la corrección (RH56, dx 3 cm, red
`politica_bc_ancho_suave_DR_dx3.pt`; `final_dx3_*_fixpista.csv`):

| escenario | experto | BC | BC − experto |
|---|---|---|---|
| `ancho` (x 0.30–0.45) | 878 (97.6%) | 880 (97.8%) | +0.02 fr/ep, p = 0.48 (empate) |
| `exigente` | 888 (98.7%; antes 864, p = 0.0002) | 883 (98.1%; antes 864, p = 0.003) | −0.06 fr/ep, p = 0.06 (empate) |
| `rapido` (0.08–0.10 m/s) | 785 (87.2%) | **809 (89.9%)** | **+0.27 fr/ep, IC95 [+0.14, +0.39], p = 0.0002** |

Con faja rápida la BC SUPERA al experto con significancia (ya pasaba con
las manos viejas, +0.51 fr/ep): pierde 89 frutas de largo contra 113. El
experto planifica con el modelo de tiempos, que con la faja rápida queda
justo; la BC, que imita sus decisiones en los estados que ella misma
visita (DAgger), suaviza esos errores. Con las RH56 corregidas, las manos
oficiales ya no cuestan frente a las viejas (exigente 888/883 vs 880/882).
