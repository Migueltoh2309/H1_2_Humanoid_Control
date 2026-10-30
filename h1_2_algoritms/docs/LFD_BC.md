# LfD, fase B: biblioteca de habilidades + behavior cloning simbólico

> Fase A (DMP de alcance): [LFD_DMP.md](LFD_DMP.md). Citas [n]:
> [REFERENCIAS_VISUAL_SERVOING.md](REFERENCIAS_VISUAL_SERVOING.md).

La fase A aprendió **cómo** mover el brazo hacia una fruta. La fase B aprende
**qué hacer**:

- cuando un brazo queda libre, ¿qué fruta agarra, o espera?
- con varias frutas llegando por la faja, ¿cómo se reparten entre los dos
  brazos?

Es la "política de asignación aprendida" del manuscrito (§3.3, decisión 3),
y la regla FIFO es su línea base.

Por ahora el demostrador es un **experto en simulación** (un planificador con
el modelo de tiempos de las habilidades). Las demostraciones humanas (fase C)
entrarán por la misma interfaz, (estado simbólico, acción), cuando la
captura multivista esté lista.

| Archivo | Qué contiene |
|---|---|
| `aprendizaje/escena_faja.py` | escena de N frutas (generada al cargar), faja cinemática, asistencia de agarre por fruta |
| `vision/multi_tracking.py` | detección de todas las frutas (un blob por fruta, esfera) y `MultiFruitTracker` (pistas de Kalman) |
| `aprendizaje/habilidades.py` | `ArmExecutor`: habilidades agarrar / depositar / volver por brazo; modelo de tiempos medido |
| `aprendizaje/alto_nivel.py` | estado simbólico, factibilidad temporal, `FifoPolicy`, `ExpertPolicy`, características, red y `BCPolicy` |
| `demos/bc_faja.py` | `calibrar` / `generar` / `entrenar` (BC + DAgger) / `evaluar` / `video` |
| `test/test_alto_nivel.py` | tests sin MuJoCo |
| `results/bc/` | datos (`datos/*.pkl`), red (`politica_bc.pt`), evaluaciones, videos |

---

## 1. Jerarquía

```text
 cámara -> blobs -> esfera -> MultiFruitTracker ──> estado simbólico
                                                    {frutas: (p, v)}, {brazos: ocupado, t restante, meta}
                                                         │
                         política de alto nivel (FIFO | experto | BC)
                         "brazo libre s: agarrar(pista k) | esperar"
                                                         │
                   ArmExecutor izq.            ArmExecutor der.
                   agarrar = DMP de alcance -> cerrar -> levantar -> mostrar/verificar
                   depositar -> volver
                                     │ metas cartesianas de las dos muñecas
                          QP bimanual (dampers brazo-brazo y entorno)
```

Esta separación es la de las arquitecturas de "habilidades + política":

- las habilidades son primitivas parametrizadas, con precondiciones y
  efectos;
- la política elige entre ellas sobre un estado simbólico.

El nivel alto no ve articulaciones ni imágenes. Por eso aprende con poca
data y en una GPU de 4 GB (o en la CPU).

## 2. La escena

`escena_faja.build_xml` parte de la escena de una fruta: quita la mandarina
única y agrega N frutas (aquí N = 8), cada una con un weld de asistencia por
brazo.

**La faja es cinemática.** Una fruta apoyada recibe la velocidad de la faja
en cada paso. El carro de la escena original lleva una sola fruta y recorre
±0.5 m, así que no sirve para varias.

**Llegadas por episodio** (`bc_faja.ESCENARIOS`, opción `--escenario`):

| Escenario | Velocidad de la faja | Separación entre frutas | Frutas |
|---|---|---|---|
| normal | U(0.03, 0.07) m/s | max(0.15 m / v, Exp(3 s)) | 8 |
| exigente | U(0.05, 0.08) m/s | max(0.15 m / v, Exp(2 s)) | 10 |
| alcanzable | como el exigente | como el exigente | 10 |

En normal y exigente, x ~ U(0.30, 0.40) m. En el exigente las frutas llegan
más rápido de lo que los dos brazos las atienden, y la asignación importa
(§6.3). El alcanzable usa x ~ U(0.30, 0.37), la franja que los brazos
cubren en toda su ventana en y (§6.5).

La superficie de la faja va de x = 0.28 a 0.49 m (`escena_faja.BELT_X`), y
la fruta tiene 4 cm de radio: con x < 0.30, la fruta queda al borde y se cae
al menor toque.

Las frutas entran por y = −0.52 (lado derecho del robot) y se pierden en
y = +0.52.

**Depósito: es una abstracción.** La mano va a una pose al costado (fruta en
(0.15, ±0.40, −0.05), dedos 60° hacia abajo y afuera, alcanzable por IK) y
abre. La fruta se cuenta como depositada; el canasto no se modela.

## 3. Habilidades y su modelo de tiempos

| Habilidad | Qué hace | Duración medida (mediana) |
|---|---|---|
| agarrar: alcance | DMP de la fase A, τ = 0.7·τ_demo = 2.92 s, meta móvil | 3.25 s |
| agarrar: cerrar, levantar, mostrar y verificar | `GraspServoing` (pose de mostrar con la palma arriba) | 2.90 s |
| depositar | servo ARTICULAR a la postura de depósito (IK en la rama de home) y abrir | 1.45 s |
| volver | servo articular a la postura de home | 0.90 s |

Las duraciones se midieron con `bc_faja.py calibrar` (8–16 episodios,
~50–100 ciclos) y quedan en `habilidades.TIMING`. **Un ciclo completo dura
~8.5 s por brazo.**

Depositar y volver van en espacio articular (`ArmExecutor.joint_target` ->
`joint_targets` del QP, ganancia 6). Con metas cartesianas, el QP local se
iba a otra rama de la IK (§6.1).

**Perfiles de tiempos** (`bc_faja.PERFILES`, opción `--perfil`; el modelo
de tiempos de cada perfil está medido con él):

| Perfil | Qué cambia | Ciclo |
|---|---|---|
| `base` (por omisión) | lo de la tabla de arriba | 8.55 s |
| `rapido` | alcance τ = 0.6·τ_demo; cierre 0.6 s, subida 0.6 s sin pausa; abre a 0.25 rad del depósito | 7.20 s |
| `tactil` | base, pero verifica el agarre por la fuerza de los dedos, sin la pose de mostrar | 8.10 s |

**Verificación táctil** (`GraspServoing(touch=...)`). Con la fruta ya
levantada, 3 lecturas seguidas de ≥ 1 N en los dedos verifican el agarre.
Sin fuerza a los 0.5 s, el agarre se da por fallido. En el simulador, la
fuerza es `FajaMulti.grip_force`: la fuerza normal de los contactos de la
mano contra lo que no es el robot. Es la misma emulación que hace el puente
sim2real del registro `FORCE_ACT` de la Inspire, así que en el robot se
leería ese registro.

## 4. Factibilidad temporal

Para un brazo s y una fruta con posición p y velocidad v (la faja va en +y),
el alcance puede empezar en el intervalo [s_min, s_max]
(`alto_nivel.start_interval`):

- **al agarrar** (inicio + T_alcance), la fruta está dentro de la ventana del
  brazo, con 2 cm de margen;
- **al empezar**, la fruta ya está a menos de 8 cm de la ventana (franja
  rastreable).

Ventanas en y: derecho (−0.30, 0.05), izquierdo (−0.05, 0.30).

**Mapa de alcance (opcional, `--mapa-alcance`).** `alto_nivel.window(side,
x)` cruza esa ventana con una ventana en y que depende de x: el intervalo
donde la IK del agarre llega con un error < 8 mm y < 5°, interpolado en x.
Con x > 0.39 m no hay alcance. El experto y las características de la BC lo
heredan a través de `start_interval`, y FIFO no cambia. Está apagado por
omisión, porque no mejora (§6.5).

La segunda condición es necesaria porque la DMP sigue a la fruta (meta
móvil, LFD_DMP §4.4). Sin ella, el experto empezaba alcances con la fruta
fuera del alcance del brazo:

| Experto (antes de las correcciones de §6.1) | Depositadas (12 episodios) | Abortos |
|---|---|---|
| sin franja | 40/96 | 28 |
| con franja | **60/96** | **10** |

## 5. Las tres políticas

**FIFO (línea base).** Es la regla de `GraspServoing`: la fruta más antigua
cuya posición dentro de 1.5 s cae en la ventana del brazo.

**Experto (demostrador).** Toma las K ≤ 6 frutas más urgentes y enumera las
asignaciones fruta → {izquierdo, derecho, ninguno} (3^K ≤ 729). Para cada
asignación:

- cada brazo atiende sus frutas por fecha límite (EDF);
- cada fruta empieza en max(brazo libre, s_min) y debe ser ≤ s_max;
- después, el brazo queda ocupado un ciclo;
- dos agarres de brazos distintos a < 20 cm en y y < 2 s en el tiempo se
  descartan, porque en el centro los dampers del QP frenarían a los dos;
- objetivo: (más frutas, empezar antes).

El brazo libre ejecuta su primera fruta del mejor plan si puede empezar ya;
si no, espera. Es lo que un operador haría mirando la faja: dejar pasar una
fruta que el otro brazo alcanza más adelante, no lanzarse sobre una que se
va a escapar, y esperar a la que está por entrar.

**BC.** Imita al experto con una red que puntúa cada opción:

- s(fruta k) = MLP([φ_k, φ_ctx]) y s(esperar) = MLP(φ_ctx);
- softmax sobre {frutas, esperar};
- es invariante al orden y a la cantidad de frutas.

Las características son **relativas al brazo**: el izquierdo se refleja en y,
como la DMP, así que la misma red sirve para los dos brazos. Son las
siguientes:

- φ_k (12 por fruta):
  - posición y velocidad;
  - y al agarrar;
  - tiempos al inicio y al fin del intervalo factible de *este* brazo y del
    *otro*;
  - si es factible ya;
  - si el otro brazo la alcanza cuando quede libre;
  - distancia al agarre del otro brazo;
  - urgencia relativa.
- φ_ctx (7):
  - número de frutas;
  - si el otro brazo está ocupado y cuánto le falta;
  - cuántas frutas son factibles para cada brazo;
  - la mínima holgura.

**Entrenamiento.** Cross-entropy, con la clase "esperar" reponderada (domina
las muestras porque se decide cada 0.25 s mientras un brazo está libre).

**DAgger** [Ross, Gordon & Bagnell, AISTATS 2011]:

1. la política aprendida conduce;
2. el experto etiqueta los estados que visita;
3. se agregan esas muestras y se reentrena.

Corrige el desvío de distribución típico del BC: el clon comete un error,
llega a estados que el experto nunca visitó, y ahí no sabe qué hacer.

## 6. Resultados

### 6.1 Correcciones del simulador (antes de comparar políticas)

La primera evaluación (20 episodios del escenario normal) dio FIFO 89,
experto 87 y BC 84 frutas de 160, sin diferencias. Al revisar los eventos
episodio por episodio, aparecieron tres fallas que afectaban a todas las
políticas por igual y tapaban las diferencias entre ellas:

1. **Brazo trabado.** El giro de muñeca hacia el depósito (−90° de azimut),
   con meta cartesiana, llevaba al QP local a otra rama de la IK: las
   articulaciones 2 y 4 quedaban en sus límites (2.66 y −2.75 rad). El brazo
   ya no volvía a home (timeout de 3 s, a 38 mm y 0.24 rad), y todos los
   alcances siguientes de ese brazo abortaban a 260–290 mm.
   *Corrección:* depositar y volver con servo articular hacia posturas
   precalculadas por IK en la rama de home.
2. **No determinismo.** Con varios procesos en paralelo, BLAS/OpenMP
   multihilo cambiaba el orden de las sumas. Como el episodio es caótico
   (contactos), la misma semilla daba 1 o 7 frutas según se corriera en serie
   o en el pool. *Corrección:* un hilo por proceso. Ahora cada semilla es
   reproducible y la comparación pareada tiene sentido.
3. **Velocidad de la faja subestimada.** El brazo izquierdo fallaba el 52% de
   sus cierres y el derecho el 23%.
   - No era la física del agarre. El Kalman de cada pista (σ_acc chico) se
     dejaba arrastrar por las oclusiones parciales: la fruta pasa bajo un
     brazo y la esfera se corre atrás y luego adelante.
   - Así subestimaba la velocidad: −5 mm/s de media y −10 a −13 mm/s en los
     agarres fallidos. Tras un alcance de 3 s, la mano cerraba 15–30 mm
     detrás de la fruta.
   - *Corrección* (`MultiFruitTracker`, `shared_velocity`): como la faja
     lleva a todas las frutas a la misma velocidad constante, se estima una
     sola velocidad, por mínimos cuadrados sobre la historia de todas las
     pistas. La posición de cada pista es el promedio de sus medidas del
     último segundo, llevadas al instante pedido.
   - Resultado: el error de la estimación al cerrar bajó de 5–16 mm a
     ~1 mm, y los cierres fallidos de 9–22 a 0 de 88.
   - Se probó también adelantar el punto de agarre (meta p + v·lead), y fue
     peor: 60 → 45 (lead 0.2 s) → 33 (lead 0.4 s) frutas de 128.

Con las tres correcciones, FIFO pasó de 60 a 88 frutas de 128 en las mismas
16 semillas. Después se acortó el ciclo de 9.2 a 8.5 s, con la ganancia del
servo articular de 3 a 6: depositar pasó de 1.80 a 1.45 s y volver de 1.20 a
0.90 s.

### 6.2 Escenario normal

30 episodios no vistos (semillas 10000–10029, 240 frutas). Ciclo de 9.2 s
(ganancia 3); experto y BC con demos y DAgger del mismo escenario.

| Política | Depositadas | Intentos | Abortos |
|---|---|---|---|
| FIFO | 183 (76%) | 243 | 52 |
| experto | 192 (80%) | 208 | 9 |
| BC + DAgger | 188 (78%) | 206 | 8 |

Diferencias pareadas por episodio (IC 95% por bootstrap, prueba de Wilcoxon):

| | Diferencia media | IC 95% | Gana / empata / pierde | p |
|---|---|---|---|---|
| experto − FIFO | +0.30 frutas | [−0.10, +0.73] | 10/13/7 | 0.19 |
| BC − FIFO | +0.17 frutas | [−0.33, +0.70] | 12/8/10 | 0.55 |

**No hay diferencia significativa.** Con la faja lenta, cualquier regla
alcanza casi todo: con v < 0.045 m/s, FIFO 86/96 y experto 86/96. Solo
con 0.045–0.06 m/s aparece la ventaja (experto 91/120, FIFO 83/120), y
hay pocos episodios así.

### 6.3 Escenario exigente

30 episodios no vistos (300 frutas); ciclo de 8.5 s; 40 episodios de demos
del experto y 2 rondas de DAgger de 40 episodios en el mismo escenario.

| Política | Depositadas | Intentos | Abortos | Distancia mínima entre brazos |
|---|---|---|---|---|
| FIFO | 131 (44%) | 267 | 123 | 30 mm |
| experto | 184 (61%) | 209 | 9 | 30 mm |
| **BC + DAgger** | **186 (62%)** | 209 | 10 | 30 mm |

| | Diferencia media | IC 95% | Gana / empata / pierde | p |
|---|---|---|---|---|
| experto − FIFO | +1.77 frutas | [+1.20, +2.33] | 25/3/2 | < 10⁻⁴ |
| **BC − FIFO** | **+1.83 frutas (+42%)** | [+1.10, +2.50] | 25/2/3 | **0.0002** |
| BC − experto | +0.07 frutas | [−0.33, +0.47] | 6/20/4 | 0.80 |

Entrenamiento:

| Ronda | Muestras | Acierto fuera de línea (total / agarrar / esperar) | Episodios con la BC al mando |
|---|---|---|---|
| 0 (solo experto) | 2552 | 97.8 / 94.6 / 98.1% | 230/400 frutas |
| 1 | 5150 | 99.0 / 95.4 / 99.4% | 233/400 |
| 2 | 7871 | 98.6 / 93.1 / 99.1% | (red final) |

Las demos del experto, en esos mismos 40 episodios de entrenamiento,
depositaron 218/400 frutas.

### 6.4 Discusión

- **La política aprendida supera a la regla FIFO cuando la decisión
  importa:** +42% de frutas, en 25 de 30 episodios, con 22% menos intentos.
- **Casi toda la ganancia es no empezar alcances inútiles.** FIFO aborta
  123 alcances, y cada uno deja al brazo ~4 s ocupado sin resultado. La BC
  aborta 10. FIFO mira solo la posición de la fruta dentro de 1.5 s; la
  política usa el modelo de tiempos de las habilidades (si la fruta va a
  estar en la ventana al terminar el alcance) y lo que va a hacer el otro
  brazo.
- **La BC iguala a su demostrador** (p = 0.80) y no lo supera: el clon
  queda acotado por el experto. DAgger corrigió el desvío de distribución:
  en lazo cerrado, la red llega a donde llega el experto.
- **Las tres políticas son seguras por igual:** la distancia mínima entre
  brazos es de 30 mm en todos los casos. La separación la impone el QP con
  dampers, no la política.
- **Con brazos no saturados (escenario normal) no hay qué aprender.** Es un
  resultado útil: la política aprendida no es peor que FIFO en ningún
  régimen.
- **El límite que queda es mecánico, no de la política** (§6.5). Ni el mapa
  de alcance, ni ciclos más cortos, ni acercar la faja, ni la verificación
  táctil superan a la configuración base. Una cota de capacidad: llegan 10
  frutas en ~20 s, cada una pasa ~9 s por la zona de trabajo, y cada brazo
  tarda ~8.5 s por ciclo. Eso da un máximo de ~65–70%, y la BC logra el 62%.
  Bajando el ciclo a 5.0 s (§6.6), el experto llega a 68% y la BC a 67%; con
  los brazos amortiguados (§6.7), a 84%.

### 6.5 Estudios de sensibilidad (escenario exigente)

Todos con las mismas semillas, salvo que se indique otra cosa; la
referencia es §6.3 (perfil base, faja en su lugar, sin mapa).

**Mapa de alcance por IK.** Experto con el perfil base, 30 episodios, según
el umbral de error de la IK:

| Umbral | Alcance hasta x | Depositadas / 300 | Intentos |
|---|---|---|---|
| 8 mm | 0.395 m | 160 | 180 |
| 12 mm | 0.405 m | 166 | 181 |
| 16 mm | 0.415 m | 173 | 193 |
| 20 mm | 0.415 m | 176 | 195 |
| 25 mm | 0.425 m | 184 | 207 |
| sin mapa | — | **184** | 209 |

El mapa solo quita frutas. El agarre asistido del simulador tolera hasta
2 cm (`try_weld`), así que recoge frutas que la IK estricta da por
inalcanzables. Con el perfil rápido sí ayuda (experto 153 → 170), porque ahí
los agarres lejanos fallan más.

Con perfil rápido y mapa de 8 mm, el pipeline completo dio FIFO 149, experto
170 y BC 187. La BC le gana a FIFO por +1.27 frutas por episodio (p =
0.0015), y aquí también al experto (+0.57, p = 0.028). Es equivalente a la
BC de §6.3 (186).

**Perfil de tiempos.** El rápido le sirve a FIFO (131 → 149, porque cada
aborto le cuesta menos) y perjudica al experto en las 30 semillas de
evaluación (184 → 153). En 16 semillas distintas (3000–3015), los cuatro
perfiles probados quedan dentro del ruido:

| Perfil (experto, 16 episodios) | Ciclo | Depositadas / 160 |
|---|---|---|
| base | 8.55 s | 94 |
| rápido | 7.20 s | 97 |
| alcance rápido + agarre base | 8.05 s | 93 |
| alcance base + agarre rápido | 7.55 s | 91 |

**Escenario alcanzable** (x ~ U(0.30, 0.37), perfil rápido, pipeline
completo): FIFO 149, experto 166, BC 164. No hay diferencia significativa
(BC − FIFO +0.50, p = 0.34). Cuando todas las frutas son alcanzables, la
asignación importa menos, y el límite es la capacidad. Una primera versión
con x ~ U(0.27, 0.34) dejaba frutas al borde de la faja, que se caían (31
por política); quedó en `results/bc/previo_alcanzable_x027/`.

**Faja 5 cm más cerca del robot** (`--faja-dx 0.05`, perfil base, pipeline
completo): FIFO 102, experto 110, BC 118, frente a 131 / 184 / 186.
Empeora mucho (experto −2.47 frutas por episodio, p < 10⁻⁴). Con la fruta
a x ≈ 0.25 m, pegada al torso, la bajada no llega al agarre (abortos del
experto 2 → 7 de 16 episodios) y el antebrazo empuja frutas fuera de la faja
(caídas 2 → 10). Esto no lo ve la IK, que solo comprueba la pose de la
muñeca. La franja útil en x es angosta (~0.28–0.38 m), y la faja actual ya
está centrada en ella. Con 8 cm, la estructura de la faja choca con la
cadera.

**Verificación táctil.** Como detector es exacta: ningún falso positivo
(todo lo verificado se depositó), y cada lectura de 0 N fue un agarre que
de verdad no sujetó la fruta. Pero no sube el rendimiento:

| Variante (experto) | Depositadas | Agarres fallidos | Ciclo |
|---|---|---|---|
| base (cámara), 30 episodios de evaluación | 184 / 300 | 3 | 8.55 s |
| táctil, 30 episodios de evaluación | 183 / 300 | 23 | 8.10 s |
| base, 16 episodios (3000–3015) | 94 / 160 | 0 | 8.55 s |
| táctil + subida de 20 cm + abrir antes, 16 episodios | 92 / 160 | 11 | 7.95 s |

Sin la pose de mostrar, el brazo va a depositar desde poca altura sobre la
faja y empuja las frutas que siguen en ella. Luego el otro brazo cierra a
~3.5 cm de donde de verdad está la fruta. Subir más la fruta reduce los
empujones, pero el depósito desde abajo es más largo (1.85–2.05 s frente a
1.50 s), y el ahorro neto de ciclo es de solo ~0.5 s. La pose de mostrar,
además de servir para verificar, saca al brazo de la zona de trabajo.

Junto con este estudio, el tracker descarta las detecciones a más de 5 cm
sobre la superficie de la faja (`bc_faja.Z_BELT_FRUIT`). Así, la fruta que
lleva una mano no contamina las pistas. Con el perfil base no cambia nada:
94/160 en las semillas 3000–3015, igual que antes del filtro.

### 6.6 Ciclo corto: depósito corto, pose de espera y cierre táctil (exigente)

§6.5 mostró que el límite es la capacidad (ciclo de ~8.5 s). Esta sección
acorta el ciclo físico sin tocar la política, dentro de los límites del
H1-2: el pico articular medido es 3.9 rad/s, frente a 9–31 rad/s del URDF.

**Cambios** (todos opcionales; el perfil `base` no cambia):
- **Depósito corto** (`habilidades.DEPOSIT_SHORT`, `ArmExecutor(deposit_pose=)`):
  la fruta va a (0.20, ∓0.40, −0.05) con elevación −60° y azimut 0°, con
  los dedos hacia delante. La mayor rotación articular desde home es
  0.77 rad, frente a 1.65 rad del depósito a ±90°.
- **Cierre asentado** (`GraspServoing(close_max_vrel=0.05)`): la mano
  cierra solo si su velocidad relativa a la fruta es menor a 5 cm/s. Los
  agarres fallidos cerraban a 0.075 m/s; los buenos, a 0.035 m/s. La fruta
  se escapaba durante CLOSE, sin llegar a la soldadura asistida.
- **Cierre táctil adaptativo** (`close_touch=True`): pasa a LIFT en cuanto
  la fuerza de agarre (`FajaMulti.grip_force`, análogo de FORCE_ACT)
  supera el umbral en varios pasos seguidos, en vez de esperar el
  `close_hold` fijo. Con subida de 6 cm en 0.5 s.
- **Pose de espera** (`--perfil espera`, `home_pose(espera)`): el brazo
  espera sobre la faja, en (0.34, ∓0.10, z de la fruta + 0.12), en vez de
  en home. Junto con `reach_tau_scale 0.5`.
- **Experto ajustado al ciclo nuevo**: la regla anti-interferencia pasa de
  la de §5 a `sep_t 1.0 s, sep_y 0.15 m` (`PERFILES["espera"]["experto"]`).

**Ciclo por fase** (experto, perfil `espera`): alcance 2.45 s,
cierre→verificación 1.40 s, depósito 1.15 s, vuelta 0.40 s; en total 5.0 s,
frente a 8.55 s del perfil base. Un alcance aún más rápido (τ × 0.4) empeora:
96 frente a 119 de 160 en 16 semillas.

**Ajuste del experto** (semillas de desarrollo 5000–5047, no las de
evaluación). Con la regla por omisión: 302/480; con `sep_t 1.0, sep_y 0.15`:
324/480 (+0.46 por episodio, p = 0.068). Relajar más no agrega nada, y
ensanchar 4 cm las ventanas es mucho peor (112 frente a 159 de 240).

**Resultados.** Pipeline completo por perfil (40 episodios de demos, 2
rondas de DAgger), evaluado en los mismos 90 episodios (semillas
10000–10089, 900 frutas):

| Perfil | Ciclo | FIFO | Experto | BC |
|---|---|---|---|---|
| base | 8.55 s | 409 (45%) | 542 (60%) | 560 (62%) |
| `corto` (depósito corto + cierre asentado) | — | 474 (53%) | 574 (64%) | 589 (65%) |
| `espera` con la regla anti-interferencia anterior | 5.0 s | 527 (59%) | 558 (62%) | 579 (64%) |
| **`espera`** | **5.0 s** | 535 (59%) | **616 (68%)** | **607 (67%)** |

| Comparación pareada (90 episodios) | Diferencia media por episodio | IC 95% | p (Wilcoxon) |
|---|---|---|---|
| experto `espera` − experto base | +0.82 | [+0.22, +1.43] | **0.011** |
| BC `espera` − BC base | +0.52 | [−0.08, +1.11] | 0.085 |
| BC `corto` − BC base | +0.32 | — | 0.062 |
| BC `espera` − FIFO `espera` | +0.80 | — | **0.006** |
| experto `espera` − FIFO `espera` | +0.90 | — | **0.002** |
| BC `espera` − experto `espera` | −0.10 | — | 0.41 |

- **El ciclo corto sube el techo**: el experto pasa de 60% a 68% (p = 0.011).
  La BC sigue la misma tendencia (62% → 67%), aún no significativa con 90
  episodios.
- **FIFO es el que más gana con el ciclo corto** (+126 frutas), porque cada
  alcance inútil cuesta menos. Aun así, planificar sigue valiendo +0.8 a +0.9
  frutas por episodio.
- **La BC imita bien** (99.1% de acierto en validación) y queda empatada con
  el experto.
- **Seguridad sin cambios**: distancia mínima entre brazos de 28 mm (FIFO
  24 mm), sin frutas caídas con experto ni BC. El experto `espera` tuvo 16
  agarres fallidos y 22 abortos.
- **Sesgo de selección**: las 16 semillas usadas para afinar (3000–3015)
  inflaban al experto (74% frente a 62% en las de evaluación). Por eso el
  ajuste final se hizo en semillas de desarrollo aparte (5000–5047).

### 6.7 Brazos amortiguados (perfil `suave`, exigente)

En los videos, al volver del depósito el brazo rebotaba. Con un episodio
instrumentado (`scripts_diagnostico/vuelta.py`) se midió que la muñeca
oscila 4–5 cm con período ~0.5 s durante ~1 s, y recorre 27–33 cm en el
segundo siguiente a volver. La causa es el PD articular que emula los
motores: kd de `DEFAULT_GAINS` (hombro kp 60, kd 2) con armature 0.01 y
damping 0.05 da un factor de amortiguamiento de ~0.2. A eso se suman la
referencia en escalones de 50 ms y el paso al control cartesiano con el
brazo todavía a ~0.5 m/s.

Perfil `suave` = `espera` + `kd_escala 4` (ζ ≈ 0.7) + `interp` (la
referencia se interpola en cada paso de física) + `home_vel_tol 0.5 rad/s`
(volver termina con el brazo asentado). Tiempos re-medidos: depositar
1.50 s y volver 0.65 s, frente a 1.15 y 0.40 s; el ciclo sube ~0.6 s.

| Medida (3 episodios) | `espera` | kd ×4 | `suave` | `suave_kp2` (+ kp ×2) |
|---|---|---|---|---|
| Recorrido de la muñeca en 1 s tras volver | 27–33 cm | 10 cm | 11–14 cm | 8–19 cm |
| Aceleración de la muñeca, p99 | 8.8–9.8 m/s² | 6.3 | **4.5** | 6.1–7.6 |
| Depositar / volver | 1.15 / 0.40 s | — | 1.50 / 0.65 s | 1.25 / 0.50 s |

Pasar `dq_des` (feedforward de velocidad) al PD no ayudó.

**Resultados** (pipeline completo, mismos 90 episodios que §6.6):

| | FIFO | Experto | BC |
|---|---|---|---|
| `espera` | 535 (59%) | 616 (68%) | 607 (67%) |
| **`suave`** | 547 (61%) | **753 (84%)** | **757 (84%)** |

- BC `suave` − BC `espera`: +1.67 frutas por episodio, IC 95%
  [+1.12, +2.22], gana en 60 de 90, p = 4·10⁻⁷. Experto: +1.52, p = 9·10⁻⁷.
- BC − FIFO: +2.33, gana en 73 de 90, p ≈ 10⁻¹³. BC − experto: +0.04,
  p = 0.58 (acierto en validación 97.6%).
- FIFO no cambia (p = 0.67). El experto hace más intentos (852 frente a
  748), con 10 agarres fallidos en vez de 16 y sin caídas. La distancia
  mínima entre brazos es de 27–29 mm.

**De dónde sale la ganancia** (experto, semillas de desarrollo 5000–5023,
de 240 frutas; `scripts_diagnostico/aislar.py`):

| Variante | Depositadas | vs `espera` |
|---|---|---|
| `espera` | 171 | — |
| `espera` con los tiempos de `suave` | 165 | −0.25/ep, p = 0.20 |
| `espera` + solo asentado (tiempos de `suave`) | 172 | +0.29/ep frente a la anterior, p = 0.49 |
| **`espera` + solo kd ×4** (tiempos de `suave`) | **202** | **+1.29/ep, p = 0.007** |
| `suave` | 192 | +0.88/ep, p = 0.09 |
| `suave` con los tiempos de `espera` | 185 | −0.29/ep frente a `suave`, p = 0.08 |

La ganancia viene del **amortiguamiento**, no del modelo de tiempos ni del
fin de vuelta asentado. Solo con kd ×4 sale incluso algo mejor que `suave`
(+0.42/ep, p = 0.07, no significativo), aunque con más aceleración (p99
6.3 frente a 4.5 m/s²). Las pérdidas por la faja caen a la mitad (284 → 147
del experto). Falta identificar el mecanismo; una hipótesis es que el brazo
que oscila arranca peor el alcance siguiente o perturba la percepción.

**En el robot real**, kp y kd se mandan en el `LowCmd`. La reductora real
amortigua más que el modelo, así que kd hay que calibrarlo en el robot,
empezando por valores bajos.

## 7. Uso

```bash
source ~/venvs/h12/bin/activate && source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash
cd ~/humanoid_ws/src/h1_2_algoritms
MUJOCO_GL=egl python3 demos/bc_faja.py calibrar --episodios 8      # tiempos de las habilidades
MUJOCO_GL=egl python3 demos/bc_faja.py generar --episodios 40      # demos del experto (~15 min)
MUJOCO_GL=egl python3 demos/bc_faja.py entrenar --dagger 2         # BC + 2 rondas de DAgger
MUJOCO_GL=egl python3 demos/bc_faja.py evaluar --episodios 20      # FIFO vs experto vs BC
MUJOCO_GL=egl python3 demos/bc_faja.py video --politica bc --seed 10003

# escenario exigente (datos en results/bc/datos_exigente/, red politica_bc_exigente.pt)
MUJOCO_GL=egl python3 demos/bc_faja.py generar  --escenario exigente --episodios 40 --workers 8
MUJOCO_GL=egl python3 demos/bc_faja.py entrenar --escenario exigente --dagger 2 --workers 8
MUJOCO_GL=egl python3 demos/bc_faja.py evaluar  --escenario exigente --episodios 30 --workers 8 \
    --out evaluacion_exigente.csv

# estudios de §6.5 (cada opción guarda datos y red con su sufijo: _mapa, _mapa_<json>, _dx5)
... --perfil rapido|tactil                  # perfil de tiempos (por omisión: base)
... --mapa-alcance [mapa.json]              # mapa de alcance por IK ({"right": [[x, [y_lo, y_hi]], ...], "xmax": x})
... --faja-dx 0.05                          # faja 5 cm más cerca del robot
... --escenario alcanzable

# §6.6, perfil espera (datos en datos_exigente_espera/, red politica_bc_exigente_espera.pt);
# todo perfil distinto de base agrega el sufijo _<perfil> a datos y red
A="--escenario exigente --perfil espera --workers 6"   # 8 workers se quedan sin memoria (15 GB)
MUJOCO_GL=egl python3 demos/bc_faja.py generar  $A --episodios 40
MUJOCO_GL=egl python3 demos/bc_faja.py entrenar $A --dagger 2
MUJOCO_GL=egl python3 demos/bc_faja.py evaluar  $A --episodios 90 --seed 10000 \
    --out evaluacion_exigente_espera.csv
```

# §6.7: igual con --perfil suave (datos _suave, red politica_bc_exigente_suave.pt);
# pipeline + videos en results/bc/scripts_diagnostico/pipeline_suave.sh (≈ 2 h 30 min)

El mismo pipeline completo está en
`results/bc/scripts_diagnostico/pipeline_espera.sh` (≈ 2 h 10 min con 6 workers).
Los diagnósticos de §6.6 están en esa carpeta, junto con `comparar.py` para
las comparaciones pareadas; ver su `README.md`.

Resultados en `results/bc/`:
- `evaluacion.csv` (normal) y `evaluacion_exigente.csv`, con sus `*_log.txt`;
- §6.5: `evaluacion_exigente_mapa.csv` (perfil rápido + mapa),
  `evaluacion_exigente_base_{mapa,m12,m16,m20,m25}_experto.csv`,
  `evaluacion_exigente_rapido_experto.csv`, `evaluacion_alcanzable.csv`,
  `evaluacion_exigente_dx5.csv` y `evaluacion_exigente_tactil_fe.csv`;
- §6.6: `evaluacion_exigente_espera.csv` (90 episodios),
  `evaluacion_exigente_corto.csv` (30) y
  `evaluacion_exigente_{base,corto}_s10030.csv` (60 episodios más, semillas
  10030–10089), con sus datos y redes `politica_bc_exigente_{corto,espera}.pt`;
  `previo_espera_sep2/`: el perfil `espera` con la regla anti-interferencia
  anterior;
- §6.7: `evaluacion_exigente_suave.csv` (90 episodios), la red
  `politica_bc_exigente_suave.pt` y los videos `episodio_{experto,bc}_suave_s{10002,10018}.mp4`;
- las redes `politica_bc.pt` y `politica_bc_exigente.pt`;
- `previo_2026-10-22/`: la primera evaluación, de antes de las correcciones
  de §6.1.

## 8. Pendiente

- **Generalización:** evaluar la red del escenario exigente en el normal,
  y entrenar con una mezcla de escenarios.
- **Superar al experto:** el BC queda acotado por su demostrador. Opciones:
  - un experto mejor (horizonte más largo; tiempos con su dispersión, no
    solo la mediana);
  - afinar la red con los resultados reales de los episodios (RL o BC
    ponderado por el resultado).
- **Ciclo más corto:** hecho en §6.6 (5.0 s). Falta: más episodios para
  confirmar la ganancia de la BC (p = 0.085), y validar en el robot la pose
  de espera y el depósito corto (alcanzabilidad y canasto real).
- **Robot real:** la verificación táctil con `FORCE_ACT` de la Inspire,
  como alternativa a la pose de mostrar. El umbral de 1 N es del
  simulador, y en el robot hay que calibrarlo sobre la lectura en reposo.
- **Demostraciones humanas (fase C):** entran por la misma interfaz, (estado
  simbólico, acción).
- **Prueba en vivo con ROS:** llevar `ArmExecutor` y la política al nodo.
