# Bimanualidad segura en el H1-2: teoría completa

Este documento explica **por qué** el controlador bimanual está hecho así, con
la teoría de cada pieza y los papers de donde sale. Para **cómo correrlo**,
ver la sección 7 del `README.md` raíz del workspace.

Mapa de archivos (paquete `h1_2_algoritms`):

| Archivo | Qué contiene | Sección |
|---|---|---|
| `fk_functions.py` | Cinemática directa DH (ya existía) | §2 |
| `QP_functions.py` → `solve_qp_bimanual` | El QP de 14 variables | §3, §4 |
| `collision_model.py` | Cápsulas, distancias, jacobianos de punto, velocity dampers | §5, §6 |
| `depth_obstacles.py` | Cámara de profundidad → obstáculos, memoria de vóxeles | §7 |
| `bimanual_avoidance.py` | Capa 1: une todo lo anterior en un ciclo de control | §3–§7 |
| `bimanual_coordinator.py` | Capa 2: prioridad, retirada, inversión de roles | §9 |
| `bimanual_planner.py` | Capa 3: diagnóstico de objetivos, RRT-Connect, seguimiento | §10 |
| `QP_bimanual_avoidance.py` | Nodo ROS 2 sobre el bridge MuJoCo | — |
| `demos/evaluate_bimanual_avoidance.py` | Evaluación contra MuJoCo | §11 |

---

## 1. El problema y la arquitectura en tres capas

Dos brazos de 7 GDL comparten el espacio delante del torso. Queremos que:

1. **nunca choquen** entre sí, con el torso ni con el entorno (seguridad);
2. **lleguen** a sus objetivos (desempeño).

Son dos objetivos distintos, y la literatura los separa en capas:

```
Capa 3  Planificador global (bimanual_planner.py)
        busca un CAMINO libre (RRT-Connect) cuando no hay salida local
            │ puntos de paso
Capa 2  Coordinación (bimanual_coordinator.py)
        decide QUIÉN pasa primero cuando los brazos se bloquean
            │ objetivos (el que cede recibe una pose de retirada)
Capa 1  Control reactivo (bimanual_avoidance.py)
        QP a 20 Hz: sigue los objetivos SIN chocar nunca (garantía local)
            │ q̇ de 14 joints
        Bridge MuJoCo / robot real (PD + compensación de gravedad)
```

La idea clave: **la capa 1 garantiza seguridad pero no llegada**. Es un
controlador local (mira solo el instante actual), y los controladores locales
tienen *mínimos locales*: puntos donde la tarea empuja justo contra la
restricción y el robot se queda quieto (§8). Las capas 2 y 3 existen para
eso. Este esquema es el de GGDRC (*Biomimetics* 2024) y el de Ju et al.
(*Biomimetics* 2026), y NEO (Haviland & Corke, RA-L 2021) lo recomienda
explícitamente: "reactivo local + planificador global".

---

## 2. Cinemática: frames, DH y jacobianos

### 2.1 Un solo frame para todo: `torso_link`

La cinemática DH de cada brazo (`fkine_arm_left/right_unitree`) entrega la
pose de `*_wrist_yaw_link` expresada en `torso_link`. Se comprobó contra
MuJoCo que coinciden al 1e-4 m. La cámara RGB-D también va fija a
`torso_link`. Por eso **todo** (cápsulas, objetivos, nube de puntos) se
expresa en `torso_link`, y la cintura (`torso_joint`) no afecta a nada:
gira el torso con los brazos y la cámara juntos.

### 2.2 Denavit-Hartenberg

Cada articulación *i* aporta una transformación

```
T_i = Rot_z(θ_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(α_i)
```

y la pose del eslabón *k* es `T_base · T_1 · … · T_k`. En la convención DH
estándar, **la articulación i gira alrededor del eje z del frame i−1**.
`collision_model.arm_frames` devuelve los 8 frames intermedios (no solo el
final), porque para las colisiones interesa cualquier punto del brazo, no
solo la mano.

### 2.3 Jacobiano de un punto cualquiera

Para un punto **p** rígidamente unido al eslabón *k*, su velocidad es
`ṗ = J_p(q) q̇` con columnas

```
J_p[:, i] = z_{i-1} × (p − o_{i-1})      si i ≤ k
J_p[:, i] = 0                            si i > k
```

donde `z_{i-1}` y `o_{i-1}` son el eje z y el origen del frame i−1. Es el
jacobiano geométrico clásico (Siciliano et al., *Robotics: Modelling,
Planning and Control*, cap. 3), evaluado en un punto que no es el efector.
Las articulaciones posteriores a *k* no mueven el punto: por eso sus columnas
son cero. Implementado en `collision_model.point_jacobian` y verificado
contra diferencias finitas en `test/test_collision_model.py`.

---

## 3. Control cinemático como QP

### 3.1 La ley de control de la tarea

En cada ciclo se quiere una velocidad del efector que reduzca el error:

```
ẋ_des = [ K_p · (p_d − p) ;  K_o · e_o ]        e_o = vec(q_d ⊗ q*)
```

`e_o` es la parte vectorial del cuaternión de error (`orientation_error`).
Si el sistema siguiera exactamente `J q̇ = ẋ_des`, el error caería
exponencialmente con constante de tiempo `1/K_p`: es un *servo cinemático*
de primer orden.

### 3.2 Por qué un QP y no la pseudoinversa

La pseudoinversa (o DLS) resuelve `J q̇ = ẋ_des` en mínimos cuadrados, pero
no admite **desigualdades** (límites articulares, distancias mínimas). El QP
sí:

```
min_q̇   ½ ‖J q̇ − ẋ_des‖²_W  +  ½ w_c ‖J_c q̇ − ẏ_c‖²  +  ½ w_r ‖q̇‖²
s.a.    q̇_min ≤ q̇ ≤ q̇_max
        (q_min − q)/Δt ≤ q̇ ≤ (q_max − q)/Δt       ← límites de posición
        A_col q̇ ≥ b_col                            ← distancias (§6)
```

El segundo término es el objetivo del codo (`J_c`: fila Y del jacobiano del
codo) y el tercero regulariza para que el problema sea estrictamente convexo
y suave cerca de singularidades (el mismo papel que el amortiguamiento de la
DLS). OSQP resuelve la forma estándar `min ½ xᵀPx + qᵀx  s.a. l ≤ Ax ≤ u`.
Se obtiene expandiendo los cuadrados: `P = JᵀWJ + w_c J_cᵀJ_c + w_r I` y
`q = −JᵀW ẋ_des − w_c J_cᵀ ẏ_c`.

### 3.3 Prioridades: fuertes y blandas

Si dos objetivos compiten en un mismo costo, el QP **reparte** el error
entre ellos según los pesos. Eso pasaba con el codo: con `w_elbow = 3`
(el valor de `QP_whole_body.py`), al llevar la mano derecha al centro el
error quedaba en **38 mm**, sin nada cerca, porque el codo "tiraba" hacia
afuera.

Hay dos soluciones clásicas:

* **Jerarquía estricta** (HQP, Escande, Mansard & Wieber, IJRR 2014): se
  resuelve primero la tarea de mayor prioridad y la siguiente se optimiza
  *sin empeorar* la anterior (en su espacio nulo). Es exacta, pero hay que
  resolver varios QP encadenados.
* **Prioridad blanda**: pesos muy distintos. Con `w_elbow = 0.03` (1/300 del
  peso de posición) el error bajó a **0.7 mm**, y el codo sigue usando el
  único GDL redundante que la mano deja libre (7 articulaciones − 6 de
  tarea = 1).

Se eligió la blanda (parámetro `w_elbow`). Las **distancias** no van en el
costo sino como **restricciones duras**: así están siempre por encima de la
tarea, que es la jerarquía que usan Stasse et al. 2008 y Dietrich et al.
2012.

---

## 4. Por qué un solo QP de 14 variables

`QP_whole_body.py` resuelve un QP por brazo. La distancia entre un punto del
brazo izquierdo **a** y uno del derecho **b** cambia según

```
ḋ = nᵀ (ṗ_a − ṗ_b) = nᵀ J_a q̇_L − nᵀ J_b q̇_R ,     n = (p_a − p_b)/‖p_a − p_b‖
```

Depende de **ambas** velocidades a la vez. Con dos QP separados, cada brazo
tendría que adivinar qué hará el otro. Con `q̇ = [q̇_L ; q̇_R]` en un único
QP, esa fila es simplemente una restricción lineal más. El costo sigue
siendo diagonal por bloques (cada brazo su tarea); lo único que acopla los
brazos son las filas de colisión. Así lo hace también la IK oficial de
Unitree para el H1-2 (`xr_teleoperate`, `H1_2_ArmIK`): un solo problema para
los 14 GDL. **Pero no tiene restricciones de autocolisión**: solo costo de
pose, regularización y suavizado.

---

## 5. Geometría de colisión: cápsulas

### 5.1 Por qué cápsulas

Una **cápsula** es un segmento con radio: todos los puntos a distancia ≤ r
del segmento [a, b]. Es la primitiva estándar para evasión en tiempo real
(las usan el MIT Humanoid de Khazoom et al. 2022, y Quiroz-Omaña & Adorno
2019 en forma de "líneas de Plücker"), por tres razones:

1. **Distancia en forma cerrada**: entre dos cápsulas,
   `d = dist(segmento₁, segmento₂) − r₁ − r₂`.
2. **Gradiente continuo** (salvo cuando los segmentos se cortan), así que la
   restricción del QP no salta.
3. Se evalúa en microsegundos, contra milisegundos para mallas.

### 5.2 Distancia entre segmentos

Con `P(s) = p₁ + s·d₁` y `Q(t) = p₂ + t·d₂`, s y t en [0, 1], se minimiza
`‖P(s) − Q(t)‖²`. La solución sin restricciones sale de un sistema lineal de
2×2. Después se recorta s y t a [0, 1] y se recalcula el otro. Algoritmo de
Ericson, *Real-Time Collision Detection*, §5.1.9
(`closest_points_segments`). Devuelve los **puntos más cercanos** `p_a` y
`p_b`, que es lo que necesita el jacobiano.

### 5.3 Calibración y conservadurismo

Por brazo hay tres cápsulas (brazo superior, antebrazo, mano) y una para el
torso. Sus extremos están fijos en un frame DH, y el radio es la **distancia
máxima** de los vértices de la malla de colisión del MJCF al segmento. Por
construcción, la cápsula envuelve la malla. Verificación: en 400
configuraciones cercanas, la distancia entre cápsulas **nunca** superó la
real entre mallas. El margen fue de +3 a +113 mm (mediana 29 mm): el modelo
es conservador, y lo que cuesta es algo de espacio de trabajo.

### 5.4 Lo que las cápsulas no cubren: el hombro contra el torso

El brazo superior está montado en el torso. Su cápsula y la del torso ya
están casi en contacto en reposo, así que ese par no se vigila (siempre
estaría "activo"). En la evaluación apareció un choque que ninguna cápsula
veía: con el **roll del hombro girado hacia adentro**, la malla del hombro
entra en la del torso. Aparecía más desde que el codo dejó de empujar hacia
afuera (§3.3).

Medido sobre 300 configuraciones aleatorias por valor: roll hacia adentro
de 0.20 rad → **100 %** en contacto; 0.10 rad → 12–14 % (y ese resto es
antebrazo o mano contra el torso, que sí cubren las cápsulas). Como es un
choque *de la articulación misma*, se trata con un **límite articular**:
`shoulder_roll_inward_max = 0.10` rad (el MJCF permite 0.38).

Matiz: MuJoCo colisiona con la **envolvente convexa** de cada malla. La del
torso rellena la concavidad junto al hombro, así que en el robot real el
±0.38 del URDF podría ser válido. Por eso es un parámetro: conviene
verificarlo con cuidado en el H1-2 físico antes de relajarlo.

---

## 6. La restricción de distancia: velocity damper y CBF

### 6.1 Velocity damper (Faverjon & Tournassoud 1987)

Para cada par con distancia `d < d_i` (distancia de influencia) se impone

```
ḋ ≥ −ξ · (d − d_s) / (d_i − d_s)
```

* Si `d ≥ d_i`: la restricción no se agrega (no molesta lejos).
* Si `d_s < d < d_i`: el brazo puede acercarse, pero cada vez más despacio.
* En `d = d_s`: `ḋ ≥ 0`, ya no puede acercarse.
* Si `d < d_s` (por error numérico o latencia): el lado derecho es
  positivo y **obliga a alejarse**.

Sustituyendo `ḋ` (§4) queda **lineal en q̇**:
`nᵀJ_a q̇_L − nᵀJ_b q̇_R ≥ −ξ(d − d_s)/(d_i − d_s)`, que es exactamente una
fila de `A_col q̇ ≥ b_col`.

**Por qué garantiza d ≥ d_s**: en el peor caso (igualdad), con `h = d − d_s`
queda `ḣ = −(ξ/(d_i − d_s)) h`, una ecuación diferencial cuya solución
`h(t) = h₀ e^{−γt}` decae hacia 0 sin cruzarlo nunca. Todo lo que sea `ḣ`
mayor aleja todavía más.

### 6.2 Es lo mismo que una Control Barrier Function

Una CBF (Ames et al.) exige `ḣ ≥ −γ h` para una función `h(q) ≥ 0` que
define la zona segura. Con `h = d − d_s` y `γ = ξ/(d_i − d_s)`, **el
velocity damper es una CBF lineal** en cinemática. Khazoom et al. (2022)
usan CBF porque trabajan con la **dinámica** completa del MIT Humanoid (el
QP decide pares, no velocidades). Ahí la CBF aporta algo que el damper
cinemático no da. Con control por velocidad/posición, como aquí, son
equivalentes.

### 6.3 Parámetros usados

| | Autocolisión (cinemática) | Entorno (cámara) |
|---|---|---|
| `d_s` (seguridad) | 30 mm | 50 mm |
| `d_i` (influencia) | 150 mm | 200 mm |
| `ξ` (velocidad máx. de acercamiento en `d_i`) | 0.4 m/s | 0.4 m/s |

El entorno lleva más margen porque la nube tiene ruido de voxelizado
(3 cm) y latencia de cámara (~7 Hz).

### 6.4 Tiempo discreto

La garantía es en tiempo continuo. El QP corre a 20 Hz y mantiene `q̇`
constante 50 ms, así que en un paso la distancia puede bajar un poco más de
lo previsto (error de linealización). `d_s` absorbe ese error: en todas las
pruebas la distancia real (mallas de MuJoCo) nunca bajó de 0.

---

## 7. Percepción: la cámara de profundidad

### 7.1 Qué aporta la cámara (y qué no)

* **Brazo contra brazo:** no se usa la cámara. La cinemática es exacta y no
  tiene latencia, mientras que la cámara ve mal los brazos (se tapan entre
  sí, y la parte de atrás no se ve).
* **Entorno** (mesa, faja, objetos, personas): solo la cámara lo sabe.
  Coincide con ARMOR (percepción egocéntrica en humanoides) y cuRoboV2
  (campos de distancia fusionados desde profundidad).

### 7.2 Pipeline

1. **Deproyección pinhole**: un píxel (u, v) con profundidad z da, en el
   frame óptico (X derecha, Y abajo, Z adelante), el punto
   `x = (u − c_x)·z/f_x`, `y = (v − c_y)·z/f_y`. Los intrínsecos vienen de
   `camera_info` (o del `fovy` del MJCF).
2. **Cambio de frame**: `p_torso = R·p_opt + t`, con `(R, t)` la pose fija de
   la cámara en el torso (TF `torso_link → camera_depth_optical_frame`).
3. **Recorte** a la caja de trabajo de los brazos.
4. **Auto-filtrado**: la cámara VE los propios brazos. Si no se quitan, cada
   brazo se toma como obstáculo de sí mismo y queda congelado. Se descarta
   todo punto a menos de `r + padding` de las cápsulas del robot, usando la q
   del momento de la imagen. El `padding` cubre el error del modelo y el
   desfase temporal.
5. **Voxelizado** (3 cm): un punto por celda, de ~19 000 puntos a ~800.
6. Cada cápsula toma su **punto más cercano** y se agrega un damper
   cápsula-punto (§6), con un solo brazo en la fila porque el obstáculo no se
   mueve con q.

### 7.3 Oclusión y memoria con borrado por espacio libre

La cámara está detrás y por encima de las manos. Cuando una mano baja hacia
la faja, **tapa justo la zona que tiene debajo**. Con solo el último cuadro,
el punto más cercano que queda visible está unos 5 cm al costado y la mano
sigue bajando. Medido: 94 pasos con contacto en el escenario `table`.

Solución (`ObstacleMemory`): un vóxel visto se **recuerda**. Solo se borra
cuando hay evidencia de espacio libre: se proyecta a la imagen nueva y el
píxel mide una profundidad **mayor** que la del vóxel, o sea que la cámara ve
*a través* de él. Si mide menos, algo lo tapa (el brazo) y se conserva. Es
el principio del *ray casting* de OctoMap (Hornung et al., 2013), reducido a
una proyección por vóxel. Resultado: `table` pasó a **0 contactos**, con la
mano detenida a 47 mm de la faja.

---

## 8. Mínimos locales: por qué la capa 1 no alcanza

### 8.1 El mecanismo

El damper solo quita la componente de velocidad que **acerca**. Si el
objetivo queda "detrás" del otro brazo, la tarea empuja justo en la
dirección prohibida. El QP encuentra entonces que lo mejor es quedarse casi
quieto: la fuerza de la tarea y la restricción se equilibran. Es un **mínimo
local** del problema, igual que en los campos potenciales (Khatib 1986).
Todos los papers de control reactivo lo reconocen (NEO; Ju et al. 2026:
*"our reactive control approach is local in nature and does not guarantee
global optimality"*).

### 8.2 Primero: ¿el objetivo es alcanzable?

Antes de culpar al controlador hay que distinguir dos casos:

* **Objetivo incompatible**: la configuración final misma choca. Con IK de
  cada brazo a su objetivo, en los escenarios `converge` y `swap` los
  antebrazos quedarían unos **12 cm uno dentro del otro**. Ningún
  algoritmo puede llegar, y detenerse a `d_s` es **lo correcto**.
* **Objetivo compatible con camino bloqueado**: el final es posible, pero
  ir directo choca. Aquí sí hay algo que mejorar.

Hay un caso intermedio muy común en manipulación: **objetivos compatibles
en secuencia pero no a la vez**. Ejemplo: los dos brazos quieren tomar algo
del mismo punto de la faja. Es el escenario `shared`, y es el problema
que resuelve la capa 2.

(También se midió que, con la mano horizontal, la muñeca no alcanza
z < ~0.08 m en `torso_link`, por el límite de pitch ±0.46 rad. Parte del
"error final" de los escenarios viejos con z = 0.05 era de alcance, no de
colisión.)

---

## 9. Capa 2: coordinación por prioridad

### 9.1 Idea

Cuando los dos se bloquean, **uno pasa y el otro cede**. Es la estrategia
maestro/esclavo de Ju et al. (2026) y la de "retrasar un robot" de la
*path-velocity decomposition* (Shin & Zheng). La capa 1 sigue activa todo
el tiempo: la capa 2 solo cambia **objetivos**, nunca desactiva la seguridad.

### 9.2 Máquina de estados (`bimanual_coordinator.py`)

```
LIBRE ──(bloqueo)──► CEDE[esclavo]
  ▲                     │ maestro llegó / se alejó
  └──(reanuda)──────────┘
                        │ el maestro TAMBIÉN se bloquea (tras dejar retirarse al esclavo)
                        └─► se invierten los roles (máx. max_switches)
                            └─► ESCALAR: hace falta la capa 3
```

* **Detectar el bloqueo** requiere tres condiciones a la vez:
  1. hay una restricción brazo-brazo activa (`d < d_i`);
  2. ningún brazo avanzó más de 1 cm hacia **su objetivo de tarea** en 1 s;
  3. al menos uno no llegó.

  Si no avanzan pero no hay restricción activa, el problema es de alcance y
  la capa 2 no interviene.
* **Quién pasa**: el que tiene **menos camino por recorrer** ("shortest job
  first": libera antes el espacio compartido). Si empatan, el brazo
  `default_master`. Si solo uno está pendiente, el que ya llegó cede.
* **Qué hace el que cede**: no se queda quieto sino que va a una **pose de
  retirada** de su lado (`DEFAULT_RETREAT`). El bloqueo es mutuo: si el
  esclavo solo se detuviera, su cuerpo seguiría tapando el camino.
* **Cuándo reanuda** el esclavo, en dos casos:
  1. la tarea mandó al maestro a otro sitio y ya se alejó (`d > d_libre`);
  2. el maestro llegó y se queda, pero **no ocupa el objetivo del esclavo**
     (su mano y sus cápsulas quedan a más de `d_i` de ese punto).

  El caso 2 importa en la zona compartida. Si el esclavo volviera apenas el
  maestro llega, se encontraría al maestro todavía "agarrando" en el punto
  común y los dos se turnarían en falso (lo detectó el test de regresión).
* **Qué distancia mira**: solo la **brazo-brazo** (`diag["d_arms_min"]`).
  Los pares brazo-torso o brazo-entorno no son un conflicto *entre brazos*,
  y dar prioridad no los resuelve. En una versión anterior se usaba la
  mínima con torso incluido, y la capa 2 se disparaba en `table` con los
  brazos a 24 cm uno del otro.
* **Inversión de roles**: si el maestro tampoco avanza **después** de que
  el esclavo tuvo tiempo de retirarse (2 ventanas), se prueba el orden
  inverso.
* **Escalar**: si ninguna prioridad funciona, o los brazos se turnan sin
  terminar (objetivos incompatibles, §8.2), se declara que hace falta la
  capa 3. Mientras tanto la capa 1 sigue manteniendo la seguridad.
* **Detalles que evitan falsos positivos** (encontrados en las pruebas):
  el historial de progreso se reinicia cuando la tarea cambia de objetivo,
  porque el error salta. Y al maestro no se le juzga hasta que el esclavo
  terminó de retirarse.

### 9.3 Capa de tarea (`TaskSequence`)

Cada brazo ejecuta una lista de (pose, espera). Por ejemplo, `shared`:
ir al punto común, quedarse 1 s ("agarrar") y volver a casa. La secuencia
avanza con el error a **su** objetivo, así que mientras el brazo cede la
tarea espera y no se salta pasos.

---

## 10. Capa 3: planificador global (`bimanual_planner.py`)

### 10.1 Cuándo hace falta

Hay dos situaciones en que las capas 1 y 2 no alcanzan:

* **Obstáculo ancho o plano delante del objetivo.** Contra un obstáculo
  pequeño y convexo (una esfera, un poste delgado) la capa 1 **ya llega**:
  el damper solo quita la componente que acerca, y la mano "resbala" por el
  costado. Se verificó en una búsqueda sistemática, donde la capa 1 llegó en
  todos los casos con esfera. Pero contra una **bandeja** horizontal justo
  encima de la mano, la dirección hacia el objetivo apunta de frente a la
  placa: no hay componente tangencial que aproveche, y la mano se queda
  debajo con ~160 mm de error. Dar prioridad a un brazo no cambia nada,
  porque el problema es con el entorno.
* **La capa 2 escaló**: ninguna prioridad entre brazos resolvió el bloqueo.

### 10.2 Qué hace

1. **Detecta** que un brazo no avanza (menos de 1 cm en 1.5 s y todavía
   lejos del objetivo). Solo lo hace si la capa 2 está libre o ya escaló;
   mientras la capa 2 arbitra, un brazo que espera en retirada "no avanza"
   a propósito.
2. **Diagnostica** antes de buscar camino:
   * ¿Existe una **postura final válida** para ese brazo? Se prueba la IK
     desde varias semillas y se descartan las posturas con colisión. Si
     ninguna sirve, el objetivo está dentro de un obstáculo o fuera de
     alcance (escenario `table`): se informa y no se intenta.
   * Si los **dos** brazos están trabados: ¿existe un **par** de posturas
     finales que no choquen entre sí? Si no, los objetivos son
     incompatibles (escenarios `swap`, `converge`), y se informa.
3. **Planifica** con **RRT-Connect** [Kuffner & LaValle 2000] en los 7 GDL
   del brazo trabado. El otro brazo queda **congelado** y cuenta como
   obstáculo: es la *planificación priorizada* de Erdmann & Lozano-Pérez
   (1987), la misma idea de "uno pasa, el otro espera" de la capa 2.
4. **Suaviza**: atajos aleatorios (si dos puntos del camino se ven en línea
   recta libre, se borra lo del medio) y remuestreo a pasos de ≤ 0.2 rad.
5. **Ejecuta** punto a punto con la capa 1, cambiando la tarea de la mano
   por una tarea articular (§10.4). Al llegar al último punto, vuelve el
   control normal por la pose de la mano.

### 10.3 RRT-Connect en una página

Se hacen crecer dos árboles de posturas válidas, uno desde el inicio y otro
desde el objetivo. En cada iteración:

* **extend**: el árbol A toma una postura aleatoria `q_rand`, busca su
  nodo más cercano y avanza un paso (0.15 rad) hacia ella si el tramo es
  libre;
* **connect**: el árbol B intenta llegar a ese nodo nuevo en línea recta,
  tantos pasos como pueda;
* si B lo alcanza, los árboles se tocaron y hay camino. Si no, A y B
  intercambian papeles.

Un **tramo** es libre si todas las posturas intermedias (cada 0.04 rad en la
articulación que más se mueve) son válidas. El algoritmo es
*probabilísticamente completo*: si existe camino, la probabilidad de
encontrarlo tiende a 1 con el tiempo. No es óptimo; por eso después se
suaviza.

### 10.4 Márgenes y seguimiento

* **Márgenes del planificador mayores que los de la capa 1**: 40 mm entre
  brazos (contra `d_s` = 30) y 60 mm con el entorno (contra 50). Así el
  damper no frena el camino planificado. Si el brazo ya arranca más cerca
  que eso (la capa 1 lo deja llegar hasta `d_s`), los márgenes se relajan
  **lo justo** para que el punto de partida cuente como válido.
* **Por qué seguir en espacio articular**: la misma posición de la mano
  admite muchas posturas del brazo (7 GDL). Si solo se le pasaran a la
  capa 1 posiciones de la mano, podría elegir un codo que vuelva a chocar.
  Se agrega al QP la tarea `½ w_joint ‖q̇ − K_q (q_wp − q)‖²` (dominante) y
  la de la mano se atenúa a 1 %. Las **restricciones de distancia no
  cambian**: aunque el camino tuviera un error, la capa 1 no deja chocar.
* **Asíncrono en el nodo ROS**: la planificación (~1–3 s) corre en un hilo.
  El lazo de 20 Hz sigue mandando a los dos brazos quedarse quietos, así que
  el robot nunca queda sin control.

### 10.5 Límites

* Planifica en **posturas**, no en tiempo: el otro brazo debe estar quieto.
  Si los dos tienen que moverse a la vez por un pasillo estrecho, haría
  falta planificar en 14 GDL o en espacio-tiempo [Erdmann & Lozano-Pérez].
* La nube de la cámara es la del momento de planificar. Un obstáculo que
  aparece después lo sigue cuidando la capa 1 (no chocar), pero el camino
  no se replanifica solo, salvo que el brazo se vuelva a trabar.
* El verificador usa las cápsulas: hereda su conservadurismo (§5.3).

## 11. Resultados y cómo reproducirlos

```bash
cd ~/humanoid_ws/src/h1_2_algoritms
python3 demos/evaluate_bimanual_avoidance.py --seconds 12
python3 -m pytest test/test_collision_model.py
```

La verdad de terreno **no** sale de las cápsulas del controlador (sería
circular) sino de MuJoCo: contactos con penetración entre las mallas
reales, distancia entre vértices de las mallas y alturas analíticas. Tabla
de resultados: ver `README.md` raíz, sección 7, y los CSV en
`h1_2_algoritms/resultados_bimanual/`.

---

## 12. Glosario

| Término | Significado |
|---|---|
| GDL | Grados de libertad (7 por brazo, 14 en total) |
| QP | Programa cuadrático: costo cuadrático + restricciones lineales |
| OSQP | Solver de QP usado (ADMM, rápido para problemas pequeños) |
| DLS | *Damped least squares*, pseudoinversa amortiguada |
| Velocity damper | Restricción `ḋ ≥ −ξ(d − d_s)/(d_i − d_s)` |
| CBF | *Control barrier function*: `ḣ ≥ −γh` mantiene `h ≥ 0` |
| `d_s`, `d_i`, `ξ` | Distancia de seguridad, de influencia, velocidad de acercamiento |
| Mínimo local | Configuración donde el controlador reactivo se detiene sin llegar |
| Auto-filtrado | Quitar de la nube de la cámara los puntos del propio robot |
| Carving | Borrar un vóxel cuando la cámara ve a través de él |
| Maestro / esclavo | El brazo que pasa / el que cede y se retira |
| RRT-Connect | Planificador por muestreo: dos árboles de posturas que crecen hasta tocarse |
| Planificación priorizada | Planificar un robot por vez, con los demás como obstáculos |
| Postura objetivo válida | Solución de IK de la pose objetivo que además no choca |

## 13. Referencias

Todas las referencias, en formato IEEE, con volumen, páginas y DOI
verificados, y lo que se tomó de cada una, están en
[`REFERENCIAS_BIMANUAL.md`](REFERENCIAS_BIMANUAL.md). Esa es la lista que
hay que usar en la tesis; aquí se citan por autor y año para no cortar la
lectura.

Para la **teoría de base** que conviene estudiar antes de leer este
documento (álgebra lineal, rotaciones, cinemática diferencial,
optimización convexa, cámara pinhole, planificación), en orden y con
ejercicios, ver [`GUIA_ESTUDIO_BIMANUAL.md`](GUIA_ESTUDIO_BIMANUAL.md).
