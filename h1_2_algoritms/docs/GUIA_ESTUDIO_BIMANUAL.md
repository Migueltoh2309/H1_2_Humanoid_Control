# Guía de estudio: la teoría para entender el control bimanual al pie de la letra

Esta guía dice **qué estudiar, en qué orden, en qué capítulo exacto y para
qué sirve en nuestro código**. Es el paso previo a
[`BIMANUAL_TEORIA.md`](BIMANUAL_TEORIA.md), que explica *nuestro* sistema y
da por sabida esta base. Las referencias de investigación (papers) están en
[`REFERENCIAS_BIMANUAL.md`](REFERENCIAS_BIMANUAL.md) y aquí se citan como
**[n]**. Los libros de estudio van al final, como **[L1]…[L11]**.

## Cómo usar esta guía

Hay 12 módulos en orden de dependencia: cada uno usa lo anterior. Cada
módulo trae:

* **Qué necesitas saber**: la lista concreta de ideas.
* **Dónde leerlo**: libro y capítulo o sección. ★ = imprescindible,
  ☆ = para profundizar.
* **Dónde aparece en el código**: archivo y función.
* **Compruébalo tú**: un ejercicio corto, casi siempre con el código real.
  Si te sale, entendiste el módulo.

```
M1 Álgebra lineal ─┬─► M2 Rotaciones ─► M3 DH ─► M4 Jacobianos ─► M5 Redundancia ─┐
                   └─► M6 Optimización (QP) ◄──────────────────────────────────────┘
M7 EDO/estabilidad ─► M8 Barreras (damper/CBF) ◄─ M4 + M6
M9 Geometría de colisión ─► M8
M10 Cámara y nubes de puntos (independiente de M5–M9)
M11 Mínimos locales y planificación ─► M12 Coordinación de dos brazos
```

Si ya dominas un módulo, haz directamente su "Compruébalo tú"; si sale,
sigue al siguiente. Estimación total: 6–8 semanas a ritmo de tesis.

Antes de los ejercicios, activa el entorno (ver `RECONSTRUIR.md`):

```bash
source ~/venvs/h12/bin/activate && cd ~/humanoid_ws/src/h1_2_algoritms
```

---

## M1. Álgebra lineal para robótica

**Qué necesitas saber**
- Producto punto y producto cruz; la matriz antisimétrica `[a]×` tal que
  `[a]× b = a × b`.
- Rango, espacio columna y **espacio nulo**. Una matriz de 6×7 de rango 6
  tiene un espacio nulo de dimensión 1: ese es el GDL "sobrante" del brazo.
- **Mínimos cuadrados**: `min ‖Ax − b‖²` y sus ecuaciones normales
  `AᵀA x = Aᵀb`.
- **SVD** y **pseudoinversa** `A⁺`. Por qué se vuelve inestable cerca de un
  valor singular pequeño, y cómo lo arregla la **DLS**
  `Aᵀ(AAᵀ + λ²I)⁻¹`.
- Norma ponderada `‖x‖²_W = xᵀWx`.

**Dónde leerlo**
- ★ [L1] Strang, cap. 4 "Orthogonality" (sobre todo mínimos cuadrados) y
  cap. 7 "The Singular Value Decomposition (SVD)".
- ☆ [L2] Lynch & Park, cap. 6, sección de cinemática inversa numérica
  (Newton-Raphson y pseudoinversa aplicados a un brazo).

**Dónde aparece en el código**
- `ik_functions.py`: `ik_pseudo_step`, `ik_dls_step`,
  `ik_dls_step_limited` (proyector del espacio nulo `I − J⁺J`).
- `QP_functions.py`: el costo `‖J q̇ − ẋ‖²_W` es un problema de mínimos
  cuadrados ponderado.

**Compruébalo tú**
1. Para `J` de 6×7 aleatoria, verifica con numpy que `N = I − J⁺J` cumple
   `J N ≈ 0` y `N N ≈ N`. ¿Por qué eso significa que un movimiento `N v` no
   mueve la mano?
2. Expande `½‖Jx − b‖²_W` y obtén `½ xᵀ(JᵀWJ)x − (JᵀWb)ᵀx + cte`. Es
   exactamente lo que arma `solve_qp_bimanual`: compara con el código.

---

## M2. Rotaciones, matrices de rotación y cuaterniones

**Qué necesitas saber**
- El grupo SO(3): `RᵀR = I`, `det R = 1`. Rotación de un vector frente a
  cambio de frame.
- Eje-ángulo y la fórmula de Rodrigues.
- **Cuaternión unitario** `q = (w, x, y, z)`, producto `⊗`, conjugado
  `q*`, y la ambigüedad `q ≡ −q`: por eso el código invierte el signo si
  `q_d · q < 0`.
- **Error de orientación**: `e_o = vec(q_d ⊗ q*)`. Para ángulos pequeños
  vale ≈ `(θ/2)·eje`, y por eso sirve como error proporcional.

**Dónde leerlo**
- ★ [L2] Lynch & Park, cap. 3 "Rigid-Body Motions" (rotaciones,
  exponencial, eje-ángulo).
- ★ [L3] Solà, secciones 1–2: definición, producto y rotación con
  cuaterniones, con convención Hamilton `(w, x, y, z)`, la misma que
  usa nuestro código.

**Dónde aparece en el código**
- `fk_functions.rot2quat`, `TF2xyzquat`.
- `ik_functions.quat_multiply`, `quat_conjugate`, `orientation_error`.
- `bimanual_avoidance.step`: normalización y cambio de signo del cuaternión.

**Compruébalo tú**
1. Con `q_d` = rotación de 10° en z y `q` = identidad, calcula
   `orientation_error(q_d, q)` y comprueba que da ≈ `(0, 0, sin(5°))`.
2. ¿Qué pasaría en `bimanual_avoidance.step` sin el `if np.dot(...) < 0`?
   Pista: el error diría "gira 350°" en vez de "gira −10°".

---

## M3. Transformaciones homogéneas y Denavit-Hartenberg

**Qué necesitas saber**
- Matriz homogénea 4×4, composición y la inversa `T⁻¹`.
- Cambio de frame de un punto: `p_A = T_AB p_B`.
- Convención **DH estándar**: `T_i = Rot_z(θ)·Trans_z(d)·Trans_x(a)·Rot_x(α)`.
  La articulación *i* gira sobre el **eje z del frame i−1**, que es lo que
  usa el jacobiano en M4.
- Diferencia entre DH estándar y DH modificado (Craig): no hay que
  mezclarlos.

**Dónde leerlo**
- ★ [25] Siciliano et al., cap. 2 "Kinematics": secciones sobre
  transformaciones homogéneas y la convención de Denavit-Hartenberg.
- ☆ [L2] Lynch & Park, cap. 4 "Forward Kinematics" y apéndice C
  "Denavit–Hartenberg Parameters".

**Dónde aparece en el código**
- `fk_functions.dh`, `fkine_arm_left_unitree`, `fkine_arm_right_unitree`.
- `collision_model.arm_frames` (los 8 frames intermedios).
- `depth_obstacles.transform_points` (cámara → `torso_link`).

**Compruébalo tú**
1. Corre `python3 -m pytest test/test_collision_model.py -k frame7`: el
   frame 7 de `arm_frames` debe ser idéntico a `fkine_arm_*`. Lee el test y
   explica qué asegura.
2. A mano: con q = 0, ¿cuál es la posición del frame 3 del brazo izquierdo?
   Compara con `arm_frames("left", np.zeros(7))[3]`.

---

## M4. Cinemática diferencial: el jacobiano

**Qué necesitas saber**
- `ẋ = J(q) q̇`. Jacobiano **geométrico** (velocidad angular ω) frente a
  **analítico** (derivada de una parametrización de la orientación).
- Columna de una articulación de revolución, para un **punto p** del
  eslabón k: `J_i = z_{i-1} × (p − o_{i-1})` para `i ≤ k`, y 0 si `i > k`.
- **Singularidades**: dónde `J` pierde rango. Manipulabilidad
  `√det(JJᵀ)`.
- Jacobiano numérico (diferencias finitas) frente al analítico: costo y
  precisión.

**Dónde leerlo**
- ★ [25] Siciliano et al., cap. 3 "Differential Kinematics and Statics":
  secciones "Geometric Jacobian", "Jacobian Computation" y "Kinematic
  Singularities".
- ☆ [L2] Lynch & Park, cap. 5 "Velocity Kinematics and Statics" (misma
  teoría con tornillos; útil como segunda mirada).

**Dónde aparece en el código**
- `collision_model.point_jacobian` (analítico, para cualquier punto del
  brazo).
- `ik_functions.numerical_jacobian` y
  `null_control_functions.numerical_jacobian_position` (numéricos, para la
  mano y el codo).

**Compruébalo tú**
1. `python3 -m pytest test/test_collision_model.py -k jacobian`: compara el
   analítico contra diferencias finitas. Cambia en el código `i ≤ k` por
   `i ≤ 7` y observa cómo falla el test. ¿Por qué?
2. Demuestra que `ḋ = nᵀ(ṗ_a − ṗ_b)` cuando `d = ‖p_a − p_b‖`. Es la fila
   que lleva cada par al QP. El test `test_self_collision_row_is_distance_derivative`
   lo verifica numéricamente.

---

## M5. Redundancia y prioridad de tareas

**Qué necesitas saber**
- Un brazo de 7 GDL con tarea 6D tiene **1 GDL redundante**: hay infinitas
  `q̇` que producen el mismo `ẋ`.
- Solución general `q̇ = J⁺ẋ + (I − J⁺J) v`: el segundo término no mueve la
  mano.
- **Prioridad estricta**: la tarea 2 se optimiza sin tocar la 1. Puede ser
  por proyección en el espacio nulo o por QP en cascada (HQP).
- **Prioridad blanda**: todo en un costo, con pesos muy distintos. Es
  aproximada, pero cabe en un solo QP.

**Dónde leerlo**
- ★ [25] Siciliano et al., cap. 3, secciones "Analysis of Redundancy" e
  "Inverse Differential Kinematics".
- ★ [10] Escande, Mansard & Wieber: introducción y secciones de
  formulación del problema jerárquico. No hace falta el algoritmo interno.
- ☆ [11] Dietrich et al.: cómo mezclar la evasión con la jerarquía de
  tareas.

**Dónde aparece en el código**
- `bimanual_avoidance.py`: `w_elbow = 0.03` como prioridad blanda. Lee el
  comentario con la medición de 38 mm → 0.7 mm.
- `ik_functions.ik_dls_step_limited`: proyección en el espacio nulo para
  alejarse de los límites articulares.

**Compruébalo tú**
1. En `BimanualAvoidanceController`, pon `w_elbow=3.0` y lleva la mano
   derecha a `(0.34, 0, 0.15)` con el izquierdo en retirada (ver la sección
   de ajustes en `README.md`). Reproduce el error de ~38 mm. Luego baja el
   peso y mira cómo cae.
2. Explica con tus palabras por qué las distancias van como
   **restricciones** y no como términos del costo.

---

## M6. Optimización convexa y programación cuadrática

**Qué necesitas saber**
- Problema convexo, QP: `min ½xᵀPx + qᵀx  s.a.  l ≤ Ax ≤ u`.
- Por qué `P` debe ser semidefinida positiva, y por qué la regularización
  `w_reg·I` la hace **definida** positiva (solución única).
- **Condiciones KKT** y multiplicadores. Restricciones activas e
  inactivas: una restricción de distancia está "activa" cuando el brazo
  está justo en el límite del damper.
- **Factibilidad**: qué pasa si ningún `x` cumple todas las restricciones,
  y por qué el código devuelve `q̇ = 0` en ese caso (§3 de la teoría).
- Idea de ADMM (el método de OSQP) a nivel intuitivo, y por qué el
  *warm start* ayuda.

**Dónde leerlo**
- ★ [L4] Boyd & Vandenberghe: §4.4 "Quadratic optimization problems" y
  §5.5.3 "KKT optimality conditions". Es gratis en la web de los autores.
- ★ [22] Stellato et al. (OSQP): secciones 1–3 (formulación y algoritmo).
- ☆ [L5] Nocedal & Wright, cap. 16 "Quadratic Programming".

**Dónde aparece en el código**
- `QP_functions.solve_qp_arm` (un brazo) y `solve_qp_bimanual` (14 GDL,
  con `A_col`).
- `QP_functions.compute_velocity_bounds`: cómo los límites de posición se
  convierten en cotas de velocidad.

**Compruébalo tú**
1. Deriva a mano `P` y `q` de `solve_qp_bimanual` a partir del costo de la
   §3.2 de la teoría. ¿Por qué el código pasa `np.triu(P)` a OSQP?
2. Deduce `(q_min − q)/Δt ≤ q̇ ≤ (q_max − q)/Δt` a partir de
   `q_min ≤ q + q̇Δt ≤ q_max`.
3. Construye un QP infactible (dos restricciones contradictorias) y mira
   qué `status` devuelve OSQP.

---

## M7. Ecuaciones diferenciales y estabilidad (lo mínimo)

**Qué necesitas saber**
- `ẋ = −kx` tiene solución `x(t) = x₀e^{−kt}`: decae sin cruzar el cero.
  Es la base del servo cinemático y del damper.
- **Lema de comparación**: si `ẋ ≥ −kx` y `x(0) ≥ 0`, entonces `x(t) ≥ 0`
  siempre.
- Idea de estabilidad de Lyapunov: una "energía" que no crece.
- Tiempo continuo frente a discreto: con `Δt` finito, `x_{k+1} = x_k −
  kΔt·x_k` es estable solo si `kΔt < 2`, y sin oscilar solo si `kΔt < 1`.

**Dónde leerlo**
- ★ [L6] Slotine & Li, cap. 3 "Fundamentals of Lyapunov Theory" (las
  primeras secciones bastan).
- ★ [6] Ames et al.: la parte donde `ḣ ≥ −α(h)` implica invariancia del
  conjunto seguro.

**Dónde aparece en el código**
- `bimanual_avoidance.step`: `ẋ_des = K_p·e`, con `K_p = 2.5` y
  `Δt = 0.05`, así que `K_p·Δt = 0.125` (bien lejos de 1).
- El damper (M8) es exactamente `ḣ ≥ −γh`.

**Compruébalo tú**
1. ¿Con qué `K_p` el servo cinemático empezaría a oscilar a 20 Hz?
2. Simula en papel dos pasos de `h_{k+1} = h_k − γΔt·h_k` con
   `γ = ξ/(d_i − d_s) = 0.4/0.12 ≈ 3.3 1/s` y `Δt = 0.05`. ¿Cuánto baja `h`
   por paso?

---

## M8. Restricciones de seguridad: velocity damper y CBF

**Qué necesitas saber**
- El damper `ḋ ≥ −ξ(d − d_s)/(d_i − d_s)` y el papel de cada parámetro
  (`d_s`, `d_i`, `ξ`).
- La demostración de que mantiene `d ≥ d_s` (M7 + lema de comparación).
- **CBF**: con `h = d − d_s` y `γ = ξ/(d_i − d_s)`, el damper es una CBF
  lineal. Cuándo una CBF aporta algo más (control por par, con dinámica).
- Por qué las restricciones se agregan solo cuando `d < d_i` (tamaño del
  QP, y no molestar lejos).
- Qué pasa si hay muchas restricciones incompatibles (infactibilidad):
  relajarlas con criterio [8].

**Dónde leerlo**
- ★ [1] Faverjon & Tournassoud: la formulación original.
- ★ [2] Stasse et al.: el damper en el QP de un humanoide real.
- ★ [6] Ames et al.: secciones de definición y teorema de CBF.
- ☆ [7] Khazoom et al.: CBF de autocolisión con la dinámica completa.
- ☆ [4] Haviland & Corke: damper + variables de holgura.

**Dónde aparece en el código**
- `collision_model._damper_rhs`, `self_collision_constraints`,
  `obstacle_constraints`.
- `bimanual_avoidance.py`: los parámetros `d_s`, `d_i`, `ξ` y su versión
  para el entorno.

**Compruébalo tú**
1. Deriva `h(t) = h₀e^{−γt}` y demuestra `γ = ξ/(d_i − d_s)`.
2. Corre `python3 demos/evaluate_bimanual_avoidance.py --scenarios converge --modes off self --seconds 6`
   y explica la diferencia entre `off` (230 choques) y `self` (0 choques,
   ~55 mm de distancia mínima). ¿Por qué 55 mm y no 30 mm (= `d_s`)?
   Pista: cápsulas conservadoras, §5.3 de la teoría.

---

## M9. Geometría de colisión

**Qué necesitas saber**
- Volúmenes envolventes: esfera, AABB, OBB, **cápsula** (volumen barrido
  por una esfera). Por qué la cápsula es buena para eslabones alargados.
- Punto más cercano entre punto y segmento, y entre dos segmentos (casos
  degenerados: segmentos paralelos o de longitud cero).
- **Conservadurismo**: una envolvente que contiene la malla nunca
  subestima la distancia. Costo: espacio de trabajo perdido.
- Mallas en MuJoCo: la colisión usa la **envolvente convexa** de cada malla.
  Es la causa del hallazgo del hombro (§5.4 de la teoría).

**Dónde leerlo**
- ★ [24] Ericson: cap. 4 "Bounding Volumes" (§4.5 "Sphere-swept Volumes")
  y cap. 5 "Basic Primitive Tests", §5.1 (puntos más cercanos; §5.1.9,
  dos segmentos).
- ★ [L7] Documentación de MuJoCo, capítulo "Computation", parte de
  colisiones (mallas y envolvente convexa).
- ☆ [5] Quiroz-Omaña & Adorno: distancias entre primitivas en un QP.

**Dónde aparece en el código**
- `collision_model.closest_points_segments`, `points_to_segment`,
  `CAPSULES`, `TORSO_CAPSULE`, `robot_self_filter`.
- `bimanual_avoidance.py`: `shoulder_roll_inward_max` (lo que las
  cápsulas no ven).

**Compruébalo tú**
1. `python3 -m pytest test/test_collision_model.py -k segments`: compara
   contra fuerza bruta. Lee el algoritmo línea por línea junto a Ericson
   §5.1.9.
2. Explica por qué el radio de cada cápsula se toma como la distancia
   **máxima** de los vértices, y no la media.

---

## M10. Cámara de profundidad, frames y nubes de puntos

**Qué necesitas saber**
- Modelo **pinhole**: intrínsecos `f_x, f_y, c_x, c_y`; proyección
  `u = f_x·x/z + c_x`; deproyección con la profundidad medida.
- Profundidad "z" (a lo largo del eje óptico) frente a distancia euclidiana
  al punto. MuJoCo y nuestro código usan z.
- Convenciones ROS: **REP-103** (X adelante, Y izquierda, Z arriba para
  cuerpos; frames `_optical` con X derecha, Y abajo, Z adelante). La
  conversión desde la cámara de MuJoCo (mira hacia −z, y arriba).
- Nubes de puntos: recorte, **voxelizado**, auto-filtrado.
- **Oclusión** y mapas de ocupación: por qué un solo cuadro no basta, y el
  borrado de espacio libre (*ray casting*).

**Dónde leerlo**
- ★ [L8] Hartley & Zisserman, cap. 6 "Camera Models" (hasta la cámara
  proyectiva finita).
- ★ [L9] REP-103 (corto, y todo es relevante).
- ★ [19] Hornung et al. (OctoMap): sección del modelo de sensor y de la
  actualización de ocupación.
- ☆ [20] ARMOR y [21] cuRoboV2: percepción de profundidad aplicada a evasión.

**Dónde aparece en el código**
- `depth_obstacles.depth_to_points`, `camera_pose_from_mjcf_xyaxes`,
  `obstacles_from_depth`, `ObstacleMemory`.
- `perception_common.deproject` (el mismo modelo, usado por los detectores).

**Compruébalo tú**
1. A partir de `xyaxes` de `robot_rgbd_camera` en el MJCF, arma a mano
   `R_torso_cam` (frame óptico) y compara con `camera_pose_from_mjcf_xyaxes()`.
2. Corre `--scenarios table --modes self+frame self+cam`: 155 frente a 0
   pasos con contacto. Explica la diferencia usando solo la palabra
   "oclusión" y la regla de borrado de `ObstacleMemory`.

---

## M11. Mínimos locales y planificación de movimiento

**Qué necesitas saber**
- Espacio de configuraciones (C-space); obstáculos en C-space.
- Métodos **reactivos** (campos potenciales, QP con restricciones): rápidos
  pero locales, con **mínimos locales**.
- Métodos **globales**: muestreo (PRM, **RRT**, **RRT-Connect**) y
  optimización de trayectorias. Completitud probabilística.
- Distinguir un objetivo **inalcanzable** (la configuración final choca)
  de un **camino bloqueado** (§8.2 de la teoría).
- **Planificación priorizada**: un robot por vez, los demás como
  obstáculos. Es incompleta (puede fallar aunque exista solución conjunta),
  pero es simple y rápida.

**Dónde leerlo**
- ★ [18] Khatib: el paper que origina los campos potenciales y deja ver sus
  mínimos locales.
- ★ [L10] Choset et al., cap. 4 "Potential Functions" y cap. 7
  "Sampling-Based Algorithms".
- ★ [16] Kuffner & LaValle (RRT-Connect).
- ★ [27] Erdmann & Lozano-Pérez (planificación priorizada): la
  introducción y la idea de prioridades bastan.
- ☆ [L11] LaValle, cap. 5 "Sampling-Based Motion Planning" (§5.5, RRT).
  Es gratis en línea.
- ☆ [L2] Lynch & Park, cap. 10 "Motion Planning".

**Dónde aparece en el código**
- `bimanual_planner.py`: `rrt_connect`, `shortcut`, `densify`,
  `ArmCollisionChecker.edge_valid` (validez de un tramo),
  `goal_configurations` y `pair_compatible` (diagnóstico de objetivos), y
  `GlobalPlannerLayer` (cuándo planificar).
- `test/test_bimanual_planner.py`.

**Compruébalo tú**
1. Dibuja en 2D un punto que va hacia una meta detrás de un obstáculo
   convexo, con un damper. Marca dónde se detiene y por qué la dirección
   permitida no lo acerca a la meta.
2. ¿Por qué en `obstacle` (bandeja encima de la mano derecha) la capa 2
   no ayuda y hace falta la capa 3? ¿Y por qué contra una esfera pequeña la
   capa 1 sola sí llega? (§10.1 de la teoría.)
3. Corre `python3 demos/evaluate_bimanual_avoidance.py --scenarios obstacle --modes full full+plan --seconds 15`
   y sigue los eventos `[capa 3]`: detección, tiempo de planificación,
   puntos del camino y llegada.

---

## M12. Coordinación de dos brazos

**Qué necesitas saber**
- Espacio de tarea **cooperativo**: pose absoluta y relativa, para cuando
  las dos manos sujetan un mismo objeto.
- Coordinación **desacoplada**: cada brazo tiene su tarea y se arbitra el
  espacio compartido (prioridad, esperas, retiradas).
- **Descomposición camino-velocidad**: los caminos se mantienen y se
  cambia el tiempo (uno espera).
- Maestro/esclavo con cambio de rol; detectar el bloqueo; qué hacer cuando
  ninguna prioridad sirve (escalar a un planificador).

**Dónde leerlo**
- ★ [15] Ju et al. (maestro/esclavo con cambio de rol): la base directa de
  la capa 2.
- ★ [12] Shin & Zheng (retrasar un robot).
- ★ [14] Geng et al. (GGDRC): la arquitectura global + reactiva completa.
- ☆ [13] Chiacchio et al. (espacio cooperativo), para tareas con objeto
  compartido.

**Dónde aparece en el código**
- `bimanual_coordinator.BimanualCoordinator` (máquina de estados) y
  `TaskSequence` (capa de tarea).
- `test/test_bimanual_coordinator.py`.

**Compruébalo tú**
1. Corre `python3 demos/evaluate_bimanual_avoidance.py --scenarios shared --modes self+cam full --seconds 12`
   y sigue los eventos `[capa 2]` en la máquina de estados de la §9.2 de la
   teoría.
2. En `swap`, ¿por qué la capa 2 termina en "hace falta la capa 3"?
   Relaciónalo con M11.

---

## Lista de control final

Entiendes el sistema "al pie de la letra" si puedes responder, sin mirar,
estas 18 preguntas:

1. ¿Por qué todo se expresa en `torso_link` y la cintura no afecta?
2. ¿Qué dice cada columna de `point_jacobian` y por qué hay columnas cero?
3. ¿Por qué un QP y no la pseudoinversa?
4. ¿Por qué un QP de 14 variables y no dos de 7?
5. ¿Qué hace `w_reg` y qué pasaría con `w_reg = 0` en una singularidad?
6. ¿Por qué `w_elbow` bajó de 3.0 a 0.03?
7. ¿Por qué las cápsulas son conservadoras, y qué cuesta eso?
8. Deriva la fila del QP para un par brazo-brazo.
9. Demuestra que el damper mantiene `d ≥ d_s` en tiempo continuo.
10. ¿Por qué el damper es una CBF, y cuándo una CBF aportaría más?
11. ¿Por qué la autocolisión no usa la cámara?
12. ¿Qué es el auto-filtrado y qué pasa si se omite?
13. ¿Cuándo borra `ObstacleMemory` un vóxel y cuándo no?
14. ¿Qué diferencia hay entre un objetivo inalcanzable y un camino
    bloqueado, y qué capa resuelve cada uno?
15. Describe la máquina de estados de la capa 2 y sus dos protecciones
    contra falsos positivos.
16. ¿Por qué la capa 3 sigue el camino en espacio articular y no con
    posiciones de la mano?
17. ¿Por qué los márgenes del planificador son mayores que los `d_s` de la
    capa 1, y qué pasa si el brazo arranca por dentro de ellos?
18. ¿Cuándo NO debe dispararse la capa 3, y por qué?

---

## Libros y documentos de estudio

[L1] G. Strang, *Introduction to Linear Algebra*, 5th ed. Wellesley, MA,
USA: Wellesley-Cambridge Press, 2016.

[L2] K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and
Control*. Cambridge, U.K.: Cambridge Univ. Press, 2017. [Online]. Available:
http://hades.mech.northwestern.edu/index.php/Modern_Robotics (PDF gratuito
de los autores).

[L3] J. Solà, "Quaternion kinematics for the error-state Kalman filter,"
2017, arXiv:1711.02508.

[L4] S. Boyd and L. Vandenberghe, *Convex Optimization*. Cambridge, U.K.:
Cambridge Univ. Press, 2004. [Online]. Available:
https://web.stanford.edu/~boyd/cvxbook/

[L5] J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed. New
York, NY, USA: Springer, 2006.

[L6] J.-J. E. Slotine and W. Li, *Applied Nonlinear Control*. Englewood
Cliffs, NJ, USA: Prentice Hall, 1991.

[L7] MuJoCo Documentation, "Computation." [Online]. Available:
https://mujoco.readthedocs.io/en/stable/computation/index.html

[L8] R. Hartley and A. Zisserman, *Multiple View Geometry in Computer
Vision*, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2004.

[L9] T. Foote and M. Purvis, "REP 103: Standard units of measure and
coordinate conventions," ROS Enhancement Proposals, 2010. [Online].
Available: https://www.ros.org/reps/rep-0103.html

[L10] H. Choset *et al.*, *Principles of Robot Motion: Theory, Algorithms,
and Implementations*. Cambridge, MA, USA: MIT Press, 2005.

[L11] S. M. LaValle, *Planning Algorithms*. Cambridge, U.K.: Cambridge
Univ. Press, 2006. [Online]. Available: http://lavalle.pl/planning/

Los libros [24] (Ericson) y [25] (Siciliano et al.) están en
[`REFERENCIAS_BIMANUAL.md`](REFERENCIAS_BIMANUAL.md) porque también se citan
en el diseño.
