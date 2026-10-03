# Plan de percepción: detección de color + distancia para visual servoing

Documento de **planeamiento** (todavía sin código). Objetivo final: visual
servoing sobre la mandarina de `h1_2_scene_surgery_table.xml` (ver
[README de h1_2_mujoco_sim_bridge](../h1_2_mujoco_sim_bridge/README.md)).
Antes de eso, el primer hito es un nodo de percepción que, a partir de la
cámara RGB-D del torso, entregue **posición 3D del objeto** (color naranja,
circular) tanto si está quieto como si se mueve sobre la faja.

Como pide la tesis una comparación, se plantean **dos métodos de detección**
independientes (uno clásico, uno con deep learning) evaluados con las
mismas métricas.

---

## 1. Qué ya existe y de qué partimos

De `h1_2_mujoco_sim_bridge` (ver su README, sección "Cámara RGB-D"):

| Tópico | Tipo | Notas |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` (rgb8) | 640×480 @ 7.5 Hz |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | intrínsecos (fx, fy, cx, cy) |
| `/camera/depth/image_raw` | `sensor_msgs/Image` (32FC1) | profundidad en metros, mismo frame |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | intrínsecos de profundidad |

Frame `robot_rgbd_camera`, con TF publicado por `robot_state_publisher`
(pose fija respecto a `pelvis`/torso, conocida desde el MJCF).

Y dos maneras de generar la escena para probar cada caso:

- `ros2 launch h1_2_mujoco_sim_bridge sim_bridge_static.launch.py` —
  mandarina quieta en el centro de la faja.
- `ros2 launch h1_2_mujoco_sim_bridge sim_bridge_moving.launch.py` —
  mandarina viajando de una punta a la otra a velocidad constante.

**Ventaja de estar en simulación**: la posición real (ground truth) de la
mandarina es exactamente la del `qpos` del joint `mandarina_slide_joint` en
MuJoCo — se puede exponer como tópico de referencia *solo para evaluación*
(nunca como entrada del detector) y así medir error de percepción en mm sin
necesitar un sistema de motion capture real. Esto es un diferencial fuerte
para la tesis: permite curvas de error cuantitativas, no solo inspección
visual.

---

## 2. Pipeline general (por etapa)

1. **Adquisición sincronizada** — `message_filters.ApproximateTimeSynchronizer`
   sobre color + depth (+ ambos `camera_info`), ya que se publican en el
   mismo hilo de cámara pero conviene no asumir timestamps idénticos.
2. **Preprocesado** — recorte a la región de interés (mesa/faja, se conoce
   su pose en el mundo) para descartar falsos positivos fuera de la zona de
   trabajo; conversión BGR/HSV si aplica.
3. **Detección/segmentación 2D** — el bloque que cambia entre método A y B
   (sección 3).
4. **Extracción de punto 2D representativo** — centroide del contorno
   (método clásico) o centro de la bounding box / máscara (YOLO).
5. **Asociación con profundidad** — en vez de leer un solo píxel (ruidoso
   en bordes), tomar la **mediana** de la profundidad dentro del contorno o
   máscara — más robusto que el centro exacto, que puede caer en un borde
   con profundidad inválida.
6. **Deproyección a 3D** — `(u, v, depth)` + intrínsecos de `CameraInfo` →
   punto 3D en el frame de la cámara (fórmula pinhole estándar); luego TF
   (`robot_rgbd_camera` → `pelvis`/mundo) para tener la posición en el marco
   que use el control de visual servoing.
7. **Filtrado temporal** —
   - Objeto **estático**: promediar N frames antes de "declarar" la
     posición final (reduce ruido, no hay apuro).
   - Objeto **en movimiento**: filtro de Kalman (o al menos EMA +
     diferencia finita) para estimar posición *y velocidad* — necesario
     para servoing predictivo (compensar el delay entre detección y
     movimiento del brazo).
8. **Salida** — publicar `geometry_msgs/PointStamped` (o `PoseStamped` si
   más adelante importa orientación) en un tópico tipo
   `/perception/target_position`, más opcionalmente la velocidad estimada
   en un campo separado o un `TwistStamped`.

### Estático vs. móvil — qué cambia en la práctica

| | Estático | Móvil |
|---|---|---|
| Frecuencia de detección | Puede ser baja (1 detección + promedio basta) | Cada frame disponible (7.5 Hz) |
| Filtro | Promedio simple | Kalman / EMA con estimación de velocidad |
| Métrica clave | Error absoluto de posición (mm) | Error de posición + jitter frame-a-frame + error de velocidad estimada |
| Riesgo principal | Ninguno particular | Motion blur (mitigado: 7.5 Hz es bajo, pero belt es lento ~0.07 m/s, así que no debería ser problema) |

---

## 3. Dos métodos de detección (para comparar)

### Método A — Clásico: color (HSV) + contornos + profundidad

1. Convertir frame RGB a HSV.
2. Umbral de color naranja (`cv2.inRange`), rango calibrado contra el fondo
   blanco de la faja y el marrón/mostaza de la mesa — buen contraste
   esperado (naranja saturado vs. blanco/marrón, colores bien separados en
   H y S).
3. Morfología (`erode`/`dilate` u `opening`/`closing`) para limpiar ruido.
4. `cv2.findContours` → contorno más grande (o filtrar por área mínima) →
   centroide (momentos) + máscara para el paso de profundidad.

**Pros**: sin entrenamiento, muy barato en CPU/GPU (relevante porque la GPU
ya está ocupada por el render de MuJoCo — ver README de `sim_bridge`,
sección de frecuencias), predecible, fácil de depurar visualmente.
**Contras**: fràgil ante cambios de iluminación, no generaliza a otros
colores/objetos sin recalibrar, falla si el objeto queda parcialmente
ocluido por la mano/brazo del robot.

### Método B — Deep learning: YOLO preentrenado (clase "orange")

1. Correr un modelo YOLO preentrenado en COCO (p.ej. YOLOv8n vía
   `ultralytics`) directamente sobre el frame RGB — COCO **ya incluye la
   clase `orange`**, que calza casi perfecto con la mandarina sin necesidad
   de fine-tuning inicial.
2. Filtrar detecciones por clase `orange` y score mínimo.
3. Tomar la bounding box (o, si se usa una variante de segmentación
   `yolov8n-seg`, la máscara — mejor para el paso de profundidad porque
   excluye el fondo dentro del bbox).
4. Igual que el método A: mediana de profundidad dentro de la máscara/bbox
   → deproyección 3D.

**Pros**: más robusto a iluminación/variaciones de color, generaliza mejor,
con segmentación (`-seg`) da una máscara más precisa que un umbral HSV
manual, y deja camino abierto a fine-tuning si luego se cambia de fruta/
objeto o se pasa a la cámara real.
**Contras**: mucho más caro computacionalmente (la GPU ya está compartida
con el render offscreen de MuJoCo — riesgo real de cuello de botella en la
laptop), requiere manejar pesos/dependencias extra (`ultralytics`/`torch`),
y "de caja" solo detecta lo que ya conoce COCO — si el objeto real
cambiara a algo sin clase equivalente, sí haría falta anotar y entrenar.

### (Opcional / método C si da tiempo) Clustering en la nube de puntos

Generar nube de puntos a partir del depth + intrínsecos, filtrar por altura
(la mandarina está a una altura conocida sobre la faja) y hacer clustering
euclidiano (tipo PCL) para encontrar el blob esférico — no depende del
color en absoluto, buen contraste con los otros dos métodos (color vs.
forma/geometría). Se deja como extensión, no bloquea el plan de 2 métodos.

---

## 4. Métricas de comparación (A vs. B)

Gracias al ground truth de MuJoCo (sección 1), se puede medir directamente:

- **Error de posición 3D** (mm) vs. la posición real del `mandarina_slide_joint`, para varias posiciones a lo largo de la faja.
- **Tasa de detección** (recall) — frames donde el objeto está en el FOV pero no fue detectado.
- **Latencia / FPS** de cada método corriendo junto con el resto del stack (MuJoCo + bridge + RViz).
- **Jitter** en el caso móvil: varianza del error frame-a-frame (¿la posición estimada "tiembla" más con un método que con otro?).
- **Robustez a oclusión parcial**: acercar el brazo/mano del robot sobre la mandarina y ver en qué punto cada método deja de detectar.

---

## 5. Próximos pasos propuestos (orden sugerido)

1. ~~Nodo mínimo con el **Método A** (color+profundidad)~~ — **implementado**,
   ver sección 6.
2. Tópico de "ground truth" (debug-only) que exponga la posición real de la
   mandarina, para poder medir error de forma automática (por ahora se
   verificó a mano contra el `qpos` conocido, ver secciones 6 y 7).
3. ~~Nodo con el **Método B** (YOLO)~~ — **implementado**, ver sección 7.
4. ~~Script de evaluación~~ — **implementado**, ver sección 8.
5. Con eso ya se tiene material de comparación cuantitativo para la tesis,
   y una interfaz de salida (`/perception/target_position`) lista para
   conectar al control de visual servoing.

---

## 6. Método A — estado: implementado y probado

Nodo: `h1_2_algoritms/nodos/color_depth_detector_node.py` (paquete
`h1_2_algoritms`, ya compilado y con entry point registrado).

```bash
ros2 launch h1_2_mujoco_sim_bridge sim_bridge_static.launch.py   # o _moving
ros2 run h1_2_algoritms color_depth_detector_node --ros-args -p target_frame:=pelvis
```

Publica `geometry_msgs/PointStamped` en `/perception/target_position`
(frame de cámara por defecto, o `target_frame` si se pasa y hay TF hacia
ahí — usa el `tf_static` que ya publica `sim_bridge_node.py`) y una imagen
de depuración (máscara + centroide + `z` estimada) en
`/perception/debug_image`, visualizable con `rqt_image_view`.

### Hallazgo de la primera prueba: el hue solo no alcanza

Con el umbral HSV inicial (basado solo en el hue de la mandarina, H≈13) el
nodo detectaba **la mesa completa** como el "objeto naranja" más grande
(109354 px vs. ~1200 px reales de la mandarina) y publicaba una posición
sin sentido. Calculando el HSV exacto de ambos colores del MJCF:

| Objeto | H (OpenCV 0-179) | S (0-255) | V (0-255) |
|---|---|---|---|
| Mandarina (`1.0 0.45 0.05`) | ≈13 | ≈242 | ≈255 |
| Mesa (`0.58 0.38 0.14`) | ≈16 | ≈193 | ≈148 |

El hue es casi idéntico (ambos son "naranja/marrón"); lo que las separa es
que la mandarina está mucho más saturada y más clara. Subiendo
`sat_min: 110→210` y `val_min: 90→170` (por encima de los valores de la
mesa) la mesa queda excluida y solo se detecta la mandarina — son los
valores por defecto actuales del nodo. Queda documentado en el propio
código (comentario junto a los parámetros) para no repetir el error si se
cambia el objeto o la iluminación más adelante.

### Precisión medida (escena estática, contra la posición real del joint)

Posición real de la mandarina en el mundo: `x=0.34, y=-0.03, z=0.864` (ver
README de `h1_2_mujoco_sim_bridge`). Salida del nodo con `target_frame:=pelvis`
(pelvis está en `x=0,y=0` del mundo, así que xy comparan directo):

| Eje | Real | Detectado | Error |
|---|---|---|---|
| x | 0.340 | 0.331 | ~9 mm |
| y | -0.030 | -0.028 | ~2 mm |
| z (mundo, sumando pelvis_z=1.0282) | 0.864 | 0.899 | ~35 mm |

XY muy preciso (centroide del contorno es buen estimador). El error en Z es
sistemático y esperado: la profundidad medida es la de la **superficie más
cercana a la cámara** de la mandarina, no la de su centro — para una esfera
vista desde arriba/adelante eso sesga la lectura hacia "más cerca" hasta en
~1 radio (4 cm), que es justo el orden del error observado. No es ruido: es
un sesgo geométrico conocido de usar profundidad de superficie como
posición del objeto. Si hace falta corregirlo (p.ej. para el control de
visual servoing) hay dos caminos simples: sumar el radio conocido a lo
largo del rayo cámara→objeto (solo válido si el objeto es siempre esférico
de radio conocido), o comparar igual con el Método B para ver si su
estimación de profundidad (con máscara de segmentación en vez de bbox) se
comporta distinto — buen punto de comparación cuantitativo para la tesis.

---

## 7. Método B — estado: implementado y probado

Nodo: `h1_2_algoritms/nodos/yolo_color_depth_detector_node.py`. Comparte con el
Método A el pipeline de profundidad/deproyección/TF (factorizado en
`h1_2_algoritms/vision/perception_common.py` — ambos nodos importan las mismas
funciones `image_to_array`, `median_depth_in_mask`, `deproject`,
`transform_point`, etc.; solo difiere la etapa de detección 2D).

Modelo: `yolov8n-seg` (nano, con segmentación — mejor máscara para la
profundidad que un bbox) preentrenado en COCO, vía `ultralytics`. **Corre
en CPU por decisión de diseño**, no por limitación: la GTX 1650 (4 GB) ya
está ocupada por el render de MuJoCo, y sumarle inferencia competiría por
la misma VRAM — se instaló `torch`/`torchvision` desde el índice CPU-only
de PyTorch (`--index-url https://download.pytorch.org/whl/cpu`) para no
tocarla en absoluto. `device` es parámetro del nodo por si más adelante
interesa medir el costo real de moverlo a GPU como parte de la
comparación.

```bash
pip install --user --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install --user ultralytics

ros2 launch h1_2_mujoco_sim_bridge sim_bridge_static.launch.py   # o _moving
ros2 run h1_2_algoritms yolo_color_depth_detector_node --ros-args -p target_frame:=pelvis
```

Publica en `/perception/target_position_yolo` y
`/perception/debug_image_yolo` — **tópicos separados del Método A a
propósito**, para poder correr los dos nodos al mismo tiempo y comparar en
vivo sin que se pisen.

### Incidente de instalación: `pip install --user` rompió `colcon build`

Instalar `torch`/`torchvision` trajo como dependencia transitiva un
`setuptools` nuevo (78.1.0) a `~/.local/lib/.../site-packages`, que por
precedencia de `sys.path` pasó a tapar el `setuptools` del sistema
(59.6.0) para **todo** el intérprete de Python del usuario — no solo para
estos dos paquetes. Como esa versión nueva de `setuptools` requiere una
función de `packaging` (`canonicalize_version` con el kwarg
`strip_trailing_zero`) que el `packaging` 21.3 de Ubuntu 22.04/ROS Humble
no tiene, **cualquier build de un paquete `ament_python` (con
`colcon build`) empezó a fallar en todo el workspace**, no solo en
`h1_2_algoritms`. Se detectó de inmediato al recompilar y se corrigió con
`pip uninstall setuptools` (vuelve a resolver al del sistema; `torch` no
lo necesita en tiempo de ejecución, solo lo declaraba como dependencia de
instalación). Vale la pena vigilar esto si más adelante se instala otra
librería pip pesada: revisar `pip show setuptools` después de instalar
algo grande, y si señala `~/.local/...` en vez de
`/usr/lib/python3/dist-packages/...`, desinstalarlo.

### Hallazgo real: domain gap entre COCO (fotos) y el render de MuJoCo

Con el modelo apuntando solo a la clase `orange` (conf. mínima 0.25) **no
detectaba nada**. Depurando con la imagen cruda (`/tmp/frame.png`) y sin
filtro de clase ni umbral, el modelo sí ve el objeto, pero lo clasifica
como:

| Clase | Confianza |
|---|---|
| `sports ball` | 0.79 |
| `orange` | 0.12 |

Mismo bounding box en ambos casos. Es decir: el objeto SÍ es detectable,
pero el modelo (entrenado con fotos reales) lo reconoce mejor como "pelota"
que como "naranja/mandarina" — el render de MuJoCo (sombreado plano, sin
textura fotográfica de cáscara) se parece visualmente más a una pelota
lisa que a la textura real de un cítrico. Es un resultado de domain gap
esperado y documentado en la sección de riesgos original de este plan, no
un bug del nodo. Solución aplicada: el nodo acepta una **lista** de clases
candidatas (`target_class_names`, default `['orange', 'sports ball']`) y
se queda con la de mayor confianza — sigue siendo detección zero-shot (sin
fine-tuning), solo más realista sobre qué clase de COCO va a disparar en
este render en particular. Si más adelante se pasa a la cámara real (fotos
reales), es esperable que `orange` vuelva a ganar por sobre `sports ball`
— sería interesante confirmarlo y dejarlo como parte de la comparación de
la tesis (zero-shot en sim vs. zero-shot en real).

### Precisión y velocidad medidas (mismo test que el Método A)

Con `target_frame:=pelvis`, detección estable en `sports ball` (conf.
0.79), posición `(0.331, -0.028, -0.133)` m en `pelvis` → mundo
`z=1.0282-0.133=0.895` m. Mismo sesgo geométrico que el Método A (Z por
superficie más cercana, no centro) y prácticamente el mismo XY:

| | Método A (color) | Método B (YOLO) | Real |
|---|---|---|---|
| x | 0.331 | 0.331 | 0.340 |
| y | -0.028 | -0.028 | -0.030 |
| z (mundo) | 0.899 | 0.895 | 0.864 |
| latencia por frame | <5 ms (CPU, umbral+contornos) | ~150-450 ms en régimen (primer frame ~3.5 s por warmup de `torch`), CPU | — |

XY prácticamente idéntico entre métodos (buena señal: la deproyección y el
centroide, que comparten código en `perception_common.py`, no son la
fuente de discrepancia). La diferencia real de fondo entre A y B no se ve
en este escenario "de laboratorio" (un solo objeto, sin oclusión, colores
bien separados) sino en los casos límite de la sección 4 (oclusión parcial,
cambios de iluminación, generalización a otros objetos) — pendiente de un
script de evaluación sistemático (paso 4 de la sección 5).

---

## 8. Script de evaluación — estado: implementado, primeros resultados sistemáticos

`h1_2_algoritms/demos/evaluate_perception.py` — standalone (no usa ROS,
mismo patrón que `benchmark_ik.py`): instancia MuJoCo directamente,
reutiliza el mismo `detect_color`/`detect_yolo` que corren los nodos en
vivo (cero implementaciones duplicadas), y mide contra el ground truth
exacto (`data.xpos` de la mandarina) — sin necesidad de motion capture.

Dos experimentos, ambos con el robot quieto en su pose de arranque (no hace
falta moverlo, solo importa la cámara):

1. **Barrido espacial estático**: la mandarina se coloca en N posiciones
   conocidas a lo largo de toda la faja (fijando `qpos` directamente, sin
   avanzar física) — cubre el rango de trabajo completo, no solo el punto
   donde se probó a mano en las secciones 6-7.
2. **Paso dinámico**: keyframe `mandarina_moving`, física real avanzando
   (`mj_step`), muestreando cada 0.4s durante ~16s (el tiempo que tarda en
   cruzar la faja). Sin gravedad a propósito — el script no controla los 27
   motores del robot, así que sin este ajuste los brazos caerían en caída
   libre y podrían tapar la cámara durante el experimento (ver docstring
   del script).

```bash
cd src/h1_2_algoritms
python3 demos/evaluate_perception.py                # todo por defecto (15 puntos + 16s)
python3 demos/evaluate_perception.py --no-yolo       # solo Método A, más rápido
python3 demos/evaluate_perception.py --device cuda:0 # si algún día se instala torch+CUDA
```

Guarda `results/percepcion/{estatico,movimiento}.csv`
(fila por detección: ground truth, estimación, error por eje, score,
latencia) para graficar en la tesis.

### Resultado (15 posiciones estáticas + 40 muestras en movimiento, CPU)

| | recall | error 3D (media / mediana / máx) | \|x\| | \|y\| | \|z\| | latencia |
|---|---|---|---|---|---|---|
| **A (color)** — estático | 93% | 38.7 / 37.5 / 48.0 mm | 7.9 | 10.2 | 35.5 mm | 7.8 ms |
| **A (color)** — movimiento | 100% | 37.4 / 34.8 / 69.8 mm | 9.5 | 15.4 | 32.4 mm | 3.4 ms |
| **B (YOLO/cpu)** — estático | 67% | 33.9 / 33.1 / 40.2 mm | 8.2 | 6.2 | 32.1 mm | 216.9 ms |
| **B (YOLO/cpu)** — movimiento | 80% | 34.9 / 34.3 / 40.2 mm | 6.9 | 13.3 | 31.4 mm | 98.6 ms |

Tres hallazgos, dos de ellos nuevos respecto a lo que ya sabíamos de las
secciones 6-7 (que solo habían probado un punto fijo de la faja):

1. **El recall de YOLO cae fuerte lejos del centro de la faja** (67-80%
   contra 93-100% del Método A). Revisando el CSV punto por punto: YOLO
   detecta con confianza alta (`sports ball`, conf. 0.6-0.8) solo en la
   franja donde la mandarina está entre las manos del robot; en los
   extremos de la faja (lejos de esa "escena típica de sujetar algo")
   simplemente no dispara ninguna detección por encima del umbral. Es una
   dimensión de domain gap que no se veía con una sola posición de prueba:
   no solo "confunde la clase" (sección 7), sino que **su confianza
   depende del contexto visual aprendido** (manos alrededor del objeto),
   algo que un umbral de color no tiene por qué sufrir. Buen resultado
   cuantitativo para la tesis — no es una afirmación cualitativa, son 15
   puntos medidos.
2. **El sesgo de profundidad en Z (sección 6) se confirma, ahora en los
   DOS métodos y en todo el rango**: ~32-36 mm de error en Z contra
   ~6-15 mm en X/Y, consistente con el radio de la mandarina (40 mm) — es
   un sesgo geométrico del principio (profundidad de superficie, no de
   centro), no un defecto de un método en particular. Que aparezca igual
   en un pipeline de color y en uno de deep learning, con implementaciones
   de detección 2D totalmente distintas, es evidencia más sólida de que la
   causa es la geometría de la deproyección y no un artefacto de alguno de
   los dos algoritmos.
3. **Latencia confirmada a escala** (sin contención de CPU, a diferencia
   de la prueba en vivo con el viewer): Método A ~3-8 ms vs. Método B
   ~99-217 ms — consistente con lo medido antes en ROS.

Con ground truth exacto y sin motion capture, estos tres puntos (recall
espacialmente heterogéneo de YOLO, sesgo de Z algoritmo-independiente,
latencia) son exactamente el tipo de resultado cuantitativo, reproducible
y con causa explicada que sostiene los ángulos de publicación propuestos
al principio de esta conversación (comparación clásico vs. DL con
recursos acotados, domain gap, y el testbed en sí como contribución
metodológica).

---

## 9. Centro de la fruta: ajuste de esfera en los nodos — estado: implementado y probado en vivo

El sesgo de z de la sección 8 (se medía la superficie visible, no el centro)
ya está corregido en los dos nodos. El parámetro nuevo `localization` elige
cómo se pasa de máscara + profundidad al punto publicado
(`perception_common.localize_center`):

| `localization` | Qué hace |
|---|---|
| `sphere` (por defecto) | ajuste de esfera de radio `object_radius_m` (0.04) a los píxeles de la máscara → **centro** |
| `median+R` | mediana + corrección cerrada 0.71·R a lo largo del rayo |
| `median` | el comportamiento original (superficie) |

Si el ajuste no converge o no es coherente con la mediana, cae a
`median+R` en ese cuadro, con un warning.

Medido en vivo (bridge headless, `target_frame:=pelvis`, mandarina en
(0.340, −0.030, −0.164)):

| Nodo | `median` | `median+R` | `sphere` |
|---|---|---|---|
| Método A (color) | 30.2 mm | 4.8 mm | **1.7 mm** |
| Método B (YOLO-seg) | — | — | **1.7 mm** |

Offline, sobre la faja, con los mismos `detect_color` / `detect_yolo`:
- Método A, con ruido tipo D435: 33.4 → 2.4 mm (21 posiciones).
- Método B: 29.8 → 1.8 mm (7 de 10 detectadas; el recall de YOLO sigue
  siendo el de la sección 8).

Detalle y comparación con el estéreo IR en VISUAL_SERVOING_PLAN.md §3; referencias (ajuste de modelo geométrico a la nube, Huber, ruido D400) en REFERENCIAS_VISUAL_SERVOING.md [6]–[8], [12]–[13].

**Hallazgo lateral: el marcador rojo inflaba la máscara.** El site
`mandarina_target` de `h1_2_scene_surgery_table.xml` (una esfera roja 6 cm
sobre la fruta) era visible para la cámara. El rojo puro pasa el umbral HSV
naranja y agregaba ~430 px (+50%) a la máscara. Con él, el ajuste de esfera
daba 8.1 mm en vez de 1.7. Se pasó al grupo 4 (no se renderiza en la cámara,
se puede activar en el viewer). Los números de las secciones 6-8 se midieron
con el marcador visible, así que `evaluate_perception.py` conviene volver a
correrlo.

`demos/evaluate_perception.py` sigue midiendo con la mediana (la
deproyección directa, no `localize_center`), para mantener comparables los
números de la sección 8.

---

## Riesgos / cosas a vigilar

- ~~GPU compartida~~ — resuelto: el Método B corre en CPU a propósito (ver
  sección 7), no toca la GTX 1650 que usa MuJoCo. Sí vale la pena vigilar
  CPU: con `sim_bridge` headless ya usando ~1 núcleo al 100% (física
  tiempo-real + hilo de cámara), y YOLO en CPU turnándose el resto, en la
  laptop (8 núcleos) no hubo problema corriendo un nodo de percepción a la
  vez — no se probó A + B simultáneos ni con el viewer nativo abierto.
- Resolución/Hz actuales (640×480 @ 7.5 Hz) son bajos a propósito por la
  laptop (ver README de `sim_bridge`) — suficientes para probar el
  pipeline, pero conviene no usarlos como referencia de rendimiento "real".
- El umbral HSV del método A habrá que recalibrarlo si más adelante cambia
  la iluminación de la escena o el objeto deja de ser naranja puro.
- Instalar librerías pip pesadas (`torch`, etc.) con `--user` puede
  arrastrar un `setuptools` que rompa `colcon build` en todo el workspace —
  ver el incidente documentado en la sección 7 antes de repetir el patrón.
