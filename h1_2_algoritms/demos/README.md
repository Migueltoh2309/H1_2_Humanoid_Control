# Demos y evaluaciones

Scripts reproducibles que miden cada parte del sistema contra MuJoCo como
verdad de terreno. Salvo `validacion_lazo_cerrado.py`, **no usan ROS**:
instancian MuJoCo directamente y llaman a las mismas funciones que los nodos
en vivo, así que lo que se mide es exactamente lo que corre en el robot.

```bash
source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash
cd ~/humanoid_ws/src/h1_2_algoritms
export MUJOCO_GL=egl          # para renderizar sin pantalla
```

Los resultados se guardan en [`../results/`](../results).

| Demo | Qué mide | Salida |
|---|---|---|
| `benchmark_ik.py [N]` | IK con límites articulares vs. IK original (DLS) | consola |
| `evaluate_bimanual_avoidance.py` | Las 3 capas de evasión en 6 escenarios y 6 modos | `results/bimanual/` |
| `record_bimanual_videos.py` | Videos comparativos lado a lado de los escenarios bimanuales | `videos/` |
| `evaluate_perception.py` | Detección por color vs. YOLO, con la fruta quieta y en movimiento | `results/percepcion/` |
| `evaluate_ir_stereo.py` | Los 4 estimadores del centro de la fruta, con ruido y oclusión | `results/percepcion/ir_stereo.csv` |
| `visual_servoing_grasp.py` | Lazo completo: cámara → localización → Kalman → QP bimanual → agarre, con física | `results/servoing/` |
| `validacion_lazo_cerrado.py [N]` | IK → `/lowcmd`: PD puro vs. PD + integral | consola |

## IK con límites articulares

```bash
python3 demos/benchmark_ik.py 15
```

Genera objetivos alcanzables por construcción y cuenta cuántas soluciones
respetan los límites mecánicos:

| IK | Soluciones usables |
|---|---|
| Original (DLS) | ~13 % |
| **Con límites** | **~97 %** |

## Evasión bimanual

```bash
python3 demos/evaluate_bimanual_avoidance.py                       # todo
python3 demos/evaluate_bimanual_avoidance.py --scenarios shared obstacle --seconds 3
```

| Escenario | Situación |
|---|---|
| `circles` | Las manos giran en círculos en contrafase que se cruzan |
| `shared` | Las dos manos van al mismo punto (como tomar la fruta del centro) |
| `converge` / `swap` | Objetivos incompatibles o cruzados |
| `table` | Objetivos por debajo de la faja: solo la cámara evita el choque |
| `obstacle` | Bandeja desconocida por el modelo encima de la mano derecha |

Modos: `off` (sin evasión), `self` (autocolisión), `self+frame` (cámara sin
memoria), `self+cam` (con memoria de vóxeles), `full` (+ capa 2) y
`full+plan` (+ capa 3). La verdad de terreno de los choques sale de los
contactos entre mallas de MuJoCo, no del modelo de cápsulas del controlador.

Para regenerar los videos (~5 min):

```bash
python3 demos/record_bimanual_videos.py            # los 5
python3 demos/record_bimanual_videos.py --only 3   # uno solo
```

## Percepción y visual servoing

```bash
python3 demos/evaluate_perception.py --no-yolo     # solo color + profundidad
python3 demos/evaluate_ir_stereo.py
python3 demos/visual_servoing_grasp.py --quick
python3 demos/visual_servoing_grasp.py --only static:sphere:-0.15 --video
```

Resultados principales (detalle en [`../docs/VISUAL_SERVOING_PLAN.md`](../docs/VISUAL_SERVOING_PLAN.md)):

| Estimador del centro | Error con ruido D435 | Agarres (fruta quieta / faja) |
|---|---|---|
| Mediana de profundidad | ~34 mm | 0/6 · 0/3 |
| Mediana + radio | ~4 mm | 2/6 · 1/3 |
| **Ajuste de esfera** | **~2.4 mm** | **5/6 · 3/3** |
| Estéreo IR | ~13 mm | 0/6 · 0/3 |

## Lazo cerrado con `/lowcmd`

```bash
python3 demos/validacion_lazo_cerrado.py 4
```

Necesita algo que consuma `/lowcmd` y publique `/lowstate`: el robot real o
un bridge de bajo nivel. Mide el error del efector con PD puro (~75 mm por
la gravedad) y con PD + término integral (~0.07 mm).
