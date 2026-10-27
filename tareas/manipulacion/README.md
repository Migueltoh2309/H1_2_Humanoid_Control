# Manipulación del H1-2 — reto R40-RT-H1_2-0002

Enunciado: `src/Reto_Manipulacion_H1_2_Robotics40.pdf`. El H1-2, de pie en FSM 201 y **sin dar un
paso**, localiza con la D435i de la cabeza cuatro objetos sobre una mesa (86 cm en el reto, 92 aquí), los reconoce,
los coge con las manos Inspire y deja cada uno en su bandeja (A-D). Brazos solo por `rt/arm_sdk`,
manos solo por Modbus TCP. Niveles: 1 tocar dianas (≤ 2 cm), 2 coger y colocar, 3 clasificar con
las dos manos mientras el instructor mueve objetos, 4 colocar girado 90 grados sobre una cruz.

**El mismo código corre en el simulador y en el robot**, como en el seguidor de línea. El
simulador (`../../h1_2_sim2real`) habla el protocolo del robot (`rt/arm_sdk`, `rt/lowstate`, las
manos por Modbus) y publica la cámara con los tópicos de `realsense2_camera`. Los algoritmos de
movimiento son los de la tesis (`h1_2_algoritms`: cinemática, cápsulas de colisión, orientaciones
de agarre; RRT y QP para la planificación del hito 3), importados desde su código fuente.

```text
manipulacion/
├── manip/                   los bloques de la sec. 5 y su base
│   ├── tipos.py             interfaces entre bloques (INTERFACES.md)
│   ├── calibracion.py       CALIBRACIÓN cámara-torso (ArUco del dorso, AX=YB, reproyección)
│   ├── percepcion.py        PERCEPCIÓN (profundidad nativa: plano, autofiltro, identidad, pose)
│   ├── planificacion.py     PLANIFICACIÓN (IK + RRT de la tesis + rectas; comprobada contra el MJCF)
│   ├── brazo.py             BRAZO (rt/arm_sdk: rampas de peso, mínima sacudida, v <= 0.5 rad/s)
│   ├── mano.py              MANO (Modbus: FORCE_SET siempre antes de cerrar, agarre, corte)
│   ├── supervisor.py        SUPERVISOR: máquina de estados, único que cambia el peso; salida segura
│   ├── registro.py          REGISTRO: CSV a 50 Hz, eventos y métricas (JSON), imágenes
│   ├── compensacion.py      error del brazo (g(q)/kp + lazo con encoders) e inclinación del cuerpo (IMU)
│   ├── dianas.py            nivel 1: dianas en IR1
│   ├── seguridad.py         vigilante de la sec. 10 (FSM, IMU, seguimiento, cámara)
│   ├── robot_io.py · fuentes.py   rt/lowstate, GetFsmId · cámara (ROS, pyrealsense2, carpeta)
│   ├── cinematica.py · modelo.py  cinemática de la tesis · el MJCF como comprobador (choques, vista)
│   └── estacion.py · config.py
├── manipulacion.py          PROGRAMA PRINCIPAL (el mismo en el simulador y en el robot)
├── config/                  estacion · objetos · brazo · poses (generadas) · agarres (tabla del hito 2) · planificacion · supervisor
├── INTERFACES.md · RESPUESTAS.md (6.1-6.4 con números)
├── sim/construir_estacion.py  escenas de la estación (niveles 0-4 x semillas) + verdad de terreno
├── herramientas/            mapa_alcance (hito 1) · poses · calibrar · curva_pasos · fuerza_mano (hito 2) ·
│                            evaluar_percepcion · probar_plan · probar_agarre (F3) · captura_camara · rviz_estacion
├── scripts/                 sim_estacion · ver_estacion · hito2_sim
├── tests/                   26 pruebas (cinemática, estación, mano simulada, calibración, percepción, plan)
└── datos/                   alcance, calibracion, curva_pasos, fuerza_mano, dataset, percepcion, planes, agarre
```

## Plan (fases ↔ hitos del reto)

| Fase | Hito | Contenido | Estado |
|---|---|---|---|
| F0 | — | Simulador de la estación: mesa, zona, bandejas, objetos, dianas, cruz, ArUco; mano con FORCE_SET real, táctil y STATUS; profundidad nativa; verdad de terreno de objetos y pies; instructor que mueve objetos | **hecho** |
| F1 | 1 | Interfaces por escrito; mapa de alcance con el URDF y la IK; elección de d | **hecho** (d = 10 cm) |
| F2 | 2 | Pose de observación; calibración cámara-torso (0.6 mm / 0.04 grados en el sim, mesa a 92 cm); curva de pasos; fuerza frente a FORCE_SET; tabla por objeto | **hecho en el sim**; en el robot, con las mismas herramientas |
| F3 | — | Percepción sobre datos grabados (36/36, <= 6.4 mm con ruido D435); planificación RRT/IK comprobada contra el modelo; física del agarre | percepción y plan **hechos** (planes válidos 21/36 con la mesa a 92 cm); agarre **12/12 sin asistencia** (cubo, vaso de 6 cm, pelota y prisma 4x4x10) |
| F4 | — | Programa principal `manipulacion.py`: supervisor (máquina de estados, salida segura), registro (CSV 50 Hz, eventos, PNG), niveles 1-4 | **hecho**; tiradas en curso (F6) |
| F5 | 3 | Simulacro: trayectorias calculadas y reproducidas en MuJoCo sin enviarse; aborto seguro | pendiente |
| F6 | niveles | Tiradas 1-4 en el simulador con métricas de verdad de terreno | pendiente |
| F7 | cierre | RESPUESTAS completo, vídeos, lista de lo que hay que medir en el robot | pendiente |

## Simulación

`source scripts/entorno.sh` en cada terminal (venv, ROS 2, SDK de Unitree).

```bash
python3 sim/construir_estacion.py --todos            # niveles 0..4 x semillas 0..2 (una vez, o tras cambiar d)
./scripts/ver_estacion.sh 3                          # MuJoCo + RViz (GPU NVIDIA), nivel 3 semilla 0
./scripts/sim_estacion.sh 1 2 use_viewer:=false      # solo el simulador (nivel 1, semilla 2)
python3 herramientas/captura_camara.py datos/capturas/mesa   # color, IR1, profundidad alineada y nativa
python3 herramientas/mapa_alcance.py                 # hito 1 (~2.5 min con 2 procesos) -> datos/alcance/
MUJOCO_GL=egl python3 herramientas/poses.py          # pose de observación y de calibración (contra el modelo)
./scripts/hito2_sim.sh                               # hito 2 en el sim: calibración, curva de pasos, fuerza (~8 min)
python3 herramientas/evaluar_percepcion.py --tapar   # graba el conjunto de datos y evalúa la percepción
python3 herramientas/probar_plan.py                  # planes sobre lo percibido + presupuesto de tiempo
python3 herramientas/probar_agarre.py --todos --guardar   # física del agarre (sin ROS, más rápido que el real)
python3 -m pytest tests -q                           # 26 pruebas

# PROGRAMA PRINCIPAL (con el simulador del nivel arrancado: ./scripts/sim_estacion.sh N)
python3 manipulacion.py --sim --nivel 2              # coger y colocar
python3 manipulacion.py --sim --nivel 3 --instructor # clasificar; el "instructor" mueve un objeto (solo sim)
python3 manipulacion.py --sim --nivel 4 --objeto prisma   # sobre la cruz, girado 90 grados
python3 manipulacion.py --sim --nivel 1              # tocar las dianas
python3 manipulacion.py --sim --nivel 2 --simulacro  # hito 3: todo calculado y registrado, nada se mueve
# registro de cada tirada: datos/tiradas/nivelN_<fecha>/ (lazo.csv, eventos.json, *.png)
```

**El robot en esta simulación está de pie y quieto** (`MARCHA=ninguna`): la base es flotante,
con gravedad, contactos e IMU de verdad, y las piernas se sostienen rígidas. Si el brazo empuja
el cuerpo, el cuerpo se mueve, pero no da pasos. La política de marcha de `unitree_rl_gym`
(`MARCHA=politica`) **no sirve para este reto**: con orden cero pisa en el sitio y deriva (medido:
13 cm hacia atrás y 11 grados de giro en 10 s), y el reto cuenta pasos. La FSM 201 real se queda
quieta. La reacción del cuerpo real a los brazos (los pasos y el giro de la pregunta 6.3.1) no
se puede simular sin el controlador de Unitree: en el simulador se mide la perturbación física
(desplazamiento del centro de masas, par en los pies) y la curva de pasos se mide en el robot (hito 2).

**La estación** (`config/estacion.yaml`, figura de la sección 4): tablero de 80x50 cm a **92 cm**
(el reto dice 86; ver abajo) con el borde cercano a d del pecho; zona de recogida de 40x25 cm centrada en y, con su centro a 19 cm
del borde cercano; bandejas de 12x12 cm con borde de 2 cm, centradas a ±30 cm de la línea media,
A/C a 12 cm y B/D a 30 cm del borde (A y B a la izquierda). El enunciado no acota las x: se
midieron sobre el dibujo y en la sesión se miden con la regla. Objetos del ejemplo del enunciado:
cubo de madera de 6 cm, vaso con arena (6 cm de diámetro), pelota de tenis y, en lugar del rotulador (tumbado mide 3.2 cm y el reto pide 4-15 de alto), un prisma de madera de 4x4x10 cm de pie.

**El mundo** (`sim/construir_estacion.py`, aspecto en `config/estacion.yaml: ambiente`): una sala
con paredes, suelo gris oscuro y luz cenital difusa (una sola sombra y dos focos sin sombra); el
tablero es una placa blanca mate sobre un canto gris, para que el borde de la mesa y los objetos
contrasten en el color; la zona de recogida va marcada con cinta azul. La percepción trabaja
sobre la profundidad, que no depende de nada de esto; el color sirve para el ArUco y para
distinguir los objetos.

**Mesa a 92 cm en vez de 86** (decidido el 2026-10-13, con el mapa de alcance para cada altura,
`mapa_alcance.py --mesa`; d = 10 cm, agarre a 3 cm sobre el tablero):

| mesa | zona alcanzable | cada mano | franja común | bandejas al agarrar | color ve |
|---|---|---|---|---|---|
| 86 cm | 80 % | 50 % | 20 % | A, C (B y D no) | 60 % |
| 90 cm | 97 % | 67 % | 37 % | las 4 | 60 % |
| 92 cm | 100 % | 74 % | 48 % | las 4 | 70 % |
| 95 cm | 100 % | 82 % | 63 % | las 4 | 70 % |

En el robot real se consigue con calzos de 6 cm bajo las patas. Con la mesa a 92 cm, en la
postura de pie las puntas de los dedos quedan a ~1 cm del tablero y con la gravedad se apoyan en
él, así que **los brazos empiezan en la pose de observación** (manos a los lados, 4 cm sobre la
mesa): en el simulador es el keyframe de las escenas; en el robot, el protocolo es llevar los
brazos a la pose de observación antes de acercar la mesa. Los datos con 86 cm quedan en
`datos/alcance/mesa86/`.

**Lo que el simulador hace como el robot** (añadido para este reto en `h1_2_sim2real`):
* **FORCE_SET limita la fuerza** de cada dedo; de fábrica (2000) aprieta al máximo (10 N por dedo,
  15 N el pulgar). Medido: con 200/500/800 gf los dedos que tocan se quedan en 199/500/799 gf, y un
  dedo que se cierra sin nada llega a 0 con fuerza ~0 (la pista de 6.4.3).
* STATUS (0 abriendo, 1 cerrando, 2 en posición, 3 parado por fuerza, 6 calado), ERR, TEMP, borrar
  error (1004), táctil aproximado en los registros reales, y la lectura en bloque 1546..1620
  devuelve basura (11000), como la mano real (sec. 7).
* **Profundidad nativa** (`/camera/depth/image_rect_raw`, marco de IR1, campo de ~80x65 grados) además
  de la alineada al color (56x43 grados).
* La cámara está **desviada** de la pose del URDF (1.8 grados de cabeceo, 1.2 de guiñada, 0.7 de
  alabeo, ~1 cm): la pose real la tiene que encontrar la calibración del hito 2. La desviación
  queda en el JSON de la escena bajo `verdad_oculta`, que solo lee la evaluación.
* Verdad de terreno de los objetos y de los pies (`/sim/ground_truth/<cuerpo>`) y el instructor
  que mueve un objeto (`/sim/poner_cuerpo`).

**Ojo con los procesos huérfanos.** Si el simulador muere de golpe (kill -9), sus hijos (DDS, cámaras,
loco) se quedaban vivos y seguían publicando un rt/lowstate congelado: el programa veía estados
mezclados y un "error de seguimiento" falso de 15 grados (pasó el 2026-10-12). Desde entonces los
hijos reciben SIGTERM al morir el padre (PR_SET_PDEATHSIG). Si algo raro pasa:
`ps -eo pid,ppid,cmd | grep multiprocessing-fork`.

## Hito 1: espacio de trabajo y d

Ver `RESPUESTAS.md` (6.1.1, 6.1.2, elección de d) y `datos/alcance/`. En resumen, con la mesa a
92 cm y **d = 10 cm** (la mínima; cada centímetro más se pierde entero):
* **Alcance con la palma abajo:** el 100 % de la zona a 3 cm sobre la mesa. Cada mano cubre
  ~74 % y comparten una franja central del 48 %.
* **Palma de lado:** imposible a 3 cm, porque el canto de la mano toca el tablero; a 10 cm, 100 %.
* **Vista:** la cámara de color ve el 67 % de la zona y la profundidad nativa el 100 %. La
  percepción va sobre la profundidad nativa.
* **Bandejas:** cada mano llega a sus dos bandejas, también a la altura de agarre.
  **Ninguna llega a las del lado contrario**, así que en el nivel 3 hace falta la entrega de mano
  a mano en la franja central.
* **Cintura:** girarla no compensa, así que queda quieta.
