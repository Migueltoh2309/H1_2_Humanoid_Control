# Respuestas a la sección 6 (con datos)

Números del **simulador** salvo que se diga otra cosa. Lo que solo se puede medir en el robot
(la reacción del cuerpo, la fuerza real frente a FORCE_SET, la pose real de la cámara) se marca
como pendiente, junto con la herramienta que lo medirá.

> **Mesa a 92 cm (2026-10-13).** El reto dice 86 cm. Con 86, la palma abajo a 3 cm solo llega
> al 80 % de la zona y no llega a B ni a D; con 92 llega al 100 % y a las cuatro bandejas (tabla
> del README). En el robot real se consigue con calzos de 6 cm. Todas las cifras de este
> documento se rehicieron con 92 cm el 2026-10-13. Las de 86 cm quedan en `datos/alcance/mesa86/`
> y en el historial de git. Con 92 cm, los brazos empiezan en la pose de observación: en la
> postura de pie los dedos se apoyan en el tablero.

## 6.1 Espacio de trabajo y calibración

### 6.1.1 ¿Qué parte de la mesa alcanza cada mano, con la palma abajo y de lado? ¿Cómo cambia con d?

**Cómo se obtuvo.** `herramientas/mapa_alcance.py`. Rejilla de 2 cm sobre el plano de la mesa
y, para cada punto y cada mano, la IK con la cadena de la tesis. La FK de la tesis está
verificada contra el MJCF: error < 1e-6 m (`tests/test_cinematica.py`). Se prueban 39
orientaciones con la palma abajo (dedos con elevación 0, -20 y -40 grados; azimut de -90 a 90
cada 15) y 14 con la palma de lado. Un punto cuenta como alcanzado si cumple a la vez:
- la IK converge a 2 mm y 2 grados dentro de los límites del URDF (muñeca ±26.5 grados), con el
  alabeo del hombro hacia dentro ≤ 0.10 rad;
- el antebrazo y la mano quedan a ≥ 1 cm del torso;
- ninguna cápsula del brazo, ni los dedos estirados ni el pulgar abierto, baja de 5 mm sobre el
  tablero en la parte que queda encima de la mesa.

El robot está de pie con el torso a 1.012 m (medido en el simulador) y el pecho en x = 0.09 m.
Datos y figuras en `datos/alcance/` (`mapa_d10.png`, `curvas_d.png` y sus variantes `_h10` y
`_cintura20`). Otra altura de mesa: `--mesa 0.86`.

**Zona de recogida alcanzada con la palma abajo** (dedos a ≤ 20 grados de la horizontal), mesa a 92 cm:

| d (cm) | agarre a 3 cm sobre la mesa | a 10 cm | palma de lado a 3 cm | a 10 cm |
|---|---|---|---|---|
| 10 | **100 %** (izq 74, der 75) | **100 %** | 0 % | 100 % |
| 15 | 89 % | 100 % | 0 % | 100 % |
| 20 | 71 % | 97 % | 0 % | 92 % |
| 25 | 50 % | 74 % | 0 % | 69 % |
| 30 | 29 % | 55 % | 0 % | 50 % |
| 35 | 12 % | 36 % | 0 % | 31 % |

(Con la mesa a 86 cm y d = 10: 80 % a 3 cm, 100 % a 10 cm y 96 % de lado a 10 cm.)

* **El límite es la distancia, no la orientación.** Con la mesa 50 cm por debajo del hombro, la
  palma llega hasta x ≈ 0.49 m delante de la pelvis, tanto a 3 cm como a 10 cm. Cada centímetro
  que se aleja la mesa se pierde entero: **d = 10 cm (la mínima) es la mejor**.
* **Palma de lado: imposible a 3 cm.** La mano mide 8 cm de ancho y, vertical, su canto entra
  en el tablero. A 10 cm (el vaso cogido por su mitad alta, o un objeto ya levantado) sí se
  puede: 100 %.
* **Cada mano cubre su mitad y una franja central común** del 48 % de la zona (con 86 cm, el 20 %).
  Es donde se hace la entrega de mano a mano.
* **Guiñada del agarre.** En el centro de la zona, la mano izquierda solo puede coger con los dedos
  entre -90 y -30 grados (hacia el centro) y la derecha entre 30 y 90, a 3 y a 10 cm. La muñeca
  (±26.5 grados de cabeceo) no da para un rango de 90 grados con una sola mano. Para el nivel 4
  (girar 90 grados) hará falta cambiar de agarre o de mano.

### Bandejas: ¿llega una mano a las del lado contrario? (sec. 4)

Con d = 10 cm, al centro de cada bandeja:

| Bandeja | Mano propia, a 3 cm | Mano propia, a 10 cm (suelta) | Mano contraria (3 y 10 cm) | La ve el color |
|---|---|---|---|---|
| A (izq, cerca) | sí (azimut -75..30) | sí (-90..30) | **no** | no |
| B (izq, lejos) | sí (-15..45) | sí (-30..30) | **no** | sí |
| C (der, cerca) | sí (-30..75) | sí (-30..90) | **no** | no |
| D (der, lejos) | sí (-45..15) | sí (-30..30) | **no** | sí |

**Ninguna mano alcanza las bandejas del lado contrario**, a ninguna altura ni con ninguna d.
Para el nivel 3 (al menos dos objetos en el lado contrario a su bandeja), el mapa **exige la
entrega de mano a mano** en la franja central que alcanzan las dos.

**¿Ayuda girar la cintura (índice 12)?** Con 20 grados (`--cintura 0.35`), la mano derecha pasa
del 75 % al 97 % de la zona, pero la izquierda baja del 74 % al 42 %. En total queda en el 99 %,
frente al 100 % sin girar. Las bandejas contrarias siguen sin alcanzarse, y girar la cintura gira
también la cámara y el cuerpo. Decisión: **cintura quieta en su postura medida** (base de la
respuesta a 6.3.6).

### 6.1.2 ¿Qué parte de la mesa ve la cámara? ¿Es fiable la profundidad a esa distancia?

**Cómo se obtuvo.** Proyección con la pose nominal del URDF (D435 a 1.70 m del suelo, 50.8
grados bajo la horizontal) y los intrínsecos del reto, y comprobación con las imágenes del
simulador (`herramientas/captura_camara.py`, `datos/capturas/mundo92_*`).

* **El color** (56 x 43 grados) ve con d = 10 cm el **67 %** de la zona de recogida (la parte
  lejana) y no ve las bandejas A ni C.
* **La profundidad nativa** (IR1, ~80 x 65 grados) ve la zona y las cuatro bandejas enteras
  (100 %). **Conclusión: la percepción se apoya en la profundidad nativa**. Con ella se separan
  los objetos del plano de la mesa y se reconocen por su geometría: los cuatro son distinguibles
  por la altura y la forma de la nube. El color solo confirma la identidad donde llega.
* La mesa queda a ~0.8-0.95 m de la cámara, dentro del rango de la D435 (mínimo ~0.28 m). Su
  error sobre el tablero vacío (ruido y sesgo frente a la regla) **se mide en el robot**: en el
  simulador la profundidad es exacta.
* **Las manos en reposo tapan la vista y, a 92 cm, tocan la mesa.** En la postura de pie las
  puntas de los dedos quedan a ~1 cm del tablero, encima de las bandejas cercanas. Con la
  gravedad se apoyan en él. La **pose de observación** (hombros abiertos 0.15 rad) deja las manos
  a 4 cm sobre la mesa, a los lados y fuera de la vista (0 % del tablero tapado). En las escenas
  el robot empieza en ella; en el robot, los brazos van a ella antes de acercar la mesa.

### 6.1.3 ¿Cómo se obtiene la transformación cámara-torso? ¿Cuántas poses y qué residuo?

**Cómo.** Con `herramientas/calibrar.py` (`scripts/hito2_sim.sh calibrar`) y el ArUco de 4 cm del
dorso de cada mano. Cada mano pasa por 10 poses quietas, generadas por `herramientas/poses.py`
contra el modelo:
- alcanzables y sin choques;
- marcador visible y de frente (a < 50 grados);
- rotaciones alrededor de ejes distintos, que es lo que hace falta para que AX = YB tenga solución única.

En cada pose se toman 6 fotogramas de IR1 con el proyector apagado. Las juntas que se usan son las
medidas, promediadas. El ajuste (`manip/calibracion.py`) busca a la vez:
- X = T_torso_cam;
- dónde quedó pegado cada marcador, Y_izq e Y_der.

Minimiza el error de reproyección de las 4 esquinas en píxeles, que es lo que se mide de verdad.
Arranca de la pose nominal del URDF. Se calibra el marco de IR1 porque ahí vive la profundidad
nativa que usa la percepción.

**Resultado en el simulador** (20 poses; la cámara del simulador está desviada a propósito ~2
grados y ~1 cm del URDF):

| | valor |
|---|---|
| reproyección | 0.45 px rms (máx 0.85) |
| centro del marcador: cadena frente a cámara (con la profundidad del píxel) | 1.3 mm medio, 3.0 máx |
| distancia de la calibración a la pose nominal del URDF | 12.3 mm y 2.28 grados |
| **error frente a la pose verdadera de la cámara** (solo se puede medir en el simulador) | **0.6 mm y 0.04 grados** |

Con las juntas exactas, 6 poses ya bastarían (prueba sintética en `tests/test_calibracion.py`). Se usan
20 para promediar el error del brazo y poder descartar poses malas.

### 6.1.4 Con el brazo ~5 grados corto, ¿qué se calibra si se usan las consignas?

El simulador reproduce el fenómeno: con kp 100 y sin compensar la gravedad, la diferencia entre
consigna y junta medida llega a **2.0-5.5 grados** según la pose. Si se calibra con las consignas:
- la reproyección sube de 0.45 a **5.8 px rms**;
- el centro del marcador queda a 39 mm, frente a 1.3;
- **la cámara calibrada se va 58 mm y 5.6 grados**.

Es decir, se calibraría la cámara contra donde el modelo cree que está la mano, no contra donde
está. **Hay que usar las juntas medidas.** Y el residuo lo delata: si el error del brazo cambia de
pose a pose, no se puede absorber y el residuo sube (prueba en `test_calibracion.py`: un error de 2
grados en las juntas da más de 2 px).

### 6.1.5 ¿Cuánto se mueve la cámara con el robot quieto?

`calibrar.py --balanceo 15` mira durante 15 s los ArUco fijos del tablero, con el centro medido
por la profundidad del píxel. **En el simulador la desviación típica es de 0.2-0.5 mm**. No es
una medida del robot, porque las piernas son rígidas y no hay política de equilibrio. En el
robot se mide con el mismo comando.

**Validación de la calibración con los ArUco del tablero**, frente a la geometría de la config:
- **en la pose de observación** (la que usa la percepción): **5.0 y 5.7 mm**. Es el torso, no la
  calibración. La config supone el torso en x = 0 y horizontal, y en el simulador está 4 mm
  adelantado e inclinado 0.19 grados. Con la pose real del torso el error baja a **≤ 1.3 mm**;
- en las poses de calibración, con un brazo extendido: 15.6 y 20.3 mm. El brazo extendido
  inclina el torso hasta ~0.8 grados, que son ~9 mm a 0.65 m.

**Consecuencia:** el planificador no usa la altura de la mesa de la config, sino el **plano de la
mesa medido** por la percepción en el sitio de cada objeto, y la percepción se hace siempre en la
pose de observación.

## 6.2 Percepción

**Cómo se evaluó.** Con `herramientas/evaluar_percepcion.py`:
- 9 escenas de la estación (niveles 2-4, semillas 0-2) con los brazos en la pose de observación;
- grabadas como conjunto de datos y leídas con `FuenteCarpeta`, como en vivo;
- la calibración del hito 2 (no la pose verdadera);
- comparación con la verdad de terreno;
- sin ruido, y con el modelo de ruido de la D435 de la tesis (σ ≈ 4 mm a 1 m, correlado, con
  huecos en los bordes), fusionando 8 fotogramas.

| | sin ruido | ruido D435, 8 fotogramas |
|---|---|---|
| objetos reconocidos | 36/36 | 36/36 |
| error en planta medio / máx (mm) | cubo 0.4/1.3, vaso 1.3/2.1, pelota 0.9/1.0, prisma 0.3/0.7 | cubo 2.9/3.4, vaso 3.7/4.6, pelota 1.1/1.7, prisma 2.9/3.8 |
| confianza mínima | 0.69 | 0.54 |
| error en altura | ≤ 1.5 mm | ≤ 1.5 mm |
| tiempo por fotograma (PC, un hilo) | ~43 ms | ~150 ms con la fusión |

La confianza es la del **margen de la asignación** (cuánto empeora el coste total óptimo si el grupo no
fuera ese objeto: cada objeto aparece una vez) por la **fracción visible** de la huella (sin rebaja
desde el 70 %). Así el prisma, que en altura y huella se parece al vaso, sale con confianza aunque el
vaso le dispute la fila, y un objeto tapado por la mano no la tiene aunque no haya otro candidato. Con
la mano encima quedan 2 de 36 casos con 11-16 mm y confianza 0.35-0.46: por eso se percibe siempre en
la pose de observación.

**6.2.1** Profundidad. El plano de la mesa (RANSAC) separa los objetos sin depender de la luz; las
sombras del brazo no tienen altura. El color no hace falta: la cámara de color no ve la parte
cercana de la zona.

**6.2.2** Por geometría:
- altura, largo y ancho;
- "llenado" del rectángulo mínimo (0.9 una caja, pi/4 un círculo);
- "cúpula": percentil 97 menos percentil 70 de la altura, ~0 en una cara plana y 0.163 R en una esfera;
- residuo del ajuste de esfera.

Asignación húngara: cada objeto una vez. **Cubo y pelota se parecen** (6 y 6.7 cm de alto y de
huella). Con un solo fotograma ruidoso se confundían; la cúpula y el residuo de esfera, con la
fusión de 8 fotogramas, los separan.

**6.2.3** Según la forma:
- caja: rectángulo mínimo de la cara de arriba (guiñada módulo 90);
- vaso: círculo de la cara de arriba;
- pelota: ajuste de esfera de radio conocido de la tesis (locate_sphere);
- rotulador: eje largo (módulo 180).

La z sale del plano medido más la mitad de la altura conocida.

**6.2.4** Autofiltro con las cápsulas de la tesis (brazos, dedos, pulgar, torso) y las juntas
medidas del instante del fotograma; además, las paredes de las bandejas y lo que cae fuera del
tablero. En la pose de observación los brazos no tapan nada (0 % del tablero).

**6.2.5** Con la mano a 8 cm encima del objeto, **solo se ve el 33 % de su huella**. Se detecta en
9/18 casos con 32 mm de error medio y **confianza ≈ 0**: la percepción sabe que esa medida no vale.
Con la mesa a 92 cm la mano tapa también a los vecinos. De los demás objetos de esas escenas,
la mayoría siguen bien o salen con confianza ≈ 0, pero tres parcialmente tapados (vaso y
rotulador) dan 13-15 mm con confianza 0.4-0.6. Decisión: **se mide antes del descenso**, con el
robot quieto en observación, y se confía en la calibración. No se sigue al objeto durante el
descenso (tampoco hay cámaras de muñeca). Tras elevar se vuelve a observar para verificar.

**6.2.6** `percepcion.cambios()` compara dos listas: más de 2 cm o 10 grados es "se movió". Se
observa de nuevo antes de cada agarre (nivel 3). Tiempo: ~43 ms por fotograma, ~150 ms con 8
fusionados, en el PC; **en el PC2 está por medir**.

## 6.3 Brazo, cuerpo y trayectorias

**6.3.1** `herramientas/curva_pasos.py` hace el mismo movimiento (observación ⇄ pose de alcance)
con duraciones de mayor a menor: 30, 20, 10, 8 s; 5 s solo con `--permitir-5s`, porque la sec.
10 no deja bajar de 8. Mide giro e inclinación de la IMU, pasos (en el robot los cuenta el
operador con las huellas; en el simulador, los pies) y el centro de masas con el modelo.

En el simulador, con las piernas rígidas, no hay pasos ni giro (0.01 grados; inclinación
0.77 grados con el brazo extendido). Pero da la parte física de la respuesta:

| T (s) | 30 | 20 | 10 | 8 |
|---|---|---|---|---|
| desplazamiento del centro de masas | 1.9 mm | 1.9 mm | 1.9 mm | 1.9 mm |
| velocidad articular pico (rad/s) | 0.07 | 0.10 | 0.20 | 0.25 |
| aceleración pico (rad/s²) | 0.007 | 0.015 | 0.062 | 0.096 |

El desplazamiento del centro de masas no depende de la duración. Si en el robot los movimientos
cortos dan pasos y los largos no, lo que cuenta es la dinámica: la aceleración, que crece como
1/T², frente a la velocidad, que crece como 1/T. La curva de pasos que vale es la del robot.

**6.3.2** Presupuesto, con `herramientas/probar_plan.py`: cada objeto son 7 tramos (aproximar,
descender, elevar, transportar, bajar, retirar, volver) más cerrar y abrir la mano. Planes
válidos en el banco de escenas: **21/36**. Los que fallan son, sobre todo, entregas sin sitio
común libre (6, casi todas del rotulador) y caminos que el RRT no encuentra.

| duración por tramo | por objeto | 4 objetos | con cambio de mano |
|---|---|---|---|
| 8 s (la mínima) | **62 s** (83 s de media con entregas) | ~5.5 min | 124 s por objeto |
| 16 s (primeras tiradas, el doble) | ~2 min | ~11 min | |

**6.3.5** **Trayectorias en articulares.** Los tránsitos usan el RRT-Connect de la tesis con la mesa,
las bandejas y los demás objetos como nube de obstáculos. El descenso final es una recta
cartesiana, resuelta con la IK punto a punto. Altura de seguridad: el punto más bajo de la mano
(dedos y pulgar incluidos) a ≥ 10 cm sobre la mesa en todo, salvo el descenso final. Se garantiza
comprobando cada punto contra el MJCF de la estación vacía: holgura con la mesa (vértices de las
mallas contra las cajas: exacta), cuerpo, otro brazo y altura de la mano.

**6.3.6** **La cintura recibe su postura medida** al empezar y no se mueve. Girarla 20 grados no
gana zona (99 % frente a 100 %) ni acerca las bandejas contrarias, y gira la cámara y el cuerpo.

**6.3.4 El error de seguimiento: con un modelo y cerrando el lazo, sin subir kp.**
(`manip/compensacion.py`). Medido al final del descenso de un agarre: sin compensar, la muñeca queda a
**13-16 mm** de la consigna (el brazo cede 1-2 grados). Tres capas:
1. **Modelo del error:** la consigna lleva g(q)/kp (par de gravedad del MJCF y la carga del objeto
   cogido): la muñeca a **≤ 1 mm**.
2. **Lazo con los encoders** en las paradas (preagarre): **≤ 0.2 mm**. En el robot, donde además tira la
   política, es la capa que más trabajará.
3. **El cuerpo:** al extender el brazo el robot se inclina 0.3-2.4 grados y el objeto, medido desde la
   observación en el marco del torso, se desplaza hasta **2.4 cm** en ese marco. Con la IMU y el cuerpo como
   sólido que gira alrededor de un pivote efectivo (simulador: 45 cm sobre el suelo, ajustado con 9
   agarres) la posición se corrige con **≤ 0.1 mm** y el descenso se rehace en el preagarre. En el robot
   el pivote se mide en el hito 2.
Solo la capa 1 llevó el cubo de 0/3 a 3/3.

## 6.4 Mano, contacto y suelta

**6.4.2** `herramientas/fuerza_mano.py`. En el robot, el índice aprieta sobre la báscula. En el
simulador, la "báscula" es la fuerza de contacto de MuJoCo. FORCE_ACT = FORCE_SET por
construcción, y **al objeto le llega un 69-75 %** (138/212/287/363/440/596 g con FORCE_SET
200-800). FORCE_ACT es una medida del lado del motor; lo que se pasa o se queda corto el
firmware real solo lo dice la báscula. La tabla por objeto (`config/agarres.yaml`) usa ese ~73 %
para fijar FORCE_SET: hacen falta ≥ m·g·2/(2·μ) por lado.

**6.4.3** `manip/mano.py` (`Mano.agarre()`): un dedo que se para lejos de 0 con fuerza tiene algo
delante, y uno que llega a 0 está vacío. Hace falta el pulgar y al menos un dedo. Lo reproduce el
simulador:
- con un objeto, los dedos se paran en 400-500 con FORCE_ACT = FORCE_SET y STATUS 3;
- sin nada, el dedo llega a 0, con fuerza ~25 gf y STATUS 2.

**6.4.1 ¿Qué agarre para cada objeto?** `herramientas/barrido_agarre.py` (el robot simulado completo, en
paralelo) y `herramientas/banco_mano.py` (la mano suelta). Sin asistencia, mesa a 92 cm, 3 semillas:

| objeto | agarre | 3 semillas | 6 semillas, error de posición 0 / 5 mm / 1 cm / 2 cm |
|---|---|---|---|
| cubo 6 cm | desde arriba: pulgar de poste en la cara cercana, dedos por la lejana (avance 0.23) | **3/3** | – / – / 6/6 / 4/6 |
| vaso 6 cm, 250 g | igual, por su parte alta (centro de agarre a 6.5 cm, avance 0.22) | **3/3** | 5/6 / 5/6 / 3/6 / 1/6 |
| pelota de tenis | desde arriba con la palma APOYADA (la enjaula antes de que ruede), dedos 20 grados abajo | **3/3** | 5/6 / 4/6 / 1/6 / 0/6 |
| prisma 4x4x10 de pie | como el vaso (avance 0.21, a 7 cm) | **3/3** | 6/6 / 6/6 / 3/6 / 3/6 |

**Tolerancia** (lo que pregunta 6.4.1): el cubo aguanta 1 cm; vaso, pelota y prisma aguantan el error
de la percepción con ruido (≤ 5 mm) pero no 1 cm. Encaja con la cadena medida: percepción ≤ 4.6 mm +
calibración 0.6 mm + brazo ≤ 1 mm con la compensación. El rotulador del ejemplo se sustituyó por el
prisma: tumbado mide 3.2 cm y el reto pide objetos de 4 a 15 cm de alto.

Cómo se llegó (cada paso medido):
- **La mano de la Inspire:** con la rotación 0 el pulgar no se cierra contra los dedos en horizontal:
  sale perpendicular a la palma (9-10 cm abierto) y vuelve hacia ella, a 17-19 cm de la muñeca. Con la
  palma abajo es un poste vertical; por eso sirven objetos de ~6 cm entre el poste y las puntas
  (26.8 cm). El vaso de 7.2 cm no cabía: se usa uno de 6 cm (el reto deja 3-8 cm).
- **Agarre lateral (palma vertical):** la muñeca (±0.46 rad) no deja bajar la mano por debajo de ~10 cm
  con los dedos horizontales; con ellos 15 grados abajo, o el meñique toca la mesa o el pulgar queda sobre
  la boca del vaso. Descartado para estos objetos. (La orientación lateral de la tesis rodaba la mano
  con elevación y azimut a la vez; corregida en `cinematica.rotacion_agarre`.)
- **El brazo y el cuerpo** eran la causa principal de los fallos: ver 6.3.4.
- **El contacto del simulador:** cono de fricción elíptico e impratio 10 (MuJoCo, para agarrar) y la
  pelota blanda (solref 0.015).
- **Al soltar** la mano vuelve a la preforma, no se abre entera: el pulgar abierto baja 10 cm.

## Elección de d (entregable del hito 1)

**d = 10 cm**, el mínimo permitido. Es la que más zona alcanza (100 % con la palma abajo a 3 cm,
con la mesa a 92 cm; con d = 15 cm baja al 89 %), y la profundidad nativa sigue viendo el 100 %
de la zona. Lo que se pierde es la vista en color de la parte cercana, y lo cubre la profundidad
nativa.
