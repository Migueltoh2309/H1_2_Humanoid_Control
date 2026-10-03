# Videos del control bimanual (H1-2, MuJoCo)

Videos comparativos para presentaciones. Cada uno pone **lado a lado el
mismo escenario con dos modos de control**: a la izquierda sin la mejora y a
la derecha con ella. Todos corren exactamente el código de la evaluación
(`demos/evaluate_bimanual_avoidance.py`); no son animaciones aparte.

**Formato:** MP4 H.264, 1920×1080, 20 fps (tiempo real: un cuadro por ciclo
de control de 50 ms). En PowerPoint: *Insertar → Video → Este dispositivo*;
se reproduce sin instalar códecs.

## Qué se ve en cada panel

| Elemento | Significado |
|---|---|
| Esfera **roja** / **azul** | Objetivo de la mano izquierda / derecha |
| Cubos **morados** | Vóxeles de la cámara de profundidad cerca de los brazos: lo que el QP trata como obstáculo en ese instante |
| Puntos **verdes** | Camino planificado por la capa 3 (más tenues los ya recorridos) |
| Puntos **rojos** + "CHOQUE" | Contacto real entre mallas, detectado por MuJoCo |
| Texto amarillo | Último evento de las capas 2 y 3 (quién cede, cuándo se planifica…) |
| Abajo | Distancias mínimas reales y contador de pasos con choque / tareas completas |

En el **primer choque** de cualquiera de los dos paneles, el video se
**congela 1.5 s** con el aviso "PAUSA: primer choque". Sin esa pausa, el
choque dura unos 0.25 s y no se nota en una diapositiva. Es la única
alteración del tiempo real.

## Los videos

### 01 — Choque vs. evasión entre brazos (`01_choque_vs_evasion_brazos.mp4`)
- **Escenario `circles`**: las dos manos giran en círculos en contrafase que
  se cruzan en el centro.
- **Izquierda:** QP sin restricciones → los antebrazos chocan en cada vuelta.
- **Derecha:** capa 1 (QP + *velocity dampers* entre cápsulas) → nunca bajan
  de ~55 mm.
- *Texto sugerido:* "La distancia entre brazos entra al QP como restricción
  lineal `ḋ ≥ −ξ(d−dₛ)/(dᵢ−dₛ)`: la seguridad no depende de la tarea."

### 02 — Zona compartida (`02_zona_compartida_capa2.mp4`)
- **Escenario `shared`**: las dos manos deben ir al mismo punto (como tomar
  la mandarina del centro de la faja), quedarse 1 s y volver.
- **Izquierda:** solo capa 1 → se frenan mutuamente a distancia segura y
  **ninguna** completa la tarea (mínimo local).
- **Derecha:** capa 2 → detecta el bloqueo, el brazo derecho pasa primero,
  el izquierdo se retira y vuelve cuando el punto queda libre: **2/2 tareas**.
- *Texto sugerido:* "Coordinación maestro/esclavo: seguro y además
  productivo."

### 03 — Bandeja sobre la mano (`03_bandeja_capa3.mp4`)
- **Escenario `obstacle`**: una bandeja que el modelo del robot **no
  conoce** (solo la ve la cámara) está justo encima de la mano derecha, que
  debe subir por encima de ella.
- **Izquierda:** capas 1+2 → no chocan, pero la mano se traba debajo de la
  placa.
- **Derecha:** capas 1+2+3 → la capa 3 detecta que no avanza, planifica un
  camino con RRT-Connect (puntos verdes), lo recorre y llega.
- *Texto sugerido:* "Global + reactivo: el planificador encuentra el camino;
  el QP sigue garantizando que no haya choques mientras tanto."

### 04 — Oclusión y memoria de la cámara (`04_oclusion_memoria_camara.mp4`)
- **Escenario `table`**: las manos bajan hacia la faja, con objetivos por
  debajo de su superficie.
- **Izquierda:** cámara usando solo el último cuadro → la mano tapa a la
  cámara la zona que tiene debajo, "desaparece" el obstáculo y la mano choca
  con la faja.
- **Derecha:** memoria de vóxeles con borrado por espacio libre → lo ya
  visto se recuerda mientras está tapado: la mano se detiene sobre la faja.
- *Texto sugerido:* "La cámara del torso está detrás de las manos: sin
  memoria, la oclusión borra justo el obstáculo más peligroso."

### 05 — Sin cámara el robot no ve la bandeja (`05_sin_camara_atraviesa.mp4`)
- **Escenario `obstacle`**, solo capa 1.
- **Izquierda:** sin cámara → el modelo no sabe que la bandeja existe y el
  brazo la **atraviesa**.
- **Derecha:** con cámara → la evita (y se traba debajo; eso lo resuelve la
  capa 3, video 03).
- *Texto sugerido:* "La cinemática cubre brazo contra brazo; lo desconocido
  del entorno solo lo aporta la cámara."

## Regenerar

```bash
source ~/venvs/h12/bin/activate
cd ~/humanoid_ws/src/h1_2_algoritms
MUJOCO_GL=egl python3 demos/record_bimanual_videos.py            # los 5 (~5 min)
MUJOCO_GL=egl python3 demos/record_bimanual_videos.py --only 3   # uno solo
```

Los escenarios, modos, encuadres y títulos están en la lista `VIDEOS` al
principio de `demos/record_bimanual_videos.py`. La teoría detrás de cada
capa está en `docs/BIMANUAL_TEORIA.md`.
