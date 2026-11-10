# LfD de la faja con los comandos del robot real (sim2real)

Paso de la fase B (`LFD_BC.md`, perfil `suave`) al camino de comandos del H1-2. 2026-11-06.

## 1. Qué cambia entre el sim de la tesis y el robot

| | sim de la tesis (`bc_faja.py`) | robot (`h1_2_sim2real`: PC → agente → `rt/arm_sdk` + Modbus) |
|---|---|---|
| motor de brazo | PD **con compensación de gravedad ideal** | PD **puro**: `tau = tau_ff + kp·Δq + kd·Δdq`; la gravedad la pone el PC en `tau_ff` |
| ganancias | `DEFAULT_GAINS` (hombro kp 60 / kd 2; `suave`: kd ×4) | `h1_2_juntas.KP/KD`: brazos kp 100 / kd 2 (`--kp-escala`, `--kd-escala` nuevos) |
| velocidad | el QP limita a 2 rad/s | el agente rampa la consigna a `--vel-max` (1 rad/s por omisión) |
| límites | MJCF (codo hasta 3.18, muñeca pitch ±0.46) | `h1_2_juntas.LIMITES` − 0.05 (codo ≤ 1.55, muñeca pitch ≤ 0.30, yaw ≤ 0.96) |
| `tau_ff` | — | saturado a `--tau-max` (10 N·m) |
| frecuencia | QP a 20 Hz, referencia interpolada por paso | órdenes a 20 Hz, `rt/arm_sdk` a 100 Hz |
| manos | 6 rad por mano, servo de posición | Inspire 0–1000 (`[meñique, anular, medio, índice, pulgar flexión, pulgar rotación]`), recorrido 0.6 s a SPEED_SET 1000, escritas a ~30 Hz |

## 2. Implementación

* `h1_2_algoritms/aprendizaje/comandos_robot.py`. Carga la tabla maestra de `h1_2_sim2real/comun` (no la copia).
  * `AgenteEmulado` es `agente_sdk.py` sin red: `_rx` (recorte, `tau_max`), `_avanzar_objetivo` (rampa) y `_publicar` (el LowCmd de 12 a 26).
  * `ManosInspire` hace la conversión rad ↔ 0–1000 y la velocidad SPEED_SET.
  * `Registro` guarda el flujo de comandos en `.npz`.
  * `aplicar_limites_robot()` hace que el QP y el IK usen los límites del agente.
* `bc_faja.py --robot [--vel-max --tau-max --kd-robot --kp-robot --hz-agente --sin-gravedad --sin-dq --vel-mano --registro DIR]`.
  * En cada orden (20 Hz), el PC manda `q` del QP, `dq` del QP como feedforward y `g(q medida)` en `tau_ff`.
  * En cada ciclo del agente (100 Hz), el motor recibe el LowCmd.
  * Las métricas nuevas van al CSV: `vel_pedida_max`, `tau_ff_max`, `err_seguimiento_*`, `ordenes_recortadas`, `tau_ff_saturado`.
* `GraspServoing(lost_timeout=)`: si la pista se pierde durante el alcance, pasa a MISSED. En `--robot` vale 1 s; en el sim de la tesis queda desactivado para no cambiar resultados ya documentados.
* `agente_sdk.py --kp-escala/--kd-escala`: escalan solo los brazos (13–26). Por omisión valen 1, así que el comportamiento no cambia.
* `demos/reproducir_comandos.py`: manda un `.npz` grabado al agente por UDP, en lazo abierto, y mide el seguimiento.

## 3. Hallazgos

1. **Límites.** Con los límites del MJCF, el QP pedía la muñeca pitch hasta 0.46 y el codo hasta 2.85. El agente recortaba (hasta 3111 órdenes por episodio) y el `q_ref` del QP se separaba del brazo: un brazo nunca volvía a home. Con los límites del agente: 0–2 recortes por episodio y ningún brazo trabado.
2. **Pista perdida.** Al reintentar una fruta abortada, el tracker ya había borrado la pista y la FSM esperaba para siempre (`t_fin` = `t_max`). El mismo bug aparece en el sim de la tesis: hay episodios de 74 s en `dev_tesis.csv`. `lost_timeout` lo corrige.
3. **Gravedad.** `tau_ff` máximo ≈ 8–9 N·m en los hombros. Cabe en `--tau-max 10`, pero con poco margen: usar 15.
4. **Seguimiento extremo a extremo** (paso 2, abajo). El retardo PC → brazo es ~0.1 s. El error del motor contra la consigna del agente tiene p99 0.144 rad en el sim de `h1_2_sim2real`, frente a 0.136 en el emulador: **el emulador reproduce el camino real**.

### Resultados: 24 episodios de desarrollo (semillas 5000–5023, exigente, `suave`, 240 frutas)

| configuración | experto | BC |
|---|---|---|
| sim de la tesis | 192 (80%) | 190 (79%) |
| robot, agente por omisión (1 rad/s, kp 100 / kd 2, tau 10) | 194 (81%) | 195 (81%) |
| robot, `--vel-max 2 --tau-max 15` | 196 | 202 |
| robot, `--vel-max 2 --tau-max 15 --kd-escala 4` | **215 (90%)** | **215 (90%)** |

Comparaciones pareadas (Wilcoxon):

* Agente por omisión frente a tesis: igual (p = 0.80 / 0.51).
* kd ×4 frente al agente por omisión: +0.88 / +0.83 frutas por episodio (p = 0.002 / 0.005).
* La velocidad sola no cambia nada (p = 0.71 / 0.23).

Conclusión: **la política BC sobrevive al paso a los comandos del robot**. Como en `LFD_BC §6.7`, la mejora la da el amortiguamiento (kd), no la velocidad.

### Resultados finales: semillas 10000–10089 (90 episodios, exigente, `suave`, 900 frutas)

| configuración | experto | BC | episodios > 60 s |
|---|---|---|---|
| sim de la tesis (`evaluacion_exigente_suave.csv`) | 753 (83.7%) | 757 (84.1%) | 12 + 12 |
| sim de la tesis + `--lost-timeout 1` | 779 (86.6%) | 791 (87.9%) | 1 + 1 |
| robot, `--vel-max 2 --tau-max 15 --kd-robot 4` | **834 (92.7%)** | **837 (93.0%)** | 2 en total |

Diferencias pareadas por semilla, en frutas por episodio (IC 95% por bootstrap, Wilcoxon):

| comparación | experto | BC |
|---|---|---|
| `lost_timeout` − tesis (el arreglo del bug) | +0.29 [+0.13, +0.47], p = 0.002 | +0.38 [+0.20, +0.59], p = 0.0006 |
| robot kd4 − tesis + `lost_timeout` (**comparación justa**) | +0.61 [+0.38, +0.86], p = 1.5e-6 | +0.51 [+0.30, +0.70], p = 2.6e-6 |
| robot kd4 − tesis (efecto total) | +0.90 [+0.61, +1.17], p = 4e-8 | +0.89 [+0.63, +1.16], p = 1e-8 |

Robot kd4: 0–1 órdenes recortadas por episodio, sin saturación de `tau_ff` (máx 8.5 N·m), error de seguimiento p99 0.136 rad, distancia mínima entre brazos 31 mm (mediana), caídas 3 (experto) y 1 (BC).

**Las 4 caídas** (revisadas con `BC_DROP_DEBUG=1 BC_TRACE=fruta,t0,t1`; se reproducen igual al repetir la semilla):

* **experto s10029 y s10056: no son caídas.** La mano agarró la fruta solo por fricción (sin weld: la fruta quedó a más de 2 cm del punto de agarre esperado), con unos 25–35 N de fuerza. La llevó hasta el depósito y la soltó ahí. La fruta seguía marcada como `belt`, así que se contó como caída. Es un error de conteo: en realidad son 2 depósitos válidos, y además sin asistencia, que es justo lo que pasará en el robot. Corregido, el experto queda en 836 (92.9%).
* **s10070 (experto y BC, el mismo evento): caída real.** La mano derecha cierra sobre la fruta 2. La verificación táctil cierra LIFT a los ~0.6 s, cuando la muñeca solo subió 1–2 cm, y el depósito arranca con un barrido casi horizontal sobre la faja. Ese barrido empuja a la fruta 3, que estaba al lado (de x 0.301 a 0.263, en el borde de la faja), y la tira 0.4 s después. En el modo tesis, con la misma semilla, la mano también la roza (3 mm), pero no la tira. El riesgo es de diseño y existe en los dos modos: el depósito no sube antes de desplazarse. Posible arreglo: no cerrar LIFT por tacto hasta haber subido `lift_height`, o empezar el depósito con una subida vertical.

Conclusión: de 84% a 93%. Un tercio de la mejora viene del arreglo de la pista perdida y dos tercios de los comandos del robot con kd ×4. BC sigue igualando al experto. **Ojo**: el kd ×4 sobre kp 100 está validado solo en sim (el emulador coincide con el sim de `h1_2_sim2real`, §3.4); en el robot hay que calibrarlo (§5).

Archivos: `results/bc/robot/{dev_*,final_tesis_lost,final_robot_v2_kd4}.csv`, `results/bc/robot/comandos/comandos_bc_s10000{,_medido}.npz`.

### Mesa nueva (`u_table`, 2026-11-07)

Todo lo anterior se midió con la mesa vieja: tablero a 0.63 m, fruta a z = −0.164 en `torso_link`. Con la mesa real del laboratorio, el tablero queda a 0.77 m y la superficie de la faja a 0.9645; la fruta queda a z = −0.024 (+0.140 m). Esto exigió tres cambios:

1. **Percepción (0% → funciona).** `BELT_BOX` (tracker) y `GraspServoing.belt_box` tenían la z de la mesa vieja escrita a mano, así que se descartaban todas las detecciones. Ahora ambas salen de `GG.Z_FRUTA_FAJA`.
2. **Poses de tarea.** El depósito (`DEPOSIT_FRUIT`, `DEPOSIT_SHORT`) y la pose de mostrar (`SHOW_FRUIT`) estaban en z fija del torso. Con la faja más alta, el depósito quedaba 2.6 cm *debajo* de la fruta: el brazo volvía bajo y por detrás, y el alcance siguiente arrancaba fuera de la distribución de las demos de la DMP. Resultado: la muñeca se quedaba 11 cm corta, con la muñeca pitch en su límite ("la bajada no llegó al agarre", 24 de 47 abortos). Ahora esas poses suben con la faja (`GG.DZ_MESA`). La espera de `suave` ya dependía de la altura de la faja.
3. Los tests `test_alto_nivel` y `test_dmp` pasan a usar la altura de la faja.

En cinemática, la mesa nueva es mejor. Con el agarre de −20°, son alcanzables 25 de 25 posiciones de la faja (antes, 17 de 25). Experto, 12 episodios de desarrollo (5000–5011): mesa vieja 96/120 → mesa nueva sin ajustar 68/120 → **ajustada 112/120 (93%)**, sin fallos de agarre. Los tiempos de las habilidades casi no cambian, así que `TIMING` de `suave` sigue valiendo.

Con la mesa nueva se rehízo el dataset y la BC (40 episodios del experto + 2 rondas de DAgger) y la evaluación final: `results/bc/mesa_u/`. Los datos y la red de la mesa vieja quedaron en `results/bc/mesa_vieja/`.

4. **Depósito a −45° (antes −60°).** En modo `--robot`, el depósito elevado a −60° no tiene IK dentro de los límites del agente. La IK caía en una rama lejana: shoulder yaw −2.58, a 2 rad de home. Al ir y volver, el brazo barría frutas de la faja: 86–87% de éxito y 35 caídas. A −45° sí tiene solución; además, `ArmExecutor` ahora avisa (`warnings.warn`) si el depósito no tiene IK.

Evaluación final de la mesa nueva (semillas 10000–10089, 900 frutas, `--lost-timeout 1`, depósito −45°: `mesa_u/final_*_dep45.csv`):

| modo | experto | BC | caídas | Δ BC frente a la mesa vieja (frutas/ep, IC95, Wilcoxon) |
|---|---|---|---|---|
| tesis | 834 (92.7%) | 834 (92.7%) | 0 | +0.48 [+0.30, +0.67], p = 4e-6 (vieja: 791) |
| robot kd×4 | 876 (97.3%) | **878 (97.6%)** | 0 | +0.46 [+0.28, +0.64], p = 8e-6 (vieja: 837) |

* Con la mesa nueva, ambos modos mejoran de forma significativa, también el experto: tesis +0.61 frutas/ep, robot +0.47. Las pérdidas en la faja bajan de 62 a 22 (robot, BC).
* El depósito a −45° frente al de −60° en la mesa nueva: en modo robot, +1.1 frutas/ep (p ≈ 1e-10), con 0 caídas (antes 22). En modo tesis el efecto es nulo para la BC (p = 0.55).
* La BC sigue empatada con el experto (Δ +0.00 tesis, +0.02 robot; n.s.). El modo robot (kd×4) sigue siendo mejor que el de la tesis: +0.49 frutas/ep.
* La BC no se reentrenó tras el cambio a −45°: el depósito es una habilidad y la red solo decide cuándo usarla.

**Barrido de velocidad de la faja** (2026-11-09). Semillas de desarrollo 5000–5023 (240 frutas por política). Se usa el modo robot kd×4, una llegada cada 2 s y la BC del exigente. Los CSV están en `results/bc/mesa_u/vel/`.

| escenario | v (m/s) | experto | BC | frutas/min (BC) |
|---|---|---|---|---|
| exigente | 0.05–0.08 | 93.3% | 93.8% | 14.6 |
| `rapido` | 0.08–0.10 | 76.7% | 78.8% | 14.4 |
| `muy_rapido` | 0.10–0.12 | 67.9% | 66.2% | 13.0 |
| `extremo` | 0.12–0.15 | 57.5% | 54.6% | 11.6 |

Subir la velocidad de la faja no aumenta las frutas por minuto: se mantienen en ~13–14.6, limitadas por el ciclo de cada brazo (~8–9 s). Las pérdidas son frutas que pasan por la ventana de alcance sin que haya un brazo libre, no agarres fallidos (0 caídas; los fallos de agarre suben poco, de 2–4 a 10–11). Entre la BC y el experto no hay diferencia significativa a ninguna velocidad (p ≥ 0.2). Para procesar más fruta hay que acortar el ciclo de las habilidades, no subir la faja.
**Acortar el ciclo** (2026-11-09). Las variantes son combinables sobre `suave` (`--perfil suave_D`, `suave_DR`, ...; ver `_VARIANTES` en `bc_faja.py`):

* **D:** depósito 10 cm más cerca, en y = 0.30 y a −30°, y soltar en movimiento (abre a 0.30 rad, en 0.3 s).
* **R:** volver sin asentarse (libre a 0.25 rad y |q̇| < 1.5 rad/s).
* **A:** alcance con intercepción y empezar hasta 25 cm antes de la ventana (`alto_nivel.TRACK_BAND`).

Los tiempos se recalibraron con `calibrar` (12 episodios) y se pasaron con `--timing`. D ahorra 0.3 s al depositar y R 0.15 s al volver; A no cambia la duración del alcance. Modo robot kd×4, semillas 5000–5023, 240 frutas; CSV en `results/bc/mesa_u/vel/ciclo_*.csv`. Valores como experto / BC:

| perfil | exigente (0.05–0.08) | rápido (0.08–0.10) |
|---|---|---|
| suave | 93 / 94% | 77 / 79% |
| D | 98 / 98% | 82 / 85% |
| R | 97 / 97% | 79 / 81% |
| A | 95 / 94% | 73 / 73% |
| **DR** | **98 / 98%** | **84 / 89%** |
| DA | 96 / 93% | 76 / 78% |
| RA | 91 / 91% | 74 / 76% |
| DRA | 94 / 95% | 79 / 79% |

* **DR frente a suave** (pareado, Wilcoxon): exigente +0.50 / +0.46 fr/ep (p = 0.003 / 0.005); rápido +0.71 / +1.00 (p < 0.001 / 0.001). Ninguna caída.
* **R solo** ayuda en el exigente (BC p = 0.01), pero no en el rápido (p ≥ 0.3); sumado a D, sí.
* **A empeora todas las combinaciones.** Empezar con la fruta lejos multiplica los agarres fallidos (6 → 17–22 en el rápido). Confirma el resultado negativo de `reach_intercept` (LFD_DMP §4.4): haría falta una DMP entrenada con demos de intercepción.
* **La BC se usó sin reentrenar**, con la red de `suave` y los tiempos de cada perfil.

**Final DR** (2026-11-09; BC reentrenada con `suave_DR` en modo robot: 40 episodios del experto + 2 rondas de DAgger, validación 99.1%; semillas 10000–10089, 900 frutas; `mesa_u/final_robot_DR*.csv`, `final_robot_suave_rapido.csv`):

| escenario | suave (experto / BC) | DR (experto / BC) | Δ BC (fr/ep, IC95, Wilcoxon) |
|---|---|---|---|
| exigente | 876 / 878 (97.6%) | 880 / 882 (98.0%) | +0.04, p = 0.49 (techo) |
| rápido | 722 / 746 (82.9%) | 773 / **819 (91.0%)** | +0.81 [+0.61, +1.01], p = 1.5e-9 |

* Con la faja actual DR ya no mejora: suave estaba en el techo (los 24 episodios de desarrollo eran más difíciles).
* Con la faja rápida, DR gana +8 puntos con la BC (experto +0.57 fr/ep, p = 4e-7).
* En la faja rápida, la BC supera al experto: con DR, +0.51 fr/ep (p = 3e-6); con suave, +0.27 (p = 1e-3). El experto planifica con un modelo de tiempos fijo, y la BC aprende de los DAgger cuándo vale la pena intentar.
* Para el robot real queda por ver que soltar en movimiento (D) deje la fruta en el contenedor.


## 4. Escalera hacia el robot

| paso | qué | estado |
|---|---|---|
| 1 | comandos del robot emulados dentro del sim de la tesis (`--robot`) | hecho |
| 2 | flujo grabado → `agente_sdk.py` real → sim de `h1_2_sim2real` (lazo abierto) | hecho (25 s, BC s10000) |
| 3 | lo mismo en el robot **colgado**: `--lento 3`, agente `--vel-max 0.5`, primero sin manos | pendiente |
| 4 | robot de pie (FSM 201), arm_sdk, velocidad real, sin faja | pendiente |
| 5 | lazo cerrado: percepción de la D435 real + QP + habilidades en el PC, órdenes al agente | pendiente: falta la D435 en ROS 2 y un nodo que corra `run_episode` contra `/joint_states` |

Comandos del paso 2 (y del 3, cambiando la IP):

```bash
~/humanoid_ws/src/h1_2_sim2real/scripts/sim.sh robot use_viewer:=true            # terminal 1
python3 robot/agente_sdk.py --sim --vel-max 2 --tau-max 15 --kd-escala 4          # terminal 2 (en h1_2_sim2real)
python3 demos/reproducir_comandos.py results/bc/robot/comandos/comandos_bc_s10000.npz --hasta 25   # terminal 3
# robot colgado:
#   agente_robot.sh lowcmd ... --vel-max 0.5 --kd-escala 4 ; reproducir_comandos.py ... --ip 192.168.0.143 --real --lento 3 --sin-manos
```

## 5. Lo que el sim todavía no modela (riesgos para el robot)

* **Asistencia de agarre (weld).** El robot no la tiene. En el reto de manipulación, con el agarre sin asistencia, una tolerancia de ~5 mm funcionaba. Es el mayor riesgo.
* **Tacto.** `touch` usa la fuerza de contacto en N (`grip_force`, umbral 1 N). En el robot, la fuerza llega por FORCE_ACT en gf por dedo, por Modbus a ~30 Hz. Hay que definir la conversión y el umbral con la mano real.
* **kd ×4 en el robot.** El motor real tiene fricción y una inercia reflejada distintas. Hay que calibrar kd con el robot colgado: mirar el rebote al volver a home.
* La percepción y la latencia de la cámara real, la faja real y la deriva del torso (IMU; en el reto de manipulación, 4 mm y 0.19°).
