# Referencias: bimanualidad y evasión de colisiones (formato IEEE)

Referencias que sustentan el controlador bimanual del H1-2
(`bimanual_avoidance.py`, `bimanual_coordinator.py`, `collision_model.py`,
`depth_obstacles.py`). Están agrupadas por tema y numeradas de forma
correlativa, para citarlas tal cual en la tesis. Debajo de cada una se indica
**qué se tomó de ahí** y **dónde aparece en el código o en la teoría**
([`BIMANUAL_TEORIA.md`](BIMANUAL_TEORIA.md)).

Los datos bibliográficos (volumen, número, páginas, DOI) se verificaron en
IEEE Xplore, las editoriales, arXiv, dblp o los repositorios de los autores
(búsqueda del 25/09/2026). Cuando no se pudo confirmar un DOI, se omite en
vez de adivinarlo y se da el identificador de arXiv.

---

## A. Evasión reactiva de colisiones con restricciones de velocidad

[1] B. Faverjon and P. Tournassoud, "A local based approach for path
planning of manipulators with a high number of degrees of freedom," in
*Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, Raleigh, NC, USA, 1987,
vol. 4, pp. 1152–1159, doi: 10.1109/ROBOT.1987.1087982.
> Origen del **velocity damper** `ḋ ≥ −ξ(d−d_s)/(d_i−d_s)`: la anti-colisión
> como restricción separada de la tarea. → `collision_model._damper_rhs`,
> teoría §6.1.

[2] O. Stasse, A. Escande, N. Mansard, S. Miossec, P. Evrard, and
A. Kheddar, "Real-time (self)-collision avoidance task on a HRP-2 humanoid
robot," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, Pasadena, CA, USA,
2008, pp. 3200–3205, doi: 10.1109/ROBOT.2008.4543698.
> Autocolisión y colisión con el entorno como restricciones de un QP en
> velocidad, en un humanoide real (HRP-2); necesidad de un gradiente de
> distancia continuo. → Estructura de la capa 1, teoría §3.2 y §6.

[3] F. Kanehiro, F. Lamiraux, O. Kanoun, E. Yoshida, and J.-P. Laumond, "A
local collision avoidance method for non-strictly convex polyhedra," in
*Proc. Robotics: Science and Systems IV*, Zurich, Switzerland, 2008,
doi: 10.15607/RSS.2008.IV.020.
> Velocity damper aplicado entre poliedros en HRP-2; continuidad de las
> restricciones de distancia. → Docstring de `collision_model.py`.

[4] J. Haviland and P. Corke, "NEO: A novel expeditious optimisation
algorithm for reactive motion control of manipulators," *IEEE Robot. Autom.
Lett.*, vol. 6, no. 2, pp. 1043–1050, Apr. 2021,
doi: 10.1109/LRA.2021.3056060.
> QP reactivo con velocity dampers y variables de holgura; recomienda
> combinarlo con un planificador global para escapar de mínimos locales.
> → Arquitectura en capas, teoría §1 y §8.

[5] J. J. Quiroz-Omaña and B. V. Adorno, "Whole-body control with (self)
collision avoidance using vector field inequalities," *IEEE Robot. Autom.
Lett.*, vol. 4, no. 4, pp. 4048–4053, Oct. 2019,
doi: 10.1109/LRA.2019.2928783.
> Distancias entre primitivas (línea-línea, como nuestras cápsulas) como
> desigualdades del QP en un humanoide; discusión sobre prioridades.
> → Modelo de cápsulas, teoría §5.1.

## B. Barreras de control (CBF) y seguridad con varias restricciones

[6] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and
P. Tabuada, "Control barrier functions: Theory and applications," in *Proc.
18th Eur. Control Conf. (ECC)*, Naples, Italy, 2019, pp. 3420–3431,
doi: 10.23919/ECC.2019.8796030.
> Definición de CBF `ḣ ≥ −γh`; permite demostrar que el velocity damper es
> una CBF lineal. → Teoría §6.2.

[7] C. Khazoom, D. Gonzalez-Diaz, Y. Ding, and S. Kim, "Humanoid
self-collision avoidance using whole-body control with control barrier
functions," in *Proc. IEEE-RAS 21st Int. Conf. Humanoid Robots
(Humanoids)*, Ginowan, Japan, 2022, pp. 558–565. [Online]. Available:
arXiv:2207.00692.
> Esferas y cápsulas + CBF dentro del controlador de cuerpo completo del MIT
> Humanoid. Aporta frente al damper cuando se usa la dinámica. → Teoría §5.1
> y §6.2.

[8] R. Chen, Y. Sun, and C. Liu, "Dexterous safe control for humanoids in
cluttered environments via projected safe set algorithm," 2025,
arXiv:2502.02858.
> Seguridad con muchas restricciones de eslabón en un **Unitree G1**;
> relajación ordenada de restricciones incompatibles. → Posible mejora
> (holgura en las restricciones de la cámara).

[9] S. J. Jorgensen and R. Bhadeshiya, "Effective virtual reality
teleoperation of an upper-body humanoid with modified task Jacobians and
relaxed barrier functions for self-collision avoidance," presented at the
IROS 2022 Workshop Horizons of an Extended Robotics Reality, 2024,
arXiv:2411.07534.
> Barreras relajadas para autocolisión en la teleoperación de un humanoide
> de torso superior. → Revisión del estado del arte.

## C. Prioridades de tareas y control de cuerpo completo

[10] A. Escande, N. Mansard, and P.-B. Wieber, "Hierarchical quadratic
programming: Fast online humanoid-robot motion generation," *Int. J. Robot.
Res.*, vol. 33, no. 7, pp. 1006–1028, 2014,
doi: 10.1177/0278364914521306.
> Jerarquía estricta de tareas (HQP). Justifica pasar el codo a prioridad
> blanda (`w_elbow = 0.03`) y poner las distancias como restricciones
> duras. → `bimanual_avoidance.py`, teoría §3.3.

[11] A. Dietrich, T. Wimböck, A. Albu-Schäffer, and G. Hirzinger,
"Integration of reactive, torque-based self-collision avoidance into a task
hierarchy," *IEEE Trans. Robot.*, vol. 28, no. 6, pp. 1278–1293, Dec. 2012,
doi: 10.1109/TRO.2012.2208667.
> Autocolisión reactiva en el humanoide Justin, con prioridad dinámica entre
> tarea y evasión. → Teoría §3.3.

## D. Coordinación de dos brazos

[12] K. G. Shin and Q. Zheng, "Minimum-time collision-free trajectory
planning for dual-robot systems," *IEEE Trans. Robot. Autom.*, vol. 8,
no. 5, pp. 641–644, Oct. 1992, doi: 10.1109/70.163787.
> Descomposición camino-velocidad: se retrasa un robot para que el otro
> pase. → Idea de "uno espera" de la capa 2, teoría §9.1.

[13] P. Chiacchio, S. Chiaverini, and B. Siciliano, "Direct and inverse
kinematics for coordinated motion tasks of a two-manipulator system," *J.
Dyn. Syst., Meas., Control*, vol. 118, no. 4, pp. 691–697, 1996,
doi: 10.1115/1.2802344.
> Espacio de tarea cooperativo (pose absoluta + relativa). → Base para
> tareas en que ambas manos sujetan el mismo objeto (trabajo futuro).

[14] Z. Geng, Z. Yang, W. Xu, W. Guo, and X. Sheng, "A globally guided
dual-arm reactive motion controller for coordinated self-handover in a
confined domestic environment," *Biomimetics*, vol. 9, no. 10, Art. no. 629,
Oct. 2024, doi: 10.3390/biomimetics9100629.
> GGDRC: planificador global que guía + QP reactivo con velocity damper
> modificado, en dos brazos. → Arquitectura de las capas 1 y 3
> (`bimanual_planner.GlobalPlannerLayer`), teoría §1 y §10.

[15] F. Ju, Z. Wang, M. Ge, H. Jin, and J. Zhao, "A reactive synchronized
motion controller for dual-arm cooperation with closed-chain constraints,"
*Biomimetics*, vol. 11, no. 5, Art. no. 298, 2026,
doi: 10.3390/biomimetics11050298.
> Esquema **maestro/esclavo** con cambio de rol y replanificación global
> cuando el error crece. → Directamente la capa 2
> (`bimanual_coordinator.py`), teoría §9.

## E. Planificación global (capa 3)

[16] J. J. Kuffner and S. M. LaValle, "RRT-connect: An efficient approach to
single-query path planning," in *Proc. IEEE Int. Conf. Robot. Autom.
(ICRA)*, San Francisco, CA, USA, 2000, pp. 995–1001,
doi: 10.1109/ROBOT.2000.844730.
> Algoritmo de la capa 3: dos árboles que crecen desde el inicio y el
> objetivo hasta tocarse. → `bimanual_planner.rrt_connect`, teoría §10.

[17] B. Sundaralingam *et al.*, "CuRobo: Parallelized collision-free robot
motion generation," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*,
London, U.K., 2023, pp. 8112–8119.
> Optimización de trayectorias sin colisión en GPU, incluidos sistemas de
> dos brazos. → Alternativa "pesada" a RRT-Connect para la capa 3, teoría §10.

[18] O. Khatib, "Real-time obstacle avoidance for manipulators and mobile
robots," *Int. J. Robot. Res.*, vol. 5, no. 1, pp. 90–98, 1986,
doi: 10.1177/027836498600500106.
> Campos potenciales y el origen del problema de los **mínimos locales**.
> → Teoría §8.1.

## F. Percepción con profundidad para evasión

[19] A. Hornung, K. M. Wurm, M. Bennewitz, C. Stachniss, and W. Burgard,
"OctoMap: An efficient probabilistic 3D mapping framework based on
octrees," *Auton. Robots*, vol. 34, no. 3, pp. 189–206, 2013,
doi: 10.1007/s10514-012-9321-0.
> Mapa de ocupación con borrado de espacio libre por *ray casting*.
> → `ObstacleMemory` (depth_obstacles.py), teoría §7.3.

[20] D. Kim, M. Srouji, C. Chen, and J. Zhang, "ARMOR: Egocentric
perception for humanoid robot collision avoidance and motion planning,"
2024, arXiv:2412.00396.
> Percepción de profundidad egocéntrica para evasión en humanoides.
> → Respaldo a usar la cámara del torso para el entorno, teoría §7.1.

[21] B. Sundaralingam, A. Murali, and S. Birchfield, "cuRoboV2:
Dynamics-aware motion generation with depth-fused distance fields for
high-DoF robots," 2026, arXiv:2603.05493.
> Campos de distancia fusionados desde profundidad para robots de muchos
> GDL. → Teoría §7.1.

## G. Herramientas, libros y software

[22] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, "OSQP:
An operator splitting solver for quadratic programs," *Math. Program.
Comput.*, vol. 12, no. 4, pp. 637–672, 2020,
doi: 10.1007/s12532-020-00179-2.
> Solver de todos los QP. → `QP_functions.py`.

[23] E. Todorov, T. Erez, and Y. Tassa, "MuJoCo: A physics engine for
model-based control," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst.
(IROS)*, Vilamoura, Portugal, 2012, pp. 5026–5033,
doi: 10.1109/IROS.2012.6386109.
> Simulador y verdad de terreno de la evaluación.
> → `demos/evaluate_bimanual_avoidance.py`.

[24] C. Ericson, *Real-Time Collision Detection*. San Francisco, CA, USA:
Morgan Kaufmann, 2005.
> Distancia entre segmentos (§5.1.9). → `closest_points_segments`.

[25] B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, *Robotics:
Modelling, Planning and Control*. London, U.K.: Springer, 2009,
doi: 10.1007/978-1-84628-642-1.
> Denavit-Hartenberg y jacobiano geométrico (caps. 2–3).
> → `collision_model.point_jacobian`, teoría §2.

[26] Unitree Robotics, "xr_teleoperate: Teleoperation of Unitree humanoid
robots using XR devices," GitHub repository. [Online]. Available:
https://github.com/unitreerobotics/xr_teleoperate (accessed Sep. 25, 2026).
> IK oficial del H1-2 (`H1_2_ArmIK`): 14 GDL en un solo problema, **sin
> restricciones de autocolisión**. → Punto de comparación, teoría §4.

## H. Planificación multi-robot (agregada con la capa 3)

[27] M. Erdmann and T. Lozano-Pérez, "On multiple moving objects,"
*Algorithmica*, vol. 2, no. 1–4, pp. 477–521, 1987,
doi: 10.1007/BF01840371.
> **Planificación priorizada**: se asigna un orden y se planifica un objeto
> por vez, con los demás como obstáculos. → Capa 3: se planifica un brazo
> con el otro congelado (`bimanual_planner.plan_arm`), teoría §10.

*(Va al final de la numeración para no renumerar las demás.)*

---

## Cómo citarlas en la tesis (ejemplos)

* "Las restricciones de distancia se imponen como *velocity dampers* [1],
  [2], que en control cinemático equivalen a CBF lineales [6]."
* "Siguiendo la arquitectura global-reactiva de [4], [14], la coordinación
  entre brazos se resuelve con un esquema maestro/esclavo [15], [12], y los
  bloqueos restantes con planificación priorizada [27] mediante
  RRT-Connect [16]."
* "La IK oficial del fabricante [26] no considera autocolisiones."

Para LaTeX, cada entrada se pasa a BibTeX sin cambios: los campos ya están
separados (autores, título, revista o congreso, vol., no., pp., año, DOI).
