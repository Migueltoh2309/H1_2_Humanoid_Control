# Referencias: visual servoing, percepción de la fruta, agarre y LfD (formato IEEE)

Referencias que sustentan la etapa de visual servoing y agarre del H1-2:

- percepción de la fruta: `fruit_localization.py` y `perception_common.localize_center`;
- geometría del agarre: `grasp_geometry.py`;
- lazo cerrado: `demos/visual_servoing_grasp.py`;
- escena con las manos: `h1_2_description/scripts/build_scene_hands.py`;
- evaluación del par IR: `demos/evaluate_ir_stereo.py`.

Complementa a [`REFERENCIAS_BIMANUAL.md`](REFERENCIAS_BIMANUAL.md), que ya
cubre el QP, los velocity dampers, OSQP, MuJoCo, OctoMap y las demás
referencias del controlador (se citan como **[B-n]** para no duplicarlas).
Los resultados que estas referencias justifican están en
[`VISUAL_SERVOING_PLAN.md`](VISUAL_SERVOING_PLAN.md) y
[`PERCEPTION_PLAN.md`](PERCEPTION_PLAN.md) §9.

Como en el documento bimanual, debajo de cada referencia se indica **qué
se tomó de ahí** y **dónde aparece**. Los datos bibliográficos se
verificaron en IEEE Xplore, las editoriales, arXiv, dblp, PMLR y las
páginas de los fabricantes (búsqueda del 25/09/2026). Si no se pudo
confirmar un DOI, se omite en vez de adivinarlo.

---

## A. Visual servoing y control cinemático

[1] S. Hutchinson, G. D. Hager, and P. I. Corke, "A tutorial on visual servo
control," *IEEE Trans. Robot. Autom.*, vol. 12, no. 5, pp. 651–670, Oct.
1996, doi: 10.1109/70.538972.
> Taxonomía clásica: basado en posición (PBVS) vs. basado en imagen
> (IBVS), cámara en la mano vs. fija (*eye-to-hand*). → Decisión PBVS con
> cámara en el torso, plan §1.

[2] F. Chaumette and S. Hutchinson, "Visual servo control. I. Basic
approaches," *IEEE Robot. Autom. Mag.*, vol. 13, no. 4, pp. 82–90, Dec.
2006, doi: 10.1109/MRA.2006.250573.
> Formulación estándar del error de PBVS, e = pose deseada − pose actual,
> con la pose del objeto estimada por la cámara. Es exactamente la tarea
> del QP: la muñeca deseada es función de la fruta vista. → Docstring de
> `visual_servoing_grasp.py`, plan §1.

[3] F. Chaumette, "Potential problems of stability and convergence in
image-based and position-based visual servoing," in *The Confluence of
Vision and Control* (Lecture Notes in Control and Information Sciences,
vol. 237), D. J. Kriegman, G. D. Hager, and A. S. Morse, Eds. London,
U.K.: Springer, 1998, pp. 66–78, doi: 10.1007/BFb0109663.
> PBVS es sensible a los errores de calibración de la cámara y de
> estimación de pose, porque el error se cierra en 3D. → Justifica el
> pendiente "calibración cámara-torso" (plan §6.4): el ajuste de esfera
> no corrige un error extrínseco.

[4] G. Flandin, F. Chaumette, and E. Marchand, "Eye-in-hand / eye-to-hand
cooperation for visual servoing," in *Proc. IEEE Int. Conf. Robot. Autom.
(ICRA)*, San Francisco, CA, USA, 2000, pp. 2741–2746,
doi: 10.1109/ROBOT.2000.846442.
> Ventajas y límites de la cámara fija (*eye-to-hand*): ve la escena
> completa, pero la propia mano la ocluye en la aproximación final. →
> Hallazgo de la oclusión total en el pre-agarre (plan §7, brecha 4).

[5] L. Sciavicco and B. Siciliano, "A solution algorithm to the inverse
kinematic problem for redundant manipulators," *IEEE J. Robot. Autom.*,
vol. 4, no. 4, pp. 403–410, Aug. 1988, doi: 10.1109/56.804.
> Cinemática inversa en lazo cerrado (CLIK): ẋ = ẋ_d + K·e. El término
> feedforward ẋ_d es el que faltaba para seguir la fruta en la faja. →
> `x_dot_ff` de `BimanualAvoidanceController.step`; plan §5 (retraso de
> 21 mm sin él, v/Kp).

## B. Percepción y localización de la fruta

[6] A. Gongal, S. Amatya, M. Karkee, Q. Zhang, and K. Lewis, "Sensors and
systems for fruit detection and localization: A review," *Comput. Electron.
Agric.*, vol. 116, pp. 8–19, 2015, doi: 10.1016/j.compag.2015.05.021.
> Revisión de la detección y localización de fruta: segmentación por
> color, sensores 3D, oclusión y variación de la iluminación. → Marco de
> la segmentación HSV (Método A) y de sus riesgos.

[7] C. Lehnert, I. Sa, C. McCool, B. Upcroft, and T. Perez, "Sweet pepper
pose detection and grasping for automated crop harvesting," in *Proc. IEEE
Int. Conf. Robot. Autom. (ICRA)*, Stockholm, Sweden, 2016, pp. 2428–2434,
doi: 10.1109/ICRA.2016.7487394.
> Ajuste de un modelo geométrico (superelipsoide) por mínimos cuadrados no
> lineales a la nube RGB-D segmentada de la fruta, para obtener su pose y
> agarrarla. Es el mismo principio que nuestro ajuste de esfera de radio
> conocido. → `fruit_localization.locate_sphere`.

[8] P. J. Huber, "Robust estimation of a location parameter," *Ann. Math.
Statist.*, vol. 35, no. 1, pp. 73–101, Mar. 1964,
doi: 10.1214/aoms/1177703732.
> Pérdida de Huber (M-estimador), cuadrática cerca de cero y lineal lejos.
> → Pesos del Gauss-Newton en `locate_sphere` (umbral de 5 mm), para que
> los píxeles de borde mal medidos no arrastren el centro.

[9] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look
once: Unified, real-time object detection," in *Proc. IEEE Conf. Comput.
Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 779–788,
doi: 10.1109/CVPR.2016.91.
> Detector de una etapa. → Método B (`yolo_color_depth_detector_node`).

[10] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," version
8.0.0, 2023. [Software]. Available: https://github.com/ultralytics/ultralytics
> Implementación y pesos `yolov8n-seg` usados en el Método B (la máscara de
> instancia alimenta el ajuste de esfera).

[11] T.-Y. Lin *et al.*, "Microsoft COCO: Common objects in context," in
*Proc. Eur. Conf. Comput. Vis. (ECCV)* (Lecture Notes in Computer Science,
vol. 8693), Zurich, Switzerland, 2014, pp. 740–755,
doi: 10.1007/978-3-319-10602-1_48.
> Clases `orange` y `sports ball` del modelo preentrenado, y origen del
> *domain gap* con el render de MuJoCo (PERCEPTION_PLAN §7-8).

## C. Cámara de profundidad y estéreo IR

[12] L. Keselman, J. I. Woodfill, A. Grunnet-Jepsen, and A. Bhowmik, "Intel
RealSense stereoscopic depth cameras," in *Proc. IEEE Conf. Comput. Vis.
Pattern Recognit. Workshops (CVPRW)*, Honolulu, HI, USA, 2017,
pp. 1267–1276, doi: 10.1109/CVPRW.2017.167.
> La profundidad de la D400 sale de estéreo activo por correspondencia
> densa en el ASIC, a partir del mismo par IR. → Explica por qué
> triangular solo dos centros de silueta en IR no le gana a la
> profundidad del ASIC (plan §3).

[13] A. Grunnet-Jepsen, J. N. Sweetser, and J. Woodfill, "Tuning depth
cameras for best performance," Intel RealSense, white paper. [Online].
Available: https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance
(accedido: 25/09/2026).
> Error RMS de profundidad = Z²·(error subpíxel)/(f·B), con ~0.08 px
> típico. → Modelo de ruido `realsense_depth_noise` (σ_z ∝ z²) y el cálculo
> de ~28 mm de profundidad por píxel de disparidad del plan §3.

[14] Intel Corporation, "Intel RealSense Product Family D400 Series
Datasheet," Doc. 337029, rev. 017, Sep. 2023.
> Módulo D430 de la D435: **línea de base de 50 mm**, campo de
> profundidad **87°×58°**, imagers OV9282 de **obturador global** y RGB
> OV2740 de **obturador rodante** (Tabla 3-11 y tablas de sensores). →
> Geometría de `robot_ir_left`/`robot_ir_right` en `build_scene_hands.py`,
> y argumento de que el IR solo es interesante en el robot real (plan §3).

[15] D. Gallup, J.-M. Frahm, P. Mordohai, and M. Pollefeys, "Variable
baseline/resolution stereo," in *Proc. IEEE Conf. Comput. Vis. Pattern
Recognit. (CVPR)*, Anchorage, AK, USA, 2008, doi: 10.1109/CVPR.2008.4587671.
> En estéreo con línea de base fija el error de profundidad crece
> cuadráticamente con la distancia. → Límite físico del par IR de 50 mm a
> ~0.8 m (plan §3).

[16] R. Hartley and A. Zisserman, *Multiple View Geometry in Computer
Vision*, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2004,
doi: 10.1017/CBO9780511811685.
> Triangulación en un par rectificado, Z = f·B/d, y proyección de una
> esfera como cónica. → `locate_stereo_ir`.

[17] N. Otsu, "A threshold selection method from gray-level histograms,"
*IEEE Trans. Syst., Man, Cybern.*, vol. SMC-9, no. 1, pp. 62–66, Jan.
1979, doi: 10.1109/TSMC.1979.4310076.
> Umbral automático para separar la fruta de la faja en la ventana IR. →
> `fruit_localization._ir_blob_center`.

[18] A. Fitzgibbon, M. Pilu, and R. B. Fisher, "Direct least square fitting
of ellipses," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 21, no. 5,
pp. 476–480, May 1999, doi: 10.1109/34.765658.
> Ajuste de elipse al contorno para obtener un centro subpíxel (es el
> método de `cv2.fitEllipse`). → `_ir_blob_center`.

[19] C.-K. Liang, L.-W. Chang, and H. H. Chen, "Analysis and compensation of
rolling shutter effect," *IEEE Trans. Image Process.*, vol. 17, no. 8,
pp. 1323–1330, Aug. 2008, doi: 10.1109/TIP.2008.925384.
> Distorsión del obturador rodante con objetos en movimiento. → Argumento
> para probar en hardware las IR (obturador global) frente a la RGB sobre
> la faja (plan §3). MuJoCo no modela este efecto.

## D. Objetos en movimiento: estimación y agarre sobre faja

[20] R. E. Kalman, "A new approach to linear filtering and prediction
problems," *J. Basic Eng.*, vol. 82, no. 1, pp. 35–45, Mar. 1960,
doi: 10.1115/1.3662552.
> Filtro de Kalman. → `FruitTracker`.

[21] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, *Estimation with
Applications to Tracking and Navigation: Theory, Algorithms and Software*.
New York, NY, USA: Wiley, 2001, doi: 10.1002/0471221279.
> Modelo cinemático de velocidad constante con ruido de aceleración
> blanco (la Q discreta usada), y compuerta (*gating*) de la innovación
> para descartar outliers. → `FruitTracker._Q`, `gate`.

[22] P. K. Allen, A. Timcenko, B. Yoshimi, and P. Michelman, "Automated
tracking and grasping of a moving object with a robotic hand-eye system,"
*IEEE Trans. Robot. Autom.*, vol. 9, no. 2, pp. 152–165, Apr. 1993,
doi: 10.1109/70.238279.
> Referencia clásica de seguimiento visual con **predicción** para
> interceptar y agarrar un objeto en movimiento. → Kalman + seguimiento
> de la meta predicha (plan §5).

[23] F. Islam, O. Salzman, A. Agarwal, and M. Likhachev, "Provably
constant-time planning and replanning for real-time grasping objects off a
conveyor belt," *Int. J. Robot. Res.*, vol. 40, no. 12–14, pp. 1370–1384,
2021, doi: 10.1177/02783649211027194.
> Agarre en faja actualizando la meta mientras el brazo ya se mueve, y el
> compromiso entre esperar una mejor estimación y perder la ventana. →
> Plan §7 (brecha 4: la meta se congela en los últimos ~0.3 s por oclusión).

[24] S. D. Han, S. W. Feng, and J. Yu, "Toward fast and optimal robotic
pick-and-place on a moving conveyor," *IEEE Robot. Autom. Lett.*, vol. 5,
no. 2, pp. 446–453, Apr. 2020, doi: 10.1109/LRA.2019.2961605.
> Las reglas voraces (FIFO) no son óptimas en el tiempo. → La selección de
> brazo del demo es esa línea base voraz, contra la que se comparará la
> asignación aprendida (plan §7).

## E. Mano, agarre y simulación de contacto

[25] Inspire Robots, "RH56DFX dexterous hand — product specifications."
[Online]. Available: https://en.inspire-robots.com/product/rh56dfx/
(accedido: 25/09/2026).
> 6 GDL y 12 articulaciones (las falanges acopladas: los `<equality joint>`
> de la escena), 10 N en la punta de cada dedo y 15 N en el pulgar, con
> umbrales de fuerza configurables. Para el H1-2 se vende la variante
> RH56DFTP, anunciada con hasta 30 N en la punta. → `HAND_ACTUATORS` de
> `build_scene_hands.py` (cierre suave por debajo del máximo).

[26] T. Feix, J. Romero, H.-B. Schmiedmayer, A. M. Dollar, and D. Kragic,
"The GRASP taxonomy of human grasp types," *IEEE Trans. Human-Mach. Syst.*,
vol. 46, no. 1, pp. 66–77, Feb. 2016, doi: 10.1109/THMS.2015.2470657.
> Agarres de potencia vs. de precisión sobre esferas. → Encuadre del
> agarre "desde arriba con la palma" y de la alternativa de pinza (plan §4).

[27] A. Bicchi and V. Kumar, "Robotic grasping and contact: A review," in
*Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, San Francisco, CA, USA, 2000,
vol. 1, pp. 348–353, doi: 10.1109/ROBOT.2000.844081.
> Cierre de fuerza y forma: un agarre que no envuelve el objeto por debajo
> de su ecuador depende solo de la fricción. → Causa 3 del plan §4.

[28] M. T. Mason, "Mechanics and planning of manipulator pushing
operations," *Int. J. Robot. Res.*, vol. 5, no. 3, pp. 53–71, 1986,
doi: 10.1177/027836498600500303.
> Mecánica del empuje: un contacto unilateral desplaza el objeto en vez de
> sujetarlo. → Causas 1 y 3 del plan §4 (el pulgar empuja la fruta, que
> rueda).

[29] P. C. Horak and J. C. Trinkle, "On the similarities and differences
among contact models in robot simulation," *IEEE Robot. Autom. Lett.*,
vol. 4, no. 2, pp. 493–499, Apr. 2019, doi: 10.1109/LRA.2019.2891085.
> Los modelos de contacto de los simuladores difieren entre sí y de la
> realidad, sobre todo en fricción y contactos múltiples. → El éxito del
> agarre en simulación es optimista y hay que validarlo en el robot
> (plan §4, §7 brecha 5).

[30] C. Li *et al.*, "iGibson 2.0: Object-centric simulation for robot
learning of everyday household tasks," in *Proc. 5th Conf. Robot Learn.
(CoRL)* (Proc. Mach. Learn. Res., vol. 164), 2021, pp. 455–465.
> **Agarre asistido** en un simulador de referencia: una restricción
> adicional palma-objeto que se activa al superar un umbral de agarre,
> para evitar deslizamientos. → Precedente directo de nuestra asistencia
> (`*_grasp_assist`, `_try_weld`), que además exige contacto real de ≥2
> dedos o palma y un error < 2 cm.

Motor de física y herramientas ya citados: MuJoCo **[B-23]** (escena,
`mj_ray`, restricciones `weld`/`joint`), OSQP **[B-22]**, OctoMap **[B-19]**
(`ObstacleMemory`), velocity dampers **[B-1]**, libro de Siciliano *et al.*
**[B-25]**.

## F. Aprendizaje por demostración (encaje con la tesis)

[31] A. J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal,
"Dynamical movement primitives: Learning attractor models for motor
behaviors," *Neural Comput.*, vol. 25, no. 2, pp. 328–373, Feb. 2013,
doi: 10.1162/NECO_a_00393.
> DMP con meta g y duración τ re-parametrizables. → La percepción entrega
> g = p(t_grasp) (plan §7).

[32] P. Pastor, H. Hoffmann, T. Asfour, and S. Schaal, "Learning and
generalization of motor skills by learning from demonstration," in *Proc.
IEEE Int. Conf. Robot. Autom. (ICRA)*, Kobe, Japan, 2009, pp. 763–768,
doi: 10.1109/ROBOT.2009.5152385.
> DMP de pick-and-place con la meta dada por visión y obstáculos evitados
> en línea: la misma división de roles que proponemos (DMP para la forma,
> percepción para g, una capa aparte para la seguridad). → Plan §7.

[33] M. Saveriano, F. J. Abu-Dakka, A. Kramberger, and L. Peternel,
"Dynamic movement primitives in robotics: A tutorial survey," *Int. J.
Robot. Res.*, vol. 42, no. 13, pp. 1133–1184, 2023,
doi: 10.1177/02783649231201196.
> Formulaciones unificadas: escalado espacial, orientación, restricciones.
> → Brechas 1 y 3 del plan §7 (orientación y retargeting del espacio de
> trabajo).

[34] A. Koutras and Z. Doulgeri, "Dynamic movement primitives for moving
goals with temporal scaling adaptation," in *Proc. IEEE Int. Conf. Robot.
Autom. (ICRA)*, Paris, France, 2020, pp. 144–150,
doi: 10.1109/ICRA40945.2020.9196765.
> DMP con **meta en movimiento** y adaptación de la escala temporal. → Es
> justo el caso de la fruta sobre la faja: la meta del Kalman se mueve
> durante el alcance (plan §7).

[35] M. Saveriano and D. Lee, "Learning barrier functions for constrained
motion planning with dynamical systems," in *Proc. IEEE/RSJ Int. Conf.
Intell. Robots Syst. (IROS)*, Macau, China, 2019, pp. 112–119,
doi: 10.1109/IROS40897.2019.8967981.
> Sistemas dinámicos aprendidos ejecutados bajo restricciones impuestas
> con un QP. → Justifica usar el QP bimanual como capa de ejecución segura
> de la DMP (plan §7).

[36] C. L. Nehaniv and K. Dautenhahn, "The correspondence problem," in
*Imitation in Animals and Artifacts*, K. Dautenhahn and C. L. Nehaniv,
Eds. Cambridge, MA, USA: MIT Press, 2002, pp. 41–61.
> El problema de correspondencia demostrador-imitador. → Brecha 1 del
> plan §7: la mano humana y la Inspire no agarran igual una fruta de
> 6-8 cm; la orientación del agarre la pone el modelo del robot.

[37] V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and
M. Grundmann, "BlazePose: On-device real-time body pose tracking," 2020,
arXiv:2006.10204.
> MediaPipe Pose: 33 puntos; por mano, además de la muñeca, solo pulgar,
> índice y meñique. Con ellos se puede estimar una orientación **gruesa**
> de la palma, no fiable para el agarre (y el roadmap de demostraciones
> hoy usa solo hombro, codo y muñeca). → Brecha 1 del plan §7.

[38] F. Zhang *et al.*, "MediaPipe Hands: On-device real-time hand
tracking," 2020, arXiv:2006.10214.
> 21 puntos por mano: permite estimar la apertura y el cierre y la
> orientación de la mano. → Propuesta para cerrar las brechas 1 y 2 del
> plan §7 (estado de agarre en las demostraciones).

[39] F. Krebs and T. Asfour, "A bimanual manipulation taxonomy," *IEEE
Robot. Autom. Lett.*, vol. 7, no. 4, pp. 11031–11038, Oct. 2022,
doi: 10.1109/LRA.2022.3196158.
> Tarea débilmente acoplada (cada mano con su objeto, con espacio y tiempo
> compartidos). → Necesidad de la capa de seguridad bimanual en W∩
> (plan §7).

---

## Qué afirmación se apoya en qué

| Afirmación (resultado propio) | Respaldo |
|---|---|
| PBVS con cámara fija y error cartesiano es la elección natural sobre el QP existente | [1], [2], [4] |
| La mediana de profundidad mide la superficie y no el centro; un modelo geométrico ajustado a la nube lo corrige (33 → 2.4 mm) | [6], [7], [8] (y la derivación 0.71·R del plan §3) |
| El par IR de 50 mm no mejora a ~0.8 m (~28 mm por píxel de disparidad) | [12], [13], [14], [15], [16] |
| Aun así, el IR merece una prueba en hardware (obturador global) | [14], [19] |
| Sin feedforward, un lazo P queda v/Kp detrás de un objetivo móvil | [5] |
| Kalman de velocidad constante con compuerta para predecir g | [20], [21], [22] |
| La cámara del torso pierde la fruta en el tramo final (oclusión por la mano) | [4], [23] |
| El agarre físico de una esfera rígida falla por empuje y falta de cierre de forma | [26], [27], [28] |
| La asistencia de agarre es práctica aceptada en simulación, y el éxito simulado es optimista | [29], [30] |
| La fuerza de cierre usada está dentro de lo que da la mano real | [25] |
| La arquitectura sirve para el LfD: DMP con meta visual y móvil, ejecutada por un QP seguro | [31]–[35], [39] |
| Las brechas de orientación y de estado de agarre en las demostraciones | [33], [36], [37], [38] |
