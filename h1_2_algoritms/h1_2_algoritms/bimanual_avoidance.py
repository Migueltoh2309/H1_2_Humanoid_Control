"""Controlador bimanual QP con evasión de colisiones (núcleo sin ROS).

Misma ley de control que QP_whole_body.py (servo cinemático de pose del
efector + objetivo secundario del codo en Y + límites articulares), pero:

  * los dos brazos se resuelven en UN solo QP de 14 variables
    (`solve_qp_bimanual`), porque la condición "no chocar entre sí" depende
    de las velocidades de ambos;
  * se añaden restricciones de distancia (velocity dampers) de dos fuentes:
      - autocolisión brazo-brazo y brazo-torso, desde la cinemática (exacta,
        sin latencia, no necesita cámara);
      - obstáculos del entorno, desde la nube de la cámara de profundidad
        (mesa, faja, objetos, una persona...: lo que el modelo no conoce).

Se separa del nodo ROS para que `demos/evaluate_bimanual_avoidance.py`
corra exactamente el mismo código contra MuJoCo directo.
"""
import numpy as np

from h1_2_algoritms.fk_functions import TF2xyzquat, fkine_arm_left_unitree, fkine_arm_right_unitree
from h1_2_algoritms.ik_functions import orientation_error, numerical_jacobian
from h1_2_algoritms.null_control_functions import (
    fk_elbow_left_unitree, fk_elbow_right_unitree, numerical_jacobian_position)
from h1_2_algoritms.QP_functions import solve_qp_bimanual
from h1_2_algoritms.collision_model import (
    arm_frames, self_collision_constraints, obstacle_constraints)
from h1_2_algoritms import joint_limits as JL

SIDES = ("left", "right")
FK = {"left": fkine_arm_left_unitree, "right": fkine_arm_right_unitree}
FK_ELBOW = {"left": fk_elbow_left_unitree, "right": fk_elbow_right_unitree}


class BimanualAvoidanceController:

    def __init__(self, dt=0.05, avoid_self=True, avoid_obstacles=True,
                 d_safe=0.03, d_influence=0.15, xi=0.4,
                 d_safe_env=None, d_influence_env=None, w_elbow=0.03,
                 shoulder_roll_inward_max=0.10):
        self.dt = dt
        self.avoid_self = avoid_self
        self.avoid_obstacles = avoid_obstacles

        # ===== Ganancias / pesos (los de QP_whole_body.py salvo w_elbow) =====
        self.Kp_c = 2.5
        self.Kp_o = 1.0
        self.W_ee = np.diag([10.0, 10.0, 10.0, 2.0, 2.0, 2.0])
        # Prioridad blanda: el codo debe usar SOLO la redundancia que deja
        # la mano (1 GDL). Con el w_elbow = 3.0 de QP_whole_body.py el codo
        # competía con la mano y el QP repartía el error: medido, 38 mm de
        # error al llevar la mano derecha al centro (0.34, 0, 0.15), sin
        # ningún choque cerca. Con 0.03 (1/300 del peso de posición) el
        # error baja a 0.7 mm y el codo sigue actuando cuando la mano ya
        # está resuelta — la versión "blanda" de la jerarquía estricta de
        # HQP (Escande et al., IJRR 2014).
        self.w_elbow = w_elbow
        self.w_reg = 1e-3
        self.k_elbow_qp = 5.0

        # ===== Seguimiento de caminos articulares (capa 3) =====
        # Servo articular q̇ = Kp_joint (q_wp − q) con peso dominante; la
        # tarea de la mano se atenúa (no se apaga) mientras dura el camino.
        self.Kp_joint = 3.0
        self.w_joint = 10.0
        self.ee_scale_following = 0.01
        self.elbow_y = {"left": 0.3, "right": -0.3}

        # ===== Límites (reales, de joint_limits.py) =====
        lo_l, hi_l = JL.get_limits("left")
        lo_r, hi_r = JL.get_limits("right")
        self.q_min = np.hstack((lo_l, lo_r))
        self.q_max = np.hstack((hi_l, hi_r))
        # Hombro contra torso: con el roll del hombro girado hacia ADENTRO
        # más de ~0.15 rad la malla del hombro entra en la del torso (medido
        # en MuJoCo sobre 300 configuraciones aleatorias por valor: 0.20 rad
        # -> 100% en contacto, 0.10 rad -> 12-14%, y ese resto es antebrazo
        # o mano, que ya cubren las cápsulas). Las cápsulas no ven este
        # choque porque ocurre en la articulación misma. Ojo: MuJoCo usa la
        # envolvente CONVEXA de la malla del torso, que rellena la concavidad
        # junto al hombro; en el robot real el ±0.38 del URDF podría ser
        # válido. Por eso es un parámetro.
        self.q_min[1] = max(self.q_min[1], -shoulder_roll_inward_max)   # izq: adentro = negativo
        self.q_max[8] = min(self.q_max[8], shoulder_roll_inward_max)    # der: adentro = positivo
        self.dq_max = np.full(14, 2.0)
        self.dq_min = -self.dq_max

        # ===== Velocity damper =====
        # d_s: distancia mínima garantizada [m]; d_i: a partir de qué
        # distancia empieza a actuar; xi: velocidad máxima de acercamiento
        # permitida justo en d_i [m/s]. El entorno (cámara) lleva su propio
        # par (d_s, d_i) porque la nube tiene ruido y latencia: conviene más
        # margen que con la cinemática.
        self.d_s = d_safe
        self.d_i = d_influence
        self.xi = xi
        self.d_s_env = d_safe_env if d_safe_env is not None else d_safe + 0.02
        self.d_i_env = d_influence_env if d_influence_env is not None else d_influence + 0.05

    def step(self, q, targets, obstacles=None, logger=None, joint_targets=None, x_dot_ff=None):
        """Un ciclo de control.

        q: 14 (izq + der) — configuración actual (o de referencia).
        targets: {"left": xd(7), "right": xd(7)} en torso_link, [x y z qw qx qy qz].
        obstacles: nube Nx3 en torso_link (o None).
        joint_targets: {"left": q7 o None, "right": q7 o None}. Si un brazo
            trae q7, sigue ESA postura (punto de paso de la capa 3) con una
            tarea articular dominante; la tarea de la mano queda atenuada.
            Las restricciones de distancia siguen igual: la seguridad no
            depende de que el camino planificado sea correcto.
        x_dot_ff: {"left": v3 o None, ...} velocidad lineal de referencia
            (feedforward) de la muñeca en torso_link. Sin ella la tarea es
            ẋ = Kp·e y un objetivo que se mueve a v queda atrás v/Kp (medido:
            21 mm con la faja a 0.05 m/s). Es también la entrada natural de
            una trayectoria aprendida (DMP): ẋ = ẋ_ref + Kp·(x_ref − x).

        Devuelve (dq, diag). diag trae las distancias de todos los pares
        vigilados, cuántas restricciones entraron al QP y el error de pose.
        """
        q = np.asarray(q, dtype=float)
        qs = {"left": q[:7], "right": q[7:]}
        tasks, err = [], {}

        for side in SIDES:
            qi = qs[side]
            xd = np.asarray(targets[side], dtype=float).copy()
            x = TF2xyzquat(FK[side](qi))
            xd[3:] /= np.linalg.norm(xd[3:]) + 1e-8
            x[3:] /= np.linalg.norm(x[3:]) + 1e-8
            if np.dot(xd[3:], x[3:]) < 0:
                xd[3:] = -xd[3:]
            ep = xd[0:3] - x[0:3]
            eo = orientation_error(xd[3:], x[3:])
            err[side] = (float(np.linalg.norm(ep)), float(np.linalg.norm(eo)))

            ff = np.zeros(3)
            if x_dot_ff is not None and x_dot_ff.get(side) is not None:
                ff = np.asarray(x_dot_ff[side], dtype=float)
            e_elbow = self.elbow_y[side] - FK_ELBOW[side](qi)[1, 3]
            task = {
                "J": numerical_jacobian(FK[side], qi, TF2xyzquat),
                "x_dot": np.hstack((self.Kp_c * ep + ff, self.Kp_o * eo)),
                "J_elbow": numerical_jacobian_position(FK_ELBOW[side], qi)[1, :],
                "y_dot_elbow": self.k_elbow_qp * e_elbow,
            }
            qwp = None if joint_targets is None else joint_targets.get(side)
            if qwp is not None:
                task["qdot_des"] = self.Kp_joint * (np.asarray(qwp) - qi)
                task["w_joint"] = self.w_joint
                task["ee_scale"] = self.ee_scale_following
            tasks.append(task)

        # ===== Restricciones de distancia =====
        frames_l = arm_frames("left", qs["left"])
        frames_r = arm_frames("right", qs["right"])
        A_blocks, lb_blocks = [], []

        A_s, lb_s, info_self = self_collision_constraints(
            frames_l, frames_r, self.d_s, self.d_i, self.xi)
        if self.avoid_self:
            A_blocks.append(A_s)
            lb_blocks.append(lb_s)

        info_env = []
        if obstacles is not None and len(obstacles) > 0:
            A_o, lb_o, info_env = obstacle_constraints(
                frames_l, frames_r, obstacles, self.d_s_env, self.d_i_env, self.xi)
            if self.avoid_obstacles:
                A_blocks.append(A_o)
                lb_blocks.append(lb_o)

        A_col = np.vstack(A_blocks) if A_blocks else None
        lb_col = np.hstack(lb_blocks) if lb_blocks else None

        dq, ok = solve_qp_bimanual(
            tasks=tasks, q_current=q, q_min=self.q_min, q_max=self.q_max,
            dq_min=self.dq_min, dq_max=self.dq_max, dt=self.dt,
            W_ee=self.W_ee, w_elbow=self.w_elbow, w_reg=self.w_reg,
            A_col=A_col, lb_col=lb_col, logger=logger)
        dq = np.clip(dq, self.dq_min, self.dq_max)

        diag = {
            "ok": ok,
            "err": err,
            "self": info_self,
            "env": info_env,
            "n_constraints": 0 if lb_col is None else len(lb_col),
            "d_self_min": min(d for _, d in info_self),
            # Solo brazo-brazo (sin torso): es lo que mira la capa 2.
            "d_arms_min": min(d for n, d in info_self if "torso" not in n),
            "d_env_min": min((d for _, d, _ in info_env), default=np.inf),
        }
        return dq, diag
