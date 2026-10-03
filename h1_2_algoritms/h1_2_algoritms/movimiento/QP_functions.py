import numpy as np
from scipy import sparse
import osqp


def compute_velocity_bounds(q, q_min, q_max, dq_min, dq_max, dt):
    """
    Combina:
    1. límites directos de velocidad:
        dq_min <= dq <= dq_max

    2. límites articulares:
        q_min <= q + dq*dt <= q_max

    De:
        q_min <= q + dq*dt <= q_max

    se obtiene:
        (q_min - q)/dt <= dq <= (q_max - q)/dt
    """

    dq_lower_from_q = (q_min - q) / dt
    dq_upper_from_q = (q_max - q) / dt

    lower = np.maximum(dq_min, dq_lower_from_q)
    upper = np.minimum(dq_max, dq_upper_from_q)

    return lower, upper


def build_qp_matrices(
    J,
    x_dot,
    J_elbow,
    y_dot_elbow,
    q_current,
    q_min,
    q_max,
    dq_min,
    dq_max,
    dt,
    W_ee,
    w_elbow,
    w_reg
):
    """
    Construye las matrices del QP.

    Problema:

        min 1/2 ||J dq - x_dot||^2_W
            + 1/2 w_elbow ||J_elbow dq - y_dot_elbow||^2
            + 1/2 w_reg ||dq||^2

        s.a.

            dq_lower <= dq <= dq_upper

    Forma OSQP:

        min 1/2 dq^T P dq + q^T dq
        s.a. l <= A dq <= u
    """

    n = J.shape[1]

    # ===== Costo de tarea principal EE =====
    P_ee = J.T @ W_ee @ J
    q_ee = -J.T @ W_ee @ x_dot

    # ===== Costo secundario del codo =====
    J_e = J_elbow.reshape(1, -1)

    P_elbow = w_elbow * (J_e.T @ J_e)
    q_elbow = -w_elbow * (J_e.T.flatten() * y_dot_elbow)

    # ===== Regularización =====
    P_reg = w_reg * np.eye(n)

    # ===== Costo total =====
    P = P_ee + P_elbow + P_reg
    q_vec = q_ee + q_elbow

    # ===== Restricciones de velocidad =====
    dq_lower, dq_upper = compute_velocity_bounds(
        q=q_current,
        q_min=q_min,
        q_max=q_max,
        dq_min=dq_min,
        dq_max=dq_max,
        dt=dt
    )

    # OSQP usa:
    # l <= A dq <= u
    A = np.eye(n)
    l = dq_lower
    u = dq_upper

    # OSQP trabaja con matrices sparse
    P_sparse = sparse.csc_matrix(P)
    A_sparse = sparse.csc_matrix(A)

    return P_sparse, q_vec, A_sparse, l, u


def solve_qp_arm(
    J,
    x_dot,
    J_elbow,
    y_dot_elbow,
    q_current,
    q_min,
    q_max,
    dq_min,
    dq_max,
    dt,
    W_ee,
    w_elbow,
    w_reg,
    logger=None
):
    """
    Resuelve el QP para un brazo y retorna dq.
    """

    P, q_vec, A, l, u = build_qp_matrices(
        J=J,
        x_dot=x_dot,
        J_elbow=J_elbow,
        y_dot_elbow=y_dot_elbow,
        q_current=q_current,
        q_min=q_min,
        q_max=q_max,
        dq_min=dq_min,
        dq_max=dq_max,
        dt=dt,
        W_ee=W_ee,
        w_elbow=w_elbow,
        w_reg=w_reg
    )

    solver = osqp.OSQP()

    solver.setup(
        P=P,
        q=q_vec,
        A=A,
        l=l,
        u=u,
        verbose=False,
        polish=False,
        warm_start=True,
        max_iter=100,
        eps_abs=1e-4,
        eps_rel=1e-4
    )

    result = solver.solve()

    if result.info.status_val not in [1, 2]:
        if logger is not None:
            logger.warn(
                f"OSQP no encontró solución óptima. Status: {result.info.status}"
            )
        return np.zeros_like(q_current)

    dq = result.x

    if dq is None or np.any(np.isnan(dq)):
        if logger is not None:
            logger.warn("OSQP devolvió dq inválido. Usando ceros.")
        return np.zeros_like(q_current)

    return dq

# =====================================================================
# QP BIMANUAL CON EVASIÓN DE COLISIONES
# =====================================================================
#
# solve_qp_arm resuelve cada brazo por separado: así no hay forma de
# expresar "el izquierdo no se acerque al derecho", porque esa condición
# depende de las velocidades de AMBOS. Aquí se apilan los dos brazos en un
# único QP de 14 variables, dq = [dq_left, dq_right]:
#
#     min  sum_brazos  1/2 ||J dq - x_dot||^2_W + 1/2 w_elbow ||J_e dq - y_dot||^2
#          + 1/2 w_reg ||dq||^2
#     s.a. dq_lower <= dq <= dq_upper                  (límites, igual que antes)
#          A_col dq >= lb_col                          (velocity dampers)
#
# El costo es diagonal por bloques (cada brazo sigue su propia tarea); lo
# único que acopla a los dos brazos son las filas de A_col de los pares
# brazo-brazo. Las filas vienen de collision_model.py (autocolisión, a
# partir de la cinemática) y de la cámara de profundidad (obstáculos).


def solve_qp_bimanual(
    tasks,
    q_current,
    q_min,
    q_max,
    dq_min,
    dq_max,
    dt,
    W_ee,
    w_elbow,
    w_reg,
    A_col=None,
    lb_col=None,
    logger=None
):
    """
    tasks: lista de 2 dicts (izquierdo, derecho) con J (6x7), x_dot (6),
    J_elbow (7) e y_dot_elbow (escalar). Opcional, para seguir un camino
    de la capa 3: "qdot_des" (7) y "w_joint" agregan la tarea de postura
    ½ w_joint ‖dq − qdot_des‖², y "ee_scale" (0..1) atenúa la tarea del
    efector mientras tanto. q_current, q_min, q_max, dq_min y
    dq_max son vectores de 14 en el mismo orden.

    Devuelve (dq, ok): dq (14) y ok=False si OSQP no resolvió. En ese caso
    dq = 0 — frenar es lo seguro: con dq = 0 todo damper con d > d_s se
    cumple.
    """
    n = 14
    P = w_reg * np.eye(n)
    q_vec = np.zeros(n)

    for i, t in enumerate(tasks):
        s = slice(7 * i, 7 * i + 7)
        J = t["J"]
        J_e = np.asarray(t["J_elbow"]).reshape(1, -1)
        k = t.get("ee_scale", 1.0)
        P[s, s] += k * (J.T @ W_ee @ J + w_elbow * (J_e.T @ J_e))
        q_vec[s] += -k * (J.T @ W_ee @ t["x_dot"] + w_elbow * J_e.flatten() * t["y_dot_elbow"])
        if t.get("qdot_des") is not None:
            # Tarea de postura (seguir un camino articular de la capa 3).
            w = t["w_joint"]
            P[s, s] += w * np.eye(7)
            q_vec[s] += -w * np.asarray(t["qdot_des"])

    dq_lower, dq_upper = compute_velocity_bounds(
        q=q_current, q_min=q_min, q_max=q_max,
        dq_min=dq_min, dq_max=dq_max, dt=dt)

    A = np.eye(n)
    l = dq_lower
    u = dq_upper
    if A_col is not None and len(A_col) > 0:
        A = np.vstack((A, A_col))
        l = np.hstack((l, lb_col))
        u = np.hstack((u, np.full(len(lb_col), np.inf)))

    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(np.triu(P)),
        q=q_vec,
        A=sparse.csc_matrix(A),
        l=l,
        u=u,
        verbose=False,
        polish=True,
        max_iter=4000,
        eps_abs=1e-5,
        eps_rel=1e-5
    )
    result = solver.solve()

    # 1 = solved, 2 = solved inaccurate
    if result.info.status_val not in [1, 2] or result.x is None \
            or np.any(np.isnan(result.x)):
        if logger is not None:
            logger.warn(
                f"QP bimanual sin solución ({result.info.status}); se frena.")
        return np.zeros(n), False

    return result.x, True
