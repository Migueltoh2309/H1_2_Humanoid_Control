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