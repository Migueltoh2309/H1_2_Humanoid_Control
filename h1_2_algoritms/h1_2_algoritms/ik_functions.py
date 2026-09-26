from h1_2_algoritms.fk_functions import *

# =========================
# CUATERNIONES
# =========================

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_multiply(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# =========================
# ERROR DE ORIENTACIÓN
# =========================

def orientation_error(qd, q):
    q_inv = quat_conjugate(q)
    qe = quat_multiply(qd, q_inv)
    return qe[1:]  # parte vectorial

# =========================
# ERROR TOTAL (POSE)
# =========================

def pose_error(xd, x):
    ep = xd[0:3] - x[0:3]
    eo = orientation_error(xd[3:], x[3:])
    return np.hstack((ep, eo))

# =========================
# JACOBIANO NUMÉRICO
# =========================

def numerical_jacobian(fkine, q, TF2xyzquat, delta=1e-6):
    n = len(q)
    J = np.zeros((6, n))

    x = TF2xyzquat(fkine(q))

    for i in range(n):
        dq = np.zeros(n)
        dq[i] = delta

        x_d = TF2xyzquat(fkine(q + dq))

        J[:, i] = (pose_error(x_d, x)) / delta

    return J

# ===================
# CINEMÁTICA INVERSA 
# ===================

def ik_pseudo_step(fkine, TF2xyzquat, q, xd, alpha=0.3):

    # ===== FK =====
    x = TF2xyzquat(fkine(q))

    # ===== ERROR =====
    e = pose_error(xd, x)

    # ===== JACOBIANO =====
    J = numerical_jacobian(fkine, q, TF2xyzquat)

    # ===== PSEUDOINVERSA =====
    J_pinv = np.linalg.pinv(J)

    # ===== UPDATE =====
    dq = alpha * (J_pinv @ e)

    return q + dq

# =================================================
# CINEMÁTICA INVERSA CON DAMPED LEAST SQUARE
# =================================================

def ik_dls_step(fkine, TF2xyzquat, q, xd, lamb=0.05):

    # ===== FK =====
    x = TF2xyzquat(fkine(q))

    # ===== ERROR =====
    e = pose_error(xd, x)

    # ===== JACOBIANO =====
    J = numerical_jacobian(fkine, q, TF2xyzquat)

    # ===== DLS =====
    JT = J.T
    JJ = J @ JT

    dq = JT @ np.linalg.inv(JJ + (lamb**2)*np.eye(6)) @ e

    return q + dq


# =====================================================================
# CINEMÁTICA INVERSA CON LÍMITES ARTICULARES
# =====================================================================
#
# Las funciones de arriba resuelven la pose pero no saben nada de los
# rangos mecánicos: con 7 GDL la redundancia deja que las articulaciones
# deriven libremente y la solución suele caer fuera del rango del robot.
# Lo de abajo añade tres mecanismos sobre el mismo paso DLS:
#
#   1. Proyección en el espacio nulo hacia el centro del rango, que usa la
#      redundancia para alejarse de los topes sin alterar la pose.
#   2. Bloqueo de articulaciones que empujan contra su tope: se anula su
#      columna del jacobiano para que la tarea se reparta entre las libres
#      en vez de insistir contra el límite.
#   3. Saturación dura de q y del tamaño del paso en cada iteración.
#
# `ik_solve_limited` envuelve todo con reintentos desde poses aleatorias
# admisibles cuando el arranque en frío cae en un mínimo local.


def limit_avoidance_gradient(q, q_min, q_max):
    """Gradiente del coste de cercanía a los topes.

    H(q) = (1/2n) Σ ((q_i - c_i) / (r_i/2))²  con c el centro del rango y
    r su amplitud. Moverse en -∇H empuja hacia el centro, con más fuerza
    cuanto más estrecho es el rango.
    """
    n = len(q)
    c = 0.5 * (q_min + q_max)
    half = np.maximum(0.5 * (q_max - q_min), 1e-9)
    return (q - c) / (n * half ** 2)


def ik_dls_step_limited(fkine, TF2xyzquat, q, xd, q_min, q_max,
                        lamb=0.05, k_null=0.5, dq_max=0.2):
    """Un paso de DLS que respeta los límites articulares.

    Devuelve (q_nuevo, e) con e el error de pose ANTES del paso.
    """
    q = np.clip(np.asarray(q, dtype=float), q_min, q_max)

    x = TF2xyzquat(fkine(q))
    e = pose_error(xd, x)
    J = numerical_jacobian(fkine, q, TF2xyzquat)

    n = len(q)
    free = np.ones(n, dtype=bool)
    dq = np.zeros(n)
    I6 = np.eye(6)

    # Término secundario: alejarse de los topes por el espacio nulo.
    dq_sec = -k_null * limit_avoidance_gradient(q, q_min, q_max)

    # Bucle de bloqueo: como mucho una articulación se congela por pasada.
    for _ in range(n + 1):
        Jf = J.copy()
        Jf[:, ~free] = 0.0

        JT = Jf.T
        J_dls = JT @ np.linalg.inv(Jf @ JT + (lamb ** 2) * I6)

        # Tarea principal con amortiguamiento (robusto cerca de
        # singularidades).
        dq = J_dls @ e

        # Proyector en el espacio nulo: se calcula con la pseudoinversa SIN
        # amortiguar. Con J_dls el proyector no es exacto y el término
        # secundario se filtra en la tarea, dejando un error residual de
        # pose que no baja por más iteraciones que se hagan.
        N = np.eye(n) - np.linalg.pinv(Jf, rcond=1e-6) @ Jf
        dq = dq + N @ dq_sec
        dq[~free] = 0.0

        # Paso acotado: evita saltos que rompan la linealización.
        norm = np.linalg.norm(dq)
        if norm > dq_max:
            dq *= dq_max / norm

        q_try = q + dq
        # Se congela lo que se sale del rango, salvo que ya estuviera
        # pegado al tope y el paso lo devuelva hacia adentro.
        outside = (q_try < q_min - 1e-12) | (q_try > q_max + 1e-12)
        newly_blocked = outside & free
        if not np.any(newly_blocked):
            break
        free &= ~newly_blocked
        if not np.any(free):
            break

    return np.clip(q + dq, q_min, q_max), e


def ik_solve_limited(fkine, TF2xyzquat, q0, xd, q_min, q_max,
                     lamb=0.05, k_null=0.5, dq_max=0.2,
                     max_iters=300, tol_pos=1e-4, tol_ori=1e-4,
                     restarts=8, seed=0):
    """Resuelve la IK respetando los límites, con reintentos aleatorios.

    Devuelve un dict con:
        q          solución (siempre dentro de los límites)
        ok         True si además convergió dentro de las tolerancias
        e_pos      norma del error de posición [m]
        e_ori      norma de la parte vectorial del error de orientación
        iters      iteraciones de la última tentativa
        restarts   cuántos rearranques hicieron falta (0 = ninguno)
    """
    q_min = np.asarray(q_min, dtype=float)
    q_max = np.asarray(q_max, dtype=float)
    rng = np.random.default_rng(seed)

    q_start = np.clip(np.asarray(q0, dtype=float), q_min, q_max)
    best = None

    for attempt in range(restarts + 1):
        q = q_start if attempt == 0 else rng.uniform(q_min, q_max)
        it = 0
        for it in range(1, max_iters + 1):
            q_new, e = ik_dls_step_limited(
                fkine, TF2xyzquat, q, xd, q_min, q_max,
                lamb=lamb, k_null=k_null, dq_max=dq_max)
            if np.any(~np.isfinite(q_new)):
                break
            q = q_new
            if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_ori:
                break

        x = TF2xyzquat(fkine(q))
        e = pose_error(xd, x)
        e_pos = float(np.linalg.norm(e[:3]))
        e_ori = float(np.linalg.norm(e[3:]))
        ok = e_pos < tol_pos and e_ori < tol_ori

        if best is None or (e_pos + e_ori) < (best["e_pos"] + best["e_ori"]):
            best = {"q": q, "ok": ok, "e_pos": e_pos, "e_ori": e_ori,
                    "iters": it, "restarts": attempt}
        if ok:
            break

    return best