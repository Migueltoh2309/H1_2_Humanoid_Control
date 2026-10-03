from h1_2_algoritms.movimiento.fk_functions import *

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



# =========================
# NORMALIZACIÓN
# =========================
def normalize_quaternions(xd, x):

    xd[3:] = xd[3:] / (np.linalg.norm(xd[3:]) + 1e-8)
    x[3:]  = x[3:]  / (np.linalg.norm(x[3:])  + 1e-8)

    if np.dot(xd[3:], x[3:]) < 0:
        xd[3:] = -xd[3:]

    return xd, x

# =========================
# IK STEP
# =========================
def ik_step(fkine, q, xd, use_dls=True):

    from h1_2_algoritms.movimiento.fk_functions import TF2xyzquat

    # FK
    x = TF2xyzquat(fkine(q))

    # Normalización
    xd, x = normalize_quaternions(xd, x)

    # Error
    e = pose_error(xd, x)

    # Jacobiano
    J = numerical_jacobian(fkine, q, TF2xyzquat)

    if use_dls:
        # ===== DLS =====
        JT = J.T
        dq = JT @ np.linalg.inv(J @ JT + (0.05**2)*np.eye(6)) @ e
    else:
        # ===== PSEUDOINVERSA =====
        dq = np.linalg.pinv(J) @ e

    return q + dq

# =========================
# TRAYECTORIA
# =========================
def circular_trajectory(xd, t, radius, omega):

    center = xd[0:3].copy()

    xd[1] = center[1] + radius * np.cos(omega * t)
    xd[2] = center[2] + radius * np.sin(omega * t)

    return xd


# =========================
# ELIMINACIÓN GAUSSEANA
# =========================
def gaussian_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    # Forward elimination
    for i in range(n):
        # Evitar división por cero
        if abs(A[i, i]) < 1e-12:
            raise ValueError("Pivote cercano a cero")

        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - factor * A[i, i:]
            b[j] = b[j] - factor * b[i]

    # Back substitution
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x


# =========================
# KINE CONTROL CON EG
# =========================
def ik_dls_gauss(fkine, q, xd, lamb=0.05):

    # FK
    x = TF2xyzquat(fkine(q))

    # Normalización
    xd, x = normalize_quaternions(xd, x)

    # Error
    e = pose_error(xd, x)

    # Jacobiano
    J = numerical_jacobian(fkine, q, TF2xyzquat)

    # Sistema lineal
    A = J @ J.T + (lamb**2) * np.eye(6)

    # Resolver con Gauss
    y = gaussian_elimination(A.copy(), e.copy())

    # dq final
    dq = J.T @ y

    return q + dq

# =========================
# Gauss_seidel
# =========================

def gauss_seidel(A, b, max_iter=100, tol=1e-6, y0=None):

    n = len(b)

    # Punto inicial
    if y0 is None:
        y = np.zeros(n)
    else:
        y = y0.copy()

    for k in range(max_iter):

        y_old = y.copy()

        for i in range(n):

            sum1 = np.dot(A[i, :i], y[:i])        
            sum2 = np.dot(A[i, i+1:], y_old[i+1:])  

            y[i] = (b[i] - sum1 - sum2) / A[i, i]

        if np.linalg.norm(y - y_old) < tol:
            return y, k+1

    return y, max_iter

    return y, max_iter

def ik_dls_seidel(fkine, q, xd, lamb=0.1):

    x = TF2xyzquat(fkine(q))
    xd, x = normalize_quaternions(xd, x)
    e = pose_error(xd, x)

    J = numerical_jacobian(fkine, q, TF2xyzquat)

    A = J @ J.T + (lamb**2)*np.eye(6)

    y, iters = gauss_seidel(A, e, max_iter=100, tol=1e-6)

    dq = J.T @ y

    print(f"Iteraciones Gauss-Seidel: {iters}")

    return q + dq