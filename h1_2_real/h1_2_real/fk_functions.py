import numpy as np
from copy import copy

cos=np.cos; sin=np.sin; pi=np.pi

# =========================
# DENAVIT HATTENBERG
# =========================

def dh(d, theta, a, alpha):
  """
  Calcular la matriz de transformacion homogenea asociada con los parametros
  de Denavit-Hartenberg.
  Los valores d, theta, a, alpha son escalares.
  """
  # Escriba aqui la matriz de transformacion homogenea en funcion de los valores de d, theta, a, alpha
  sth = np.sin(theta)
  cth = np.cos(theta)
  sa  = np.sin(alpha)
  ca  = np.cos(alpha)
  T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                [sth,  ca*cth, -sa*cth, a*sth],
                [0.0,      sa,      ca,     d],
                [0.0,     0.0,     0.0,   1.0]])
  return T

# =========================
# KFINE DE BRAZO IZQUIERDO
# =========================

def fkine_arm_left_unitree(q):
    """
    Cinemática directa del brazo izquierdo del Unitree H1-2
    
    q: numpy array de 7 elementos
       [shoulder_pitch,
        shoulder_roll,
        shoulder_yaw,
        elbow,
        wrist_roll,
        wrist_pitch,
        wrist_yaw]
    """
    # =========================
    # DH estructural 7R
    # =========================

    Tbase = np.array([
            [0.0, -1.0,  0.0, 0.0],
            [0.25881905, 0.0, 0.96592583,  0.20794643],
            [-0.96592583, 0.0, 0.25881905, 0.43937656],
            [0.0, 0.0, 0.0, 1.0]
        ])
    
    T1 = dh(0.0, q[0],                           0.0060011,  np.pi / 2.0)
    T2 = dh(0.0, q[1] - (np.pi / 2.0 + 0.2618),  0.0,        np.pi / 2.0)
    T3 = dh(-0.3276, q[2] + np.pi / 2.0,         0.006,     -np.pi / 2.0)
    T4 = dh(0.0, q[3] + np.pi / 2.0,             0.011,      np.pi / 2.0)
    T5 = dh(0.208, q[4] + np.pi,                 0.0,        np.pi / 2.0)
    T6 = dh(0.0, q[5] + np.pi / 2.0,             0.020,      np.pi / 2.0)
    T7 = dh(0.0, q[6],                           0.0,        0.0)

    # Transformación total
    T = Tbase.dot(T1).dot(T2).dot(T3).dot(T4).dot(T5).dot(T6).dot(T7)

    return T

# =========================
# KFINE DE BRAZO DERECHO
# =========================

def fkine_arm_right_unitree(q):
    """
    Cinemática directa del brazo derecho del Unitree H1-2
    
    q: numpy array de 7 elementos
       [shoulder_pitch,
        shoulder_roll,
        shoulder_yaw,
        elbow,
        wrist_roll,
        wrist_pitch,
        wrist_yaw]
    """
    # =========================
    # DH estructural 7R
    # =========================

    Tbase = np.array([
            [0.0, -1.0,  0.0, 0.0],
            [-0.25881905, 0.0, 0.96592583, -0.20794643],
            [-0.96592583, 0.0, -0.25881905, 0.43937656],
            [0.0, 0.0, 0.0, 1.0],
        ])
    
    T1 = dh(0.0, q[0],                           0.0060011,  np.pi / 2.0)
    T2 = dh(0.0, q[1] - (np.pi / 2.0 - 0.2618),  0.0,        np.pi / 2.0)
    T3 = dh(-0.3276, q[2] + np.pi / 2.0,         0.006,     -np.pi / 2.0)
    T4 = dh(0.0, q[3] + np.pi / 2.0,             0.011,      np.pi / 2.0)
    T5 = dh(0.208, q[4] + np.pi,                 0.0,        np.pi / 2.0)
    T6 = dh(0.0, q[5] + np.pi / 2.0,             0.020,      np.pi / 2.0)
    T7 = dh(0.0, q[6],                           0.0,        0.0)

    # Transformación total
    T = Tbase.dot(T1).dot(T2).dot(T3).dot(T4).dot(T5).dot(T6).dot(T7)

    return T

# =========================
# MATRIZ DE ROTACIÓN A CUATERNIONES
# =========================

def rot2quat(R):
    q = np.zeros(4)
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2,1] - R[1,2]) / s
        q[2] = (R[0,2] - R[2,0]) / s
        q[3] = (R[1,0] - R[0,1]) / s

    else:
        # buscar mayor diagonal
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            q[0] = (R[2,1] - R[1,2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0,1] + R[1,0]) / s
            q[3] = (R[0,2] + R[2,0]) / s

        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            q[0] = (R[0,2] - R[2,0]) / s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,2] + R[2,1]) / s

        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            q[0] = (R[1,0] - R[0,1]) / s
            q[1] = (R[0,2] + R[2,0]) / s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = 0.25 * s

    return q / (np.linalg.norm(q) + 1e-8)

# =========================
# TF A CUATERNIONES
# =========================

def TF2xyzquat(T):
  """
  Convert a homogeneous transformation matrix into the a vector containing the
  pose of the robot.
  
  Input:
   T -- A homogeneous transformation
  Output:
   X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
        is Cartesian coordinates and the last part is a quaternion
  """
  quat = rot2quat(T[0:3,0:3])
  res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
  return np.array(res)

# ===== FUNCIÓN AUXILIAR =====
def compute_pose(self, fk_func, q):
    T = fk_func(q)
    x = TF2xyzquat(T)
    return T, x

