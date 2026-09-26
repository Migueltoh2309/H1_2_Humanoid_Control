import numpy as np
from copy import copy
from h1_2_algoritms.fk_functions import *

def fk_elbow_left_unitree(q):
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

    T = Tbase @ T1 @ T2 @ T3 @ T4

    return T

def fk_elbow_right_unitree(q):
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

    T = Tbase @ T1 @ T2 @ T3 @ T4

    return T

def numerical_jacobian_position(fk_func, q, delta=1e-6):
    n = len(q)
    J = np.zeros((3, n))

    x = fk_func(q)[0:3, 3]

    for i in range(n):
        q_d = q.copy()
        q_d[i] += delta

        x_d = fk_func(q_d)[0:3, 3]

        J[:, i] = (x_d - x) / delta

    return J

