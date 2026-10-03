"""Obstáculos a partir de la cámara de profundidad del torso.

Pipeline (numpy puro, sin ROS, para que lo usen igual el nodo en vivo y el
script de evaluación offline):

    depth (HxW, metros, 32FC1)
      -> submuestreo (stride) + rango válido
      -> deproyección pinhole a frame óptico (X der., Y abajo, Z adelante)
      -> transformación a torso_link (T_torso_cam, estática)
      -> recorte a la caja de trabajo de los brazos
      -> auto-filtrado: fuera los puntos del propio robot (collision_model)
      -> voxelizado (un punto por celda)

El resultado (Nx3 en torso_link) entra directo a
`collision_model.obstacle_constraints`.

La pose de la cámara es fija respecto a torso_link (va montada en el
torso). `camera_pose_from_mjcf_xyaxes` la reconstruye a partir del
`<camera pos=... xyaxes=...>` del MJCF, que es lo mismo que publica el
bridge como TF estático torso_link -> camera_depth_optical_frame; el nodo
prefiere leer el TF y usa esto como respaldo.
"""
import numpy as np

from h1_2_algoritms.bimanual.collision_model import robot_self_filter

# robot_rgbd_camera en h1_2_scene_*.xml (cuelga de torso_link).
MJCF_CAMERA_POS = np.array([0.11109, 0.01750, 0.68789])
MJCF_CAMERA_XYAXES = np.array([0.0, -1.0, 0.0, 0.774503060, 0.0, 0.632570162])

# Caja de trabajo en torso_link [m]: frente al robot, a la altura de las
# manos. Todo lo que quede fuera no puede tocar los brazos en el horizonte
# de un ciclo de control, así que no vale la pena pasarlo al QP.
DEFAULT_WORKSPACE = (np.array([-0.10, -0.65, -0.45]), np.array([0.85, 0.65, 0.75]))


def camera_pose_from_mjcf_xyaxes(pos=MJCF_CAMERA_POS, xyaxes=MJCF_CAMERA_XYAXES):
    """T (4x4) del frame ÓPTICO de ROS expresado en torso_link.

    En MuJoCo la cámara mira hacia -z_cam con y_cam hacia arriba; el frame
    óptico de ROS (REP-103) tiene X = x_cam, Y = -y_cam, Z = -z_cam."""
    x = xyaxes[:3] / np.linalg.norm(xyaxes[:3])
    y = xyaxes[3:] / np.linalg.norm(xyaxes[3:])
    z = np.cross(x, y)
    T = np.eye(4)
    T[0:3, 0:3] = np.column_stack((x, -y, -z))
    T[0:3, 3] = pos
    return T


def depth_to_points(depth, fx, fy, cx, cy, stride=4, depth_min=0.1, depth_max=2.0):
    """Nube Nx3 en el frame óptico. `stride` submuestrea filas/columnas:
    640x480 con stride 4 son ~19k puntos, de sobra para voxeles de 2-3 cm."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[0:h:stride, 0:w:stride]
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)
    z = z[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.column_stack((x, y, z))


def transform_points(T, P):
    return P @ T[0:3, 0:3].T + T[0:3, 3]


def crop_box(P, lo, hi):
    return P[np.all((P >= lo) & (P <= hi), axis=1)]


def voxel_downsample(P, voxel):
    """Un punto (el centroide) por celda cúbica de lado `voxel`."""
    if len(P) == 0 or voxel <= 0:
        return P
    keys = np.floor(P / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.zeros((len(counts), 3))
    np.add.at(out, inv, P)
    return out / counts[:, None]


def remove_near(P, centers, radius):
    """Quita los puntos a menos de `radius` de cualquiera de `centers`. Se
    usa para no tratar como obstáculo al objeto que se quiere agarrar."""
    if len(P) == 0 or centers is None or len(centers) == 0:
        return P
    keep = np.ones(len(P), dtype=bool)
    for c in centers:
        keep &= np.linalg.norm(P - c, axis=1) > radius
    return P[keep]


def obstacles_from_depth(depth, intrinsics, T_torso_cam, frames_l, frames_r,
                         stride=4, voxel=0.03, self_padding=0.04,
                         workspace=DEFAULT_WORKSPACE, depth_max=2.0,
                         exclude_centers=None, exclude_radius=0.0):
    """Pipeline completo. intrinsics = (fx, fy, cx, cy); frames_* son los
    frames DH de cada brazo en la configuración en que se tomó la imagen.
    Devuelve la nube de obstáculos (Nx3, torso_link)."""
    fx, fy, cx, cy = intrinsics
    P = depth_to_points(depth, fx, fy, cx, cy, stride=stride, depth_max=depth_max)
    P = transform_points(T_torso_cam, P)
    P = crop_box(P, *workspace)
    P = robot_self_filter(P, frames_l, frames_r, self_padding)
    P = remove_near(P, exclude_centers, exclude_radius)
    return voxel_downsample(P, voxel)


class ObstacleMemory:
    """Memoria de vóxeles de obstáculo con borrado por espacio libre.

    Problema que resuelve — la OCLUSIÓN: la cámara está en el torso, detrás
    y por encima de las manos. Cuando una mano baja hacia la mesa, tapa
    justo la zona de la mesa que tiene debajo; si solo se usa el último
    cuadro, el punto de obstáculo más cercano que queda visible está
    varios cm al costado y el damper deja que la mano siga bajando hasta
    tocar (medido en demos/evaluate_bimanual_avoidance.py, escenario
    'table').

    Regla: un vóxel observado se recuerda hasta que la cámara vea A TRAVÉS
    de él, es decir, cuando al proyectarlo en la imagen nueva el píxel mide
    una profundidad mayor que la del vóxel (+ `carve_tol`): ahí ya no hay
    nada. Si el píxel mide MENOS (algo delante, p.ej. el brazo) el vóxel
    sigue ocluido y se mantiene. Fuera del campo de visión se mantiene
    hasta `max_age` s. Es el mismo principio que el ray-casting de
    OctoMap, reducido a una proyección por vóxel (barato: ~1k vóxeles).
    """

    def __init__(self, voxel=0.03, carve_tol=0.04, max_age=30.0):
        self.voxel = voxel
        self.carve_tol = carve_tol
        self.max_age = max_age
        self.points = np.zeros((0, 3))
        self.stamps = np.zeros(0)

    def update(self, new_points, depth, intrinsics, T_torso_cam, t, depth_min=0.1):
        """new_points: obstáculos del cuadro actual (ya filtrados, torso_link).
        depth: el mismo cuadro, para decidir qué vóxeles viejos borrar."""
        keep = self._still_occupied(depth, intrinsics, T_torso_cam, depth_min)
        keep &= (t - self.stamps) <= self.max_age
        old_p, old_t = self.points[keep], self.stamps[keep]
        P = np.vstack((new_points, old_p)) if len(new_points) else old_p
        ts = np.hstack((np.full(len(new_points), t), old_t))
        if len(P) == 0:
            self.points, self.stamps = P, ts
            return self.points
        # Un vóxel por celda; ante duplicados gana el más reciente (va primero).
        keys = np.floor(P / self.voxel).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        self.points, self.stamps = P[first], ts[first]
        return self.points

    def _still_occupied(self, depth, intrinsics, T_torso_cam, depth_min):
        n = len(self.points)
        if n == 0:
            return np.zeros(0, dtype=bool)
        fx, fy, cx, cy = intrinsics
        R, p = T_torso_cam[0:3, 0:3], T_torso_cam[0:3, 3]
        Pc = (self.points - p) @ R          # torso -> óptico
        z = Pc[:, 2]
        keep = np.ones(n, dtype=bool)
        in_front = z > 1e-3
        u = np.full(n, -1)
        v = np.full(n, -1)
        u[in_front] = np.round(Pc[in_front, 0] * fx / z[in_front] + cx).astype(int)
        v[in_front] = np.round(Pc[in_front, 1] * fy / z[in_front] + cy).astype(int)
        h, w = depth.shape
        vis = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        meas = depth[v[vis], u[vis]]
        # Solo se borra con evidencia positiva de espacio libre: un píxel
        # inválido (NaN, o algo pegado a la lente) no dice nada.
        valid = np.isfinite(meas) & (meas > depth_min)
        seen_through = valid & (meas > z[vis] + self.carve_tol)
        keep[np.flatnonzero(vis)[seen_through]] = False
        return keep


def intrinsics_from_fovy(width, height, fovy_deg):
    """Mismos intrínsecos que publica el bridge (msg_builders.camera_info_msg):
    píxeles cuadrados, fovy vertical."""
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return fy, fy, width / 2.0, height / 2.0
