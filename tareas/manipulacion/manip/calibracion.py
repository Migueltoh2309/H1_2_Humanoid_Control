"""Bloque de CALIBRACIÓN cámara-torso (pregunta 6.1.3), con el ArUco del dorso de la mano.

Cada pose quieta i del brazo da dos medidas del mismo marcador:
  * la cadena del robot: T_torso_muneca(q_i) por la FK de la tesis (con las juntas MEDIDAS);
  * la cámara: dónde ve las 4 esquinas del marcador en la imagen (y, por solvePnP,
    T_cam_marcador_i).
Las incógnitas son X = T_torso_cam (lo que se busca) e Y_lado = T_muneca_marcador (dónde
quedó pegado el marcador en cada mano: nadie lo sabe al milímetro). La ecuación
T_torso_muneca(q_i) · Y = X · T_cam_marcador_i es la calibración mano-ojo "robot-mundo"
(AX = YB; Shah 2013, Li et al. 2010).

Se resuelve en dos pasos:
  1. arranque: X y las Y nominales (URDF y dónde se pegó el marcador), refinadas con un
     ajuste de puntos 3D (centros de los marcadores);
  2. ajuste fino: mínimos cuadrados no lineales (scipy, pérdida robusta) del ERROR DE
     REPROYECCIÓN de las 4 esquinas de todos los marcadores, en píxeles. Es el error que de
     verdad se mide (solvePnP con un marcador de 4 cm tiene mucha incertidumbre en la
     profundidad; las esquinas en la imagen no).
Residuo declarado: px (reproyección) y mm (centro del marcador por la cadena frente al de
solvePnP), medio y máximo, y por pose (para ver una mala).

La cámara calibrada es la de IR1, donde vive la profundidad nativa que usa la percepción.
El marcador se detecta en la imagen de IR1 (ve toda la mesa; el color solo la mitad lejana).
"""
import datetime
import json
import math
import os

import numpy as np

from . import cinematica as C

ESQUINAS_ARUCO = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=float) / 2.0  # x lado


# ------------------------------------------------------------------ utilidades SE(3)
def T_de(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return T_de(R.T, -R.T @ t)


def rotvec_a_R(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def T_de_vec(p):
    return T_de(rotvec_a_R(p[:3]), p[3:6])


def vec_de_T(T):
    return np.concatenate([C.log_rot(T[:3, :3]), T[:3, 3]])


def ang_deg(R1, R2):
    return math.degrees(math.acos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


# ------------------------------------------------------------------ detección
def detector(diccionario="DICT_4X4_50"):
    import cv2
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, diccionario))
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dic, par)


def detectar(img, K, lado_m, det=None, ids_validos=None):
    """{id: {'esquinas': 4x2 px, 'T_cam_marcador': 4x4, 'reproy_px': float}} en `img` (gris)."""
    import cv2
    det = det or detector()
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    esquinas, ids, _ = det.detectMarkers(img)
    out = {}
    if ids is None:
        return out
    obj = ESQUINAS_ARUCO * lado_m
    for e, i in zip(esquinas, ids.ravel()):
        i = int(i)
        if ids_validos is not None and i not in ids_validos:
            continue
        e = e.reshape(4, 2).astype(np.float64)
        ok, rv, tv = cv2.solvePnP(obj, e, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        proy, _ = cv2.projectPoints(obj, rv, tv, K, None)
        out[i] = {"esquinas": e, "T_cam_marcador": T_de(cv2.Rodrigues(rv)[0], tv.ravel()),
                  "reproy_px": float(np.sqrt(np.mean(np.sum((proy.reshape(4, 2) - e) ** 2, axis=1))))}
    return out


def centro_3d(esquinas, prof, K, ventana=3):
    """Centro del marcador en 3D (marco de la cámara) con la PROFUNDIDAD del píxel central
    (mediana de una ventana): mucho más preciso a 1 m que la distancia de solvePnP con un
    marcador de 4 cm. La profundidad nativa está en el marco de IR1, el mismo de la imagen."""
    u, v = np.mean(esquinas, axis=0)
    iu, iv = int(round(u)), int(round(v))
    vent = prof[max(iv - ventana, 0):iv + ventana + 1, max(iu - ventana, 0):iu + ventana + 1]
    z = vent[vent > 0]
    if z.size == 0:
        return None
    z = float(np.median(z))
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def proyectar(K, P):
    """Puntos Nx3 del marco óptico -> píxeles Nx2."""
    return np.column_stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]])


# ------------------------------------------------------------------ nominales
def camara_nominal(cfg, ir=True):
    """T_torso_cam nominal (URDF de Unitree) del marco óptico de IR1 (o del color)."""
    cam = cfg["camara"]
    t = math.radians(cam["nominal"]["inclinacion_deg"])
    xc, yc = np.array([0, -1.0, 0]), np.array([math.sin(t), 0, math.cos(t)])
    Ro = np.column_stack([xc, -yc, -np.cross(xc, yc)])
    pos = np.array(cam["nominal"]["pos"]) + (np.array([0, 0.015, 0]) if ir else 0.0)
    return T_de(Ro, pos)


def marcador_nominal(lado):
    """T_muneca_marcador donde se pegó (centro del dorso, 'arriba' del marcador hacia los dedos)."""
    centro, normal = C.DORSO[lado]
    ey = np.array([1.0, 0.0, 0.0])
    ex = np.cross(ey, normal)
    return T_de(np.column_stack([ex, ey, normal]), centro)


# ------------------------------------------------------------------ ajuste
def resolver(muestras, K, lado_m, X0, Y0, usar="medida"):
    """muestras: lista de dicts {'lado', 'q_medida', 'q_consigna', 'esquinas' (4x2),
    'T_cam_marcador'}. Devuelve (X, {lado: Y}, informe)."""
    from scipy.optimize import least_squares
    lados = sorted({m["lado"] for m in muestras})
    T_tm = [C.fk(m["lado"], m["q_" + usar]) for m in muestras]
    obj = ESQUINAS_ARUCO * lado_m
    obj_h = np.column_stack([obj, np.ones(4)])

    def desempaquetar(p):
        X = T_de_vec(p[:6])
        Ys = {lado: T_de_vec(p[6 + 6 * k:12 + 6 * k]) for k, lado in enumerate(lados)}
        return X, Ys

    def residuos(p):
        X, Ys = desempaquetar(p)
        Xi = inv(X)
        r = []
        for m, T in zip(muestras, T_tm):
            P = (Xi @ T @ Ys[m["lado"]] @ obj_h.T).T[:, :3]
            if np.any(P[:, 2] <= 0.05):
                r.append(np.full(8, 100.0))
                continue
            r.append((proyectar(K, P) - m["esquinas"]).ravel())
        return np.concatenate(r)

    p0 = np.concatenate([vec_de_T(X0)] + [vec_de_T(Y0[lado]) for lado in lados])
    sol = least_squares(residuos, p0, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=2000)
    X, Ys = desempaquetar(sol.x)
    # informe: reproyección (px) y centro del marcador cadena vs solvePnP (mm), por pose
    r = residuos(sol.x).reshape(len(muestras), 4, 2)
    px = np.sqrt(np.mean(np.sum(r ** 2, axis=2), axis=1))
    # centro medido: con la profundidad del píxel si la hay (precisa), si no el de solvePnP
    mm = []
    for m, T in zip(muestras, T_tm):
        c_cadena = (inv(X) @ T @ Ys[m["lado"]])[:3, 3]
        c_med = m["centro_3d"] if m.get("centro_3d") is not None else m["T_cam_marcador"][:3, 3]
        mm.append(1000 * np.linalg.norm(c_cadena - c_med))
    mm = np.array(mm)
    # la parte de la profundidad de solvePnP es la más ruidosa: también en el plano de la imagen
    mm_lat = []
    for m, T in zip(muestras, T_tm):
        c = (inv(X) @ T @ Ys[m["lado"]])[:3, 3]
        d = c - m["T_cam_marcador"][:3, 3]
        u = m["T_cam_marcador"][:3, 3] / np.linalg.norm(m["T_cam_marcador"][:3, 3])
        mm_lat.append(1000 * np.linalg.norm(d - (d @ u) * u))
    informe = {"n_poses": len(muestras), "usar": usar, "reproy_px_rms": float(np.sqrt(np.mean(px ** 2))),
               "reproy_px_max": float(px.max()), "mm_medio": float(mm.mean()), "mm_max": float(mm.max()),
               "mm_lateral_medio": float(np.mean(mm_lat)), "mm_lateral_max": float(np.max(mm_lat)),
               "por_pose_px": px.round(3).tolist(), "por_pose_mm": mm.round(2).tolist(),
               "coste": float(sol.cost), "evaluaciones": int(sol.nfev), "convergio": bool(sol.success)}
    return X, Ys, informe


def validar_tablero(X, detecciones, posiciones_torso):
    """Marcadores fijos del tablero vistos por la cámara y llevados al torso con X, frente a
    donde están (regla en el robot; geometría de la escena en el simulador). mm por id."""
    out = {}
    for i, det in detecciones.items():
        if i in posiciones_torso:
            p = (X @ det["T_cam_marcador"])[:3, 3]
            out[int(i)] = float(1000 * np.linalg.norm(p - np.asarray(posiciones_torso[i])))
    return out


def guardar(ruta_dir, X, Ys, informe, K, extra=None):
    fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(ruta_dir, exist_ok=True)
    d = {"fecha": fecha, "marco_camara": "camera_infra1_optical_frame (profundidad nativa)",
         "T_torso_cam": X.tolist(), "T_muneca_marcador": {k: v.tolist() for k, v in Ys.items()},
         "K": np.asarray(K).tolist(), "informe": informe, "metodo":
         "AX=YB con el ArUco del dorso: arranque nominal + ajuste de reproyección de esquinas (scipy, soft_l1)"}
    if extra:
        d.update(extra)
    ruta = os.path.join(ruta_dir, f"calib_{fecha}.json")
    with open(ruta, "w") as fh:
        json.dump(d, fh, indent=1)
    with open(os.path.join(ruta_dir, "ultima.json"), "w") as fh:
        json.dump(d, fh, indent=1)
    return ruta


def cargar(ruta):
    """La calibración (la última si `ruta` es una carpeta). Única fuente de T_torso_cam."""
    if os.path.isdir(ruta):
        ruta = os.path.join(ruta, "ultima.json")
    with open(ruta) as fh:
        d = json.load(fh)
    d["T_torso_cam"] = np.array(d["T_torso_cam"])
    d["K"] = np.array(d["K"])
    return d
