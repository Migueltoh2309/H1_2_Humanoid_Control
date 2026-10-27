#!/usr/bin/env python3
"""Posturas fijas del reto, calculadas contra el modelo (sin robot): config/poses.yaml.

  * observacion: los dos brazos colgando a los lados, FUERA de la vista de la mesa (en la
    postura de pie las manos quedan delante del pecho, a ~12 cm sobre la mesa, y tapan el
    7.7 % del tablero visto desde IR1). Se busca en una rejilla de hombro y codo la postura
    que menos tablero tapa, con >= 3 cm a la cadera y al torso, y cerca de la de pie (menos
    movimiento, menos reacción del cuerpo).
  * calibracion: por mano, poses con el ArUco del dorso visible y de frente a la cámara
    (pregunta 6.1.3). Para que AX = YB tenga solución única las rotaciones deben girar
    alrededor de ejes distintos: palma abajo con la mano inclinada ±15-25 grados en dos
    ejes y el azimut variado, a 10-25 cm sobre la mesa, repartidas sobre la zona visible.
    Cada candidata: IK de la tesis, holgura >= 3 cm a mesa, cuerpo y otro brazo (mallas del
    MJCF), marcador dentro de la imagen de IR1, normal a < 50 grados de la cámara y el
    marcador visto de verdad en el render (no lo tapa la propia mano). Se eligen N lo más
    distintas posible (máxima distancia mínima en posición y orientación).

La visibilidad se calcula con la pose NOMINAL de la cámara (la real aún no se conoce: es lo
que se va a calibrar); hay margen de sobra para los ~2 grados de diferencia.

Uso:  MUJOCO_GL=egl python3 herramientas/poses.py [--n 10]
"""
import argparse
import math
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sim"))
from manip import cinematica as C  # noqa: E402
from manip.config import cargar, cargar_objetos  # noqa: E402
from manip.estacion import Estacion  # noqa: E402
from manip.modelo import Modelo, Vista  # noqa: E402

ESPEJO = np.array([1, -1, -1, 1, -1, 1, -1], dtype=float)   # izq -> der (caja_cuadrado SIGNO_ESPEJO)
HOLGURA = 0.03


def rot(eje, ang):
    c, s = math.cos(ang), math.sin(ang)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[eje]


def observacion(mod, vista, q_pie, est, altura_seguridad=0.10):
    """Mejor postura colgando para la mano izquierda (la derecha es la espejo). También a la
    altura de seguridad sobre la mesa: es el principio y el final de todos los tránsitos."""
    mejor = None
    z_min = est.z_mesa - est.torso_z + altura_seguridad + 0.01
    for sp in np.arange(-0.1, 0.61, 0.1):
        for sr in np.arange(0.15, 0.61, 0.05):
            for el in np.arange(-0.9, 0.31, 0.1):
                q = np.array([sp, sr, 0.0, el, 0.0, 0.0, 0.0])
                if C.mano_baja("izq", q, est.tablero) < z_min:
                    continue
                mod.poner(q, q * ESPEJO)
                h = mod.holguras("izq", con_otro_brazo=False)
                if min(h.values()) < HOLGURA:
                    continue
                tapa, _ = vista.tablero_tapado()
                coste = tapa * 100 + 0.2 * np.abs(q - q_pie).sum()
                if mejor is None or coste < mejor[0]:
                    mejor = (coste, q, tapa, min(h.values()))
    return mejor


def proyectar(p_cam, K):
    return np.array([K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2], K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]])


def candidatas_calibracion(lado, mod, vista, est, cfg, q_obs, rng, n_cand=500):
    """Poses del brazo `lado` con el marcador del dorso bien visible."""
    m, d = mod.m, mod.d
    cam = m.camera("robot_ir_left").id
    w, h = (int(v) for v in m.cam_resolution[cam])
    f = h / 2 / math.tan(math.radians(m.cam_fovy[cam]) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    s = 1.0 if lado == "izq" else -1.0
    centro_m, normal_m = C.DORSO[lado]
    ids_marcador = {g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(f"aruco_{lado}")}
    out = []
    zona = est.zona
    for _ in range(n_cand):
        # muñeca sobre la mitad de la mesa de ese lado, a 10-25 cm, palma abajo inclinada
        x = rng.uniform(zona.cx - zona.fondo / 2 - 0.05, zona.cx + zona.fondo / 2)
        y = s * rng.uniform(-0.05, zona.ancho / 2 + 0.10)
        z = est.z_mesa + rng.uniform(0.10, 0.25) - est.torso_z
        az = s * rng.uniform(-70, 10)
        R = C.rotacion_agarre(lado, "arriba", rng.uniform(-25, 10), az) @ rot("x", math.radians(rng.uniform(-25, 25)))
        q, ep, eo, ok = C.ik(lado, [x, y, z], R, nombre="palma")
        if not ok:
            continue
        qi, qd = (q, q_obs["der"]) if lado == "izq" else (q_obs["izq"], q)
        mod.poner(qi, qd)
        hol = mod.holguras(lado)
        if min(hol.values()) < HOLGURA:
            continue
        # marcador en el marco de la cámara (MuJoCo: -z adelante) -> óptico
        T = C.fk(lado, q)
        c_t = T[:3, 3] + T[:3, :3] @ centro_m
        n_t = T[:3, :3] @ normal_m
        Rc = d.cam_xmat[cam].reshape(3, 3)
        pc = d.cam_xpos[cam]
        tor = m.body("torso_link").id
        Rt, pt = d.xmat[tor].reshape(3, 3), d.xpos[tor]
        c_w, n_w = pt + Rt @ c_t, Rt @ n_t
        v = Rc.T @ (c_w - pc)
        p_opt = np.array([v[0], -v[1], -v[2]])
        if p_opt[2] < 0.3:
            continue
        uv = proyectar(p_opt, K)
        if not (40 < uv[0] < w - 40 and 40 < uv[1] < h - 40):
            continue
        hacia_cam = (pc - c_w) / np.linalg.norm(pc - c_w)
        ang = math.degrees(math.acos(np.clip(n_w @ hacia_cam, -1, 1)))
        if ang > 50:
            continue
        # el render lo ve (no lo tapa la propia mano ni el otro brazo)
        _, seg = vista.tablero_tapado()
        u, vv = int(uv[0]), int(uv[1])
        parche = seg[vv - 3:vv + 4, u - 3:u + 4]
        if np.mean(np.isin(parche, list(ids_marcador))) < 0.6:
            continue
        out.append({"q": q, "p": c_t, "R": T[:3, :3], "ang_camara": ang, "uv": uv,
                    "holgura": min(hol.values())})
    return out


def elegir(cands, n):
    """Máxima distancia mínima: posición (m) + orientación (rad * 0.15 m)."""
    if not cands:
        return []
    elegidas = [cands[int(np.argmin([c["ang_camara"] for c in cands]))]]
    while len(elegidas) < min(n, len(cands)):
        def dist(a, b):
            ang = math.acos(np.clip((np.trace(a["R"].T @ b["R"]) - 1) / 2, -1, 1))
            return np.linalg.norm(a["p"] - b["p"]) + 0.15 * ang
        k = int(np.argmax([min(dist(c, e) for e in elegidas) for c in cands]))
        elegidas.append(cands[k])
    return elegidas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="poses de calibración por mano")
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    cfg = cargar()
    est = Estacion(cfg)
    from construir_estacion import construir
    import tempfile
    sin_desvio = {k: 0.0 for k in ("cabeceo", "guinada", "alabeo", "dx", "dy", "dz")}
    xml, _ = construir(0, 0, cfg, cargar_objetos(), desvio=sin_desvio, salida=tempfile.mkdtemp())
    mod = Modelo(xml)
    vista = Vista(mod)
    q_pie = mod.d.qpos[mod.qadr["izq"]].copy()
    lejos = {"izq": np.array([0.3, 1.2, 0, -0.5, 0, 0, 0]), "der": np.array([0.3, -1.2, 0, -0.5, 0, 0, 0])}
    vista.tablero_tapado(lejos)
    mod.poner()
    tapa_pie, _ = vista.tablero_tapado()
    coste, q_obs, tapa, hol = observacion(mod, vista, q_pie, est)
    print(f"de pie: tapa el {tapa_pie * 100:.1f} % del tablero (IR1). Observación: {np.round(q_obs, 2).tolist()} "
          f"tapa el {tapa * 100:.2f} %, holgura {hol * 100:.1f} cm")
    q_obs = {"izq": q_obs, "der": q_obs * ESPEJO}
    rng = np.random.default_rng(a.semilla)
    poses = {"observacion": {k: [round(float(x), 4) for x in v] for k, v in q_obs.items()},
             "calibracion": {}}
    for lado in C.LADOS:
        cands = candidatas_calibracion(lado, mod, vista, est, cfg, q_obs, rng)
        sel = elegir(cands, a.n)
        print(f"{lado}: {len(cands)} candidatas válidas, elegidas {len(sel)}; ángulo a la cámara "
              f"{min(c['ang_camara'] for c in sel):.0f}-{max(c['ang_camara'] for c in sel):.0f} grados, "
              f"holgura mín {min(c['holgura'] for c in sel) * 100:.1f} cm")
        poses["calibracion"][lado] = [[round(float(x), 4) for x in c["q"]] for c in sel]
    vista.cerrar()
    import yaml
    ruta = os.path.join(RAIZ, "config", "poses.yaml")
    with open(ruta, "w") as fh:
        fh.write("# GENERADO por herramientas/poses.py (contra el modelo, cámara nominal). Juntas en el orden\n"
                 "# de la tesis: [hombro cabeceo, alabeo, guiñada, codo, muñeca alabeo, cabeceo, guiñada].\n")
        yaml.safe_dump(poses, fh, default_flow_style=None, sort_keys=False)
    print("escrito", ruta)


if __name__ == "__main__":
    main()
