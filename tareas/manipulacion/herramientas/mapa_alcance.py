#!/usr/bin/env python3
"""Hito 1 — mapa de alcance y elección de la distancia d (preguntas 6.1.1 y 6.1.2).

Para cada punto de una rejilla sobre el plano de la mesa (a `--altura` sobre el tablero:
el centro de agarre de un objeto de ~6 cm) y para cada mano, resuelve la IK (la cadena de
la tesis, manip/cinematica.py) con dos familias de orientación:
  * "arriba": palma hacia abajo, dedos con elevación 0 / -20 / -40 grados y azimut de
    -90 a 90 cada 15 (la guiñada del agarre que se puede pedir en ese punto);
  * "lado": palma vertical mirando a la línea media, elevación 0 / -20, azimut -60..60;
y da el punto por alcanzado con una orientación si:
  * la IK converge (2 mm, 2 grados) dentro de los límites del URDF (muñeca ±26.5 grados) y
    con el alabeo del hombro hacia dentro <= 0.10 rad;
  * antebrazo y mano a >= 1 cm de la cápsula del torso (modelo de colisión de la tesis);
  * ninguna cápsula del brazo, ni los dedos estirados ni el pulgar abierto, por debajo
    del tablero (+5 mm) en la parte del brazo que queda encima de la mesa. Esto depende
    de d (dónde empieza la mesa), así que se evalúa para cada d de la lista.

Lo que ve la cámara: proyección con la pose NOMINAL del URDF (la real sale del hito 2)
y los intrínsecos del reto, en el color (56 x 43 grados) y en la profundidad nativa (IR1,
80 x 65 grados), con la profundidad mínima de la D435. No descuenta lo que tapen los brazos
(en reposo están delante del pecho, a 12 cm sobre la mesa: hace falta una pose de
observación, hito 2).

Salida (datos/alcance/): mapa_alcance.npz (todo), mapa_alcance.json (resumen y d
elegida), mapa_d<d>.png (vista en planta para cada d) y curvas_d.png.

Uso:  python3 herramientas/mapa_alcance.py                    # ~2 min (2 procesos)
      python3 herramientas/mapa_alcance.py --paso 0.03 --rapido   # comprobación rápida
      python3 herramientas/mapa_alcance.py --cintura 0.35        # con la cintura girada (rad)
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
from manip import cinematica as C  # noqa: E402
from manip.config import cargar  # noqa: E402
from manip.estacion import Estacion  # noqa: E402

FAMILIAS = {
    "arriba": {"elev": [0.0, -20.0, -40.0], "az": list(range(-90, 91, 15))},
    "lado": {"elev": [0.0, -20.0], "az": list(range(-60, 61, 20))},
}
D_LISTA = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
MARGEN_MESA = 0.005
MARGEN_TORSO = 0.01
# dedos estirados (4 rectas de la base de los dedos a la punta) y pulgar abierto, en el
# marco de la muñeca (mallas del MJCF; el pulgar de la tesis, THUMB_CAPSULE)
DEDOS = [(np.array([0.165, 0.0, z]), np.array([0.268, 0.0, z]), 0.010) for z in (-0.025, 0.0, 0.015, 0.03)]
PULGAR = (np.array([0.11, 0.0, 0.02]), np.array([0.17, 0.0, 0.11]), 0.012)


def puntos_colision(lado, q, fr):
    """(P, r): puntos de muestra de todas las cápsulas (brazo + dedos + pulgar) en
    torso_link y su radio."""
    P, r = [], []
    for c in C.capsulas(lado, q, fr):
        for s in np.linspace(0, 1, 6):
            P.append(c["a"] + s * (c["b"] - c["a"]))
            r.append(c["r"])
    T = fr[7]
    sgn = 1.0 if lado == "izq" else -1.0
    for a, b, rr in DEDOS + [PULGAR]:
        a, b = a * np.array([1, sgn, 1]), b * np.array([1, sgn, 1])
        for s in np.linspace(0, 1, 4):
            P.append(T[:3, 3] + T[:3, :3] @ (a + s * (b - a)))
            r.append(rr)
    return np.array(P), np.array(r)


def holgura_mesa(P, r, est_d, z_mesa_t, ancho):
    """Para cada d: holgura mínima (m) de los puntos que quedan encima del tablero."""
    out = []
    for x_borde in est_d:
        sel = (P[:, 0] + r >= x_borde) & (np.abs(P[:, 1]) - r <= ancho / 2)
        out.append(np.min(P[sel, 2] - r[sel] - z_mesa_t) if sel.any() else np.inf)
    return np.array(out)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def trabajar(args):
    """Un brazo, toda la rejilla. Devuelve los booleanos de alcance y las posturas."""
    lado, xs, ys, z_t, cintura, x_bordes, ancho, z_mesa_t = args
    Rw = rot_z(-cintura)                       # suelo -> torso girado por la cintura
    lim = C.limites(lado)
    fams = {f: [(e, a) for e in v["elev"] for a in v["az"]] for f, v in FAMILIAS.items()}
    res = {f: np.zeros((len(xs), len(ys), len(o), len(x_bordes)), bool) for f, o in fams.items()}
    pos_ok = np.zeros((len(xs), len(ys)), bool)
    q_pos = np.full((len(xs), len(ys), 7), np.nan)
    q_ant = {f: [None] * len(o) for f, o in fams.items()}
    q_prev_punto = None
    R_or = {f: [C.rotacion_agarre(lado, "arriba" if f == "arriba" else "lado", e, a) for e, a in o]
            for f, o in fams.items()}
    for ix, x in enumerate(xs):
        for iy in range(len(ys)):
            y = ys[iy]
            p = Rw @ np.array([x, y, z_t])
            ok = False
            for s in (q_prev_punto, C.postura_inicial(lado)):
                if s is None:
                    continue
                q, ep, eo, ok = C.ik(lado, p, None, q0=s, lim=lim, tol_pos=0.003, iters=80)
                if ok:
                    break
            if not ok:
                continue
            pos_ok[ix, iy] = True
            q_pos[ix, iy] = q
            q_prev_punto = q
            for f, ors in fams.items():
                for k in range(len(ors)):
                    Rd = Rw @ R_or[f][k]
                    for s in (q_ant[f][k], q):
                        if s is None:
                            continue
                        qk, ep, eo, okk = C.ik(lado, p, Rd, q0=s, lim=lim, iters=80)
                        if okk:
                            break
                    if not okk:
                        continue
                    q_ant[f][k] = qk
                    fr = C.frames(lado, qk)
                    if C.dist_torso(lado, qk, fr) < MARGEN_TORSO:
                        continue
                    P, r = puntos_colision(lado, qk, fr)
                    res[f][ix, iy, k] = holgura_mesa(P, r, x_bordes, z_mesa_t, ancho) >= MARGEN_MESA
    return lado, pos_ok, q_pos, res


def visibilidad(cfg, est, X, Y, z, cintura=0.0, margen_px=4):
    """(color, profundidad nativa): booleanos sobre la rejilla, con la pose nominal."""
    cam = cfg["camara"]
    t = math.radians(cam["nominal"]["inclinacion_deg"])
    xc, yc = np.array([0, -1.0, 0]), np.array([math.sin(t), 0, math.cos(t)])
    Ro = np.column_stack([xc, -yc, -np.cross(xc, yc)])        # óptico: x der, y abajo, z adelante
    pos = np.array(cam["nominal"]["pos"])
    Rw = rot_z(cintura)                                       # torso -> suelo
    P = np.stack([X, Y, np.full_like(X, z - est.torso_z)], -1)
    P = P @ Rw                                                # suelo -> torso (Rw^T p)
    out = []
    for nombre, off in (("color", 0.0), ("ir", 0.015)):
        k = cam[nombre]
        cx, cy = k.get("cx", k["ancho"] / 2), k.get("cy", k["alto"] / 2)
        pc = (P - (pos + np.array([0, off, 0]))) @ Ro
        zc = pc[..., 2]
        u = k["fx"] * pc[..., 0] / np.maximum(zc, 1e-6) + cx
        v = k["fy"] * pc[..., 1] / np.maximum(zc, 1e-6) + cy
        out.append((zc > cam["profundidad_min"]) & (u >= margen_px) & (u < k["ancho"] - margen_px)
                   & (v >= margen_px) & (v < k["alto"] - margen_px))
    return out


def dibujar(ruta, xs, ys, est, alc, vis_c, vis_d, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ext = [ys[0] - (ys[1] - ys[0]) / 2, ys[-1] + (ys[1] - ys[0]) / 2, xs[0] - (xs[1] - xs[0]) / 2,
           xs[-1] + (xs[1] - xs[0]) / 2]
    img = np.ones((len(xs), len(ys), 3))
    izq, der = alc["izq"], alc["der"]
    img[izq & ~der] = (0.55, 0.70, 1.0)
    img[der & ~izq] = (1.0, 0.62, 0.55)
    img[izq & der] = (0.72, 0.55, 0.90)
    ax.imshow(img, origin="lower", extent=ext, aspect="equal", interpolation="nearest")
    # y a la izquierda del robot = a la izquierda de la figura (vista desde arriba, robot abajo)
    ax.contour(ys, xs, vis_c.astype(float), levels=[0.5], colors="green", linewidths=1.5)
    ax.contour(ys, xs, vis_d.astype(float), levels=[0.5], colors="darkgreen", linewidths=1.2, linestyles="--")
    for r_, col, lw in [(est.tablero, "k", 1.5), (est.zona, "gray", 1.2)] + \
            [(b, "saddlebrown", 1.2) for b in est.bandejas.values()]:
        ax.add_patch(Rectangle((r_.cy + r_.ancho / 2, r_.cx - r_.fondo / 2), -r_.ancho, r_.fondo,
                               fill=False, ec=col, lw=lw))
    for letra, b in est.bandejas.items():
        ax.text(b.cy, b.cx, letra, ha="center", va="center", fontsize=11, weight="bold", color="saddlebrown")
    ax.plot([0.0], [est.pecho_x], "k^", ms=10)
    ax.set_xlim(ext[1], ext[0])      # +y (izquierda del robot) a la izquierda de la figura
    ax.set_ylim(min(ext[2], est.pecho_x - 0.05), ext[3])
    ax.set_xlabel("y (m, + izquierda del robot)")
    ax.set_ylabel("x (m, adelante)")
    ax.set_title(titulo, fontsize=10)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=(0.55, 0.70, 1.0), label="solo mano izq"),
                       Patch(color=(1.0, 0.62, 0.55), label="solo mano der"),
                       Patch(color=(0.72, 0.55, 0.90), label="las dos"),
                       Line2D([], [], color="green", label="ve el color"),
                       Line2D([], [], color="darkgreen", ls="--", label="ve la profundidad nativa")],
              loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--altura", type=float, default=0.03, help="altura del centro de agarre sobre la mesa (m)")
    ap.add_argument("--paso", type=float, default=0.02)
    ap.add_argument("--cintura", type=float, default=0.0, help="guiñada de la cintura (rad; + gira a la izquierda)")
    ap.add_argument("--mesa", type=float, default=None, help="altura del tablero (m; por defecto la de la config)")
    ap.add_argument("--procesos", type=int, default=2)
    ap.add_argument("--rapido", action="store_true", help="solo elevación 0 y azimut cada 30")
    ap.add_argument("--salida", default=os.path.join(RAIZ, "datos", "alcance"))
    a = ap.parse_args()
    if a.rapido:
        FAMILIAS["arriba"] = {"elev": [0.0], "az": list(range(-90, 91, 30))}
        FAMILIAS["lado"] = {"elev": [0.0], "az": [-30, 0, 30]}
    cfg = cargar()
    if a.mesa is not None:
        cfg["mesa"]["altura"] = a.mesa
    est0 = Estacion(cfg, d=D_LISTA[0])
    xs = np.round(np.arange(est0.x_borde - 0.04, est0.pecho_x + D_LISTA[-1] + cfg["mesa"]["fondo"] + 1e-9, a.paso), 4)
    ys = np.round(np.arange(-cfg["mesa"]["ancho"] / 2 - 0.04, cfg["mesa"]["ancho"] / 2 + 0.04 + 1e-9, a.paso), 4)
    z_t = est0.z_mesa + a.altura - est0.torso_z
    x_bordes = [est0.pecho_x + d for d in D_LISTA]
    print(f"rejilla {len(xs)} x {len(ys)} (paso {a.paso} m), agarre a {a.altura * 100:.0f} cm sobre la mesa "
          f"(z = {z_t:+.3f} en torso_link), cintura {math.degrees(a.cintura):.0f} grados; "
          f"orientaciones: arriba {len(FAMILIAS['arriba']['elev']) * len(FAMILIAS['arriba']['az'])}, "
          f"lado {len(FAMILIAS['lado']['elev']) * len(FAMILIAS['lado']['az'])}", flush=True)
    t0 = time.time()
    trabajos = [(lado, xs, ys, z_t, a.cintura, x_bordes, cfg["mesa"]["ancho"], est0.z_mesa - est0.torso_z)
                for lado in C.LADOS]
    with get_context("spawn").Pool(min(a.procesos, 2)) as pool:
        resultados = {r[0]: r[1:] for r in pool.map(trabajar, trabajos)}
    print(f"IK resueltas en {time.time() - t0:.0f} s", flush=True)

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    vis_c, vis_d = visibilidad(cfg, est0, X, Y, est0.z_mesa + a.altura, a.cintura)
    os.makedirs(a.salida, exist_ok=True)
    sufijo = ("" if a.altura == 0.03 else f"_h{a.altura * 100:.0f}") + \
        ("" if a.cintura == 0 else f"_cintura{math.degrees(a.cintura):.0f}") + \
        ("" if a.mesa is None else f"_mesa{a.mesa * 100:.0f}")
    np.savez_compressed(os.path.join(a.salida, f"mapa_alcance{sufijo}.npz"), xs=xs, ys=ys, d_lista=D_LISTA,
                        vis_color=vis_c, vis_prof=vis_d,
                        **{f"{lado}_pos": resultados[lado][0] for lado in C.LADOS},
                        **{f"{lado}_q": resultados[lado][1] for lado in C.LADOS},
                        **{f"{lado}_{f}": resultados[lado][2][f] for lado in C.LADOS for f in FAMILIAS})

    def alcanza(lado, f, kd, elev_max=20.0):
        """Alcanzado con alguna orientación de la familia con |elevación| <= elev_max."""
        ors = [(e, az) for e in FAMILIAS[f]["elev"] for az in FAMILIAS[f]["az"]]
        sel = [k for k, (e, _) in enumerate(ors) if abs(e) <= elev_max]
        return resultados[lado][2][f][:, :, sel, kd].any(-1)

    def az_posibles(lado, f, kd, ix, iy, elev_max=20.0):
        ors = [(e, az) for e in FAMILIAS[f]["elev"] for az in FAMILIAS[f]["az"]]
        r = resultados[lado][2][f]
        return sorted({az for k, (e, az) in enumerate(ors) if abs(e) <= elev_max and r[ix, iy, k, kd]})

    def celda(x, y):
        return int(np.argmin(np.abs(xs - x))), int(np.argmin(np.abs(ys - y)))

    resumen = {"altura_agarre": a.altura, "altura_mesa": est0.z_mesa,
               "cintura_rad": a.cintura, "paso": a.paso, "por_d": {},
               "familias": FAMILIAS, "criterio": "palma abajo = elevacion de los dedos 0 o -20 grados"}
    curvas = {k: [] for k in ("zona_alcanza", "zona_alcanza_ve_prof", "zona_alcanza_ve_color", "zona_ve_color",
                              "zona_ve_prof", "zona_lado")}
    for kd, d in enumerate(D_LISTA):
        est = Estacion(cfg, d=d)
        zona = est.zona.contiene(X, Y)
        alc = {lado: alcanza(lado, "arriba", kd) for lado in C.LADOS}
        lado_ = {lado: alcanza(lado, "lado", kd) for lado in C.LADOS}
        cualquiera = alc["izq"] | alc["der"]
        n = max(int(zona.sum()), 1)
        fila = {
            "x_borde": est.x_borde,
            "zona_alcanza": float((cualquiera & zona).sum() / n),
            "zona_alcanza_izq": float((alc["izq"] & zona).sum() / n),
            "zona_alcanza_der": float((alc["der"] & zona).sum() / n),
            "zona_lado": float(((lado_["izq"] | lado_["der"]) & zona).sum() / n),
            # con los dedos hasta 40 grados hacia abajo (palma inclinada)
            "zona_alcanza_40": float(((alcanza("izq", "arriba", kd, 40.0) | alcanza("der", "arriba", kd, 40.0))
                                      & zona).sum() / n),
            "zona_ve_color": float((vis_c & zona).sum() / n),
            "zona_ve_prof": float((vis_d & zona).sum() / n),
            "zona_alcanza_ve_color": float((cualquiera & vis_c & zona).sum() / n),
            "zona_alcanza_ve_prof": float((cualquiera & vis_d & zona).sum() / n),
            "x_max_alcanza": float(X[cualquiera & zona].max()) if (cualquiera & zona).any() else None,
            "bandejas": {},
        }
        for letra, b in est.bandejas.items():
            ix, iy = celda(b.cx, b.cy)
            propia = est.lado_bandeja(letra)
            otra = "der" if propia == "izq" else "izq"
            fila["bandejas"][letra] = {
                "centro": [b.cx, b.cy], "mano_propia": propia,
                "alcanza_propia": bool(alc[propia][ix, iy]), "alcanza_contraria": bool(alc[otra][ix, iy]),
                "azimuts_propia": az_posibles(propia, "arriba", kd, ix, iy),
                "ve_color": bool(vis_c[ix, iy]), "ve_prof": bool(vis_d[ix, iy])}
        ix, iy = celda(est.zona.cx, est.zona.cy)
        fila["centro_zona"] = {"azimuts_izq": az_posibles("izq", "arriba", kd, ix, iy),
                               "azimuts_der": az_posibles("der", "arriba", kd, ix, iy)}
        resumen["por_d"][f"{d:.2f}"] = fila
        for k in curvas:
            curvas[k].append(fila[k])
        titulo = (f"d = {d * 100:.0f} cm · palma abajo a {a.altura * 100:.0f} cm sobre la mesa"
                  + (f" · cintura {math.degrees(a.cintura):.0f} grados" if a.cintura else "") +
                  f"\nzona: alcanza {fila['zona_alcanza'] * 100:.0f} %, alcanza y la ve la profundidad "
                  f"{fila['zona_alcanza_ve_prof'] * 100:.0f} %, el color {fila['zona_alcanza_ve_color'] * 100:.0f} %")
        dibujar(os.path.join(a.salida, f"mapa_d{d * 100:.0f}{sufijo}.png"), xs, ys, est, alc, vis_c, vis_d, titulo)

    # d elegida: lo máximo de la zona que se alcanza Y se ve en profundidad; a igualdad, la menor
    pun = np.array(curvas["zona_alcanza_ve_prof"]) + 0.5 * np.array(curvas["zona_alcanza"])
    kd = int(np.flatnonzero(pun >= pun.max() - 1e-9)[0])
    resumen["d_elegida"] = D_LISTA[kd]
    resumen["criterio_d"] = "max(zona alcanzada y vista en profundidad + 0.5 x zona alcanzada); a igualdad, la menor d"
    with open(os.path.join(a.salida, f"mapa_alcance{sufijo}.json"), "w") as fh:
        json.dump(resumen, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    etiquetas = {"zona_alcanza": "alcanza (palma abajo, alguna mano)",
                 "zona_alcanza_ve_prof": "alcanza y ve la profundidad",
                 "zona_alcanza_ve_color": "alcanza y ve el color", "zona_ve_color": "ve el color",
                 "zona_ve_prof": "ve la profundidad", "zona_lado": "alcanza con la palma de lado"}
    for k, et in etiquetas.items():
        ax.plot(np.array(D_LISTA) * 100, np.array(curvas[k]) * 100, "o-" if "alcanza" in k else "s--", label=et)
    ax.axvline(D_LISTA[kd] * 100, color="k", lw=0.8)
    ax.set_xlabel("d: borde cercano de la mesa al pecho (cm)")
    ax.set_ylabel("% de la zona de recogida")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.salida, f"curvas_d{sufijo}.png"), dpi=110)

    print(f"\n{'d':>5} {'alcanza':>8} {'izq':>5} {'der':>5} {'lado':>5} {'ve col':>7} {'ve prof':>8} "
          f"{'alc+prof':>9} {'alc+col':>8}  x max   bandejas propia/contraria")
    for d, fila in resumen["por_d"].items():
        bj = " ".join(f"{k}:{'s' if v['alcanza_propia'] else 'n'}{'s' if v['alcanza_contraria'] else 'n'}"
                      for k, v in fila["bandejas"].items())
        print(f"{d:>5} {fila['zona_alcanza'] * 100:7.0f}% {fila['zona_alcanza_izq'] * 100:4.0f}% "
              f"{fila['zona_alcanza_der'] * 100:4.0f}% {fila['zona_lado'] * 100:4.0f}% "
              f"{fila['zona_ve_color'] * 100:6.0f}% "
              f"{fila['zona_ve_prof'] * 100:7.0f}% {fila['zona_alcanza_ve_prof'] * 100:8.0f}% "
              f"{fila['zona_alcanza_ve_color'] * 100:7.0f}%  {fila['x_max_alcanza'] or 0:.2f}  {bj}")
    print(f"\nd elegida: {D_LISTA[kd]:.2f} m  ({resumen['criterio_d']})")
    print(f"salida: {a.salida}")


if __name__ == "__main__":
    main()
