"""Percepción del NIVEL 1: las dianas (pegatinas de 2 cm con una cruz) sobre el tablero.

Son planas: la profundidad no las ve. Se buscan en IR1 (el marco calibrado de la profundidad
nativa, que ve toda la zona; el color solo ve su parte lejana):
  1. mediana 5x5: quita los puntos del emisor de la D435 (píxeles sueltos muy brillantes);
  2. cada píxel se lleva al plano de la mesa MEDIDO (rayo de IR1 con la calibración ∩ plano)
     y solo cuenta lo que cae dentro de la zona de recogida, a 1.5 cm de su borde (la cinta);
  3. más oscuro que el tablero (fracción de su mediana): la pegatina sobre el blanco mate;
  4. componentes conexas con el área de un disco de 2 cm a esa distancia (0.4-2.2 veces),
     compactas (llenado de la caja >= 0.5, proporción <= 1.8);
  5. centro = centroide del disco, llevado al plano de la mesa (marco del torso).

En el robot: a 850 nm una pegatina naranja puede verse casi tan clara como el tablero. Se
comprueba en el hito 2 con las pegatinas de la sesión; si no contrastan, se usan oscuras o el
color (donde llega).
"""
import numpy as np


def rayos_a_plano(us, vs, K, T_torso_cam, plano):
    """Píxeles (u, v) de la cámara -> puntos del plano n·p + d = 0 (marco del torso)."""
    n, d = np.asarray(plano[:3]), float(plano[3])
    r_cam = np.column_stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones(len(us))])
    R, o = T_torso_cam[:3, :3], T_torso_cam[:3, 3]
    r = r_cam @ R.T
    t = -(n @ o + d) / (r @ n)
    return o + r * t[:, None]


def detectar(f, T_torso_cam, plano, zona, diametro=0.02, oscuro=0.85, margen=0.015, paso=1):
    """Dianas en el Fotograma `f` (IR1 + K_prof). Devuelve [(centro_torso (3,), area_px, contraste)]."""
    import cv2
    if f.ir is None or plano is None or not np.all(np.isfinite(plano)):
        return []
    ir = cv2.medianBlur(np.asarray(f.ir, np.uint8), 5).astype(np.float32)
    h, w = ir.shape
    v, u = np.mgrid[0:h:paso, 0:w:paso]
    P = rayos_a_plano(u.ravel().astype(float), v.ravel().astype(float), f.K_prof, T_torso_cam, plano)
    en_zona = zona.contiene(P[:, 0], P[:, 1], margen=margen).reshape(u.shape)
    if en_zona.sum() < 100:
        return []
    ref = float(np.median(ir[::paso, ::paso][en_zona]))
    cand = (ir[::paso, ::paso] < oscuro * ref) & en_zona
    n, lab, stats, cent = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    out = []
    o = T_torso_cam[:3, 3]
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        cx, cy = cent[k]
        p = rayos_a_plano(np.array([cx * paso]), np.array([cy * paso]), f.K_prof, T_torso_cam, plano)[0]
        dist = float(np.linalg.norm(p - o))
        d_px = f.K_prof[0, 0] * diametro / dist / paso
        a_esp = np.pi * d_px ** 2 / 4
        if not (0.4 * a_esp <= area <= 2.2 * a_esp):
            continue
        if area / max(bw * bh, 1) < 0.5 or max(bw, bh) / max(min(bw, bh), 1) > 1.8:
            continue
        contraste = 1.0 - float(ir[::paso, ::paso][lab == k].mean()) / max(ref, 1.0)
        out.append((p, int(area), contraste))
    return out


def detectar_cruz(f, T_torso_cam, plano, centro, radio=0.06, oscuro=0.6):
    """Nivel 4: la cruz negra (10 cm, trazo de 8 mm) cerca de `centro` (x, y del torso; el centro
    del tablero de la config). Se mide con la MISMA cámara que el objeto: el sesgo de la pose del
    torso (unos mm) se cancela al comparar los dos. Devuelve (x, y, z) en el torso o None."""
    import cv2
    if f.ir is None or plano is None or not np.all(np.isfinite(plano)):
        return None
    ir = cv2.medianBlur(np.asarray(f.ir, np.uint8), 5).astype(np.float32)
    h, w = ir.shape
    v, u = np.mgrid[0:h, 0:w]
    P = rayos_a_plano(u.ravel().astype(float), v.ravel().astype(float), f.K_prof, T_torso_cam, plano)
    cerca = (np.hypot(P[:, 0] - centro[0], P[:, 1] - centro[1]) < radio).reshape(h, w)
    if cerca.sum() < 50:
        return None
    ref = float(np.percentile(ir[cerca], 80))               # el blanco del tablero alrededor
    cand = ((ir < oscuro * ref) & cerca).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < 20:
        return None
    sel = (lab == k).ravel()
    return P[sel].mean(axis=0)
