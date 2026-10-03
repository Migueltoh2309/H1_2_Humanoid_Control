"""Localización 3D del CENTRO de la mandarina: cuatro estimadores
comparables (ver VISUAL_SERVOING_PLAN.md, sección "Percepción: depth vs.
estéreo IR").

Todos parten de la misma detección 2D en la imagen RGB (máscara HSV del
Método A, `perception_common.detect_color`) y devuelven el centro de la
esfera en el frame óptico de la cámara RGB (X der., Y abajo, Z adelante):

  median     profundidad mediana dentro de la máscara, deproyectada en el
             centroide. Es lo que hacen hoy los nodos de percepción. Mide la
             SUPERFICIE visible, no el centro: sesgo de ~0.7 R hacia la
             cámara (PERCEPTION_PLAN.md §8 midió ~35 mm en z).
  median+R   la anterior corrida a lo largo del rayo por 0.707·R: la mediana
             de la profundidad sobre el casquete visible de una esfera cae,
             para una distribución uniforme en el área del disco, en
             ρ² = 1/2, es decir a z_c − R·√(1−1/2) del centro. Corrección
             geométrica cerrada, sin iterar.
  sphere     ajuste de esfera de radio conocido a la nube deproyectada de
             la máscara (erosionada para no usar los píxeles de borde, que
             en un sensor real son los peores). Gauss-Newton sobre
             Σ(|p_i − c| − R)², unas pocas iteraciones.
  stereo_ir  triangulación de la silueta en el par IR (izq./der., 50 mm de
             línea de base, imágenes rectificadas). No usa la profundidad:
             el centro de la silueta de una esfera en cada imagen es (casi)
             la proyección de su centro, así que la disparidad de los dos
             centros da directamente la profundidad del CENTRO. La
             detección en IR no puede usar color (es monocromo): se proyecta
             la estimación RGB a cada imagen IR como región de interés y ahí
             se segmenta por intensidad (Otsu) + ajuste de elipse al
             contorno.

Sin ROS: lo usan igual el script de evaluación (`demos/evaluate_ir_stereo.py`)
y el de visual servoing (`demos/visual_servoing_grasp.py`).
"""
from dataclasses import dataclass

import cv2
import numpy as np

MANDARINA_RADIUS = 0.04   # [m] — el de la escena; en real, medirlo o estimarlo


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_fovy(width, height, fovy_deg):
        """Igual que el bridge (msg_builders.camera_info_msg): píxeles
        cuadrados, fovy vertical, sin distorsión."""
        f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
        return Intrinsics(f, f, width / 2.0, height / 2.0)

    def project(self, p):
        return np.array([self.fx * p[0] / p[2] + self.cx, self.fy * p[1] / p[2] + self.cy])

    def ray(self, u, v):
        d = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        return d / np.linalg.norm(d)


# ---------------------------------------------------------------- RGB + depth
def segment_orange(rgb, hue_range=(0, 22), sat_min=210, val_min=170, min_area=40):
    """Máscara (uint8) del blob naranja más grande, o None. Mismo umbral
    HSV que el Método A (perception_common.DEFAULT_*): con S/V más
    permisivos (probado: S≥150, V≥100) la mesa mostaza entra como el blob
    más grande y el error salta a ~1 m."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([hue_range[0], sat_min, val_min], np.uint8),
                       np.array([hue_range[1], 255, 255], np.uint8))
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, [c], -1, 255, cv2.FILLED)
    return out


def _centroid(mask):
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def _valid_depth(depth, mask, zmin=0.05, zmax=3.0):
    z = depth[mask.astype(bool)]
    return z[np.isfinite(z) & (z > zmin) & (z < zmax)]


def locate_median(mask, depth, K: Intrinsics, radius=MANDARINA_RADIUS):
    uv = _centroid(mask)
    z = _valid_depth(depth, mask)
    if uv is None or z.size == 0:
        return None
    zm = float(np.median(z))
    return np.array([(uv[0] - K.cx) * zm / K.fx, (uv[1] - K.cy) * zm / K.fy, zm])


def locate_median_radius(mask, depth, K: Intrinsics, radius=MANDARINA_RADIUS):
    p = locate_median(mask, depth, K)
    if p is None:
        return None
    # La mediana es de z (profundidad, no distancia): se corrige z y se
    # reescala x,y a lo largo del mismo rayo.
    zc = p[2] + radius * np.sqrt(0.5)
    return p * (zc / p[2])


def locate_sphere(mask, depth, K: Intrinsics, radius=MANDARINA_RADIUS, erode_px=2, iters=10):
    m = mask
    if erode_px > 0:
        m = cv2.erode(mask, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
        if m.sum() == 0:
            m = mask
    v, u = np.nonzero(m)
    z = depth[v, u]
    ok = np.isfinite(z) & (z > 0.05) & (z < 3.0)
    if ok.sum() < 10:
        return None
    u, v, z = u[ok], v[ok], z[ok]
    P = np.column_stack(((u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z))
    # Arranque: punto medio empujado un radio a lo largo del rayo.
    c = P.mean(axis=0)
    c = c + radius * c / np.linalg.norm(c)
    for _ in range(iters):
        d = P - c
        n = np.linalg.norm(d, axis=1) + 1e-12
        r = n - radius
        J = -d / n[:, None]
        # Mínimos cuadrados robustos (Huber, 5 mm): un par de píxeles de
        # borde mal medidos no deben arrastrar el centro.
        w = np.where(np.abs(r) < 0.005, 1.0, 0.005 / np.abs(r))
        JTJ = (J * w[:, None]).T @ J
        dc = np.linalg.solve(JTJ + 1e-9 * np.eye(3), -(J * w[:, None]).T @ r)
        c = c + dc
        if np.linalg.norm(dc) < 1e-6:
            break
    return c


# ---------------------------------------------------------------- estéreo IR
def _ir_blob_center(ir, uv_pred, r_pred_px, search=2.5):
    """Centro subpíxel de la silueta de la fruta en una imagen IR, buscando
    en una ventana alrededor de la predicción. Otsu separa fruta (gris
    medio) de faja (blanca) dentro de la ventana; se toma la componente
    conexa que contiene/está más cerca de la predicción y se ajusta una
    elipse a su contorno (usa todos los puntos de borde: precisión
    subpíxel, mucho mejor que el centroide de píxeles enteros)."""
    h, w = ir.shape
    half = int(max(12, search * r_pred_px))
    u0, v0 = int(round(uv_pred[0])), int(round(uv_pred[1]))
    x0, x1 = max(0, u0 - half), min(w, u0 + half)
    y0, y1 = max(0, v0 - half), min(h, v0 + half)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    roi = cv2.GaussianBlur(ir[y0:y1, x0:x1], (5, 5), 0)
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return None
    pred = np.array([uv_pred[0] - x0, uv_pred[1] - y0])
    area_pred = np.pi * r_pred_px ** 2
    best, best_cost = None, np.inf
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.2 * area_pred or area > 3.0 * area_pred:
            continue
        cost = np.linalg.norm(cents[i] - pred) / r_pred_px + abs(np.log(area / area_pred))
        if cost < best_cost:
            best, best_cost = i, cost
    if best is None:
        return None
    comp = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=len)
    # Solo el tramo del contorno que NO toca el borde de la ventana (si la
    # fruta está parcialmente tapada por la mano, el borde del oclusor no
    # es borde de la esfera — el ajuste de elipse lo ignora así).
    pts = c[:, 0, :].astype(float)
    inner = (pts[:, 0] > 1) & (pts[:, 0] < roi.shape[1] - 2) & (pts[:, 1] > 1) & (pts[:, 1] < roi.shape[0] - 2)
    pts = pts[inner]
    if len(pts) < 6:
        return None
    (cx, cy), (a, b), _ = cv2.fitEllipse(pts.astype(np.float32))
    return np.array([cx + x0, cy + y0]), 0.25 * (a + b)


def locate_stereo_ir(ir_left, ir_right, K_ir: Intrinsics, baseline, prior_left,
                     radius=MANDARINA_RADIUS, max_dev=0.03):
    """prior_left: estimación grosera del centro en el frame óptico del IR
    izquierdo (de la RGB-D). Devuelve el centro en ese mismo frame, o None.

    Par rectificado (misma orientación, desplazado `baseline` en +X): una
    fruta en (X, Y, Z) cae en u_L = f X/Z + cx, u_R = f (X − B)/Z + cx, v
    igual en ambas. Z = f B / (u_L − u_R)."""
    if prior_left is None or prior_left[2] <= 0.05:
        return None
    r_px = K_ir.fx * radius / prior_left[2]
    uvL = K_ir.project(prior_left)
    uvR = K_ir.project(prior_left - np.array([baseline, 0.0, 0.0]))
    L = _ir_blob_center(ir_left, uvL, r_px)
    R = _ir_blob_center(ir_right, uvR, r_px)
    if L is None or R is None:
        return None
    (uL, vL), _ = L
    (uR, vR), _ = R
    disp = uL - uR
    if disp <= 0.5 or abs(vL - vR) > 3.0:     # rectificado: v debe coincidir
        return None
    Z = K_ir.fx * baseline / disp
    v = 0.5 * (vL + vR)
    c = np.array([(uL - K_ir.cx) * Z / K_ir.fx, (v - K_ir.cy) * Z / K_ir.fy, Z])
    # Consistencia con la RGB-D: con B = 50 mm a ~0.8 m, 1 px de disparidad
    # son ~28 mm de profundidad; si el blob elegido en una de las dos
    # imágenes no es la fruta (bordes del campo, la mano), el salto es de
    # decímetros. Se descarta en vez de propagarlo.
    if np.linalg.norm(c - prior_left) > max_dev:
        return None
    return c


# ---------------------------------------------------------------- ruido
def realsense_depth_noise(depth, rng, f_px, baseline=0.05, subpixel=0.08,
                          corr_px=3, edge_invalid=True):
    """Modelo de ruido de la D435 para una profundidad perfecta (MuJoCo):

    * RMS según la hoja de datos de Intel: σ_z = z² · subpixel / (f · B)
      (subpixel ≈ 0.08 px, f del imager IR, B = 50 mm). Espacialmente
      correlacionado (el matcher agrega ventanas): ruido blanco suavizado
      con un gaussiano de `corr_px` y re-escalado a σ_z.
    * "Flying pixels" / huecos en discontinuidades: el matcher estéreo no
      encuentra correspondencia en los bordes de ocultamiento; se invalidan
      (NaN) los píxeles donde la profundidad salta > 2 cm entre vecinos.

    Es un MODELO (no una medida): sirve para comparar métodos bajo el mismo
    ruido, no para predecir el error absoluto del robot real."""
    z = depth.astype(np.float32).copy()
    n = rng.standard_normal(z.shape).astype(np.float32)
    if corr_px > 0:
        n = cv2.GaussianBlur(n, (0, 0), corr_px)
        n /= n.std() + 1e-9
    sigma = z ** 2 * subpixel / (f_px * baseline)
    z = z + sigma * n
    if edge_invalid:
        gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
        gy = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
        edge = cv2.dilate(((gx > 0.02) | (gy > 0.02)).astype(np.uint8), np.ones((3, 3), np.uint8))
        z[edge.astype(bool)] = np.nan
    return z


def image_noise(img, rng, sigma=3.0):
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
