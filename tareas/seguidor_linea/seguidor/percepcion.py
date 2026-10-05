"""Bloque de PERCEPCION (reto, sec. 5 y 6.2): fotograma IR -> MedidaLinea en el suelo.

El mismo codigo para la camara en vivo (RealSense en el PC2), el simulador (ROS 2)
y el conjunto de datos grabado: solo recibe un Fotograma (+ roll/pitch opcional).

Pasos:
 1. Mediana (5 px): borra el patron de puntos del emisor IR (6.2.1).
 2. Vista cenital (BEV) del suelo a 1 cm/px en el marco del robot, con la
    homografia de geometria.py (compensada con roll/pitch de la IMU si se da).
    En la BEV la cinta mide 5 px de ancho a cualquier distancia: los umbrales de
    anchura son en metros.
 3. Cinta = mas oscura que su fondo local. Fondo = cierre morfologico con un nucleo
    de `fondo_m` (15 cm, el triple de la cinta): rellena lo oscuro y estrecho (cinta)
    y deja lo oscuro y ancho (sombras). Contraste relativo c = (fondo - I) / fondo:
    una sombra multiplica fondo y cinta por lo mismo, asi que c no cambia (6.2.2).
    Alternativa para comparar: umbral fijo de gris.
 4. Franjas de 5 cm de cerca a lejos; en cada una, tramos oscuros contiguos:
      ancho_min..ancho_max          candidato a linea
      >= transversal_min            barra de fin o esquina (6.2.4)
    Asociacion: se sigue la linea desde la franja mas cercana, con una puerta
    lateral respecto de lo extrapolado (la primera, respecto de la ultima medida).
 5. Ajuste y = c0 + c1 x + c2 x^2 por minimos cuadrados con un paso de rechazo de
    atipicos. Salidas en x = 0 (bajo el robot): desplazamiento, angulo, curvatura;
    objetivo a la distancia de mira; confianza.
 6. Barra: tramo transversal que cruza la linea por los dos lados (+-15 cm) con la
    linea acabando ahi. Esquina: tramo transversal hacia UN lado donde la linea acaba.

Confianza (6.2.5) = fraccion de franjas visibles con linea * calidad del ajuste
(residuo) * contraste medio, en 0..1.
"""
import math
import time

import cv2
import numpy as np

# Un solo hilo en OpenCV: con todos los nucleos ocupados (simulador + RViz + viewer en
# el PC, o los procesos de Unitree en el PC2), su pool de hilos se pisa con el resto y
# la percepcion paso de ~15 ms a ~415 ms por fotograma (medido). Las imagenes son
# pequenas: un hilo basta y el tiempo es predecible.
cv2.setNumThreads(1)

from .geometria import ModeloCamara, RejillaBEV, matriz_bev
from .tipos import Fotograma, MedidaLinea


class Percepcion:
    def __init__(self, cfg, cam: ModeloCamara = None):
        self.cfg = cfg
        p = cfg["percepcion"]
        self.p = p
        self.cam = cam
        self.rej = RejillaBEV(p["bev_x"][0], p["bev_x"][1], p["bev_y"][0], p["bev_y"][1], p["resolucion"])
        self.compensar = cfg["camara"].get("compensar_imu", True)
        k = max(3, int(round(p["fondo_m"] / p["resolucion"])) | 1)
        self.nucleo = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        self.ultimo_y0 = 0.0          # desplazamiento de la ultima medida buena (para asociar)
        self._mascara_fija = None

    # ------------------------------------------------------------------ BEV
    def asegurar_camara(self, f: Fotograma):
        if self.cam is None or (self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy) != tuple(f.K):
            self.cam = ModeloCamara.desde_config(self.cfg, K=f.K, ancho=f.ir.shape[1], alto=f.ir.shape[0])

    def bev(self, ir, d_roll=0.0, d_pitch=0.0):
        """(imagen cenital float32, mascara de pixeles validos)."""
        n_f, n_c = self.rej.forma
        M = matriz_bev(self.cam, self.rej, d_roll, d_pitch)
        flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        img = cv2.warpPerspective(ir.astype(np.float32), M, (n_c, n_f), flags=flags, borderValue=0)
        valido = cv2.warpPerspective(np.ones(ir.shape, np.uint8), M, (n_c, n_f), flags=cv2.WARP_INVERSE_MAP |
                                     cv2.INTER_NEAREST, borderValue=0)
        valido = cv2.erode(valido, np.ones((5, 5), np.uint8)).astype(bool)
        for x0, x1, y0, y1 in self.p.get("mascara_robot", []):
            c0, f1 = self.rej.a_pixel(x0, y1)
            c1, f0 = self.rej.a_pixel(x1, y0)
            valido[max(0, int(f0)):max(0, int(math.ceil(f1))), max(0, int(c0)):max(0, int(math.ceil(c1)))] = False
        return img, valido

    def mascara_cinta(self, img, valido):
        """(binaria de cinta, contraste relativo)."""
        if self.p["umbral"] == "fijo":
            m = (img < self.p["umbral_fijo"]) & valido
            return m, np.where(m, 1.0, 0.0).astype(np.float32)
        fondo = cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.nucleo)
        c = (fondo - img) / np.maximum(fondo, 8.0)
        m = (c > self.p["contraste_min"]) & valido
        # fuera los pixeles sueltos (ruido, puntos del emisor que sobrevivan)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        return m, c

    # ------------------------------------------------------------------ medida
    def procesar(self, f: Fotograma, roll=0.0, pitch=0.0, mira=1.0) -> MedidaLinea:
        t0 = time.perf_counter()
        self.asegurar_camara(f)
        k = self.p["mediana"]
        ir = cv2.medianBlur(f.ir, k) if k and k > 1 else f.ir
        dr, dp = (roll, pitch) if self.compensar else (0.0, 0.0)
        img, valido = self.bev(ir, dr, dp)
        m, contraste = self.mascara_cinta(img, valido)
        med = self._lineas(m, contraste, valido, f.t, mira)
        med.ms = (time.perf_counter() - t0) * 1000.0
        self.ultimo_bev = (img, valido, m)
        return med

    def _asociar(self, candidatos, bandas_c, y0, puerta0):
        """Sigue la linea franja a franja: la primera con `puerta0` alrededor de y0, las
        siguientes con la puerta normal alrededor de lo extrapolado."""
        puerta = self.p["puerta"]
        puntos, contrastes = [], []
        y_pred, pend, x_ant = y0, None, None
        for x, k, cand in candidatos:
            yp = y_pred if pend is None or x_ant is None else y_pred + pend * (x - x_ant)
            g = puerta if puntos else puerta0
            mejor, d_mejor = None, g
            for y_c, c0, c1 in cand:
                d = abs(y_c - yp)
                if d < d_mejor:
                    mejor, d_mejor = (y_c, c0, c1), d
            if mejor is None:
                continue
            y_c, c0, c1 = mejor
            if puntos:
                pend = (y_c - puntos[-1][1]) / (x - puntos[-1][0])
            puntos.append((x, y_c))
            contrastes.append(float(bandas_c[k, c0:c1].mean()))
            y_pred, x_ant = y_c, x
        return puntos, contrastes

    def _tramos(self, fila_bool):
        """[(col0, col1)] de tramos True contiguos."""
        d = np.diff(np.concatenate([[0], fila_bool.astype(np.int8), [0]]))
        ini, fin = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        return list(zip(ini, fin))

    def _lineas(self, m, contraste, valido, t, mira):
        p, rej = self.p, self.rej
        res = rej.res
        alto_franja = max(1, int(round(p["franja"] / res)))
        n_f, n_c = rej.forma
        # Estadisticas de TODAS las franjas de una vez (reshape): con un bucle de
        # operaciones numpy pequenas por franja, con la CPU ocupada (simulador + RViz +
        # viewer) la percepcion tardaba ~0.3 s por fotograma.
        n_b = n_f // alto_franja
        r0 = n_f - n_b * alto_franja                  # filas sobrantes arriba (lo mas lejano)
        bandas_m = m[r0:].reshape(n_b, alto_franja, n_c).mean(1) > 0.4
        bandas_v = valido[r0:].reshape(n_b, alto_franja, n_c).mean((1, 2))
        bandas_c = contraste[r0:].reshape(n_b, alto_franja, n_c).mean(1)
        # 1) candidatos de cada franja (de cerca a lejos) y tramos transversales
        transversales, visibles, x_visibles, candidatos = [], 0, [], []
        for k in range(n_b - 1, -1, -1):              # de cerca (abajo) a lejos
            f0 = r0 + k * alto_franja
            x = rej.x1 - (f0 + alto_franja / 2) * res
            if bandas_v[k] < 0.3:
                continue
            visibles += 1
            x_visibles.append(x)
            perfil = bandas_m[k]
            if not perfil.any():
                continue
            cand = []
            for c0, c1 in self._tramos(perfil):
                ancho = (c1 - c0) * res
                ya, yb = rej.y1 - (c1 - 1) * res, rej.y1 - c0 * res      # extremos en y
                if ancho >= p["transversal_min"]:
                    transversales.append((x, ya, yb))
                elif p["ancho_min"] <= ancho <= p["ancho_max"]:
                    cand.append((rej.y1 - (c0 + c1 - 1) / 2 * res, c0, c1))
            if cand:
                candidatos.append((x, k, cand))
        # 2) asociacion desde la ultima posicion buena; si no sale linea, otra vez desde el
        #    centro del robot con puerta amplia. (Sin ese segundo intento, una medida mala
        #    -con la marcha real, el robot enganchado a otra cinta junto al cuadro de inicio-
        #    dejaba la prediccion a 0.42 m y la linea real nunca entraba en la puerta.)
        puntos, contrastes = self._asociar(candidatos, bandas_c, self.ultimo_y0, max(p["puerta"], 0.35))
        if len(puntos) < p["puntos_min"] and abs(self.ultimo_y0) > 0.05:
            puntos, contrastes = self._asociar(candidatos, bandas_c, 0.0, 0.6)
        P = np.array(puntos) if puntos else np.zeros((0, 2))
        x_vis = (rej.x0, rej.x1)
        if len(P) < p["puntos_min"]:
            med = MedidaLinea(t=t, valida=False, puntos=P, x_visible=x_vis)
            med.barra, med.esquina = self._fin_y_esquina(P, None, transversales)
            return med
        # ajuste LOCAL: lo que usa el control es la linea cerca del robot; una sola
        # cuadratica no aproxima a la vez una recta y la curva que viene detras
        local = P[P[:, 0] <= P[0, 0] + p.get("ventana_ajuste", 1.4)]
        if len(local) < p["puntos_min"]:
            local = P[:p["puntos_min"]]
        coef, resid = self._ajuste(local)
        c0_, c1_, c2_ = coef
        x_m, y_m = self._punto_a_distancia(P, mira)
        # Confianza (6.2.5): continuidad entre el primer y el ultimo punto de linea (que la
        # linea ACABE en la barra o en una esquina no la baja; que tenga huecos si), por
        # la longitud vista, la calidad del ajuste y el contraste.
        xv = np.array(x_visibles)
        entre = int(((xv >= P[:, 0].min() - 1e-6) & (xv <= P[:, 0].max() + 1e-6)).sum())
        continuidad = len(P) / max(1, entre)
        largo = min(1.0, (P[:, 0].max() - P[:, 0].min()) / 0.8)
        # calidad: ¿forman los puntos una curva coherente? (no: ¿caben en una parabola?). Con la
        # marcha real, en la S del nivel 3 el robot va oblicuo a una curva de R 1.2 m que gira
        # ~90 grados dentro de la vista: la parabola y(x) dejaba residuos de > 5 cm, la confianza
        # caia a ~0 con la linea bien vista y el robot se perdia. Residuo de una CUBICA.
        resid_q = self._residuo_cubico(local) if len(local) >= 8 else resid
        calidad = math.exp(-(resid_q / 0.03) ** 2)
        cont = float(np.clip(np.median(contrastes) / 0.8, 0.0, 1.0)) if self.p["umbral"] != "fijo" else 1.0
        conf = float(np.clip(continuidad * 1.2, 0.0, 1.0) * largo * calidad * (0.5 + 0.5 * cont))
        med = MedidaLinea(t=t, valida=True, desplazamiento=float(c0_), angulo=float(math.atan(c1_)),
                          curvatura=float(2 * c2_ / (1 + c1_ * c1_) ** 1.5), objetivo=(x_m, float(y_m)),
                          confianza=conf, puntos=P, coef=coef, x_visible=x_vis)
        med.barra, med.esquina = self._fin_y_esquina(P, coef, transversales)
        if conf > self.cfg["estimacion"]["confianza_min"]:
            self.ultimo_y0 = float(np.polyval(coef[::-1], P[0, 0]))
        return med

    @staticmethod
    def _residuo_cubico(P):
        """RMS del ajuste cubico x -> y (sigue S y arcos oblicuos), con un paso de rechazo."""
        c = np.polyfit(P[:, 0], P[:, 1], 3)
        r = np.abs(np.polyval(c, P[:, 0]) - P[:, 1])
        ok = r < max(0.03, 2.5 * np.median(r))
        if ok.sum() >= 6:
            c = np.polyfit(P[ok, 0], P[ok, 1], 3)
            r = np.abs(np.polyval(c, P[ok, 0]) - P[ok, 1])
        return float(np.sqrt(np.mean(r ** 2)))

    @staticmethod
    def _punto_a_distancia(P, L):
        """Primer punto de la polilinea detectada a distancia L del robot (pure
        pursuit), interpolando; si no llega, el ultimo."""
        r = np.hypot(P[:, 0], P[:, 1])
        k = np.flatnonzero(r >= L)
        if len(k) == 0:
            return float(P[-1, 0]), float(P[-1, 1])
        i = int(k[0])
        if i == 0:
            return float(P[0, 0]), float(P[0, 1])
        a = (L - r[i - 1]) / max(r[i] - r[i - 1], 1e-9)
        q = P[i - 1] + a * (P[i] - P[i - 1])
        return float(q[0]), float(q[1])

    def _ajuste(self, P):
        """Cuadratica (o recta si hay poco tramo) con mas peso cerca del robot: lo que
        se extrapola a x = 0 (bajo el robot, zona ciega) depende sobre todo de los
        primeros puntos. Un paso de rechazo de atipicos."""
        grado = 2 if (P[:, 0].max() - P[:, 0].min()) > 0.6 and len(P) >= 6 else 1
        cerca = np.exp(-(P[:, 0] - P[0, 0]) / self.p.get("peso_cerca_m", 0.5))
        w = cerca.copy()
        for _ in range(2):
            c = np.polyfit(P[:, 0], P[:, 1], grado, w=np.sqrt(w))
            r = np.abs(np.polyval(c, P[:, 0]) - P[:, 1])
            w = cerca * np.where(r < max(0.03, 2.5 * np.median(r)), 1.0, 0.05)
        c = c[::-1]
        if grado == 1:
            c = np.array([c[0], c[1], 0.0])
        resid = float(np.sqrt(np.average(r ** 2, weights=w)))
        return c, resid

    def _recta_al_final(self, P, tramo=0.25, max_deg=12.0):
        """Una esquina en angulo recto llega por una RECTA; una curva (R 1.2 m del
        nivel 3), al ponerse horizontal en la imagen, tambien deja un tramo
        transversal, pero su rumbo cambia ~24 grados en los ultimos 50 cm. Se compara
        el rumbo de los dos ultimos tramos de 25 cm."""
        if len(P) < 6:
            return False
        xf = P[:, 0].max()
        a = P[(P[:, 0] > xf - tramo)]
        b = P[(P[:, 0] <= xf - tramo) & (P[:, 0] > xf - 2 * tramo)]
        if len(a) < 3 or len(b) < 3:
            return False
        th_a = math.atan(np.polyfit(a[:, 0], a[:, 1], 1)[0])
        th_b = math.atan(np.polyfit(b[:, 0], b[:, 1], 1)[0])
        return abs(math.degrees(th_a - th_b)) < max_deg

    def _fin_y_esquina(self, P, coef, transversales):
        """Barra de fin: tramo transversal que cubre la linea a ambos lados y donde
        la linea acaba. Esquina: tramo transversal que sale hacia un solo lado del
        final de la linea."""
        if not transversales:
            return None, None
        if coef is None and len(P) == 0:
            y_ref = lambda x: self.ultimo_y0          # noqa: E731
            x_fin = -math.inf
        else:
            cc = coef if coef is not None else np.array([P[:, 1].mean(), 0.0, 0.0])
            y_ref = lambda x: cc[0] + cc[1] * x + cc[2] * x * x      # noqa: E731
            x_fin = P[:, 0].max() if len(P) else -math.inf
        barra = esquina = None
        p_b0, p_b1 = self.p.get("barra_ancho", [0.45, 0.85])
        for x, ya, yb in sorted(transversales):
            if x < x_fin - 0.15:
                continue                               # la linea sigue mas alla: no es un final
            yl = y_ref(x)
            if not (ya - 0.10 <= yl <= yb + 0.10):
                continue
            # Barra o esquina por el ANCHO TOTAL del tramo transversal, que no depende de
            # donde se estime la linea: la barra mide 60 cm; en la esquina la cinta sigue
            # 2 m hacia un lado (hasta el borde de la vista). Con la marcha real, el criterio
            # anterior ("sale +-15 cm por los dos lados de la linea") alternaba barra y
            # esquina en la barra de fin del nivel 2 segun bailaba la estimacion de la linea.
            ancho = yb - ya
            izq, der = yb - yl, yl - ya                # cuanto sale hacia cada lado
            if p_b0 <= ancho <= p_b1 and min(izq, der) > 0.05:
                barra = x if barra is None else min(barra, x)
            elif ancho > p_b1 and min(izq, der) < 0.12 and self._recta_al_final(P):
                if esquina is None or x < esquina[0]:
                    esquina = (x, 1 if izq > der else -1)
        return barra, esquina
