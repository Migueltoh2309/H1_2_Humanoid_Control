"""Piezas comunes de los videos de los hitos (grabar_video.py, tarjeta_video.py): escritor
H.264 y texto con tildes. Copia de la del seguidor de línea, con los estados del supervisor.

H.264 + yuv420p con el ffmpeg que trae imageio-ffmpeg: se reproduce en cualquier
navegador, en PowerPoint y en el movil (el mp4v de OpenCV no siempre)."""
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANCHO, ALTO = 1280, 720
FUENTES = "/usr/share/fonts/truetype/dejavu"
_cache = {}

FONDO = (24, 26, 32)
GRIS = (150, 156, 168)
BLANCO = (235, 238, 242)
COLOR_ESTADO = {           # RGB, estados del supervisor (manip/supervisor.py)
    "ENTRADA": (120, 130, 145), "OBSERVAR": (60, 140, 230), "PLANIFICAR": (60, 140, 230),
    "APROXIMAR": (40, 170, 90), "ASENTAR": (40, 170, 90), "DESCENDER": (40, 170, 90), "SONDEAR": (40, 170, 90),
    "CERRAR": (150, 90, 220), "VERIFICAR": (150, 90, 220), "ELEVAR": (40, 170, 90), "TRANSPORTAR": (40, 170, 90),
    "BAJAR": (40, 170, 90), "SOLTAR": (150, 90, 220), "RETIRAR": (40, 170, 90), "VOLVER": (40, 170, 90),
    "SUBIR": (40, 170, 90), "SALIDA": (120, 130, 145), "SALIDA_SEGURA": (220, 50, 50),
}


def fuente(tam, negrita=False, mono=False):
    nombre = "DejaVuSansMono" if mono else "DejaVuSans"
    if negrita:
        nombre += "-Bold"
    k = (nombre, tam)
    if k not in _cache:
        try:
            _cache[k] = ImageFont.truetype(os.path.join(FUENTES, nombre + ".ttf"), tam)
        except OSError:
            _cache[k] = ImageFont.load_default()
    return _cache[k]


class Lienzo:
    """Lienzo RGB en numpy con texto de PIL (cv2 no dibuja tildes ni eñes)."""

    def __init__(self, ancho=ANCHO, alto=ALTO, fondo=FONDO):
        self.img = np.zeros((alto, ancho, 3), np.uint8)
        self.img[:] = fondo
        self._pil = None

    def pegar(self, x, y, img):
        h, w = img.shape[:2]
        self._sin_pil()
        self.img[y:y + h, x:x + w] = img if img.ndim == 3 else img[..., None]

    def rect(self, x0, y0, x1, y1, color, relleno=True, alfa=1.0):
        self._sin_pil()
        if alfa < 1.0:
            zona = self.img[y0:y1, x0:x1].astype(np.float32)
            self.img[y0:y1, x0:x1] = (zona * (1 - alfa) + np.array(color) * alfa).astype(np.uint8)
        else:
            cv2.rectangle(self.img, (x0, y0), (x1, y1), color, -1 if relleno else 2)

    def texto(self, x, y, s, tam=18, color=BLANCO, negrita=False, mono=False, ancla="la"):
        if self._pil is None:
            self._pil = Image.fromarray(self.img)
            self._draw = ImageDraw.Draw(self._pil)
        self._draw.text((x, y), s, font=fuente(tam, negrita, mono), fill=tuple(color), anchor=ancla)

    def ancho_texto(self, s, tam=18, negrita=False, mono=False):
        return fuente(tam, negrita, mono).getlength(s)

    def _sin_pil(self):
        if self._pil is not None:
            self.img = np.asarray(self._pil).copy()
            self._pil = None

    def final(self):
        self._sin_pil()
        return self.img


class EscritorVideo:
    def __init__(self, ruta, fps=15, ancho=ANCHO, alto=ALTO):
        import imageio_ffmpeg
        os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
        self.ruta, self.fps, self.n = ruta, fps, 0
        self.gen = imageio_ffmpeg.write_frames(
            ruta, (ancho, alto), fps=fps, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=8,
            output_params=["-crf", "22", "-preset", "veryfast", "-movflags", "+faststart"])
        self.gen.send(None)

    def escribir(self, rgb, veces=1):
        b = np.ascontiguousarray(rgb).tobytes()
        for _ in range(veces):
            self.gen.send(b)
            self.n += 1

    def cerrar(self):
        self.gen.close()
        return self.n / self.fps


def tarjeta(titulo, lineas=(), subtitulo=None, pie=None):
    """Tarjeta de titulo / de resultados. `lineas`: texto normal; una linea que empieza
    por '$ ' o por espacios se escribe en monoespaciada (salida de consola)."""
    L = Lienzo()
    L.rect(0, 0, ANCHO, 6, (40, 170, 90))
    L.texto(60, 60, titulo, 40, negrita=True)
    y = 120
    if subtitulo:
        L.texto(60, y, subtitulo, 22, GRIS)
        y += 50
    for s in lineas:
        mono = s.startswith("$ ") or s.startswith("  ")
        tam = 17 if mono else 22
        if y > ALTO - 70:
            break
        L.texto(60, y, s, tam, (200, 230, 200) if mono else BLANCO, mono=mono)
        y += tam + (8 if mono else 14)
    if pie:
        L.texto(60, ALTO - 40, pie, 16, GRIS)
    return L.final()


def tarjeta_imagen(ruta_png, titulo, pie=None):
    L = Lienzo()
    L.texto(40, 22, titulo, 28, negrita=True)
    im = cv2.cvtColor(cv2.imread(ruta_png), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    esc = min((ANCHO - 60) / w, (ALTO - 110) / h)
    im = cv2.resize(im, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)
    x = (ANCHO - im.shape[1]) // 2
    L.pegar(x, 70, im)
    if pie:
        L.texto(40, ALTO - 30, pie, 16, GRIS)
    return L.final()


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def ultimas_lineas(ruta, n=9, ancho=110):
    """Las n ultimas lineas de un fichero de consola (con \\r: lo que se ve al final)."""
    try:
        with open(ruta, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 16000))
            txt = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = []
    for s in txt.split("\n"):
        s = _ANSI.sub("", s).split("\r")[-1].rstrip()
        if s and "multicast" not in s:
            out.append(s[:ancho])
    return out[-n:]
