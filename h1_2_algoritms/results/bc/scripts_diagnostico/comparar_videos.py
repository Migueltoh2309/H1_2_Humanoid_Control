"""Une dos videos de bc_faja lado a lado (mismo episodio, mismas frutas), con
una franja de título por lado, y lo guarda en H.264 (imageio-ffmpeg).

  python3 comparar_videos.py salida.mp4 "Título" A.mp4 "Etiqueta A" B.mp4 "Etiqueta B" [--lento x0 x1]
"""
import sys, cv2, numpy as np, imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"   # con tildes (cv2.putText no las tiene)

def frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f)

def band(text, w, h=44, color=(40, 40, 40), size=20):
    im = Image.new("RGB", (w, h), color[::-1])
    dr, font = ImageDraw.Draw(im), ImageFont.truetype(FONT, size)
    x0, y0, x1, y1 = dr.textbbox((0, 0), text, font=font)
    dr.text(((w - x1 + x0) // 2 - x0, (h - y1 + y0) // 2 - y0), text, font=font, fill=(255, 255, 255))
    return np.asarray(im)[:, :, ::-1].copy()

out, title, pa, la, pb, lb = sys.argv[1:7]
A, B = frames(pa), frames(pb)
n = max(len(A), len(B))
A += [A[-1]] * (n - len(A)); B += [B[-1]] * (n - len(B))
h, w = A[0].shape[:2]
top = band(title, 2 * w + 8, 40, (25, 25, 25), 21)
heads = np.hstack([band(la, w, color=(60, 60, 140)), np.zeros((44, 8, 3), np.uint8), band(lb, w, color=(60, 120, 60))])
wr = imageio_ffmpeg.write_frames(out, (2 * w + 8, h + 84), fps=15, codec="libx264", quality=7,
                                 macro_block_size=4, pix_fmt_out="yuv420p")
wr.send(None)
for a, b in zip(A, B):
    f = np.vstack([top, heads, np.hstack([a, np.zeros((h, 8, 3), np.uint8), b])])
    wr.send(np.ascontiguousarray(f[:, :, ::-1]))
wr.close()
print(out, n, "cuadros", f"{n / 15:.0f} s")
