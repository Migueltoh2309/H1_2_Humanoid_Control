#!/usr/bin/env python3
"""Utilidad para grabar ventanas (scripts/videos_hitos.sh con RVIZ=1): busca una ventana
por su titulo, la pone SIEMPRE ENCIMA y activa (que nada la tape mientras se graba) y
escribe su posicion y tamano en pantalla, para ffmpeg x11grab.

    python3 herramientas/ventana_x11.py "RViz" [--esperar 20] [--soltar]
    -> X Y ANCHO ALTO   (ancho y alto pares)
"""
import argparse
import sys
import time

from Xlib import X, display, protocol


def buscar(d, texto):
    raiz = d.screen().root
    nombres = raiz.get_full_property(d.intern_atom("_NET_CLIENT_LIST"), X.AnyPropertyType)
    for wid in (nombres.value if nombres else []):
        w = d.create_resource_object("window", wid)
        try:
            t = w.get_full_property(d.intern_atom("_NET_WM_NAME"), d.intern_atom("UTF8_STRING"))
            titulo = t.value.decode("utf-8", "replace") if t else (w.get_wm_name() or "")
        except Exception:
            continue
        if texto in titulo:
            return w
    return None


def mensaje(d, w, tipo, datos):
    raiz = d.screen().root
    ev = protocol.event.ClientMessage(window=w, client_type=d.intern_atom(tipo), data=(32, datos + [0] * (5 - len(datos))))
    raiz.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("titulo")
    ap.add_argument("--esperar", type=float, default=20.0)
    ap.add_argument("--soltar", action="store_true", help="quitar 'siempre encima'")
    a = ap.parse_args()
    d = display.Display()
    t0 = time.time()
    w = None
    while w is None and time.time() - t0 < a.esperar:
        w = buscar(d, a.titulo)
        if w is None:
            time.sleep(0.5)
    if w is None:
        sys.exit(f"no hay ninguna ventana con '{a.titulo}'")
    encima = d.intern_atom("_NET_WM_STATE_ABOVE")
    mensaje(d, w, "_NET_WM_STATE", [0 if a.soltar else 1, encima, 0, 1])
    if not a.soltar:
        mensaje(d, w, "_NET_ACTIVE_WINDOW", [1, X.CurrentTime, 0])
    d.sync()
    time.sleep(0.5)
    g = w.get_geometry()
    p = w.translate_coords(d.screen().root, 0, 0)
    print(-p.x, -p.y, g.width // 2 * 2, g.height // 2 * 2)


if __name__ == "__main__":
    main()
