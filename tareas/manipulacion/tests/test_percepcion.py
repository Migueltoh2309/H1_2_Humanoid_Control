"""Percepción sobre un fotograma de una escena conocida (render EGL del MJCF de la estación con
los brazos en observación) y sobre el mismo fotograma grabado y leído (FuenteCarpeta): mismo
resultado, los cuatro objetos reconocidos y en su sitio."""
import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sim"))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))
pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def escena():
    from construir_estacion import construir
    from evaluar_percepcion import render, verdad_torso
    from manip.config import _leer, cargar, cargar_objetos
    cfg, obj = cargar(), cargar_objetos()
    d = tempfile.mkdtemp()
    xml, geo = construir(3, 1, cfg, obj, salida=d)
    poses = _leer(os.path.join(RAIZ, "config", "poses.yaml"))
    fs, (Rt, pt) = render(xml, poses["observacion"], 0, np.random.default_rng(0))
    from calibrar import verdad_ir_sim
    calib = {"T_torso_cam": verdad_ir_sim(xml)}        # la cámara exacta: aquí se prueba la percepción
    return cfg, obj, fs[0], calib, verdad_torso(geo, Rt, pt), d


def _comprobar(lista, verdad):
    vistos = {o.clase: o for o in lista.objetos}
    assert set(vistos) == set(verdad)
    for k, (p, g) in verdad.items():
        o = vistos[k]
        assert np.linalg.norm(o.centro[:2] - p[:2]) < 0.005, k
        assert abs(o.centro[2] - p[2]) < 0.006, k
        if o.simetria > 0:
            assert abs(math.remainder(o.guinada - g, o.simetria)) < math.radians(3), k
        assert o.confianza > 0.3, k


def test_reconoce_y_localiza(escena):
    from manip.estacion import Estacion
    from manip.percepcion import Percepcion
    cfg, obj, f, calib, verdad, _ = escena
    _comprobar(Percepcion(calib, obj, Estacion(cfg))(f), verdad)


def test_igual_sobre_datos_grabados(escena):
    """Sec. 5: la percepción funciona igual sobre lo grabado que en vivo (solo ve un Fotograma)."""
    from manip.estacion import Estacion
    from manip.fuentes import FuenteCarpeta, grabar
    from manip.percepcion import Percepcion
    cfg, obj, f, calib, verdad, d = escena
    carpeta = os.path.join(d, "grabado")
    os.makedirs(carpeta)
    grabar(os.path.join(carpeta, "f000.npz"), f)
    g = FuenteCarpeta(carpeta).fotograma()
    per = Percepcion(calib, obj, Estacion(cfg))
    a, b = per(f), per(g)
    _comprobar(b, verdad)
    for oa, ob in zip(a.objetos, b.objetos):
        assert oa.clase == ob.clase and np.allclose(oa.centro, ob.centro)


def test_cambios_detecta_objeto_movido(escena):
    from dataclasses import replace

    from manip.percepcion import cambios
    from manip.tipos import ListaObjetos
    _, _, _, _, verdad, _ = escena
    from manip.tipos import ObjetoDetectado
    antes = ListaObjetos(0.0, [ObjetoDetectado(k, p.copy(), g, math.pi / 2, np.zeros(3), 1, 1, 100)
                               for k, (p, g) in verdad.items()], np.zeros(4), 0.0)
    movido = [replace(o, centro=o.centro + np.array([0.06, 0, 0])) if o.clase == "vaso" else o for o in antes.objetos]
    ahora = ListaObjetos(1.0, movido, np.zeros(4), 0.0)
    assert set(cambios(antes, ahora)) == {"vaso"}
    json.dumps({k: list(v) for k, v in cambios(antes, ahora).items()})
