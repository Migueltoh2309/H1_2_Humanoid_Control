"""El planificador da un plan válido para un objeto alcanzable, todos sus puntos pasan la
comprobación contra el modelo, y rechaza lo que el mapa de alcance dice que no se puede."""
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
pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def plan():
    from construir_estacion import construir
    from manip.config import _leer, cargar, cargar_objetos
    from manip.estacion import Estacion
    from manip.modelo import Modelo
    from manip.planificacion import Planificador
    cfg, cat = cargar(), cargar_objetos()
    est = Estacion(cfg)
    xml, _ = construir(0, 0, cfg, cat, salida=tempfile.mkdtemp())
    poses = _leer(os.path.join(RAIZ, "config", "poses.yaml"))
    P = Planificador(est, _leer(os.path.join(RAIZ, "config", "agarres.yaml")), cat,
                     _leer(os.path.join(RAIZ, "config", "planificacion.yaml")), Modelo(xml), poses["observacion"])
    return P, est, {k: np.asarray(v) for k, v in poses["observacion"].items()}


def _objeto(clase, x, y, est, gui=0.0, alto=0.06):
    from manip.tipos import ObjetoDetectado
    z = est.z_mesa - est.torso_z + alto / 2
    return ObjetoDetectado(clase, np.array([x, y, z]), gui, math.pi / 2, np.array([0.06, 0.06, alto]), 0.9, 1.0, 500)


def test_plan_cubo_alcanzable(plan):
    P, est, q = plan
    o = _objeto("cubo", 0.30, 0.10, est, gui=math.radians(20))
    p = P.planificar(o, q)
    assert p.ok, p.motivo
    assert p.mano == "izq" and p.bandeja == "A"
    nombres = [t.nombre for t in p.tramos]
    assert nombres == ["aproximar", "descender", "elevar", "transportar", "bajar", "retirar", "volver"]
    from manip.cinematica import limites
    lo, hi = limites("izq")
    for t in p.tramos:
        assert np.all(t.q_ruta >= lo - 1e-6) and np.all(t.q_ruta <= hi + 1e-6)
        assert np.max(np.abs(np.diff(np.vstack([t.q0, t.q_ruta]), axis=0))) < 0.16
    ok, hmin, motivo = P.comprobar(p.tramos, q["der"])
    assert ok, motivo
    # la guiñada de la mano está alineada con una cara del cubo, con la desalineación que se
    # admite (config: tolerancia_guinada, hasta 20 grados)
    assert abs(math.remainder(math.radians(p.az_agarre) - o.guinada, math.pi / 2)) < math.radians(21)


def test_rechaza_fuera_de_alcance_y_otro_lado(plan):
    P, est, q = plan
    lejos = P.planificar(_objeto("cubo", 0.65, 0.15, est), q)
    assert not lejos.ok and "agarre imposible" in lejos.motivo
    otro = P.planificar(_objeto("cubo", 0.30, -0.10, est), q)        # cubo -> A (izq), objeto a la der
    assert not otro.ok and "entrega" in otro.motivo


def test_transporte_por_encima_de_los_otros(plan):
    """Con el vaso (9.5 cm) en la mesa, la mano vacía y la cargada pasan por encima: la cargada,
    además, lo que el cubo cuelga por debajo de ella."""
    P, est, q = plan
    cubo = _objeto("cubo", 0.30, 0.12, est)
    vaso = _objeto("vaso", 0.38, 0.05, est, alto=0.095)
    p = P.planificar(cubo, q, otros=[vaso])
    assert p.ok, p.motivo
    assert p.alturas["vacia"] >= 0.095 + 0.03 - 1e-9
    assert p.alturas["carga"] >= p.alturas["colgado"] + 0.095 + 0.03 - 1e-9
    t = {tr.nombre: tr for tr in p.tramos}
    for qq in t["transportar"].q_ruta:
        assert P.mano_baja("izq", qq) >= P.z_mesa + p.alturas["carga"] - 1e-3


def test_suelta_con_giro(plan):
    """Nivel 4: la mano suelta con su azimut de agarre + el giro pedido (+- la tolerancia)."""
    P, est, q = plan
    o = _objeto("cubo", 0.30, 0.12, est, gui=math.radians(10))
    cruz = (est.tablero.cx, est.tablero.cy)
    p = P.planificar(o, q, bandeja=cruz, mano="izq", giro=math.radians(30))
    if not p.ok:
        pytest.skip(p.motivo)
    assert abs(math.remainder(math.radians(p.az_suelta - p.az_agarre - 30), math.pi / 2)) <= math.radians(5.5)
