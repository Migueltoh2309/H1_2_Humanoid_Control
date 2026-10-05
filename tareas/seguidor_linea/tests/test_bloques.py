"""Pruebas automaticas de los cinco bloques (reto, sec. 9: "pruebas automaticas").
Sin robot, sin MuJoCo y sin ROS:   python3 -m pytest tests -q"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seguidor import config                                    # noqa: E402
from seguidor.control import Control                           # noqa: E402
from seguidor.estimacion import Estimador                      # noqa: E402
from seguidor.percepcion import Percepcion                     # noqa: E402
from seguidor.supervisor import FIN, PARADA, SEGUIMIENTO, Supervisor   # noqa: E402
from seguidor.tipos import EstadoLinea, Fotograma, MedidaLinea, Orden   # noqa: E402
import sintetico as S                                          # noqa: E402

K = (378.8, 378.8, 320.0, 240.0)


@pytest.fixture
def cfg():
    return config.cargar()


def medir(cfg, img, **k):
    per = Percepcion(cfg)
    return per.procesar(Fotograma(0.0, img, K), **k)


# ------------------------------------------------------------------ geometria
def test_ida_y_vuelta_pixel_suelo():
    cam = S.camara()
    u, v = np.array([100.0, 320.0, 600.0]), np.array([470.0, 300.0, 120.0])
    for r, p in ((0, 0), (0.03, -0.02)):
        G = cam.pixel_a_suelo(u, v, r, p)
        uv = cam.suelo_a_pixel(G[:, 0], G[:, 1], r, p)
        assert np.allclose(uv, np.column_stack([u, v]), atol=1e-6)


def test_alcance_y_fov():
    cam = S.camara()
    assert math.degrees(cam.fov_vertical()) == pytest.approx(64.72, abs=0.05)
    cerca, lejos = cam.alcance_suelo()
    assert 0.25 < cerca < 0.45 and 4.0 < lejos < 6.0


# ------------------------------------------------------------------ percepcion
@pytest.mark.parametrize("desp,ang_deg", [(0.0, 0.0), (0.15, 10.0), (-0.2, -15.0)])
def test_recta(cfg, desp, ang_deg):
    th = math.radians(ang_deg)
    img = S.imagen(S.camara(), [S.recta(-0.5, desp - 0.5 * math.tan(th), th, 4.0 / math.cos(th))])
    m = medir(cfg, img)
    assert m.valida and m.confianza > 0.8
    assert m.desplazamiento == pytest.approx(desp, abs=0.02)
    assert math.degrees(m.angulo) == pytest.approx(ang_deg, abs=1.5)


def test_sombra_adaptativo_si_fijo_no(cfg):
    """6.2.2: una sombra que multiplica la luz por 0.3 sobre el tramo central."""
    img = S.imagen(S.camara(), [S.recta(-0.5, 0.0, 0.0, 4.0)], sombra=(0.8, 2.0, -1.5, 1.5, 0.3))
    m = medir(cfg, img)
    assert m.valida and m.confianza > 0.7
    assert ((m.puntos[:, 0] > 0.9) & (m.puntos[:, 0] < 1.9)).sum() >= 15     # sigue viendo la linea en la sombra
    cfg["percepcion"]["umbral"] = "fijo"
    mf = medir(cfg, img)
    # con umbral fijo la sombra entera es "cinta": en la sombra no hay linea limpia
    assert ((mf.puntos[:, 0] > 0.9) & (mf.puntos[:, 0] < 1.9)).sum() < 5


def test_emisor_y_ruido(cfg):
    img = S.imagen(S.camara(), [S.recta(-0.5, 0.1, 0.0, 4.0)], emisor=True, ruido=8.0)
    m = medir(cfg, img)
    assert m.valida and m.desplazamiento == pytest.approx(0.1, abs=0.02)


def test_suelo_vacio_y_junta(cfg):
    assert not medir(cfg, S.imagen(S.camara(), [])).valida
    # junta del suelo: 8 mm de ancho y poco contraste -> no es linea
    assert not medir(cfg, S.imagen(S.camara(), [(S.recta(-0.5, 0.0, 0.0, 4.0), 0.008)])).valida


def test_barra_de_fin(cfg):
    linea = S.recta(-0.5, 0.0, 0.0, 2.0)
    barra = S.recta(1.5, -0.30, math.pi / 2, 0.60)
    m = medir(cfg, S.imagen(S.camara(), [linea, barra]))
    assert m.barra == pytest.approx(1.5, abs=0.06) and m.esquina is None


@pytest.mark.parametrize("lado", [1, -1])
def test_esquina(cfg, lado):
    linea = S.recta(-0.5, 0.0, 0.0, 2.0)
    tramo = S.recta(1.5, 0.0, lado * math.pi / 2, 1.5)
    m = medir(cfg, S.imagen(S.camara(), [linea, tramo]))
    assert m.esquina is not None and m.esquina[1] == lado and m.esquina[0] == pytest.approx(1.5, abs=0.06)
    assert m.barra is None


def test_curva_no_es_esquina(cfg):
    """Una curva de R 1.2 m (nivel 3) que se pone horizontal no debe parecer esquina."""
    P = np.vstack([S.recta(-0.5, 0.0, 0.0, 1.2), S.arco(0.7, 0.0, 0.0, 1.2, math.pi / 2)])
    m = medir(cfg, S.imagen(S.camara(), [P]))
    assert m.valida and m.esquina is None and m.curvatura > 0.2


def test_compensacion_imu(cfg):
    """El balanceo: la camara inclinada 2 grados de pitch; con la IMU se corrige."""
    img = S.imagen(S.camara(), [S.recta(-0.5, 0.2, 0.0, 2.0)], pitch=math.radians(2))   # acaba en x = 1.5
    sin = medir(cfg, img)
    con = medir(cfg, img, pitch=math.radians(2))
    assert abs(con.desplazamiento - 0.2) < 0.02
    assert con.puntos[:, 0].max() == pytest.approx(1.5, abs=0.04)
    assert abs(sin.puntos[:, 0].max() - 1.5) > 0.06     # sin compensar, el final sale desplazado


# ------------------------------------------------------------------ estimacion
def test_memoria_cubre_hueco(cfg):
    est = Estimador(cfg)
    pts = S.recta(0.4, 0.0, 0.0, 2.0)[::5]
    med = MedidaLinea(t=0.0, valida=True, confianza=0.95, puntos=pts, coef=np.zeros(3))
    est.actualizar(0.0, 0.0, (0.0, 0.0, 0.0), med)
    t = 0.0
    for _ in range(40):                                  # 2 s a 0.3 m/s sin medidas
        t += 0.05
        e = est.actualizar(t, 0.0, (0.3, 0.0, 0.0), None)
    assert e.valida and abs(e.desplazamiento) < 0.02 and e.sin_linea_m > 0.3


def test_deriva_estimada(cfg):
    est = Estimador(cfg)
    t, yaw = 0.0, 0.0
    for _ in range(400):                                 # 20 s andando recto con 2 grados/s de deriva
        t += 0.05
        yaw += math.radians(2.0) * 0.05
        e = est.actualizar(t, yaw, (0.3, 0.0, 0.0), None, gyro_z=math.radians(2.0))
    assert math.degrees(e.sesgo_yaw) == pytest.approx(2.0, abs=0.3)


# ------------------------------------------------------------------ control
def test_control_signos_y_topes(cfg):
    c = Control(cfg)
    est = EstadoLinea(t=0, valida=True, confianza=0.95)
    o = None
    for _ in range(100):
        o = c.calcular(est, (1.0, 0.3), 0.05)            # mira a la izquierda
    assert o.vyaw > 0 and 0 < o.vx <= cfg["control"]["vx_max"] + 1e-9
    for _ in range(100):
        o = c.calcular(est, (0.6, -2.0), 0.05)           # mira muy a la derecha
    # pure pursuit: vyaw = vx * kappa y vx baja con la curvatura -> gira, sin saturar
    assert -cfg["control"]["vyaw_max"] <= o.vyaw < 0
    cfg["control"]["ley"] = "pd"                         # con PD y un error grande, satura
    cpd = Control(cfg)
    lejos = EstadoLinea(t=0, valida=True, confianza=0.95, desplazamiento=-1.0, angulo=-0.5)
    for _ in range(100):
        o = cpd.calcular(lejos, None, 0.05)
    assert o.vyaw == pytest.approx(-cfg["control"]["vyaw_max"], abs=1e-6)
    cfg["control"]["ley"] = "pure_pursuit"
    cfg["control"]["escala"] = 0.5
    c2 = Control(cfg)
    for _ in range(100):
        o = c2.calcular(est, (1.0, 0.0), 0.05)
    assert o.vx <= 0.5 * cfg["control"]["vx_max"] + 1e-9


# ------------------------------------------------------------------ supervisor
def _sup(cfg):
    s = Supervisor(cfg, log=lambda *_: None)
    s.estado = SEGUIMIENTO
    return s


def _salud(t, **k):
    d = {"t_camara": t, "t_imu": t, "roll": 0.0, "pitch": 0.0, "motores_ok": True, "excepcion": None}
    d.update(k)
    return d


def test_vigilante(cfg):
    est = EstadoLinea(t=0, valida=True, confianza=0.9)
    s = _sup(cfg)
    assert s.paso(1.0, est, True, Orden(0.2), _salud(1.0, t_camara=0.3)).parar and s.estado == PARADA
    s = _sup(cfg)
    assert s.paso(1.0, est, True, Orden(0.2), _salud(1.0, roll=math.radians(25))).parar and s.estado == PARADA
    s = _sup(cfg)
    assert s.paso(1.0, est, True, Orden(0.2), _salud(1.0, motores_ok=False)).parar and s.estado == PARADA


def test_linea_perdida_para(cfg):
    s = _sup(cfg)
    est = EstadoLinea(t=0, valida=True, confianza=0.1, sin_linea_s=1.0, sin_linea_m=0.2)
    o = s.paso(1.0, est, False, Orden(0.3), _salud(1.0))
    assert not o.parar and o.vx <= cfg["supervisor"]["vx_perdida"]
    est.sin_linea_m = 1.0
    assert s.paso(2.0, est, False, Orden(0.3), _salud(2.0)).parar and s.estado == FIN


def test_llegada_a_la_barra(cfg):
    s = _sup(cfg)
    est = EstadoLinea(t=0, valida=True, confianza=0.9, barra=0.8, recorrido=3.0)
    o = s.paso(1.0, est, True, Orden(0.3), _salud(1.0))
    assert not o.parar and o.vx <= 0.3
    est.barra = 0.05
    assert s.paso(2.0, est, True, Orden(0.3), _salud(2.0)).parar and s.estado == FIN


def test_se_recupera_de_una_prediccion_vieja(cfg):
    """Con la marcha real, una medida mala dejo la prediccion a -0.42 m y la linea real (en
    0) nunca volvia a entrar en la puerta: la percepcion quedaba bloqueada."""
    per = Percepcion(cfg)
    per.ultimo_y0 = -0.42
    m = per.procesar(Fotograma(0.0, S.imagen(S.camara(), [S.recta(-0.5, 0.0, 0.0, 4.0)]), K))
    assert m.valida and abs(m.desplazamiento) < 0.02
