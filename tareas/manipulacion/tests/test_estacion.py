"""Geometría de la estación y escenas generadas: las reglas de la sección 4 se cumplen."""
import json
import math
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sim"))
from manip.config import cargar, cargar_objetos  # noqa: E402
from manip.estacion import LETRAS, Estacion  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return cargar()


def test_geometria(cfg):
    e = Estacion(cfg, d=0.10)
    assert e.x_borde == pytest.approx(cfg["robot"]["pecho_x"] + 0.10)
    # la zona cabe en el tablero y las bandejas no la pisan
    for x, y in e.zona.esquinas():
        assert e.tablero.contiene(x, y, -1e-9)
    for letra in LETRAS:
        b = e.bandejas[letra]
        assert abs(abs(b.cy) - 0.30) < 1e-9
        for x, y in b.esquinas():
            assert e.tablero.contiene(x, y, -1e-9)
            assert not e.zona.contiene(x, y, 0.0)
    assert e.lado_bandeja("A") == "izq" and e.lado_bandeja("C") == "der"
    p = e.a_torso([0.3, 0.0, e.z_mesa])
    assert p[2] == pytest.approx(e.z_mesa - e.torso_z)


def test_celdas_aruco_se_detectan():
    """Las celdas que se modelan con cajitas son las del diccionario: pintadas en una
    imagen, OpenCV lee el mismo id."""
    cv2 = pytest.importorskip("cv2")
    from construir_estacion import celdas_aruco
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(dic)
    for ident in (0, 3, 10, 11):
        cel = celdas_aruco(ident)
        img = np.full((160, 160), 255, np.uint8)
        img[20:140, 20:140] = np.kron(np.where(cel, 0, 255), np.ones((20, 20))).astype(np.uint8)
        _, ids, _ = det.detectMarkers(img)
        assert ids is not None and int(ids.ravel()[0]) == ident


@pytest.fixture(scope="module")
def escenas(cfg, tmp_path_factory):
    pytest.importorskip("cv2")
    from construir_estacion import construir
    d = tmp_path_factory.mktemp("estaciones")
    obj = cargar_objetos()
    return {(n, s): construir(n, s, cfg, obj, salida=str(d)) for n in range(5) for s in range(3)}


def test_reglas_de_colocacion(cfg, escenas):
    e = Estacion(cfg)
    obj = cargar_objetos()
    from construir_estacion import radio_huella
    for (n, s), (_, geo) in escenas.items():
        if n < 2:
            assert not geo["objetos"]
            continue
        assert set(geo["objetos"]) == set(obj)
        pos = geo["objetos"]
        for k, o in pos.items():
            assert e.zona.contiene(o["x"], o["y"], radio_huella(obj[k]) - 1e-9), (n, s, k)
            assert o["z"] == pytest.approx(e.z_mesa + o["alto"] / 2)
        nombres = list(pos)
        for i in range(len(nombres)):
            for j in range(i + 1, len(nombres)):
                a, b = pos[nombres[i]], pos[nombres[j]]
                assert math.hypot(a["x"] - b["x"], a["y"] - b["y"]) >= \
                    radio_huella(obj[nombres[i]]) + radio_huella(obj[nombres[j]])
        contrarios = sum(o["lado"] != o["lado_bandeja"] for o in pos.values())
        if n == 2:
            assert contrarios == 0
        else:
            assert contrarios >= 2
        if n == 4:
            cx, cy, _ = geo["cruz"]
            assert (cx, cy) == pytest.approx((e.tablero.cx, e.tablero.cy))
            for k, o in pos.items():
                assert math.hypot(o["x"] - cx, o["y"] - cy) >= cfg["cruz_n4"]["largo"] / 2 + radio_huella(obj[k])


def test_dianas_nivel1(cfg, escenas):
    e = Estacion(cfg)
    _, geo = escenas[(1, 0)]
    assert len(geo["dianas"]) == 4
    for x, y, _ in geo["dianas"]:
        assert e.zona.contiene(x, y)


def test_escena_carga_y_objetos_quietos(escenas):
    """El MJCF carga, el keyframe lleva los objetos en su sitio (no en el origen) y, sin
    robot que los toque, siguen sobre la mesa al cabo de 1 s."""
    mujoco = pytest.importorskip("mujoco")
    xml, geo = escenas[(3, 0)]
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(500):
        mujoco.mj_step(m, d)
    for k, o in geo["objetos"].items():
        p = d.xpos[m.body(k).id]
        assert math.hypot(p[0] - o["x"], p[1] - o["y"]) < 0.01, k
        assert abs(p[2] - o["z"]) < 0.01, k


def test_camara_desviada(escenas):
    """La pose real de la cámara no es la del URDF (la tiene que encontrar la calibración)
    y la desviación queda solo en 'verdad_oculta'."""
    _, geo = escenas[(0, 0)]
    nom, real = geo["camara_nominal"], geo["verdad_oculta"]["camara_torso"]
    assert np.linalg.norm(np.subtract(nom["pos"], real["pos"])) > 0.005
    Rn, Rr = np.array(nom["R"]), np.array(real["R"])
    ang = math.degrees(math.acos(np.clip((np.trace(Rn.T @ Rr) - 1) / 2, -1, 1)))
    assert 1.0 < ang < 5.0
    assert np.allclose(Rr.T @ Rr, np.eye(3), atol=1e-9)
    json.dumps(geo)
