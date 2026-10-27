"""La mano Inspire simulada (h1_2_sim2real) se porta como la real en lo que el reto usa:
FORCE_SET limita la fuerza, un dedo que se cierra sin nada llega a 0, STATUS, táctil y la
lectura en bloque que devuelve basura (secciones 3, 6.4 y 7)."""
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sim"))


@pytest.fixture(scope="module")
def entorno(tmp_path_factory):
    pytest.importorskip("mujoco")
    pytest.importorskip("cv2")
    from construir_estacion import S2R, construir
    sys.path.insert(0, S2R)
    from h1_2_sim2real.comun import J
    from h1_2_sim2real.robot_simulado import RobotSimulado
    from manip.config import cargar, cargar_objetos
    xml, _ = construir(2, 0, cargar(), cargar_objetos(), salida=str(tmp_path_factory.mktemp("est")))
    return xml, J, RobotSimulado


def _ordenes(J):
    n = J.N
    z = np.zeros(n)
    return {"lc_t": np.array([-1e9]), "lc_q": z, "lc_dq": z, "lc_kp": z, "lc_kd": z, "lc_tau": z, "lc_mode": z,
            "arm_t": np.array([-1e9]), "arm_q": z, "arm_dq": z, "arm_kp": z, "arm_kd": z, "arm_tau": z,
            "arm_w": np.zeros(1), "loco_v": np.zeros(3), "loco_t_fin": np.zeros(1), "fsm": np.array([201.0])}


def _cerrar_sobre_pelota(entorno, force_set, segundos=2.0):
    """Pelota sujeta delante de la palma izquierda (como en un soporte) y la mano cerrando."""
    xml, J, RobotSimulado = entorno
    sim = RobotSimulado(xml, log=lambda s: None)
    m, d = sim.model, sim.data
    mano = sim.manos["izq"]
    w = m.body("left_wrist_yaw_link").id
    off = np.array([0.17, -0.058, 0.0])          # centro de agarre (tesis, GRASP_OFFSET)
    mano.escribir(J.FORCE_SET, [force_set] * 6)
    mano.escribir(J.SPEED_SET, [300] * 6)
    mano.escribir(J.ANGLE_SET, [0, 0, 0, 0, 0, 1000])
    c, t = _ordenes(J), 0.0
    for _ in range(int(segundos / sim.dt)):
        sim.poner_cuerpo("pelota", d.xpos[w] + d.xmat[w].reshape(3, 3) @ off, [1, 0, 0, 0])
        sim.paso(t, c)
        t += sim.dt
    estado = [b for r in mano.leer(J.STATUS, 3) for b in ((r >> 8) & 255, r & 255)]
    return mano, J, mano.leer(J.ANGLE_ACT, 6), mano.leer(J.FORCE_ACT, 6), estado


@pytest.mark.parametrize("fs", [300, 700])
def test_force_set_limita(entorno, fs):
    mano, J, ang, fza, estado = _cerrar_sobre_pelota(entorno, fs)
    tocan = [k for k in range(4) if ang[k] > 100]      # dedos detenidos lejos de 0: algo delante
    assert len(tocan) >= 2
    for k in tocan:
        assert abs(fza[k] - fs) <= 0.03 * fs           # aprieta con FORCE_SET, no más
        assert estado[k] == 3                          # parado por fuerza
    for k in range(4):
        if ang[k] < 20:                                # llegó a 0: vacío, sin fuerza
            assert fza[k] < 100 and estado[k] == 2
    # el táctil de la yema de un dedo que toca sube sobre el reposo (0-3)
    k = tocan[0]
    assert max(mano.leer(J.TACTIL_DEDOS[k] + 18, 96)) > 60


def test_registros_como_la_mano_real(entorno):
    xml, J, RobotSimulado = entorno
    sim = RobotSimulado(xml, log=lambda s: None)
    mano = sim.manos["der"]
    assert mano.leer(J.FORCE_SET, 6) == [2000] * 6               # de fábrica: sin límite
    assert mano.leer(J.ANGLE_ACT, 75)[0] == 11000                # 1546..1620 de un bloque: basura
    assert len(mano.leer(J.FORCE_ACT, 6)) == 6                   # por grupos: bien
    assert mano.leer(J.ERR, 3) == [0, 0, 0]
    mano.escribir(J.BORRAR_ERROR, [1])
    assert max(mano.leer(J.TACTIL_PALMA, 112)) < 10
