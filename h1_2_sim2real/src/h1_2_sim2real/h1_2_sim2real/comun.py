"""Acceso a los modulos maestros de h1_2_sim2real/comun (h1_2_juntas,
protocolo_sim2real, modbus_mini), que tambien usa el agente del robot sin ROS.

Se buscan, por orden: $H12_S2R_COMUN, el share instalado del paquete
(share/h1_2_sim2real/comun) y la carpeta comun/ del arbol de fuentes."""
import os
import sys


def _candidatos():
    env = os.environ.get("H12_S2R_COMUN")
    if env:
        yield env
    try:
        from ament_index_python.packages import get_package_share_directory
        yield os.path.join(get_package_share_directory("h1_2_sim2real"), "comun")
    except Exception:
        pass
    yield os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "comun"))


for _d in _candidatos():
    if os.path.isfile(os.path.join(_d, "protocolo_sim2real.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        COMUN_DIR = _d
        break
else:
    raise ImportError("no encuentro la carpeta comun/ de h1_2_sim2real (exporta H12_S2R_COMUN)")

import h1_2_juntas as J            # noqa: E402,F401
import modbus_mini                 # noqa: E402,F401
import protocolo_sim2real as P     # noqa: E402,F401


def recurso(*partes):
    """Ruta a un fichero del paquete (mjcf/, urdf/, robot/...): share instalado
    o arbol de fuentes."""
    try:
        from ament_index_python.packages import get_package_share_directory
        r = os.path.join(get_package_share_directory("h1_2_sim2real"), *partes)
        if os.path.exists(r):
            return r
    except Exception:
        pass
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", *partes))
