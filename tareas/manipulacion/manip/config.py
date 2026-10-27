"""Carga de config/*.yaml (YAML; si el python del robot no trae PyYAML, acepta el mismo
contenido en JSON)."""
import json
import os

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ESTACION = os.path.join(RAIZ, "config", "estacion.yaml")
OBJETOS = os.path.join(RAIZ, "config", "objetos.yaml")


def _leer(ruta):
    with open(ruta) as f:
        texto = f.read()
    if ruta.endswith(".json"):
        return json.loads(texto)
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(f"Falta PyYAML para leer {ruta}: pip install pyyaml, o pasa la config en .json") from e
    return yaml.safe_load(texto)


def _fundir(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _fundir(base[k], v)
        else:
            base[k] = v
    return base


def cargar(ruta=None, extras=()):
    """config/estacion.yaml y, encima, los ficheros `extras` (solo las claves que traen)."""
    cfg = _leer(ruta or ESTACION)
    for e in extras or ():
        _fundir(cfg, _leer(e))
    return cfg


def cargar_objetos(ruta=None):
    return _leer(ruta or OBJETOS)
