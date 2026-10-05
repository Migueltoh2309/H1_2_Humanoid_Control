"""Carga de config/seguidor.yaml (YAML; si el python del robot no trae PyYAML,
acepta el mismo contenido en JSON)."""
import json
import os

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
POR_DEFECTO = os.path.join(RAIZ, "config", "seguidor.yaml")


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
    """La config y, encima, los ficheros `extras` (solo las claves que traen; p.ej. la
    planta de la marcha cinematica: config/planta_cinematica.yaml)."""
    cfg = _leer(ruta or POR_DEFECTO)
    for e in extras or ():
        _fundir(cfg, _leer(e))
    return cfg
