"""Carga de config/seguidor.yaml (YAML; si el python del robot no trae PyYAML,
acepta el mismo contenido en JSON)."""
import json
import os

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
POR_DEFECTO = os.path.join(RAIZ, "config", "seguidor.yaml")


def cargar(ruta=None):
    ruta = ruta or POR_DEFECTO
    with open(ruta) as f:
        texto = f.read()
    if ruta.endswith(".json"):
        return json.loads(texto)
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(f"Falta PyYAML para leer {ruta}: pip install pyyaml, o pasa la config en .json") from e
    return yaml.safe_load(texto)
