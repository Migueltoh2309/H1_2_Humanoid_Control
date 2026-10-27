# Scripts de diagnóstico y pipelines (LfD fase B, doc §6.5–6.6)

Copiados del scratchpad de la sesión del 2026-10-27. Se corren desde la raíz de
`h1_2_algoritms` con el venv `~/venvs/h12`, ROS humble sourceado y `MUJOCO_GL=egl`.
Usar 6 workers como máximo (con 8 se agota la RAM de 15 GB).

## Reentrenar (pipeline completo: generar 40 → entrenar con DAgger 2 → evaluar 90)
- `pipeline_espera.sh` — perfil `espera` (el mejor: experto 68%, BC 67%), ≈ 2 h 10 min.
- `pipeline_corto.sh` — perfil `corto`.
- Las salidas van a `results/bc/` con sufijo `_exigente_<perfil>`; los logs de las
  corridas de hoy están aquí (`pipeline_*.log`).

## Análisis
- `comparar.py` — comparación pareada por semilla (Δ/ep, IC bootstrap, Wilcoxon) entre
  evaluaciones: `python3 comparar.py A.csv:bc B.csv:bc ...`. El "base" de 90 episodios es
  `evaluacion_exigente.csv` (10000–10029) + `evaluacion_exigente_base_s10030.csv`
  (10030–10089); concatenarlos antes de comparar.
- `stats.py` — corre una política en una lista de semillas (`POL`, `SEEDS`, `W`) y resume.
- `razones.py` — razones de fallo por perfil (aborto, agarre fallido, caída...).
- `perfiles.py`, `perfiles_v.py` — duración por fase del ciclo; `_v` además mide vmax articular.
- `experto_var.py` (+ `run_expvar*.sh`) — variantes del experto (sep_t, sep_y, ventanas)
  en semillas de desarrollo 5000–5047.
- `mapa_err.py`, `m*.json`, `mapa.json` — mapa de alcance por IK (§6.5).

## Depuración del agarre/cierre (usados para llegar a `close_max_vrel` y `close_touch`)
`close_dbg.py`, `vref.py`, `contactos.py`, `tact_dbg.py`, `env_dbg.py`, `fallo.py`,
`fallo_t.py`, `fase_otro.py`, `rasgos.py`, `trabado.py`, `desfase.py`, `percep*.py`, `sesgo.py`
(este último: visibilidad de la fruta en cámara según x, y).
Son scripts de una sola vez: pueden necesitar ajustes de rutas o semillas.
