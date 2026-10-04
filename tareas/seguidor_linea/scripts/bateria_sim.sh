#!/usr/bin/env bash
# Bateria de tiradas automaticas en el simulador: niveles x semillas (la semilla
# cambia el signo de la deriva de rumbo de la marcha simulada). Tabla final.
#   ./scripts/bateria_sim.sh                 # niveles 1-4, semillas 1-3
#   NIVELES="3" SEMILLAS="1 2 3 4 5" ./scripts/bateria_sim.sh
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
NIVELES=${NIVELES:-"1 2 3 4"}
SEMILLAS=${SEMILLAS:-"1 2 3"}
INICIO=$(date +%s)
for n in $NIVELES; do
  for s in $SEMILLAS; do
    echo "=== nivel $n, semilla $s"
    SEMILLA=$s "$AQUI/scripts/tirada_sim.sh" "$n" "$@" 2>&1 | grep -E "EXITO|^FIN:" 
  done
done
python3 - "$AQUI/datos/tiradas" "$INICIO" <<'PY'
import glob, json, os, sys
filas = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_verdad.json"))):
    if os.path.getmtime(f) >= float(sys.argv[2]):
        filas.append(json.load(open(f)))
print(f"\n{'nivel':>5} {'exito':>5} {'err medio':>9} {'err max':>8} {'pie max':>8} {'parada':>7} {'tiempo':>7}  fin")
for v in filas:
    print(f"{v['nivel']:>5} {('SI' if v['exito'] else 'NO'):>5} {v.get('error_medio_m', float('nan')):9.3f} "
          f"{v.get('error_max_m', float('nan')):8.3f} {v.get('pie_max_m', float('nan')):8.3f} {v['parada_m']:+7.3f} "
          f"{v['tiempo_s']:7.1f}  {v['motivo_fin'][:40]}")
print(f"\nexito en {sum(v['exito'] for v in filas)} de {len(filas)} tiradas")
PY
