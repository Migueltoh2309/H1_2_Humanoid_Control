#!/usr/bin/env bash
# Graba en el SIMULADOR un video por cada actividad del plan de trabajo del reto (sec. 8).
# Sin ventanas: no hace falta mirar mientras graba. Una carpeta por hito en videos/:
#   1_hito1_preparacion_medida/  2_conjunto_datos/  3_hito2_percepcion/  4_hito3_simulacro/
#   5_tiradas_niveles/  6_cierre/
# Los .mp4 (1280x720, H.264) son SOLO la simulacion; las graficas van aparte en .png y los
# resultados en .txt; LEEME.txt de cada carpeta dice que muestra cada video.
#
# RVIZ=1 ./scripts/videos_hitos.sh   ademas abre RViz2 EN TU PANTALLA (siempre encima: no la
#   tapes) y graba su ventana -> rviz_<video>.mp4, y saca una foto del visor de MuJoCo de cada
#   pista -> mujoco_pista_nivelN.png en las carpetas que la usan.
#
#   ./scripts/videos_hitos.sh                    # todos, en orden (~20 min)
#   ./scripts/videos_hitos.sh hito3 nivel2       # solo esos
#
# Actividades: hito1 datos escalon hito2 hito3 nivel1 nivel2 nivel3 nivel4 cierre
#   hito2 usa el ultimo datos/dataset_n3_*; cierre, las ultimas tiradas de cada nivel.
# Lo que en el robot hace una persona con wasd.py, aqui lo hace herramientas/operador_sim.py
# (sigue la cinta real con la verdad de terreno del simulador): se dice en los LEEME.txt.
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AQUI" || exit 1
source "$AQUI/scripts/entorno.sh"
export PYTHONUNBUFFERED=1
OUT="$AQUI/videos"
mkdir -p "$OUT"
TMP=$(mktemp -d /tmp/videos_hitos_XXXX)
CON="$TMP/consola.log"
SIM= REC= FF=
RVIZ=${RVIZ:-0}
FFMPEG=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
ACTIVIDADES=("$@")
[ ${#ACTIVIDADES[@]} -eq 0 ] && ACTIVIDADES=(hito1 datos escalon hito2 hito3 nivel1 nivel2 nivel3 nivel4 cierre)

limpiar() {
    [ -n "$FF" ] && kill -INT "$FF" 2>/dev/null && wait "$FF" 2>/dev/null
    [ -n "$REC" ] && kill -INT "$REC" 2>/dev/null && wait "$REC" 2>/dev/null
    [ -n "$SIM" ] && pkill -INT -P "$SIM" 2>/dev/null && wait "$SIM" 2>/dev/null
    REC= SIM=
}
trap 'limpiar; rm -rf "$TMP"' EXIT
trap 'exit 130' INT TERM

arrancar_sim() {    # nivel [args de sim.launch.py]
    local n=$1; shift
    local rv=()
    [ "$RVIZ" = 1 ] && rv=(visor:=true rviz_config:="$AQUI/rviz/seguidor.rviz")
    "$AQUI/scripts/sim_pista.sh" "$n" use_viewer:=false "${rv[@]}" "$@" > "$TMP/sim.log" 2>&1 &
    SIM=$!
    until grep -q "D435 simulada" "$TMP/sim.log"; do
        sleep 0.5; kill -0 $SIM 2>/dev/null || { tail -20 "$TMP/sim.log"; exit 1; }
    done
    sleep 3
}

leeme() {           # carpeta video titulo [lineas...]: lo que muestra el video, en el LEEME.txt de la carpeta
    local dir=$1 vid=$2 tit=$3; shift 3
    { echo "== $vid — $tit"; for l in "$@"; do echo "   $l"; done
      [ "$RVIZ" = 1 ] && [ "${vid%.mp4}" != percepcion ] && echo "   rviz_$vid: la misma ejecución, grabada de la ventana de RViz2."
      echo; } >> "$OUT/$dir/LEEME.txt"
}

carpeta() {         # crea (o vacia) la carpeta de un hito
    rm -rf "${OUT:?}/$1"; mkdir -p "$OUT/$1"
}

grabar() {          # nivel carpeta/video.mp4 titulo subtitulo [lineas para el LEEME...]
    local n=$1 sal=$2 tit=$3 sub=$4; shift 4
    leeme "$(dirname "$sal")" "$(basename "$sal")" "$tit" "$sub" "$@"
    : > "$CON"
    # grabadoras con prioridad baja: con ellas a la par, la percepcion pasaba de 7 a 33 ms, el
    # lazo bajaba de 20 a ~10 Hz y los niveles 3 y 4 fallaban (la grabacion estropeaba la prueba)
    nice -n 15 python3 herramientas/grabar_video.py --nivel "$n" --salida "$OUT/$sal" --titulo "$tit" \
        --consola "$CON" > "$TMP/rec.log" 2>&1 &
    REC=$!
    until grep -q "grabando" "$TMP/rec.log"; do
        sleep 0.3; kill -0 $REC 2>/dev/null || { cat "$TMP/rec.log"; exit 1; }
    done
    if [ "$RVIZ" = 1 ]; then          # la ventana de RViz, encima de todo, grabada tal cual
        local g
        g=$(python3 herramientas/ventana_x11.py "RViz" --esperar 30) || exit 1
        read -r X Y W H <<< "$g"
        renice -n 10 -p $(pgrep -f "[r]viz2 -d") > /dev/null 2>&1
        sleep 2
        nice -n 15 "$FFMPEG" -v error -y -f x11grab -framerate 15 -video_size "${W}x${H}" -i "$DISPLAY+$X,$Y" \
            -c:v libx264 -preset ultrafast -threads 2 -crf 23 -pix_fmt yuv420p -movflags +faststart \
            "$OUT/$(dirname "$sal")/rviz_$(basename "$sal")" < /dev/null > "$TMP/ff.log" 2>&1 &
        FF=$!
    fi
}

terminar() {        # [segundos de cola]
    sleep "${1:-3}"
    if [ -n "$FF" ]; then
        kill -INT "$FF"; wait "$FF"; FF=
        python3 herramientas/ventana_x11.py "RViz" --soltar --esperar 1 > /dev/null 2>&1
    fi
    kill -INT "$REC"; wait "$REC"; REC=
    pkill -INT -P "$SIM"; wait "$SIM" 2>/dev/null; SIM=
    tail -1 "$TMP/rec.log"
}

cmd() {             # muestra el comando en la consola del video y lo ejecuta
    echo "\$ $*" >> "$CON"
    "$@" >> "$CON" 2>&1
}

tapar() { ros2 topic pub -w 1 --once /sim/tapar_camara std_msgs/msg/Bool "{data: $1}" > /dev/null 2>&1; }

ultimo() { ls -td $1 2>/dev/null | head -1; }

foto_mujoco() {     # nivel carpeta: foto del visor de MuJoCo con la pista (una vez por pista, luego se copia)
    local n=$1 dir=$2 f="$TMP/mujoco_pista_nivel$1.png"
    [ "$RVIZ" = 1 ] || return 0
    if [ ! -f "$f" ]; then
        "$AQUI/scripts/sim_pista.sh" "$n" > "$TMP/foto.log" 2>&1 &
        local P=$!
        until grep -q "D435 simulada" "$TMP/foto.log"; do
            sleep 0.5; kill -0 $P 2>/dev/null || { tail -20 "$TMP/foto.log"; exit 1; }
        done
        local g; g=$(python3 herramientas/ventana_x11.py "MuJoCo" --esperar 20)
        read -r X Y W H <<< "$g"
        sleep 4
        "$FFMPEG" -v error -y -f x11grab -video_size "${W}x${H}" -i "$DISPLAY+$X,$Y" -frames:v 1 "$f" < /dev/null
        python3 herramientas/ventana_x11.py "MuJoCo" --soltar --esperar 1 > /dev/null 2>&1
        pkill -INT -P $P; wait $P 2>/dev/null
    fi
    cp "$f" "$OUT/$dir/"
}

# ---------------------------------------------------------------- actividades
hito1() {
    echo "== Hito 1: preparación y medida"
    local D=1_hito1_preparacion_medida
    carpeta $D
    # robot QUIETO de pie para medir (con la politica pisa sin avanzar y la medida baila
    # +-0.4 grados); en el robot real, igual: de pie y sin mandar Move
    foto_mujoco 1 $D
    MARCHA=ninguna arrancar_sim 1
    grabar 1 $D/hito1_preparacion_medida.mp4 "Hito 1 · Preparación y medida" \
        "Entregable: altura, inclinación y zona ciega medidas" \
        "Interfaces de la sección 5 por escrito: INTERFACES.md." \
        "1. La cámara abre y da fotogramas (topic hz de la IR izquierda)." \
        "2. Geometría de la cámara: FOV, zona ciega y alcance sobre el suelo (6.1)." \
        "3. Altura e inclinación MEDIDAS: plano del suelo con la profundidad (RANSAC)," \
        "   con el robot quieto de pie."
    sleep 2
    echo "\$ ros2 topic hz /camera/infra1/image_rect_raw" >> "$CON"
    timeout 6 ros2 topic hz /camera/infra1/image_rect_raw >> "$CON" 2>&1
    cmd python3 herramientas/geometria_camara.py
    sleep 4
    cmd python3 herramientas/calibrar_camara.py --sim
    cp datos/calibracion.json "$OUT/$D/"
    python3 - "$OUT/$D/resultados.txt" <<'EOF'
import json, subprocess, sys
c = json.load(open("datos/calibracion.json"))
g = subprocess.run([sys.executable, "herramientas/geometria_camara.py"], capture_output=True, text=True).stdout
import re
ir = [l.strip() for l in g.splitlines() if "suelo visible" in l][-1]
cerca, lejos = re.findall(r"(\d+\.\d+) m", ir)[:2]
open(sys.argv[1], "w").write("\n".join([
    "Hito 1 · Resultado",
    f"Altura del centro óptico: {c['altura']} m (± {c['dispersion']['altura_mm']} mm)",
    f"Inclinación bajo la horizontal: {c['inclinacion_deg']}° (± {c['dispersion']['inclinacion_deg']}°)",
    f"Alabeo: {c['alabeo_deg']}°   ·   {c['fotogramas']} fotogramas de profundidad, fuente: {c['fuente']}",
    "",
    "IR izquierda (fx 378.8): FOV vertical 64.7°, horizontal 80.4°",
    "  " + ir,
    f"Zona ciega: los primeros {cerca} m delante del robot (alcance en el eje: {lejos} m).",
    "Los pies y las manos tapan además la parte de abajo de la imagen: config percepcion.mascara_robot.",
    "",
    "En el robot real: los mismos comandos en el PC2 (calibrar_camara.py con la D435i).",
]) + "\n")
EOF
    terminar 5
}

datos() {
    local D=2_conjunto_datos
    carpeta $D
    for n in 1 2 3; do
        echo "== Conjunto de datos: nivel $n"
        foto_mujoco "$n" $D
        arrancar_sim "$n"
        grabar "$n" "$D/recorrido_nivel$n.mp4" "Conjunto de datos · recorrido nivel $n" \
            "Entregable: al menos tres recorridos grabados (niveles 1 a 3) y la respuesta al escalón" \
            "Se graban las imágenes IR y la IMU con marca de tiempo (grabar_dataset.py)" \
            "mientras un operador lleva el robot por la pista." \
            "" \
            "En el robot: una persona con wasd.py. En el simulador: operador_sim.py," \
            "que sigue la cinta con la pose real del simulador (no usa la cámara)."
        echo "\$ python3 herramientas/grabar_dataset.py --sim --nivel $n &" >> "$CON"
        # su contador de fotogramas reescribe la linea con \r: a lineas, para que no se pegue al operador
        python3 herramientas/grabar_dataset.py --sim --nivel "$n" > >(awk -v RS='\r' 'length($0) { print; fflush() }' >> "$CON") 2>&1 &
        local G=$!
        sleep 2
        cmd python3 herramientas/operador_sim.py --nivel "$n" --vx 0.25
        kill -TERM $G; wait $G
        local d; d=$(ultimo "datos/dataset_n${n}_*/")
        python3 - "$d" "$OUT/$D/recorrido_nivel$n.txt" "$n" <<'EOF'
import csv, os, sys
d, sal, n = sys.argv[1], sys.argv[2], sys.argv[3]
F = list(csv.DictReader(open(os.path.join(d, "indice.csv"))))
imu = sum(1 for _ in open(os.path.join(d, "imu.csv"))) - 1
dur = float(F[-1]["t"]) - float(F[0]["t"]) if F else 0
open(sal, "w").write("\n".join([
    f"Recorrido nivel {n} grabado",
    f"Carpeta: {os.path.relpath(d)}",
    f"{len(F)} fotogramas IR en {dur:.1f} s ({len(F) / max(dur, 1e-3):.1f} fps), {imu} muestras de IMU",
    "",
    "indice.csv: t, imagen, intrínsecos, roll/pitch/yaw y gyro_z de la IMU en ese instante",
    "            (+ pose real de la pelvis: solo en el simulador, para evaluar)",
    "imu.csv:    rt/lowstate a ~100 Hz",
    "",
    "Sobre estos datos se desarrolla y se prueba la percepción sin robot (hito 2).",
]) + "\n")
EOF
        terminar 3
    done
}

escalon() {
    echo "== Respuesta al escalón de vyaw"
    local D=2_conjunto_datos
    mkdir -p "$OUT/$D"; rm -f "$OUT/$D"/escalon_vyaw.*
    arrancar_sim 1
    grabar 1 $D/escalon_vyaw.mp4 "Conjunto de datos · respuesta al escalón de vyaw (6.3)" \
        "Entregable: la respuesta al escalón" \
        "Move(0, 0, ±0.3 rad/s) durante 4 s, tres veces, alternando el sentido." \
        "El giroscopo de la IMU (rt/lowstate, ~200 Hz) mide el giro real." \
        "Ajuste de primer orden con retardo: w(t) = K · w_c · (1 − exp(−(t − T) / tau))." \
        "Con esto se fijan retardo_marcha, tau_marcha (y la ganancia) en config/seguidor.yaml."
    sleep 2
    cmd python3 herramientas/escalon_vyaw.py --sim --png
    cp datos/escalon_vyaw.png datos/escalon_vyaw.json "$OUT/$D/"
    terminar 4
}

hito2() {
    echo "== Hito 2: percepción sin robot"
    local d; d=$(ultimo "datos/dataset_n3_*/")
    [ -n "$d" ] || { echo "falta un datos/dataset_n3_*: ./scripts/videos_hitos.sh datos"; return 1; }
    python3 -m pytest tests -q > "$TMP/pytest.log" 2>&1
    tail -1 "$TMP/pytest.log"
    local D=3_hito2_percepcion
    carpeta $D
    cp "$TMP/pytest.log" "$OUT/$D/pruebas_automaticas.txt"
    leeme $D percepcion.mp4 "Hito 2 · Percepción sin robot, sobre el recorrido grabado del nivel 3 ($(basename "$d"))" \
        "Izquierda: umbral ADAPTATIVO (contraste relativo al fondo local). Derecha: umbral FIJO (reto 6.2.2)." \
        "Verde: puntos de la línea detectada, reproyectados sobre la imagen." \
        "Recuadro: vista cenital; en rojo, lo que el umbral considera cinta." \
        "Abajo: % de fotogramas con detección y ms por fotograma, acumulados." \
        "Resultados: percepcion_resultados.txt / .json · pruebas: pruebas_automaticas.txt" \
        "Este hito es SIN robot ni simulador (sobre datos grabados): no hay RViz ni MuJoCo."
    python3 herramientas/video_percepcion.py "$d" --salida "$OUT/$D/percepcion.mp4" --pytest "$TMP/pytest.log" | tail -1
}

hito3() {
    echo "== Hito 3: control y supervisor en simulacro"
    local D=4_hito3_simulacro
    carpeta $D
    foto_mujoco 2 $D
    arrancar_sim 2
    grabar 2 $D/simulacro.mp4 "Hito 3 · Control y supervisor en simulacro" \
        "Entregable: registro con el signo de las órdenes y el vigilante parando al tapar la cámara" \
        "seguidor.py --simulacro: cámara en vivo, percepción, estimación, control y supervisor" \
        "completos; las órdenes se calculan y se registran pero NO se envían (sin Move)." \
        "El robot lo lleva el operador (en el robot real, wasd.py; aquí operador_sim.py)." \
        "" \
        "A mitad del recorrido se tapa la cámara: el vigilante tiene que pasar a PARADA."
    echo "\$ python3 seguidor.py --sim --simulacro --nivel 2 &" >> "$CON"
    python3 seguidor.py --sim --rviz --simulacro --nivel 2 >> "$CON" 2>&1 &
    local S=$!
    sleep 2
    echo "\$ python3 herramientas/operador_sim.py --nivel 2 &" >> "$CON"
    python3 herramientas/operador_sim.py --nivel 2 --espera 3 >> "$CON" 2>&1 &
    local O=$!
    sleep 30
    echo "\$ (se tapa la cámara)  ros2 topic pub /sim/tapar_camara std_msgs/Bool '{data: true}'" >> "$CON"
    tapar true
    wait $S
    sleep 3
    kill -INT $O 2>/dev/null; wait $O
    tapar false
    local csv; csv=$(ls -t datos/tiradas/nivel2_*.csv | head -1)
    cmd python3 herramientas/grafica_simulacro.py "$csv"
    cp "${csv%.csv}_simulacro.png" "$OUT/$D/signo_ordenes_y_parada.png"
    cp "$csv" "$OUT/$D/registro_simulacro.csv"
    {
        echo "Hito 3 · Resultado"
        echo "Registro: ${csv}"
        grep "signo de vyaw" "$CON" | tail -1
        grep "\[supervisor\]" "$CON" | tail -4 | sed 's/^/  /'
        grep "^FIN:" "$CON" | tail -1
        echo ""
        echo "En simulacro no se compensa la deriva de rumbo: se estima como giroscopo − vyaw pedido,"
        echo "y lo pedido no se ejecuta (el robot lo lleva el operador)."
    } > "$OUT/$D/resultados.txt"
    terminar 4
}

nivel() {
    local n=$1
    echo "== Tirada nivel $n"
    local D=5_tiradas_niveles
    mkdir -p "$OUT/$D"; rm -f "$OUT/$D"/*nivel$n.*
    foto_mujoco "$n" $D
    arrancar_sim "$n"
    local desc=([1]="Recta de 4 m y barra de fin." [2]="Curvas en S de radio amplio." \
                [3]="Curvas cerradas (R 1.2 m), un hueco de 0.40 m y una zona en sombra." \
                [4]="Cuadrado con esquinas de 90°: giro sobre el sitio con la IMU.")
    grabar "$n" "$D/tirada_nivel$n.mp4" "Pruebas en el robot (simulado) · nivel $n" "${desc[$n]}" \
        "Tirada autónoma: seguidor.py con la palabra de confirmación (SEGUIR)." \
        "El supervisor es el único que llama a Move / StopMove; vigilante activo." \
        "Al terminar: métricas de la sección 9 con la verdad de terreno del simulador."
    echo "\$ echo SEGUIR | python3 seguidor.py --sim --nivel $n" >> "$CON"
    echo SEGUIR | python3 seguidor.py --sim --rviz --nivel "$n" >> "$CON" 2>&1
    local csv; csv=$(ls -t datos/tiradas/nivel"$n"_*.csv | head -1)
    cmd python3 herramientas/metricas.py "$csv" --png
    cp "${csv%.csv}.png" "$OUT/$D/trayectoria_nivel$n.png"
    cp "$csv" "$OUT/$D/registro_nivel$n.csv"
    { echo "Nivel $n · métricas (sec. 9)"; python3 herramientas/metricas.py "$csv" 2>/dev/null | sed 's/^/  /'; } \
        > "$OUT/$D/metricas_nivel$n.txt"
    if [ "$RVIZ" = 1 ] && ! grep -q "^Nota:" "$OUT/$D/LEEME.txt"; then
        { echo "Nota: estas tiradas se grabaron con RViz y dos grabadoras a la vez. La percepción mide"
          echo "20-35 ms por fotograma (sin grabar, ~7 ms), y eso deja menos margen en los niveles 3 y 4."; echo; } \
            >> "$OUT/$D/LEEME.txt"
    fi
    terminar 3
}

cierre() {
    echo "== Cierre: métricas"
    local D=6_cierre
    carpeta $D
    python3 - "$OUT/$D/metricas.txt" <<'EOF'
import glob, os, subprocess, sys
filas = ["Cierre · métricas de las tiradas (sec. 9)", ""]
filas.append("  nivel   éxito   error lateral real (medio / máx)   pie fuera (máx)   parada (pie-barra)   tiempo")
for n in (1, 2, 3, 4):
    c = sorted(glob.glob(f"datos/tiradas/nivel{n}_*.csv"), key=os.path.getmtime)
    if not c:
        continue
    o = subprocess.run([sys.executable, "herramientas/metricas.py", c[-1]], capture_output=True, text=True).stdout
    def buscar(clave):
        return next((l for l in o.splitlines() if clave in l), "")
    import re
    em = re.findall(r"[-+]?\d+\.\d+", buscar("error lateral real"))
    pf = re.findall(r"[-+]?\d+\.\d+", buscar("pie fuera"))
    pa = re.findall(r"[-+]?\d+\.\d+", buscar("parada:"))
    ti = re.findall(r"[-+]?\d+\.\d+", buscar("tiempo_recorrido_s"))
    ex = "SI" if "EXITO (reto, sec. 9): SI" in o else "NO"
    filas.append(f"  {n:5d}   {ex:5s}   {em[0] if em else '?':>8s} / {em[1] if len(em) > 1 else '?':>6s} m"
                 f"                {pf[0] if pf else '?':>6s} m           {pa[0] if pa else '?':>7s} m      "
                 f"{ti[0] if ti else '?':>5s} s")
filas += ["", "Informe: RESPUESTAS.md (preguntas del reto) e INTERFACES.md (sección 5).",
          "Código: seguidor.py + seguidor/ (percepción, estimación, control, supervisor, registro).",
          "Registros: datos/tiradas/*.csv|json|png (uno por tirada) y datos/dataset_n*/.",
          "", "Todo en el simulador: en el robot real falta repetir las medidas y las tiradas."]
open(sys.argv[1], "w").write("\n".join(filas) + "\n")
EOF
    cat "$OUT/$D/metricas.txt"
    python3 - "$OUT/$D/metricas.txt" "$OUT/$D/metricas.png" <<'EOF'
import sys
import cv2
sys.path.insert(0, "herramientas")
from video_comun import tarjeta
L = open(sys.argv[1], encoding="utf-8").read().splitlines()
cv2.imwrite(sys.argv[2], cv2.cvtColor(tarjeta(L[0], L[1:]), cv2.COLOR_RGB2BGR))
EOF
}

for act in "${ACTIVIDADES[@]}"; do
    case $act in
        nivel[1-4]) nivel "${act#nivel}" ;;
        hito1|datos|escalon|hito2|hito3|cierre) $act ;;
        *) echo "actividad desconocida: $act"; exit 1 ;;
    esac
done
echo "Resultados en $OUT:"
(cd "$OUT" && find . -mindepth 1 | sort)
