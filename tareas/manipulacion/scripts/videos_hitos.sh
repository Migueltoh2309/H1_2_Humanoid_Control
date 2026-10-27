#!/usr/bin/env bash
# Graba en el SIMULADOR un vídeo por actividad del plan de trabajo del reto (sec. 8), sin
# ventanas, en videos/<actividad>/, y uno con todas seguidas (videos/todas_las_actividades.mp4).
# Solo la vista de MuJoCo (herramientas/grabar_video.py --solo-mujoco), como si se grabara su
# visor, con una tarjeta de título corta antes de cada parte. Acelerados (lo dice la tarjeta).
#
#   ./scripts/videos_hitos.sh                  # todas (~1 h)
#   ./scripts/videos_hitos.sh hito3 nivel2     # solo esas
#
# Actividades: hito1 hito2 hito3 nivel1 nivel2 nivel3 nivel4
#   hito1   la estación: el robot de pie (FSM 201) ante la mesa a la distancia d elegida
#   hito2   calibración cámara-torso con el ArUco del dorso, y curva de pasos frente a la duración
#   hito3   simulacro sin arm_sdk, sus trayectorias reproducidas, un movimiento real a la pose de
#           observación y vuelta, y los abortos con salida segura (cámara tapada, giro del cuerpo)
#   nivelN  una tirada de cada nivel (semilla 0; el nivel 3 con el instructor moviendo un objeto)
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AQUI" || exit 1
source scripts/entorno.sh > /dev/null 2>&1
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
OUT="$AQUI/videos"
TMP=$(mktemp -d /tmp/videos_manip_XXXX)
FFMPEG=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
PIE="Reto R40-RT-H1_2-0002 · H1-2 en FSM 201 · simulador MuJoCo"
SIM= REC=

parar_sim() {
    local pid
    for pid in $(ps -eo pid,cmd | grep -E "ros2 launch h1_2_sim2real|h1_2_sim2real/sim --ros-args" | grep -v grep | awk '{print $1}'); do
        kill -TERM "$pid" 2>/dev/null
    done
    [ -n "$SIM" ] && wait "$SIM" 2>/dev/null
    SIM=
    sleep 3
}
limpiar() {
    [ -n "$REC" ] && kill -INT "$REC" 2>/dev/null && wait "$REC" 2>/dev/null
    parar_sim
}
trap 'limpiar; rm -rf "$TMP"' EXIT
trap 'exit 130' INT TERM

tarjeta() {         # salida.mp4 segundos "título" ["línea"...]
    local sal=$1 seg=$2; shift 2
    printf '%s\n' "$@" > "$TMP/t.txt"
    python3 herramientas/tarjeta_video.py --salida "$sal" --segundos "$seg" --fps 30 "$TMP/t.txt" > /dev/null
}

acelerar() {        # entrada salida factor: 1280x720 a 30 fps, H.264
    "$FFMPEG" -v error -y -i "$1" -an -vf "setpts=PTS/$3,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x181a20,fps=30" \
        -c:v libx264 -preset veryfast -crf 21 -pix_fmt yuv420p "$2"
}

arrancar_sim() {    # nivel semilla [argumentos de sim.launch.py]
    local n=$1 s=$2; shift 2
    ./scripts/sim_estacion.sh "$n" "$s" use_viewer:=false "$@" > "$TMP/sim.log" 2>&1 &
    SIM=$!
    until grep -q "D435 simulada" "$TMP/sim.log"; do
        sleep 0.5; kill -0 $SIM 2>/dev/null || { tail -20 "$TMP/sim.log"; exit 1; }
    done
    sleep 2
}

grabar() {          # nivel semilla [argumentos de grabar_video.py]: empieza a grabar la estación
    local n=$1 s=$2; shift 2
    # con prioridad baja: la grabadora no debe frenar el simulador ni la percepción
    nice -n 15 python3 herramientas/grabar_video.py --solo-mujoco --nivel "$n" --semilla "$s" \
        --salida "$TMP/vivo.mp4" "$@" > "$TMP/rec.log" 2>&1 &
    REC=$!
    until grep -q "grabando" "$TMP/rec.log"; do
        sleep 0.3; kill -0 $REC 2>/dev/null || { cat "$TMP/rec.log"; exit 1; }
    done
}

cortar() {          # salida.mp4 factor: deja de grabar, para el simulador y acelera
    sleep 4
    kill -INT "$REC"; wait "$REC"; REC=
    parar_sim
    acelerar "$TMP/vivo.mp4" "$1" "$2"
}

tirada() {          # salida.mp4 factor nivel semilla [argumentos de manipulacion.py]
    local sal=$1 f=$2 n=$3 s=$4; shift 4
    arrancar_sim "$n" "$s"
    grabar "$n" "$s"
    timeout 2400 python3 -u manipulacion.py --sim --nivel "$n" --semilla "$s" "$@" < /dev/null 2>&1 \
        | grep --line-buffered -E "COLOCADO|colocados|tocadas|SALIDA SEGURA|instructor\]|sobre la cruz|Traceback"
    cortar "$sal" "$f"
}

unir() {            # salida.mp4 partes...
    local sal=$1; shift
    : > "$TMP/lista.txt"
    for p in "$@"; do echo "file '$p'" >> "$TMP/lista.txt"; done
    "$FFMPEG" -v error -y -f concat -safe 0 -i "$TMP/lista.txt" -c:v libx264 -preset veryfast -crf 21 \
        -pix_fmt yuv420p -r 30 -movflags +faststart "$sal"
    echo "vídeo: $sal ($("$FFMPEG" -i "$sal" 2>&1 | grep -o 'Duration: [0-9:.]*'))"
}

hito1() {
    local P="$TMP/hito1"; mkdir -p "$P"
    tarjeta "$P/00.mp4" 5 "Hito 1 · Preparación y espacio de trabajo" "$PIE" "" \
        "El robot de pie en FSM 201, sin dar pasos, ante la mesa del reto." \
        "Tablero a 92 cm (6 cm de calzas sobre los 86 del reto), a d = 10 cm del robot:" \
        "con esa d la mano llega palma abajo al 100 % de la zona de recogida y a las 4 bandejas." \
        "Ningún brazo se mueve en este hito."
    arrancar_sim 0 0
    grabar 0 0 --orbita 8 --camara 0.3,0,0.95,2.6,150,-22
    sleep 45
    cortar "$P/01.mp4" 1
    unir "$OUT/hito1/hito1_estacion.mp4" "$P"/[0-9][0-9].mp4
}

hito2() {
    local P="$TMP/hito2"; mkdir -p "$P"
    tarjeta "$P/00.mp4" 5 "Hito 2 · Calibración y caracterización" "$PIE" "" \
        "1. Calibración cámara-torso: el brazo lleva el ArUco del dorso de la mano a un conjunto" \
        "   de poses fijas delante de la D435 (emisor IR apagado)." \
        "2. Curva de pasos: el mismo movimiento con duraciones cada vez más cortas, midiendo" \
        "   pasos y giro del cuerpo." "" "Vídeo acelerado ×4."
    tarjeta "$P/01.mp4" 3 "1 · Calibración cámara-torso (×4)"
    arrancar_sim 0 0 emisor_ir:=false
    grabar 0 0 --camara 0.2,0.05,1.2,1.9,200,-15
    python3 herramientas/calibrar.py --sim --escena sim/estaciones/estacion_n0_s0.xml 2>&1 | grep -v multicast | tail -3
    cortar "$P/02.mp4" 4
    tarjeta "$P/03.mp4" 3 "2 · Curva de pasos frente a la duración (×4)"
    arrancar_sim 0 0 emisor_ir:=false
    grabar 0 0
    python3 herramientas/curva_pasos.py --sim --lado izq 2>&1 | grep -v multicast | tail -3
    cortar "$P/04.mp4" 4
    unir "$OUT/hito2/hito2_calibracion.mp4" "$P"/[0-9][0-9].mp4
}

hito3() {
    local P="$TMP/hito3"; mkdir -p "$P"
    tarjeta "$P/00.mp4" 5 "Hito 3 · Simulacro y salida segura" "$PIE" "" \
        "1. Simulacro: cámara y planes en vivo; NO se publica arm_sdk, la mano abre y cierra en el aire" \
        "2. Las trayectorias registradas, reproducidas en MuJoCo" \
        "3. Un único movimiento real: a la pose de observación y vuelta, sin objeto" \
        "4. Aborto con salida segura al TAPAR LA CÁMARA" \
        "5. Aborto con salida segura al simular un GIRO DEL CUERPO de 6°" "" "Vídeo acelerado ×2."
    tarjeta "$P/01.mp4" 3 "1 · Simulacro: los brazos no se mueven" "Se observa, se planifica y se registra; la mano abre y cierra en el aire."
    tirada "$P/02.mp4" 2 2 0 --simulacro --t-max 150
    local R
    R=$(ls -td datos/tiradas/nivel2_* | head -1)
    tarjeta "$P/03.mp4" 3 "2 · Las trayectorias del simulacro, reproducidas (×4)"
    python3 herramientas/reproducir.py "$R" > /dev/null 2>&1
    # la reproducción lleva su rótulo arriba: se recorta (solo la vista de MuJoCo)
    "$FFMPEG" -v error -y -i "$R/reproduccion.mp4" -vf "crop=iw:ih-60:0:60" -c:v libx264 -preset veryfast \
        -crf 21 -pix_fmt yuv420p "$TMP/repro.mp4"
    acelerar "$TMP/repro.mp4" "$P/04.mp4" 4
    tarjeta "$P/05.mp4" 3 "3 · Movimiento real a la pose de observación y vuelta"
    tirada "$P/06.mp4" 2 2 0 --solo-observacion
    tarjeta "$P/07.mp4" 4 "4 · Aborto: cámara tapada" "A los 60 s, con el brazo yendo hacia el prisma, se tapa la D435." \
        "Salida segura: congelar, abrir la mano, subir a la altura de seguridad, peso a 0."
    tirada "$P/08.mp4" 2 2 0 --tapar-camara 60
    tarjeta "$P/09.mp4" 4 "5 · Aborto: giro del cuerpo" "A los 60 s se simula en la IMU un giro de 6° (el vigilante para a más de 5°)." \
        "Salida segura igual que con la cámara tapada."
    tirada "$P/10.mp4" 2 2 0 --simular-giro 60
    unir "$OUT/hito3/hito3_simulacro_y_abortos.mp4" "$P"/[0-9][0-9].mp4
}

nivel() {           # N título líneas... -- [argumentos de manipulacion.py]
    local n=$1 tit=$2; shift 2
    local lineas=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do lineas+=("$1"); shift; done
    shift
    local P="$TMP/nivel$n"; mkdir -p "$P"
    tarjeta "$P/00.mp4" 5 "Nivel $n · $tit" "$PIE" "" "${lineas[@]}" "" "Vídeo acelerado ×4."
    tirada "$P/01.mp4" 4 "$n" 0 "$@"
    unir "$OUT/nivel$n/nivel${n}.mp4" "$P"/[0-9][0-9].mp4
}

ACTS=("$@")
[ ${#ACTS[@]} -eq 0 ] && ACTS=(hito1 hito2 hito3 nivel1 nivel2 nivel3 nivel4)
for h in "${ACTS[@]}"; do
    mkdir -p "$OUT/$h"
    case $h in
        hito1|hito2|hito3) $h ;;
        nivel1) nivel 1 "Tocar dianas" "Cuatro dianas al azar en la zona de recogida: la yema del índice toca" \
                    "el centro de cada una, detecta el contacto por la fuerza y se retira." -- ;;
        nivel2) nivel 2 "Coger y colocar" "Los cuatro objetos, uno a uno, cada uno en su bandeja con la mano de su lado." -- ;;
        nivel3) nivel 3 "Clasificar con las dos manos" "Los cuatro a la vez, dos en el lado contrario a su bandeja:" \
                    "cambio de mano dejándolos en la mesa. Mientras transporta uno, el instructor" \
                    "mueve otro 5-10 cm (aquí lo hace el simulador)." -- --instructor ;;
        nivel4) nivel 4 "Colocar con orientación" "El prisma sobre la cruz del centro, girado 90°:" \
                    "por pasos, observando tras cada uno (≤ 1.5 cm y ≤ 15°)." -- --objeto prisma ;;
        *) echo "actividad desconocida: $h"; exit 1 ;;
    esac
done
TODOS=()
for h in hito1 hito2 hito3 nivel1 nivel2 nivel3 nivel4; do
    for v in "$OUT/$h"/*.mp4; do [ -f "$v" ] && TODOS+=("$v"); done
done
[ ${#TODOS[@]} -gt 1 ] && unir "$OUT/todas_las_actividades.mp4" "${TODOS[@]}"
