"""Utilidades compartidas por los nodos de percepción (Método A: color+
profundidad, Método B: YOLO+profundidad — ver PERCEPTION_PLAN.md) Y por
demos/evaluate_perception.py (mismo algoritmo de detección, sin ROS, para
medir contra el ground truth exacto del simulador). Factor común porque
las tres partes hacen exactamente lo mismo salvo cómo obtienen la
imagen/quién las llama: (des)serializar sensor_msgs/Image a mano (mismo
criterio que msg_builders.py del sim_bridge, evita depender de cv_bridge),
detectar el objeto 2D, sacar la profundidad mediana dentro de una máscara,
deproyectar a 3D y (los nodos ROS) transformar con TF.

`detect_yolo` NO importa ultralytics/torch a nivel de módulo — recibe el
modelo ya cargado por quien la llama — así este módulo sigue siendo liviano
para quien solo use el Método A (color_depth_detector_node no necesita
cargar torch nunca).
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf2_geometry_msgs import do_transform_point

# Umbral HSV del Método A y clases candidatas del Método B — ver el porqué
# de estos valores en color_depth_detector_node.py / yolo_color_depth_detector_node.py
# y en PERCEPTION_PLAN.md (secciones 6 y 7). Centralizados aquí para que el
# script de evaluación mida exactamente lo mismo que corre en vivo.
DEFAULT_HUE_RANGE = (0, 22)
DEFAULT_SAT_MIN = 210
DEFAULT_VAL_MIN = 170
DEFAULT_MORPH_KERNEL_SIZE = 5
DEFAULT_MIN_CONTOUR_AREA_PX = 40

DEFAULT_YOLO_MODEL_VARIANT = "yolov8n-seg.pt"
DEFAULT_YOLO_TARGET_CLASSES = ["orange", "sports ball"]
DEFAULT_YOLO_CONF_THRESHOLD = 0.25
DEFAULT_MIN_MASK_AREA_PX = 40

DEFAULT_DEPTH_MIN_VALID_M = 0.05
DEFAULT_DEPTH_MAX_VALID_M = 5.0

# Cómo pasar de la detección (máscara + profundidad) al punto 3D publicado
# (ver `localize_center` y VISUAL_SERVOING_PLAN.md §3):
#   "sphere"    ajuste de esfera de radio conocido -> CENTRO del objeto
#   "median+R"  mediana de profundidad + corrección cerrada de 0.71·R
#   "median"    mediana de profundidad sin corregir (comportamiento original:
#               mide la SUPERFICIE, sesgo de ~30 mm en z para R = 4 cm)
LOCALIZATION_MODES = ("sphere", "median+R", "median")
DEFAULT_LOCALIZATION = "sphere"
DEFAULT_OBJECT_RADIUS_M = 0.04   # mandarina de la escena; medir la real
# Fracción mínima del disco esperado que tiene que verse para publicar: con
# la mano delante, una silueta parcial sesga el centroide (y el ajuste) hacia
# el lado visible; mejor no publicar y que el consumidor prediga. Es el mismo
# criterio con el que se midió demos/visual_servoing_grasp.py (sin él, en
# vivo la mano se fue a cerrar ~12 cm al lado de la fruta).
DEFAULT_MIN_VISIBLE_FRACTION = 0.35


@dataclass
class Detection:
    u: float        # centroide en píxeles (col)
    v: float         # centroide en píxeles (fila)
    z: float          # profundidad mediana dentro de la máscara [m]
    score: float      # área en px (Método A) o confianza (Método B)
    label: str        # "color" o el nombre de clase COCO detectado
    mask: np.ndarray  # máscara booleana/uint8, mismo H×W que la imagen


def image_to_array(msg: Image) -> np.ndarray:
    if msg.encoding == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    raise ValueError(f"encoding no soportado: {msg.encoding}")


def array_to_rgb_msg(rgb: np.ndarray, frame_id: str, stamp) -> Image:
    msg = Image()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height, msg.width = rgb.shape[0], rgb.shape[1]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return msg


def median_depth_in_mask(depth: np.ndarray, mask: np.ndarray,
                          depth_min: float, depth_max: float) -> Optional[float]:
    """mask: array booleano/uint8 del mismo H×W que depth. None si no hay
    ningún píxel de profundidad válida dentro de la máscara."""
    depth_vals = depth[mask.astype(bool)]
    valid = depth_vals[np.isfinite(depth_vals) & (depth_vals > depth_min) & (depth_vals < depth_max)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def deproject(u: float, v: float, z: float, fx: float, fy: float,
              cx: float, cy: float) -> tuple:
    """Pinhole estándar (convención "optical frame" de ROS: X derecha,
    Y abajo, Z adelante) — z es la profundidad ya medida a lo largo del
    eje óptico, no la distancia euclidiana al punto."""
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return float(x), float(y), float(z)


def make_point_stamped(x: float, y: float, z: float, frame_id: str, stamp) -> PointStamped:
    pt = PointStamped()
    pt.header = Header(frame_id=frame_id, stamp=stamp)
    pt.point.x, pt.point.y, pt.point.z = x, y, z
    return pt


def transform_point(tf_buffer: tf2_ros.Buffer, pt: PointStamped, target_frame: str,
                     logger) -> PointStamped:
    """Intenta llevar pt a target_frame vía TF; si falla, devuelve pt sin
    tocar (mismo frame de origen) y deja un warning (throttled)."""
    if not target_frame or target_frame == pt.header.frame_id:
        return pt
    try:
        import rclpy.time
        tf = tf_buffer.lookup_transform(target_frame, pt.header.frame_id, rclpy.time.Time())
        out = do_transform_point(pt, tf)
        out.header.frame_id = target_frame
        return out
    except tf2_ros.TransformException as ex:
        logger.warn(f"TF {pt.header.frame_id}->{target_frame} falló: {ex} "
                    f"— publico en frame de cámara", throttle_duration_sec=2.0)
        return pt


def detect_color(rgb: np.ndarray, depth: np.ndarray,
                  hue_range=DEFAULT_HUE_RANGE, sat_min=DEFAULT_SAT_MIN, val_min=DEFAULT_VAL_MIN,
                  kernel_size=DEFAULT_MORPH_KERNEL_SIZE, min_area=DEFAULT_MIN_CONTOUR_AREA_PX,
                  depth_min=DEFAULT_DEPTH_MIN_VALID_M, depth_max=DEFAULT_DEPTH_MAX_VALID_M
                  ) -> Optional[Detection]:
    """Método A: umbral HSV + contorno más grande + profundidad mediana."""
    lower = np.array([hue_range[0], sat_min, val_min], dtype=np.uint8)
    upper = np.array([hue_range[1], 255, 255], dtype=np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    m = cv2.moments(largest)
    if m['m00'] == 0:
        return None
    u, v = m['m10'] / m['m00'], m['m01'] / m['m00']

    contour_mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [largest], -1, 255, thickness=cv2.FILLED)
    z = median_depth_in_mask(depth, contour_mask, depth_min, depth_max)
    if z is None:
        return None
    return Detection(u, v, z, float(area), "color", contour_mask)


def detect_yolo(model, class_ids: List[int], has_masks: bool, bgr: np.ndarray, depth: np.ndarray,
                 conf_threshold=DEFAULT_YOLO_CONF_THRESHOLD, device="cpu",
                 min_mask_area=DEFAULT_MIN_MASK_AREA_PX,
                 depth_min=DEFAULT_DEPTH_MIN_VALID_M, depth_max=DEFAULT_DEPTH_MAX_VALID_M
                 ) -> Optional[Detection]:
    """Método B: YOLO (bbox o, si el modelo es -seg, máscara de instancia)
    + profundidad mediana. `model` ya debe estar cargado (`YOLO(weights)`) —
    esta función no importa ultralytics, así que no le agrega esa
    dependencia a quien solo use el Método A."""
    results = model.predict(bgr, conf=conf_threshold, classes=class_ids,
                             device=device, verbose=False)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return None

    confs = boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    conf = float(confs[best])
    cls_name = model.names[int(boxes.cls[best])]

    mask = None
    if has_masks and results.masks is not None:
        # results.masks.data ya viene reescalada por ultralytics al tamaño
        # original de la imagen (mismo H×W que depth).
        mask = results.masks.data[best].cpu().numpy().astype(np.uint8)
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        if int(mask.sum()) < min_mask_area:
            mask = None

    if mask is not None:
        m = cv2.moments(mask, binaryImage=True)
        if m['m00'] == 0:
            mask = None
        else:
            u, v = m['m10'] / m['m00'], m['m01'] / m['m00']

    if mask is None:
        # Sin máscara (modelo sin segmentación, o máscara descartada): se
        # cae a la caja — centro del bbox y una máscara rectangular
        # recortada al 60% central para no mezclar profundidad de fondo.
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = (x2 - x1), (y2 - y1)
        mask = np.zeros(depth.shape, dtype=np.uint8)
        cx1, cy1 = int(u - 0.3 * bw), int(v - 0.3 * bh)
        cx2, cy2 = int(u + 0.3 * bw), int(v + 0.3 * bh)
        mask[max(cy1, 0):cy2, max(cx1, 0):cx2] = 1

    z = median_depth_in_mask(depth, mask, depth_min, depth_max)
    if z is None:
        return None
    return Detection(u, v, z, conf, cls_name, mask)


def localize_center(det: Detection, depth: np.ndarray, intrinsics: tuple,
                    mode: str = DEFAULT_LOCALIZATION,
                    radius: float = DEFAULT_OBJECT_RADIUS_M) -> tuple:
    """Punto 3D (frame óptico) de una detección, y el modo que se usó.

    La profundidad dentro de la máscara es la de la superficie visible, no
    la del centro: la mediana queda ~0.71·R más cerca de la cámara (medido
    en simulación: 33 mm de error; con ajuste de esfera, 2.4 mm con ruido
    tipo D435 y 6 mm con la mano tapando parte de la fruta). "sphere" cae a
    "median+R" si el ajuste no converge o da algo incoherente con la
    mediana (p.ej. máscara con muy pocos píxeles de profundidad válida).
    Devuelve ((x, y, z), modo_usado)."""
    if mode not in LOCALIZATION_MODES:
        raise ValueError(f"localization '{mode}' no es una de {LOCALIZATION_MODES}")
    fx, fy, cx, cy = intrinsics
    if mode == "sphere":
        # Import local: fruit_localization no depende de ROS, pero así
        # este módulo no carga nada extra si se usa "median".
        from h1_2_algoritms.vision.fruit_localization import Intrinsics, locate_sphere
        c = locate_sphere(det.mask, depth, Intrinsics(fx, fy, cx, cy), radius)
        # El centro debe quedar detrás de la superficie medida, a menos de
        # ~R de ella (0.71·R en el caso ideal).
        if c is not None and np.all(np.isfinite(c)) and -0.5 * radius < c[2] - det.z < 2.0 * radius:
            return (float(c[0]), float(c[1]), float(c[2])), "sphere"
        mode = "median+R"
    x, y, z = deproject(det.u, det.v, det.z, fx, fy, cx, cy)
    if mode == "median+R":
        k = (z + radius * np.sqrt(0.5)) / z
        x, y, z = x * k, y * k, z * k
    return (float(x), float(y), float(z)), mode


def visible_fraction(det: Detection, center_z: float, fx: float,
                     radius: float = DEFAULT_OBJECT_RADIUS_M) -> float:
    """Área de la máscara / área del disco que proyectaría una esfera de
    radio `radius` cuyo centro está a profundidad `center_z`."""
    if center_z <= 0:
        return 0.0
    expected = np.pi * (fx * radius / center_z) ** 2
    return float(np.count_nonzero(det.mask)) / expected
