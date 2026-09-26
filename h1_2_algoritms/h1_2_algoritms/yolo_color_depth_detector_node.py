#!/usr/bin/env python3
"""Método B del plan de percepción (ver PERCEPTION_PLAN.md): detección con
YOLO (modelo preentrenado en COCO, clase "orange" — calza con la mandarina
sin fine-tuning) + profundidad, sobre la misma RGB-D de
h1_2_mujoco_sim_bridge. Mismo pipeline de profundidad/deproyección/TF que
el Método A (color_depth_detector_node.py, ver perception_common.py); lo
único que cambia es la etapa de detección 2D: en vez de un umbral HSV, una
red yolov8*-seg (segmentación) cuya máscara de instancia reemplaza al
contorno de color. La detección en sí (perception_common.detect_yolo)
también la usa demos/evaluate_perception.py para medir contra el ground
truth del simulador, sin pasar por ROS.

Corre en CPU por defecto (`device:=cpu`): la GPU (GTX 1650, 4 GB) ya está
ocupada por el render de MuJoCo (nativo + offscreen de cámara), y
contender por ella degradaría a ambos. Puede compararse contra
`device:=cuda:0` como parte de la comparación cuantitativa (ver
PERCEPTION_PLAN.md, sección de métricas) si se quiere medir el costo de
mover la inferencia a GPU.

Requiere `pip install --user torch torchvision --index-url
https://download.pytorch.org/whl/cpu` y `pip install --user ultralytics`
(no son paquetes ROS/rosdep, se instalan aparte, igual que pymeshlab para
la malla de la faja).

Uso:
    ros2 launch h1_2_mujoco_sim_bridge sim_bridge_static.launch.py
    ros2 run h1_2_algoritms yolo_color_depth_detector_node

    # Método A y B a la vez, para comparar en vivo (tópicos de salida
    # distintos, no chocan):
    ros2 run h1_2_algoritms color_depth_detector_node
    ros2 run h1_2_algoritms yolo_color_depth_detector_node

Para ver la máscara/caja en vivo:
    ros2 run rqt_image_view rqt_image_view /perception/debug_image_yolo
"""
import os
import time

import cv2
import numpy as np
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped
import tf2_ros

from h1_2_algoritms.perception_common import (
    image_to_array, array_to_rgb_msg, localize_center, make_point_stamped, transform_point,
    DEFAULT_LOCALIZATION, DEFAULT_OBJECT_RADIUS_M, LOCALIZATION_MODES,
    DEFAULT_MIN_VISIBLE_FRACTION, visible_fraction,
    detect_yolo, DEFAULT_YOLO_MODEL_VARIANT, DEFAULT_YOLO_TARGET_CLASSES,
    DEFAULT_YOLO_CONF_THRESHOLD, DEFAULT_MIN_MASK_AREA_PX,
    DEFAULT_DEPTH_MIN_VALID_M, DEFAULT_DEPTH_MAX_VALID_M)

# Ultralytics/torch son pesados de importar (~1-2 s) — se hace a nivel de
# módulo (una sola vez) y no dentro del callback.
from ultralytics import YOLO

_WEIGHTS_CACHE_DIR = os.path.expanduser("~/.cache/h1_2_algoritms")


class YoloColorDepthDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_color_depth_detector_node')

        # ===== Parámetros =====
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        # Tópicos de salida distintos a los del Método A a propósito: así
        # los dos nodos pueden correr al mismo tiempo para comparar.
        self.declare_parameter('output_topic', '/perception/target_position_yolo')
        self.declare_parameter('debug_image_topic', '/perception/debug_image_yolo')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('target_frame', '')

        # Punto publicado: 'sphere' (default) = CENTRO de la fruta por ajuste
        # de esfera de radio conocido; 'median' = mediana de profundidad sin
        # corregir (el comportamiento original: mide la superficie, ~30 mm
        # más cerca de la cámara); 'median+R' = corrección cerrada. Ver
        # perception_common.localize_center y VISUAL_SERVOING_PLAN.md §3.
        self.declare_parameter('localization', DEFAULT_LOCALIZATION)
        self.declare_parameter('object_radius_m', DEFAULT_OBJECT_RADIUS_M)
        # 0 = publicar siempre (comportamiento anterior).
        self.declare_parameter('min_visible_fraction', DEFAULT_MIN_VISIBLE_FRACTION)

        # '' = usar model_variant y cachear en ~/.cache/h1_2_algoritms/
        # (se descarga solo una vez, ~7 MB). Poner una ruta propia si ya
        # se hizo fine-tuning del modelo.
        self.declare_parameter('weights_path', '')
        self.declare_parameter('model_variant', DEFAULT_YOLO_MODEL_VARIANT)
        # Lista de clases COCO aceptadas como "el objeto" — se queda con la
        # de mayor confianza entre todas. Por qué más de una clase: el
        # modelo preentrenado, sobre el render sintético (sombreado plano,
        # sin textura fotográfica), clasifica la mandarina como "sports
        # ball" (conf. ~0.79) con más confianza que como "orange" (conf.
        # ~0.12, insuficiente) — domain gap esperado entre fotos reales
        # (COCO) y render de MuJoCo. Ver PERCEPTION_PLAN.md.
        self.declare_parameter('target_class_names', list(DEFAULT_YOLO_TARGET_CLASSES))
        self.declare_parameter('conf_threshold', DEFAULT_YOLO_CONF_THRESHOLD)
        self.declare_parameter('device', 'cpu')

        self.declare_parameter('min_mask_area_px', DEFAULT_MIN_MASK_AREA_PX)
        self.declare_parameter('depth_min_valid_m', DEFAULT_DEPTH_MIN_VALID_M)
        self.declare_parameter('depth_max_valid_m', DEFAULT_DEPTH_MAX_VALID_M)

        p = {name: self.get_parameter(name).value for name in (
            'color_topic', 'depth_topic', 'camera_info_topic', 'output_topic',
            'debug_image_topic', 'publish_debug_image', 'target_frame',
            'localization', 'object_radius_m', 'min_visible_fraction',
            'weights_path', 'model_variant', 'target_class_names',
            'conf_threshold', 'device', 'min_mask_area_px',
            'depth_min_valid_m', 'depth_max_valid_m')}
        self._p = p
        if p['localization'] not in LOCALIZATION_MODES:
            raise ValueError(f"localization '{p['localization']}' no es una de {LOCALIZATION_MODES}")

        weights_path = p['weights_path'] or os.path.join(_WEIGHTS_CACHE_DIR, p['model_variant'])
        os.makedirs(_WEIGHTS_CACHE_DIR, exist_ok=True)
        self.get_logger().info(f"Cargando modelo YOLO ({weights_path}, device={p['device']})"
                                " — primera vez descarga los pesos, puede tardar...")
        t0 = time.time()
        self._model = YOLO(weights_path)
        self._model.to(p['device'])
        self.get_logger().info(f"Modelo cargado en {time.time() - t0:.1f}s")

        name_to_id = {v: k for k, v in self._model.names.items()}
        unknown = [n for n in p['target_class_names'] if n not in name_to_id]
        if unknown:
            raise RuntimeError(
                f"{unknown} no son clases de este modelo. Clases disponibles: "
                f"{sorted(name_to_id.keys())}")
        self._target_class_ids = [name_to_id[n] for n in p['target_class_names']]
        self._has_masks = 'seg' in p['model_variant'] or 'seg' in weights_path

        self._intrinsics = None
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ===== I/O =====
        self._pub_point = self.create_publisher(PointStamped, p['output_topic'], 10)
        self._pub_debug = (self.create_publisher(Image, p['debug_image_topic'], 5)
                            if p['publish_debug_image'] else None)

        self.create_subscription(CameraInfo, p['camera_info_topic'],
                                  self._on_camera_info, qos_profile_sensor_data)

        self._color_sub = message_filters.Subscriber(
            self, Image, p['color_topic'], qos_profile=qos_profile_sensor_data)
        self._depth_sub = message_filters.Subscriber(
            self, Image, p['depth_topic'], qos_profile=qos_profile_sensor_data)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub], queue_size=5, slop=0.1)
        self._sync.registerCallback(self._on_frames)

        self.get_logger().info(
            f"yolo_color_depth_detector_node: clases={p['target_class_names']} "
            f"conf>={p['conf_threshold']} mask={self._has_masks} -> '{p['output_topic']}'")

    def _on_camera_info(self, msg: CameraInfo):
        if self._intrinsics is None:
            fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
            self._intrinsics = (fx, fy, cx, cy)
            self.get_logger().info(f"Intrínsecos recibidos: fx={fx:.1f} fy={fy:.1f} "
                                    f"cx={cx:.1f} cy={cy:.1f}")

    def _on_frames(self, color_msg: Image, depth_msg: Image):
        if self._intrinsics is None:
            self.get_logger().warn("Sin CameraInfo todavía, se descarta el frame",
                                    throttle_duration_sec=2.0)
            return

        p = self._p
        rgb = image_to_array(color_msg)
        depth = image_to_array(depth_msg)
        # Ultralytics asume BGR (misma convención que cv2) para un array numpy.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        t0 = time.time()
        det = detect_yolo(self._model, self._target_class_ids, self._has_masks, bgr, depth,
                           conf_threshold=p['conf_threshold'], device=p['device'],
                           min_mask_area=p['min_mask_area_px'],
                           depth_min=p['depth_min_valid_m'], depth_max=p['depth_max_valid_m'])
        infer_ms = (time.time() - t0) * 1000.0

        if det is None:
            self.get_logger().warn(f"Sin detección válida ({infer_ms:.0f} ms inferencia)",
                                    throttle_duration_sec=2.0)
            if self._pub_debug is not None:
                self._publish_debug(rgb, None, color_msg.header.stamp)
            return

        (x, y, z), used = localize_center(det, depth, self._intrinsics,
                                          self._p['localization'], self._p['object_radius_m'])
        if used != self._p['localization']:
            self.get_logger().warn(f"localization '{self._p['localization']}' no convergió en "
                                   f"este cuadro; se usó '{used}'", throttle_duration_sec=2.0)
        frac = visible_fraction(det, z, self._intrinsics[0], self._p['object_radius_m'])
        if frac < self._p['min_visible_fraction']:
            self.get_logger().warn(f"Fruta ocluida (se ve el {100 * frac:.0f}% del disco esperado): "
                                   f"no se publica", throttle_duration_sec=1.0)
            return

        pt = make_point_stamped(x, y, z, depth_msg.header.frame_id, depth_msg.header.stamp)
        pt = transform_point(self._tf_buffer, pt, self._p['target_frame'], self.get_logger())

        self._pub_point.publish(pt)
        self.get_logger().info(
            f"target en '{pt.header.frame_id}': "
            f"({pt.point.x:.3f}, {pt.point.y:.3f}, {pt.point.z:.3f}) m, "
            f"clase='{det.label}' conf={det.score:.2f}, {infer_ms:.0f} ms",
            throttle_duration_sec=1.0)

        if self._pub_debug is not None:
            self._publish_debug(rgb, det, color_msg.header.stamp)

    def _publish_debug(self, rgb, det, stamp):
        overlay = rgb.copy()
        if det is not None:
            overlay[det.mask > 0] = (0.5 * overlay[det.mask > 0]
                                      + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            u, v = int(det.u), int(det.v)
            cv2.drawMarker(overlay, (u, v), (255, 0, 0), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(overlay, f"{det.label} z={det.z:.3f}m conf={det.score:.2f}", (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        self._pub_debug.publish(array_to_rgb_msg(overlay, "debug", stamp))


def main(args=None):
    rclpy.init(args=args)
    node = YoloColorDepthDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
