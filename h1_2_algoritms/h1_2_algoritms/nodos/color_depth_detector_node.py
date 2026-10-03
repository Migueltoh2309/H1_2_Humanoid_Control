#!/usr/bin/env python3
"""Método A del plan de percepción (ver PERCEPTION_PLAN.md): detección por
color (HSV) + profundidad sobre la RGB-D de h1_2_mujoco_sim_bridge.

Sincroniza /camera/color/image_raw y /camera/depth/image_raw, segmenta la
mandarina por color naranja, y con la profundidad dentro del contorno + los
intrínsecos de /camera/color/camera_info estima el CENTRO 3D de la fruta
(parámetro `localization`: por defecto ajuste de esfera de radio conocido;
'median' = la mediana de profundidad original, que mide la superficie y
queda ~30 mm corta en z) — publicado como geometry_msgs/PointStamped en
/perception/target_position (frame por defecto: camera_depth_optical_frame,
el mismo que publica el bridge; con target_frame se transforma vía TF, p.ej.
a "pelvis").

La detección en sí (perception_common.detect_color) también la usa
demos/evaluate_perception.py para medir contra el ground truth del
simulador, sin pasar por ROS.

No usa cv_bridge (mismo criterio que msg_builders.py del sim_bridge: armar
el array a mano es trivial y evita esa dependencia frágil).

Uso:
    ros2 launch h1_2_mujoco_sim_bridge sim_bridge_static.launch.py
    ros2 run h1_2_algoritms color_depth_detector_node

Para ver la máscara/contorno en vivo:
    ros2 run rqt_image_view rqt_image_view /perception/debug_image
"""
import cv2
import numpy as np
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped
import tf2_ros

from h1_2_algoritms.vision.perception_common import (
    image_to_array, array_to_rgb_msg, localize_center, make_point_stamped, transform_point,
    DEFAULT_LOCALIZATION, DEFAULT_OBJECT_RADIUS_M, LOCALIZATION_MODES,
    DEFAULT_MIN_VISIBLE_FRACTION, visible_fraction,
    detect_color, DEFAULT_HUE_RANGE, DEFAULT_SAT_MIN, DEFAULT_VAL_MIN,
    DEFAULT_MORPH_KERNEL_SIZE, DEFAULT_MIN_CONTOUR_AREA_PX,
    DEFAULT_DEPTH_MIN_VALID_M, DEFAULT_DEPTH_MAX_VALID_M)


class ColorDepthDetectorNode(Node):

    def __init__(self):
        super().__init__('color_depth_detector_node')

        # ===== Parámetros =====
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('output_topic', '/perception/target_position')
        self.declare_parameter('debug_image_topic', '/perception/debug_image')
        self.declare_parameter('publish_debug_image', True)
        # Vacío = publicar en el frame de la cámara (header del depth, ya
        # optical-frame de ROS). Si se llena, p.ej. "pelvis", se transforma
        # con TF (necesita el tf_static que publica sim_bridge_node.py).
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

        # Umbral HSV (convención OpenCV: H en [0,179]) calibrado para la
        # mandarina (rgba 1.0 0.45 0.05 del MJCF -> H≈13, S≈242, V≈255).
        # El hue solo NO alcanza para distinguirla de la mesa marrón/mostaza
        # (rgba 0.58 0.38 0.14 -> H≈16, casi el mismo hue): con
        # sat_min/val_min bajos (probado en vivo) la mesa entera se cuela
        # como "detección" más grande que la mandarina. La mesa tiene
        # S≈193/V≈148 (menos saturada/oscura); estos mínimos quedan justo
        # por encima para excluirla y quedarse solo con la mandarina.
        # Recalibrar si cambia el objeto o la iluminación. Valores por
        # defecto centralizados en perception_common.py.
        self.declare_parameter('hue_min', DEFAULT_HUE_RANGE[0])
        self.declare_parameter('hue_max', DEFAULT_HUE_RANGE[1])
        self.declare_parameter('sat_min', DEFAULT_SAT_MIN)
        self.declare_parameter('val_min', DEFAULT_VAL_MIN)

        self.declare_parameter('min_contour_area_px', DEFAULT_MIN_CONTOUR_AREA_PX)
        self.declare_parameter('morph_kernel_size', DEFAULT_MORPH_KERNEL_SIZE)
        self.declare_parameter('depth_min_valid_m', DEFAULT_DEPTH_MIN_VALID_M)
        self.declare_parameter('depth_max_valid_m', DEFAULT_DEPTH_MAX_VALID_M)

        p = {name: self.get_parameter(name).value for name in (
            'color_topic', 'depth_topic', 'camera_info_topic', 'output_topic',
            'debug_image_topic', 'publish_debug_image', 'target_frame',
            'localization', 'object_radius_m', 'min_visible_fraction',
            'hue_min', 'hue_max', 'sat_min', 'val_min',
            'min_contour_area_px', 'morph_kernel_size',
            'depth_min_valid_m', 'depth_max_valid_m')}
        self._p = p
        if p['localization'] not in LOCALIZATION_MODES:
            raise ValueError(f"localization '{p['localization']}' no es una de {LOCALIZATION_MODES}")

        self._intrinsics = None  # (fx, fy, cx, cy)
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
            f"color_depth_detector_node: color='{p['color_topic']}' "
            f"depth='{p['depth_topic']}' -> '{p['output_topic']}' "
            f"(HSV H[{p['hue_min']},{p['hue_max']}] S>={p['sat_min']} V>={p['val_min']}, "
            f"target_frame='{p['target_frame'] or '(cámara)'}')")

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

        det = detect_color(rgb, depth,
                            hue_range=(p['hue_min'], p['hue_max']),
                            sat_min=p['sat_min'], val_min=p['val_min'],
                            kernel_size=p['morph_kernel_size'], min_area=p['min_contour_area_px'],
                            depth_min=p['depth_min_valid_m'], depth_max=p['depth_max_valid_m'])

        if det is None:
            self.get_logger().warn("Sin detección válida (color u.o. profundidad)",
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
            f"({pt.point.x:.3f}, {pt.point.y:.3f}, {pt.point.z:.3f}) m, área={det.score:.0f}px",
            throttle_duration_sec=1.0)

        if self._pub_debug is not None:
            self._publish_debug(rgb, det, color_msg.header.stamp)

    def _publish_debug(self, rgb, det, stamp):
        overlay = rgb.copy()
        if det is not None:
            overlay[det.mask > 0] = (0.5 * overlay[det.mask > 0]
                                      + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            u, v = int(det.u), int(det.v)
            cv2.drawMarker(overlay, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(overlay, f"z={det.z:.3f}m", (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        self._pub_debug.publish(array_to_rgb_msg(overlay, "debug", stamp))


def main(args=None):
    rclpy.init(args=args)
    node = ColorDepthDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
