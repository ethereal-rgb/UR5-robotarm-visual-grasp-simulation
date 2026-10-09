#!/usr/bin/env python3
import json
import math
import os
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped transforms
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


Point2 = Tuple[float, float]
BBox = Tuple[float, float, float, float]



@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: BBox
    center: Point2
    confirmed_frames: int = 1
    face_corners: Optional[List[Point2]] = None


@dataclass
class TrackState:
    filtered_center: Point2
    filtered_bbox: BBox
    confidence: float
    class_id: int
    label: str
    confirmed_frames: int = 1
    missed_frames: int = 0
    filtered_corners: Optional[List[Point2]] = None


class ParameterReader:
    def __init__(self, namespace: str = "/detection"):
        self.namespace = namespace.rstrip("/")

    def get(self, name: str, default=None):
        private_name = "~%s" % name
        public_name = "%s/%s" % (self.namespace, name)
        if rospy.has_param(private_name):
            return rospy.get_param(private_name)
        if rospy.has_param(public_name):
            return rospy.get_param(public_name)
        return rospy.get_param(name, default)

class TemplateDigitFaceDetector:
    def __init__(
        self,
        class_names: List[str],
        template_size: int = 128,
        min_face_area_px: float = 180.0,
        max_face_area_px: float = 8000.0,
        min_aspect_ratio: float = 0.55,
        max_aspect_ratio: float = 1.8,
        face_min_value: int = 200,
        face_max_saturation: int = 85,
        face_excluded_hue_min: int = 35,
        face_excluded_hue_max: int = 95,
        face_excluded_min_saturation: int = 25,
        digit_dark_threshold: int = 105,
        min_template_score: float = 0.18,
        template_shift_px: int = 5,
    ):
        self.class_names = [str(name) for name in class_names]
        self.template_size = int(template_size)
        self.min_face_area_px = float(min_face_area_px)
        self.max_face_area_px = float(max_face_area_px)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.face_min_value = int(face_min_value)
        self.face_max_saturation = int(face_max_saturation)
        self.face_excluded_hue_min = int(face_excluded_hue_min)
        self.face_excluded_hue_max = int(face_excluded_hue_max)
        self.face_excluded_min_saturation = int(face_excluded_min_saturation)
        self.digit_dark_threshold = int(digit_dark_threshold)
        self.min_template_score = float(min_template_score)
        self.template_shift_px = max(0, int(template_shift_px))
        self.templates = self._build_templates()
        self.last_mask: Optional[np.ndarray] = None
        self.last_contour_count = 0
        self.last_rejects: List[Dict] = []
        rospy.loginfo("Loaded template digit-face detector with %d templates", len(self.templates))

    def infer(self, image: np.ndarray) -> List[Detection]:
        mask = self._face_mask(image)
        self.last_mask = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.last_contour_count = len(contours)
        self.last_rejects = []
        detections: List[Detection] = []
        for contour in contours:
            candidate = self._candidate_from_contour(image, contour)
            if candidate is None:
                continue
            detections.append(candidate)
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return self._suppress_overlaps(detections)

    def _face_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, self.face_min_value], dtype=np.uint8)
        upper = np.array([179, self.face_max_saturation, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        green_like = (
            (hue >= self.face_excluded_hue_min)
            & (hue <= self.face_excluded_hue_max)
            & (saturation >= self.face_excluded_min_saturation)
        )
        mask[green_like] = 0
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _candidate_from_contour(self, image: np.ndarray, contour: np.ndarray) -> Optional[Detection]:
        area = cv2.contourArea(contour)
        if area < self.min_face_area_px or area > self.max_face_area_px:
            self._remember_reject("area", area=area)
            return None
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        if rw <= 1.0 or rh <= 1.0:
            self._remember_reject("degenerate_rect", area=area, width=rw, height=rh)
            return None
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
            self._remember_reject("aspect", area=area, width=rw, height=rh, aspect=aspect)
            return None
        box = cv2.boxPoints(rect)
        ordered_box = self._order_points(box.astype(np.float32))
        warped = self._warp_face(image, box)
        if warped is None:
            self._remember_reject("warp", area=area, width=rw, height=rh, aspect=aspect)
            return None
        label, score = self._match_digit(warped)
        if score < self.min_template_score:
            self._remember_reject("score", area=area, width=rw, height=rh, aspect=aspect, label=label, score=score)
            return None
        x, y, w, h = cv2.boundingRect(contour)
        class_id = int(label) if label.isdigit() else self.class_names.index(label)
        return Detection(
            class_id=class_id,
            label=label,
            confidence=score,
            bbox=(float(x), float(y), float(w), float(h)),
            center=(float(cx), float(cy)),
            face_corners=[(float(point[0]), float(point[1])) for point in ordered_box],
        )

    def _remember_reject(self, reason: str, **values) -> None:
        if len(self.last_rejects) >= 12:
            return
        item = {"reason": reason}
        for key, value in values.items():
            item[key] = round(float(value), 3) if isinstance(value, (float, np.floating)) else value
        self.last_rejects.append(item)

    def _warp_face(self, image: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        ordered = self._order_points(box.astype(np.float32))
        width_a = np.linalg.norm(ordered[2] - ordered[3])
        width_b = np.linalg.norm(ordered[1] - ordered[0])
        height_a = np.linalg.norm(ordered[1] - ordered[2])
        height_b = np.linalg.norm(ordered[0] - ordered[3])
        max_width = int(round(max(width_a, width_b)))
        max_height = int(round(max(height_a, height_b)))
        if max_width < 8 or max_height < 8:
            return None
        dst = np.array(
            [
                [0, 0],
                [self.template_size - 1, 0],
                [self.template_size - 1, self.template_size - 1],
                [0, self.template_size - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(image, matrix, (self.template_size, self.template_size))

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        # Cyclic ordering keeps four distinct corners even at exactly 45 degrees.
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        ordered = points[np.argsort(angles)]
        return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)

    def _match_digit(self, face_image: np.ndarray) -> Tuple[str, float]:
        candidate = self._digit_mask(face_image)
        best_label = "unknown"
        best_score = 0.0
        for rotated in self._rotations(candidate):
            for label, template in self.templates.items():
                score = self._mask_score(rotated, template, self.template_shift_px)
                if score > best_score:
                    best_score = score
                    best_label = label
        return best_label, float(best_score)

    def _digit_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray < self.digit_dark_threshold).astype(np.uint8)
        border = max(2, self.template_size // 32)
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
        kernel = np.ones((2, 2), dtype=np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    @staticmethod
    def _rotations(mask: np.ndarray) -> List[np.ndarray]:
        return [
            mask,
            cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(mask, cv2.ROTATE_180),
            cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

    @classmethod
    def _mask_score(cls, candidate: np.ndarray, template: np.ndarray, max_shift: int = 0) -> float:
        if max_shift <= 0:
            return cls._cosine_mask_score(candidate, template)

        best = 0.0
        half_shift = max_shift // 2
        shifts = sorted({0, -max_shift, max_shift, -half_shift, half_shift})
        for dy in shifts:
            for dx in shifts:
                score = cls._cosine_mask_score(cls._shift_mask(candidate, dx, dy), template)
                if score > best:
                    best = score
        return best

    @staticmethod
    def _cosine_mask_score(candidate: np.ndarray, template: np.ndarray) -> float:
        candidate_f = candidate.astype(np.float32)
        template_f = template.astype(np.float32)
        denom = math.sqrt(float(np.sum(candidate_f * candidate_f) * np.sum(template_f * template_f)))
        if denom <= 1e-6:
            return 0.0
        return float(np.sum(candidate_f * template_f) / denom)

    @staticmethod
    def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
        if dx == 0 and dy == 0:
            return mask
        shifted = np.zeros_like(mask)
        height, width = mask.shape[:2]
        src_x1 = max(0, -dx)
        src_x2 = min(width, width - dx)
        dst_x1 = max(0, dx)
        dst_x2 = min(width, width + dx)
        src_y1 = max(0, -dy)
        src_y2 = min(height, height - dy)
        dst_y1 = max(0, dy)
        dst_y2 = min(height, height + dy)
        if src_x1 < src_x2 and src_y1 < src_y2:
            shifted[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
        return shifted

    def _build_templates(self) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        for label in self.class_names:
            templates[label] = self._render_template(label)
        return templates

    def _render_template(self, label: str) -> np.ndarray:
        try:
            from PIL import Image as PilImage
            from PIL import ImageDraw, ImageFont

            image = PilImage.new("L", (256, 256), 255)
            draw = ImageDraw.Draw(image)
            font = self._font(ImageFont, 180)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text(((256 - text_w) / 2 - bbox[0], (256 - text_h) / 2 - bbox[1] - 4), label, font=font, fill=0)
            arr = np.array(image)
        except Exception:
            arr = np.full((256, 256), 255, dtype=np.uint8)
            cv2.putText(arr, label, (62, 190), cv2.FONT_HERSHEY_SIMPLEX, 6.0, 0, 18, cv2.LINE_AA)
        if label in ("6", "9"):
            cv2.line(arr, (78, 222), (178, 222), 0, 16, cv2.LINE_AA)
        arr = cv2.resize(arr, (self.template_size, self.template_size), interpolation=cv2.INTER_AREA)
        return (arr < self.digit_dark_threshold).astype(np.uint8)

    @staticmethod
    def _font(image_font, size: int):
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ):
            if os.path.exists(path):
                return image_font.truetype(path, size)
        return image_font.load_default()

    @staticmethod
    def _suppress_overlaps(detections: List[Detection]) -> List[Detection]:
        kept: List[Detection] = []
        for detection in detections:
            if all(TemplateDigitFaceDetector._iou(detection.bbox, other.bbox) < 0.35 for other in kept):
                kept.append(detection)
        return kept

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        if union <= 0.0:
            return 0.0
        return inter / union


class MultiFrameDetectionFilter:
    def __init__(self, confirm_frames: int, center_tolerance_px: float, miss_tolerance_frames: int, ema_alpha: float):
        self.confirm_frames = max(1, int(confirm_frames))
        self.center_tolerance_px = float(center_tolerance_px)
        self.miss_tolerance_frames = max(1, int(miss_tolerance_frames))
        self.ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self.tracks: Dict[str, TrackState] = {}

    def update(self, detections: List[Detection]) -> List[Detection]:
        best_by_label = self._best_by_label(detections)
        for state in self.tracks.values():
            state.missed_frames += 1

        for label, detection in best_by_label.items():
            if label not in self.tracks:
                self.tracks[label] = TrackState(
                    filtered_center=detection.center,
                    filtered_bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    label=detection.label,
                    filtered_corners=detection.face_corners,
                )
                continue

            state = self.tracks[label]
            distance = math.hypot(
                detection.center[0] - state.filtered_center[0],
                detection.center[1] - state.filtered_center[1],
            )
            if distance <= self.center_tolerance_px:
                state.confirmed_frames += 1
                state.filtered_center = self._ema_point(state.filtered_center, detection.center)
                state.filtered_bbox = self._ema_bbox(state.filtered_bbox, detection.bbox)
            else:
                state.confirmed_frames = 1
                state.filtered_center = detection.center
                state.filtered_bbox = detection.bbox
            state.filtered_corners = detection.face_corners
            state.confidence = detection.confidence
            state.class_id = detection.class_id
            state.label = detection.label
            state.missed_frames = 0

        self._remove_stale_tracks()
        return self.confirmed()

    def confirmed(self) -> List[Detection]:
        results = []
        for state in self.tracks.values():
            if state.missed_frames or state.confirmed_frames < self.confirm_frames:
                continue
            results.append(
                Detection(
                    class_id=state.class_id,
                    label=state.label,
                    confidence=state.confidence,
                    bbox=state.filtered_bbox,
                    center=state.filtered_center,
                    confirmed_frames=state.confirmed_frames,
                    face_corners=state.filtered_corners,
                )
            )
        results.sort(key=lambda detection: detection.label)
        return results

    @staticmethod
    def _best_by_label(detections: List[Detection]) -> Dict[str, Detection]:
        best: Dict[str, Detection] = {}
        for detection in detections:
            if detection.label not in best or detection.confidence > best[detection.label].confidence:
                best[detection.label] = detection
        return best

    def _ema_point(self, previous: Point2, current: Point2) -> Point2:
        return (
            self.ema_alpha * current[0] + (1.0 - self.ema_alpha) * previous[0],
            self.ema_alpha * current[1] + (1.0 - self.ema_alpha) * previous[1],
        )

    def _ema_bbox(self, previous: BBox, current: BBox) -> BBox:
        return tuple(self.ema_alpha * current[i] + (1.0 - self.ema_alpha) * previous[i] for i in range(4))  # type: ignore

    def _remove_stale_tracks(self) -> None:
        stale = [label for label, state in self.tracks.items() if state.missed_frames > self.miss_tolerance_frames]
        for label in stale:
            del self.tracks[label]


class RvizDetectionVisualizer:
    def __init__(self, bridge: CvBridge, image_topic: str, marker_topic: str, mask_topic: Optional[str] = None):
        self.bridge = bridge
        self.image_pub = rospy.Publisher(image_topic, Image, queue_size=1, latch=True)
        self.mask_pub = rospy.Publisher(mask_topic, Image, queue_size=1, latch=True) if mask_topic else None
        self.marker_pub = rospy.Publisher(marker_topic, MarkerArray, queue_size=1)

    def publish_image(self, image: np.ndarray, candidates: List[Detection], confirmed: List[Detection], header) -> None:
        annotated = image.copy()
        confirmed_labels = {detection.label for detection in confirmed}
        for detection in candidates:
            color = (0, 220, 255)
            if detection.label in confirmed_labels:
                continue
            self._draw_detection(annotated, detection, color, "candidate")
        for detection in confirmed:
            self._draw_detection(annotated, detection, (0, 220, 0), "confirmed")
        msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        msg.header = header
        self.image_pub.publish(msg)

    def publish_mask(self, mask: Optional[np.ndarray], header) -> None:
        if self.mask_pub is None or mask is None:
            return
        msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        msg.header = header
        self.mask_pub.publish(msg)

    def publish_markers(self, payload: List[Dict], frame_id: str) -> None:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        marker_id = 0
        for detection in payload:
            center_3d = detection.get("center_3d")
            if center_3d is None:
                continue
            marker_frame = center_3d.get("frame_id", frame_id)
            markers.markers.append(self._sphere(marker_id, marker_frame, center_3d))
            marker_id += 1
            markers.markers.append(self._text(marker_id, marker_frame, center_3d, detection))
            marker_id += 1
        self.marker_pub.publish(markers)

    @staticmethod
    def _draw_detection(image: np.ndarray, detection: Detection, color: Tuple[int, int, int], state: str) -> None:
        x, y, w, h = [int(round(value)) for value in detection.bbox]
        cx, cy = [int(round(value)) for value in detection.center]
        thickness = 3 if state == "confirmed" else 2
        if detection.face_corners:
            points = np.array(detection.face_corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        cv2.circle(image, (cx, cy), 4, color, -1)
        text = str(detection.label)
        origin = (x, max(24, y - 8))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        text_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        pad = 4
        bg_x1 = max(0, origin[0] - pad)
        bg_y1 = max(0, origin[1] - text_h - baseline - pad)
        bg_x2 = min(image.shape[1] - 1, origin[0] + text_w + pad)
        bg_y2 = min(image.shape[0] - 1, origin[1] + baseline + pad)
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.putText(image, text, origin, font, font_scale, color, text_thickness, cv2.LINE_AA)

    @staticmethod
    def _sphere(marker_id: int, frame_id: str, center_3d: Dict) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "digit_detection_centers"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = center_3d["x"]
        marker.pose.position.y = center_3d["y"]
        marker.pose.position.z = center_3d["z"]
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.035
        marker.color.r = 0.0
        marker.color.g = 0.9
        marker.color.b = 0.2
        marker.color.a = 0.9
        return marker

    @staticmethod
    def _text(marker_id: int, frame_id: str, center_3d: Dict, detection: Dict) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "digit_detection_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = center_3d["x"]
        marker.pose.position.y = center_3d["y"]
        marker.pose.position.z = center_3d["z"] + 0.06
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.045
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = "%s %.2f" % (detection["label"], detection["confidence"])
        return marker


@dataclass(frozen=True)
class TransformMatrices:
    tool0_to_camera_link: np.ndarray
    camera_link_to_optical: np.ndarray
    tool0_to_camera_optical: np.ndarray
    tool0_to_gripper_tcp: np.ndarray

    @classmethod
    def from_current_urdf(cls) -> "TransformMatrices":
        tool0_to_camera_link = np.array(
            [
                [0.0, 0.0, -1.0, 0.02],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.02],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        camera_link_to_optical = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        tool0_to_gripper_tcp = np.eye(4, dtype=float)
        tool0_to_gripper_tcp[2, 3] = 0.142
        return cls(
            tool0_to_camera_link=tool0_to_camera_link,
            camera_link_to_optical=camera_link_to_optical,
            tool0_to_camera_optical=tool0_to_camera_link.dot(camera_link_to_optical),
            tool0_to_gripper_tcp=tool0_to_gripper_tcp,
        )


@dataclass(frozen=True)
class GraspPoseConfig:
    target_frame: str = "base_link"
    tcp_frame: str = "gripper_tcp"
    moveit_tip_frame: str = "tool0"
    surface_to_grasp_z_offset: float = -0.0225
    pregrasp_lift: float = 0.10
    grasp_yaw: float = 0.0


class CameraBackProjector:
    def __init__(self):
        self.camera_info: Optional[CameraInfo] = None
        self.camera_model = PinholeCameraModel()

    def update_camera_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg
        self.camera_model.fromCameraInfo(msg)

    @property
    def frame_id(self) -> str:
        if self.camera_info is None:
            return ""
        return self.camera_info.header.frame_id

    def project(self, pixel: Point2, depth: float) -> np.ndarray:
        if self.camera_info is None:
            raise RuntimeError("CameraInfo is not available")
        ray = self.camera_model.projectPixelTo3dRay((float(pixel[0]), float(pixel[1])))
        return np.array([ray[0] * depth, ray[1] * depth, ray[2] * depth], dtype=float)


class TfCoordinateTransformer:
    def __init__(
        self,
        tf_buffer: tf2_ros.Buffer,
        default_timeout: float = 0.25,
        use_latest_on_extrapolation: bool = True,
    ):
        self.tf_buffer = tf_buffer
        self.default_timeout = float(default_timeout)
        self.use_latest_on_extrapolation = bool(use_latest_on_extrapolation)

    def transform_pose(self, pose: PoseStamped, target_frame: str, timeout: Optional[float] = None) -> PoseStamped:
        duration = rospy.Duration(self.default_timeout if timeout is None else float(timeout))
        try:
            return self.tf_buffer.transform(pose, target_frame, duration)
        except tf2_ros.ExtrapolationException as exc:
            if not self.use_latest_on_extrapolation:
                raise
            latest_pose = copy.deepcopy(pose)
            latest_pose.header.stamp = rospy.Time(0)
            if rospy.core.is_initialized():
                rospy.logwarn_throttle(
                    5.0,
                    "TF transform to %s at stamp %.6f hit extrapolation (%s); retrying with latest transform",
                    target_frame,
                    pose.header.stamp.to_sec(),
                    exc,
                )
            return self.tf_buffer.transform(latest_pose, target_frame, duration)


class GraspPoseEstimator:
    def __init__(self, config: GraspPoseConfig, transforms: Optional[TransformMatrices] = None):
        self.config = config
        self.transforms = transforms or TransformMatrices.from_current_urdf()
        self._tool0_to_tcp_inv = np.linalg.inv(self.transforms.tool0_to_gripper_tcp)

    def object_pose_from_point(self, point_pose: PoseStamped) -> PoseStamped:
        object_pose = PoseStamped()
        object_pose.header = point_pose.header
        object_pose.pose.position.x = point_pose.pose.position.x
        object_pose.pose.position.y = point_pose.pose.position.y
        object_pose.pose.position.z = point_pose.pose.position.z
        object_pose.pose.orientation.w = 1.0
        return object_pose

    def grasp_pose_tcp(self, object_pose: PoseStamped, yaw: Optional[float] = None) -> PoseStamped:
        grasp_pose = PoseStamped()
        grasp_pose.header = object_pose.header
        grasp_pose.pose.position.x = object_pose.pose.position.x
        grasp_pose.pose.position.y = object_pose.pose.position.y
        grasp_pose.pose.position.z = object_pose.pose.position.z + self.config.surface_to_grasp_z_offset
        self._set_quaternion(grasp_pose, self._topdown_rotation(self.config.grasp_yaw if yaw is None else yaw))
        return grasp_pose

    def pregrasp_pose_tcp(self, grasp_pose_tcp: PoseStamped) -> PoseStamped:
        pregrasp = PoseStamped()
        pregrasp.header = grasp_pose_tcp.header
        pregrasp.pose.position.x = grasp_pose_tcp.pose.position.x
        pregrasp.pose.position.y = grasp_pose_tcp.pose.position.y
        pregrasp.pose.position.z = grasp_pose_tcp.pose.position.z + self.config.pregrasp_lift
        pregrasp.pose.orientation = grasp_pose_tcp.pose.orientation
        return pregrasp

    def tool0_pose_from_tcp(self, grasp_pose_tcp: PoseStamped) -> PoseStamped:
        tcp_matrix = self._pose_to_matrix(grasp_pose_tcp)
        tool0_matrix = tcp_matrix.dot(self._tool0_to_tcp_inv)
        tool0_pose = PoseStamped()
        tool0_pose.header = grasp_pose_tcp.header
        self._matrix_to_pose(tool0_matrix, tool0_pose)
        return tool0_pose

    @staticmethod
    def _topdown_rotation(yaw: float) -> np.ndarray:
        c = math.cos(yaw)
        s = math.sin(yaw)
        x_axis = np.array([c, s, 0.0], dtype=float)
        z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    @classmethod
    def _set_quaternion(cls, pose: PoseStamped, rotation: np.ndarray) -> None:
        qx, qy, qz, qw = cls._quaternion_from_rotation(rotation)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

    @staticmethod
    def _quaternion_from_rotation(rotation: np.ndarray) -> Tuple[float, float, float, float]:
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            diag = np.diagonal(rotation)
            axis = int(np.argmax(diag))
            if axis == 0:
                scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                qw = (rotation[2, 1] - rotation[1, 2]) / scale
                qx = 0.25 * scale
                qy = (rotation[0, 1] + rotation[1, 0]) / scale
                qz = (rotation[0, 2] + rotation[2, 0]) / scale
            elif axis == 1:
                scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                qw = (rotation[0, 2] - rotation[2, 0]) / scale
                qx = (rotation[0, 1] + rotation[1, 0]) / scale
                qy = 0.25 * scale
                qz = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                qw = (rotation[1, 0] - rotation[0, 1]) / scale
                qx = (rotation[0, 2] + rotation[2, 0]) / scale
                qy = (rotation[1, 2] + rotation[2, 1]) / scale
                qz = 0.25 * scale
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        return qx / norm, qy / norm, qz / norm, qw / norm

    @staticmethod
    def _pose_to_matrix(pose: PoseStamped) -> np.ndarray:
        qx = pose.pose.orientation.x
        qy = pose.pose.orientation.y
        qz = pose.pose.orientation.z
        qw = pose.pose.orientation.w
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )
        matrix[0, 3] = pose.pose.position.x
        matrix[1, 3] = pose.pose.position.y
        matrix[2, 3] = pose.pose.position.z
        return matrix

    @classmethod
    def _matrix_to_pose(cls, matrix: np.ndarray, pose: PoseStamped) -> None:
        pose.pose.position.x = float(matrix[0, 3])
        pose.pose.position.y = float(matrix[1, 3])
        pose.pose.position.z = float(matrix[2, 3])
        qx, qy, qz, qw = cls._quaternion_from_rotation(matrix[:3, :3])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw


class DetectionPayloadBuilder:
    def __init__(self, back_projector: CameraBackProjector, transformer: TfCoordinateTransformer, grasp_estimator: GraspPoseEstimator):
        self.back_projector = back_projector
        self.transformer = transformer
        self.grasp_estimator = grasp_estimator

    def pixel_to_target_pose(self, center: Point2, depth: float, image_header, target_frame: str) -> PoseStamped:
        point_camera = self.back_projector.project(center, depth)
        camera_pose = PoseStamped()
        camera_pose.header.stamp = image_header.stamp
        camera_pose.header.frame_id = self.back_projector.frame_id or image_header.frame_id
        camera_pose.pose.position.x = float(point_camera[0])
        camera_pose.pose.position.y = float(point_camera[1])
        camera_pose.pose.position.z = float(point_camera[2])
        camera_pose.pose.orientation.w = 1.0
        return self.transformer.transform_pose(camera_pose, target_frame)

    def grasp_payload(self, object_point_pose: PoseStamped, yaw: Optional[float] = None) -> Dict:
        object_pose = self.grasp_estimator.object_pose_from_point(object_point_pose)
        tcp_pose = self.grasp_estimator.grasp_pose_tcp(object_pose, yaw)
        tool0_pose = self.grasp_estimator.tool0_pose_from_tcp(tcp_pose)
        pregrasp_pose = self.grasp_estimator.pregrasp_pose_tcp(tcp_pose)
        return {
            "object_pose": self.pose_to_dict(object_pose),
            "grasp_pose_tcp": self.pose_to_dict(tcp_pose, self.grasp_estimator.config.tcp_frame),
            "grasp_pose_tool0": self.pose_to_dict(tool0_pose, self.grasp_estimator.config.moveit_tip_frame),
            "pregrasp_pose_tcp": self.pose_to_dict(pregrasp_pose, self.grasp_estimator.config.tcp_frame),
        }

    @staticmethod
    def pose_to_dict(pose: PoseStamped, child_frame_id: Optional[str] = None) -> Dict:
        result = {
            "frame_id": pose.header.frame_id,
            "position": {
                "x": round(float(pose.pose.position.x), 6),
                "y": round(float(pose.pose.position.y), 6),
                "z": round(float(pose.pose.position.z), 6),
            },
            "orientation": {
                "x": round(float(pose.pose.orientation.x), 6),
                "y": round(float(pose.pose.orientation.y), 6),
                "z": round(float(pose.pose.orientation.z), 6),
                "w": round(float(pose.pose.orientation.w), 6),
            },
        }
        if child_frame_id:
            result["child_frame_id"] = child_frame_id
        return result


class DigitDetectionNode:
    def __init__(self):
        self.params = ParameterReader()
        self.bridge = CvBridge()
        self.back_projector = CameraBackProjector()
        self.latest_depth: Optional[np.ndarray] = None
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transformer = TfCoordinateTransformer(
            self.tf_buffer,
            default_timeout=float(self.params.get("tf_transform_timeout", 0.25)),
            use_latest_on_extrapolation=self._parse_bool(
                self.params.get("tf_use_latest_on_extrapolation", True)
            ),
        )

        class_names = self.params.get("class_names", [str(i) for i in range(10)])
        self.target_frame = self.params.get("target_frame", "base_link")
        self.grasp_target_frame = self.params.get("grasp_target_frame", self.target_frame)
        grasp_config = GraspPoseConfig(
            target_frame=self.grasp_target_frame,
            tcp_frame=self.params.get("primary_tcp_frame", "gripper_tcp"),
            moveit_tip_frame=self.params.get("moveit_tip_frame", "tool0"),
            surface_to_grasp_z_offset=float(
                self.params.get(
                    "surface_to_grasp_z_offset",
                    -0.5 * float(self.params.get("block_height", 0.045)),
                )
            ),
            pregrasp_lift=float(self.params.get("pregrasp_lift", 0.10)),
            grasp_yaw=float(self.params.get("grasp_yaw", 0.0)),
        )
        self.payload_builder = DetectionPayloadBuilder(
            self.back_projector,
            self.transformer,
            GraspPoseEstimator(grasp_config),
        )
        self.min_confirmed_detections = int(self.params.get("min_confirmed_detections", 3))

        self.detector = self._make_detector(class_names)
        self.filter = MultiFrameDetectionFilter(
            confirm_frames=self.params.get("confirm_frames", 3),
            center_tolerance_px=self.params.get("center_tolerance_px", 18),
            miss_tolerance_frames=self.params.get("miss_tolerance_frames", 5),
            ema_alpha=self.params.get("ema_alpha", 0.45),
        )

        self.detection_pub = rospy.Publisher(self.params.get("detections_topic", "/digit_detections"), String, queue_size=5)
        self.visualizer = RvizDetectionVisualizer(
            bridge=self.bridge,
            image_topic=self.params.get("detection_image_topic", "/digit_detection_image"),
            marker_topic=self.params.get("marker_topic", "/digit_detection_markers"),
            mask_topic=self.params.get("debug_mask_topic", "/digit_detection_mask"),
        )

        rospy.Subscriber(self.params.get("depth_topic", "/wrist_camera/depth/image_raw"), Image, self._on_depth, queue_size=1)
        rospy.Subscriber(
            self.params.get("camera_info_topic", "/wrist_camera/camera_info"),
            CameraInfo,
            self._on_camera_info,
            queue_size=1,
        )
        rospy.Subscriber(self.params.get("image_topic", "/wrist_camera/color/image_raw"), Image, self._on_image, queue_size=1)

    def _make_detector(self, class_names: List[str]):
        return TemplateDigitFaceDetector(
            class_names=class_names,
            template_size=self.params.get("template_size", 128),
            min_face_area_px=self.params.get("min_face_area_px", 180),
            max_face_area_px=self.params.get("max_face_area_px", 8000),
            min_aspect_ratio=self.params.get("min_aspect_ratio", 0.55),
            max_aspect_ratio=self.params.get("max_aspect_ratio", 1.8),
            face_min_value=self.params.get("face_min_value", 135),
            face_max_saturation=self.params.get("face_max_saturation", 85),
            face_excluded_hue_min=self.params.get("face_excluded_hue_min", 35),
            face_excluded_hue_max=self.params.get("face_excluded_hue_max", 95),
            face_excluded_min_saturation=self.params.get("face_excluded_min_saturation", 25),
            digit_dark_threshold=self.params.get("digit_dark_threshold", 105),
            min_template_score=self.params.get("min_template_score", 0.18),
            template_shift_px=self.params.get("template_shift_px", 5),
        )

    @staticmethod
    def _parse_bool(value) -> bool:
        return str(value).lower() in ("1", "true", "yes", "y", "on")

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.back_projector.update_camera_info(msg)

    def _on_depth(self, msg: Image) -> None:
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Depth image conversion failed: %s", exc)

    def _on_image(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            candidates = self.detector.infer(image)
            confirmed = self.filter.update(candidates)
            payload = self._build_payload(candidates, confirmed, msg.header)
            self.detection_pub.publish(String(data=json.dumps(payload)))
            self.visualizer.publish_image(image, candidates, confirmed, msg.header)
            self.visualizer.publish_mask(getattr(self.detector, "last_mask", None), msg.header)
            self.visualizer.publish_markers(payload["detections"], self.grasp_target_frame)
            rospy.loginfo_throttle(
                2.0,
                "Digit detector frame: contours=%d candidates=%d confirmed=%d image_topic=%s",
                int(getattr(self.detector, "last_contour_count", -1)),
                len(candidates),
                len(confirmed),
                self.params.get("detection_image_topic", "/digit_detection_image"),
            )
            if len(payload["detections"]) < self.min_confirmed_detections:
                rospy.logwarn_throttle(
                    5.0,
                    "Only %d confirmed digit detections; waiting for at least %d stable targets",
                    len(payload["detections"]),
                    self.min_confirmed_detections,
                )
        except Exception as exc:
            rospy.logerr_throttle(2.0, "Digit detection failed: %s", exc)

    def _build_payload(self, candidates: List[Detection], detections: List[Detection], image_header) -> Dict:
        payload = {
            "stamp": rospy.Time.now().to_sec(),
            "frame_id": self.grasp_target_frame,
            "debug": {
                "contours": int(getattr(self.detector, "last_contour_count", -1)),
                "candidates": len(candidates),
                "confirmed": len(detections),
                "candidate_labels": [
                    {
                        "label": detection.label,
                        "confidence": round(float(detection.confidence), 4),
                        "center": [int(round(detection.center[0])), int(round(detection.center[1]))],
                        "face_corners": self._corners_to_ints(detection.face_corners),
                    }
                    for detection in candidates[:12]
                ],
                "rejects": getattr(self.detector, "last_rejects", []),
                "camera_info_ready": self.back_projector.camera_info is not None,
                "depth_ready": self.latest_depth is not None,
            },
            "detections": [],
        }
        for detection in detections:
            item = {
                "class_id": detection.class_id,
                "label": detection.label,
                "confidence": round(float(detection.confidence), 6),
                "bbox": [int(round(value)) for value in detection.bbox],
                "center": [int(round(detection.center[0])), int(round(detection.center[1]))],
                "face_corners": self._corners_to_ints(detection.face_corners),
                "confirmed_frames": int(detection.confirmed_frames),
                "center_camera_3d": None,
                "center_3d": None,
                "object_pose": None,
                "grasp_pose_tcp": None,
                "grasp_pose_tool0": None,
                "pregrasp_pose_tcp": None,
            }
            camera_point = self._pixel_to_camera_point(detection.center)
            if camera_point is not None:
                item["center_camera_3d"] = {
                    "x": round(float(camera_point[0]), 6),
                    "y": round(float(camera_point[1]), 6),
                    "z": round(float(camera_point[2]), 6),
                    "frame_id": self.back_projector.frame_id,
                }
            pose = self._pixel_to_pose(detection.center, image_header)
            if pose is not None:
                item["center_3d"] = {
                    "x": pose.pose.position.x,
                    "y": pose.pose.position.y,
                    "z": pose.pose.position.z,
                    "frame_id": pose.header.frame_id,
                }
                yaw = self._face_yaw(detection, image_header)
                item["grasp_yaw"] = yaw
                # An axis-aligned detection box alone cannot determine object yaw.
                if yaw is not None:
                    item.update(self.payload_builder.grasp_payload(pose, yaw))
            payload["detections"].append(item)
        return payload

    def _face_yaw(self, detection: Detection, image_header) -> Optional[float]:
        if not detection.face_corners:
            return None
        # Sample inside the top face, avoiding background depth at its boundary.
        corners = np.asarray(detection.face_corners, dtype=float)
        center = corners.mean(axis=0)
        points = [self._pixel_to_pose(tuple(center + 0.6 * (p - center)), image_header)
                  for p in corners]
        if any(p is None for p in points):
            return None
        xy = np.array([[p.pose.position.x, p.pose.position.y] for p in points])
        edges = np.roll(xy, -1, axis=0) - xy
        if np.any(np.linalg.norm(edges, axis=1) < 0.005):
            return None
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        # Square symmetry: average modulo 90 degrees, in the base frame.
        resultant = np.mean(np.exp(4j * angles))
        if abs(resultant) < 0.8:
            return None
        return float(np.angle(resultant) / 4.0)

    @staticmethod
    def _corners_to_ints(corners: Optional[List[Point2]]) -> Optional[List[List[int]]]:
        if corners is None:
            return None
        return [[int(round(point[0])), int(round(point[1]))] for point in corners]

    def _pixel_to_camera_point(self, center: Point2) -> Optional[np.ndarray]:
        if self.back_projector.camera_info is None or self.latest_depth is None:
            return None
        depth = self._depth_at(center)
        if depth is None:
            return None
        try:
            return self.back_projector.project(center, depth)
        except Exception:
            return None

    def _pixel_to_pose(self, center: Point2, image_header) -> Optional[PoseStamped]:
        if self.back_projector.camera_info is None or self.latest_depth is None:
            return None
        depth = self._depth_at(center)
        if depth is None:
            return None
        try:
            return self.payload_builder.pixel_to_target_pose(center, depth, image_header, self.grasp_target_frame)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "TF transform to %s failed: %s", self.grasp_target_frame, exc)
            return None

    def _depth_at(self, center: Point2) -> Optional[float]:
        if self.latest_depth is None:
            return None
        u, v = int(round(center[0])), int(round(center[1]))
        height, width = self.latest_depth.shape[:2]
        if u < 0 or u >= width or v < 0 or v >= height:
            return None
        x1, x2 = max(0, u - 2), min(width, u + 3)
        y1, y2 = max(0, v - 2), min(height, v + 3)
        roi = self.latest_depth[y1:y2, x1:x2].astype(np.float32)
        valid = roi[np.isfinite(roi) & (roi > 0)]
        if valid.size == 0:
            return None
        depth = float(np.median(valid))
        if depth > 20.0:
            depth /= 1000.0
        return depth


def main() -> None:
    rospy.init_node("digit_detector")
    DigitDetectionNode()
    rospy.loginfo("Multi-frame digit detector is running")
    rospy.spin()


if __name__ == "__main__":
    main()
