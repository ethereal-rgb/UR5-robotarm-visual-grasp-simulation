#!/usr/bin/env python3
import os
import sys
import unittest

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
sys.path.insert(0, SCRIPT_DIR)

from detection import (  # noqa: E402
    CameraBackProjector,
    GraspPoseConfig,
    GraspPoseEstimator,
    TemplateDigitFaceDetector,
    TfCoordinateTransformer,
    TransformMatrices,
)


class CoordinateTransformTests(unittest.TestCase):
    def test_center_pixel_backprojects_to_optical_z_axis(self):
        info = CameraInfo()
        info.header.frame_id = "wrist_camera_optical_frame"
        info.width = 640
        info.height = 480
        info.K = [554.256, 0.0, 320.0, 0.0, 554.256, 240.0, 0.0, 0.0, 1.0]
        info.P = [554.256, 0.0, 320.0, 0.0, 0.0, 554.256, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        projector = CameraBackProjector()
        projector.update_camera_info(info)

        point = projector.project((320.0, 240.0), 1.0)

        np.testing.assert_allclose(point, np.array([0.0, 0.0, 1.0]), atol=1e-9)

    def test_fixed_urdf_transforms_match_documented_matrices(self):
        transforms = TransformMatrices.from_current_urdf()

        expected_tool0_to_optical = np.array(
            [
                [0.0, 1.0, 0.0, 0.02],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.02],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        expected_tool0_to_tcp = np.eye(4)
        expected_tool0_to_tcp[2, 3] = 0.142

        np.testing.assert_allclose(transforms.tool0_to_camera_optical, expected_tool0_to_optical, atol=1e-9)
        np.testing.assert_allclose(transforms.tool0_to_gripper_tcp, expected_tool0_to_tcp, atol=1e-9)

    def test_tcp_grasp_pose_converts_to_tool0_pose_above_tcp(self):
        object_pose = PoseStamped()
        object_pose.header.frame_id = "base_link"
        object_pose.pose.position.x = 0.50
        object_pose.pose.position.y = -0.20
        object_pose.pose.position.z = 0.045
        object_pose.pose.orientation.w = 1.0

        estimator = GraspPoseEstimator(
            GraspPoseConfig(
                target_frame="base_link",
                surface_to_grasp_z_offset=-0.0225,
                pregrasp_lift=0.10,
                grasp_yaw=0.0,
            )
        )

        tcp_pose = estimator.grasp_pose_tcp(object_pose)
        tool0_pose = estimator.tool0_pose_from_tcp(tcp_pose)
        pregrasp_pose = estimator.pregrasp_pose_tcp(tcp_pose)

        self.assertAlmostEqual(tcp_pose.pose.position.z, 0.0225)
        self.assertAlmostEqual(tool0_pose.pose.position.x, tcp_pose.pose.position.x)
        self.assertAlmostEqual(tool0_pose.pose.position.y, tcp_pose.pose.position.y)
        self.assertAlmostEqual(tool0_pose.pose.position.z, tcp_pose.pose.position.z + 0.142)
        self.assertAlmostEqual(pregrasp_pose.pose.position.z, tcp_pose.pose.position.z + 0.10)

    def test_tf_transformer_retries_latest_transform_on_extrapolation(self):
        class FakeBuffer:
            def __init__(self):
                self.stamps = []

            def transform(self, pose, target_frame, timeout):
                self.stamps.append(pose.header.stamp)
                if pose.header.stamp != rospy.Time(0):
                    raise tf2_ros.ExtrapolationException("future extrapolation")
                result = PoseStamped()
                result.header.frame_id = target_frame
                result.header.stamp = pose.header.stamp
                result.pose = pose.pose
                return result

        pose = PoseStamped()
        pose.header.frame_id = "wrist_camera_optical_frame"
        pose.header.stamp = rospy.Time(28, 579000000)
        pose.pose.orientation.w = 1.0

        fake_buffer = FakeBuffer()
        transformer = TfCoordinateTransformer(fake_buffer, use_latest_on_extrapolation=True)

        result = transformer.transform_pose(pose, "base_link")

        self.assertEqual(result.header.frame_id, "base_link")
        self.assertEqual(fake_buffer.stamps, [rospy.Time(28, 579000000), rospy.Time(0)])


class TemplateDigitFaceDetectorTests(unittest.TestCase):
    def test_detects_light_digit_face_and_ignores_green_target_face(self):
        detector = TemplateDigitFaceDetector(
            class_names=[str(index) for index in range(1, 10)],
            template_size=128,
            min_face_area_px=1000,
            max_face_area_px=30000,
            face_max_saturation=85,
            min_template_score=0.5,
        )
        image = np.full((360, 480, 3), (80, 60, 40), dtype=np.uint8)
        self._draw_digit_face(image, detector, "5", top_left=(80, 110), background_bgr=(235, 245, 245))
        self._draw_digit_face(image, detector, "3", top_left=(280, 110), background_bgr=(220, 255, 220))

        detections = detector.infer(image)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "5")
        self.assertAlmostEqual(detections[0].center[0], 143.5, delta=2.0)
        self.assertAlmostEqual(detections[0].center[1], 173.5, delta=2.0)

    def test_orientation_marker_disambiguates_six_and_nine(self):
        detector = TemplateDigitFaceDetector(
            class_names=[str(index) for index in range(1, 10)],
            template_size=128,
            min_template_score=0.5,
        )

        for label in ("6", "9"):
            face = self._template_face(detector, label)
            rotated_face = np.rot90(face, 2).copy()

            detected_label, _score = detector._match_digit(face)
            rotated_label, _rotated_score = detector._match_digit(rotated_face)

            self.assertEqual(detected_label, label)
            self.assertEqual(rotated_label, label)

    @staticmethod
    def _draw_digit_face(image, detector, label, top_left, background_bgr):
        x, y = top_left
        face = np.full((128, 128, 3), background_bgr, dtype=np.uint8)
        face[detector.templates[label] > 0] = (15, 15, 15)
        image[y : y + 128, x : x + 128] = face

    @staticmethod
    def _template_face(detector, label):
        face = np.full((128, 128, 3), (245, 245, 245), dtype=np.uint8)
        face[detector.templates[label] > 0] = (15, 15, 15)
        return face


if __name__ == "__main__":
    unittest.main()
