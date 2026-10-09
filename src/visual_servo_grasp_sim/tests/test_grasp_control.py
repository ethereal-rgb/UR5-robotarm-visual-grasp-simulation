import math
import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from controller import (DetectionBuffer, GazeboGraspConstraint, GripperCommander,
                        MoveItGraspController, ToolPoseConverter, VisualServoGraspNode)
from detection import Detection, DigitDetectionNode, GraspPoseConfig, GraspPoseEstimator, TemplateDigitFaceDetector


class GraspRegressionTests(unittest.TestCase):
    def test_rotated_square_corners_remain_distinct(self):
        for angle in (0, 44.999, 45, 90, 135, -45):
            box = cv2.boxPoints(((100, 100), (50, 50), angle))
            ordered = TemplateDigitFaceDetector._order_points(box)
            self.assertEqual(len(np.unique(ordered, axis=0)), 4)
            self.assertAlmostEqual(abs(cv2.contourArea(ordered)), 2500, delta=0.01)

    def test_yaw_uses_transformed_edges_and_square_symmetry(self):
        node = DigitDetectionNode.__new__(DigitDetectionNode)
        for yaw in (-2.9, -0.785398, 0.3, 1.8):
            c, s = math.cos(yaw), math.sin(yaw)
            def project(pixel, header):
                p = PoseStamped()
                p.pose.position.x = c * pixel[0] - s * pixel[1] + 0.6
                p.pose.position.y = s * pixel[0] + c * pixel[1] - 0.2
                return p
            node._pixel_to_pose = project
            detection = Detection(1, '1', 1, (0, 0, 1, 1), (0, 0),
                                  face_corners=[(-.02, -.02), (.02, -.02), (.02, .02), (-.02, .02)])
            actual = node._face_yaw(detection, None)
            self.assertAlmostEqual(math.cos(4 * (actual - yaw)), 1.0)
        detection.face_corners = None
        self.assertIsNone(node._face_yaw(detection, None))

    def test_measured_yaw_reaches_tool_pose(self):
        estimator = GraspPoseEstimator(GraspPoseConfig())
        p = PoseStamped()
        p.pose.position.z = .045
        tcp = estimator.grasp_pose_tcp(p, .37)
        tool = estimator.tool0_pose_from_tcp(tcp)
        rotation = ToolPoseConverter._pose_to_matrix(tool)[:3, :3]
        np.testing.assert_allclose(rotation[:, 0], [math.cos(.37), math.sin(.37), 0], atol=1e-9)
        self.assertAlmostEqual(tool.pose.position.z, .1645)

    def test_width_mapping_matches_pad_geometry(self):
        gripper = GripperCommander.__new__(GripperCommander)
        gripper.closed_position = .78
        for width in (.041, .045, .053):
            q = gripper.position_for_width(width)
            actual = 2 * (.0127 + .03759940821 * math.cos(q) - .04303959807 * math.sin(q) - .00780198)
            self.assertAlmostEqual(actual, width)

    def test_attachment_requires_both_pads_on_requested_block(self):
        gripper = GripperCommander.__new__(GripperCommander)
        contact = SimpleNamespace(collision1_name='robot::pad', collision2_name='digit_block_1::link')
        gripper._contacts = {'left': ([contact], 0), 'right': ([contact], 0)}
        with patch('controller.rospy.is_shutdown', return_value=False), patch('controller.time.sleep'), patch('controller.time.monotonic', side_effect=[0, 0, 0, .16, .16]):
            gripper.wait_for_grasp(1)
        for states in ({'left': ([contact], 0)}, {'left': ([contact], 0), 'right': ([contact], 0)}):
            gripper._contacts = states
            with patch('controller.rospy.is_shutdown', return_value=False), patch('controller.time.sleep'), patch('controller.time.monotonic', side_effect=[0, 0, 0, 5]):
                with self.assertRaises(RuntimeError):
                    gripper.wait_for_grasp(2)

    def test_place_compensates_actual_grasp_translation(self):
        constraint = GazeboGraspConstraint.__new__(GazeboGraspConstraint)
        constraint._lock = threading.Lock()
        constraint._tcp_to_model = np.eye(4)
        constraint._tcp_to_model[:3, 3] = [.004, -.003, .009]
        target = PoseStamped()
        target.pose.position.z = .2
        target.pose.orientation.x = 1
        corrected = constraint.compensated_place_pose(target)
        matrix = ToolPoseConverter._pose_to_matrix(corrected)
        np.testing.assert_allclose((matrix @ constraint._tcp_to_model)[:3, 3], [0, 0, .2], atol=1e-9)

    def test_cartesian_trajectory_is_retimed_before_execution(self):
        motion = MoveItGraspController.__new__(MoveItGraspController)
        motion.cartesian_fraction_threshold = .999
        motion.velocity_scaling = .1
        motion.acceleration_scaling = .1
        motion.group = Mock()
        motion._ensure_pose_frame = Mock()
        motion._prepare_start_state = Mock()
        motion._compute_cartesian_path = Mock(return_value=(SimpleNamespace(joint_trajectory=SimpleNamespace(points=[1])), 1.0))
        motion._execute = Mock()
        with patch('controller.rospy.loginfo'):
            motion.move_cartesian('descend_to_place', PoseStamped())
        self.assertEqual(motion.group.retime_trajectory.call_args.kwargs['velocity_scaling_factor'], .02)
        self.assertEqual(motion.group.retime_trajectory.call_args.kwargs['acceleration_scaling_factor'], .02)
        motion.group.set_start_state_to_current_state.assert_called_once()
        motion._execute.assert_called_once_with('descend_to_place', motion.group.retime_trajectory.return_value)

    def test_pregrasp_corrects_pose_above_block_before_descent(self):
        node = VisualServoGraspNode.__new__(VisualServoGraspNode)
        node.motion = Mock()
        node.pregrasp_lift = .1
        node.grasp_yaw_offsets = [0]
        grasp = PoseStamped()
        grasp.pose.orientation.x = 1
        grasp.pose.position.z = .1645
        with patch('controller.rospy.loginfo'), patch('controller.rospy.Time.now'):
            result = node._move_to_reachable_pregrasp(grasp)
        calls = node.motion.method_calls
        self.assertEqual([call[0] for call in calls], ['move_to_pose', 'move_cartesian'])
        self.assertEqual(calls[1].args[0], 'align_pregrasp')
        self.assertAlmostEqual(calls[1].args[1].pose.position.z, .2645)
        self.assertAlmostEqual(result.pose.position.z, .1645)

    def test_stability_reset_requires_new_camera_frames(self):
        buffer = DetectionBuffer.__new__(DetectionBuffer)
        buffer._stable_counts = {}
        buffer._latest_stamp = Mock(return_value=10)
        buffer._target_from_latest = Mock(return_value=object())
        buffer.reset_stability(1)
        self.assertIsNone(buffer.stable_block(1, 2))
        buffer._latest_stamp.return_value = 11
        self.assertIsNone(buffer.stable_block(1, 2))
        self.assertIsNone(buffer.stable_block(1, 2))
        buffer._latest_stamp.return_value = 12
        self.assertIsNotNone(buffer.stable_block(1, 2))
