#!/usr/bin/env python3
import copy
import json
import os
import random
import threading
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import actionlib
import moveit_commander
import numpy as np
import rospy
import rospkg
import tf2_ros
import yaml
from gazebo_msgs.msg import ContactsState, ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    Constraints,
    MoveGroupAction,
    PlanningScene,
    PlanningSceneComponents,
    RobotTrajectory,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
from std_msgs.msg import Float64, String
from sensor_msgs.msg import JointState


def pose_from_dict(data: Dict, default_frame: str) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = data.get("frame_id") or default_frame
    pose.header.stamp = rospy.Time.now()

    position = data.get("position", {})
    orientation = data.get("orientation", {})
    pose.pose.position.x = float(position.get("x", 0.0))
    pose.pose.position.y = float(position.get("y", 0.0))
    pose.pose.position.z = float(position.get("z", 0.0))
    pose.pose.orientation.x = float(orientation.get("x", 0.0))
    pose.pose.orientation.y = float(orientation.get("y", 0.0))
    pose.pose.orientation.z = float(orientation.get("z", 0.0))
    pose.pose.orientation.w = float(orientation.get("w", 1.0))
    return pose


def lifted_pose(pose: PoseStamped, dz: float) -> PoseStamped:
    result = copy.deepcopy(pose)
    result.header.stamp = rospy.Time.now()
    result.pose.position.z += float(dz)
    return result


def rotated_pose_about_base_z(pose: PoseStamped, yaw_offset: float) -> PoseStamped:
    result = copy.deepcopy(pose)
    result.header.stamp = rospy.Time.now()
    if abs(float(yaw_offset)) < 1e-9:
        return result

    pose_matrix = ToolPoseConverter._pose_to_matrix(pose)
    c = np.cos(float(yaw_offset))
    s = np.sin(float(yaw_offset))
    yaw_matrix = np.eye(4, dtype=float)
    yaw_matrix[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rotated = yaw_matrix.dot(pose_matrix)
    rotated[:3, 3] = pose_matrix[:3, 3]
    ToolPoseConverter._matrix_to_pose(rotated, result)
    return result


def has_points(plan: RobotTrajectory) -> bool:
    return bool(plan and plan.joint_trajectory and plan.joint_trajectory.points)


ACCESSORY_LINKS = {
    "flange",
    "tool0",
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link",
    "robotiq_85_right_knuckle_link",
    "robotiq_85_left_finger_link",
    "robotiq_85_right_finger_link",
    "robotiq_85_left_inner_knuckle_link",
    "robotiq_85_right_inner_knuckle_link",
    "robotiq_85_left_finger_tip_link",
    "robotiq_85_right_finger_tip_link",
    "gripper_tcp",
    "tool0_controller",
    "wrist_camera_link",
    "wrist_camera_optical_frame",
}

MOUNT_ADJACENT_LINKS = {
    "wrist_2_link",
    "wrist_3_link",
    "flange",
    "tool0",
}


class ToolPoseConverter:
    def __init__(self, tcp_offset_z: float = 0.142):
        self.tool0_to_tcp = np.eye(4, dtype=float)
        self.tool0_to_tcp[2, 3] = float(tcp_offset_z)
        self.tcp_to_tool0 = np.linalg.inv(self.tool0_to_tcp)

    def tool0_pose_from_tcp(self, tcp_pose: PoseStamped) -> PoseStamped:
        tcp_matrix = self._pose_to_matrix(tcp_pose)
        tool0_matrix = tcp_matrix.dot(self.tcp_to_tool0)
        tool0_pose = PoseStamped()
        tool0_pose.header = tcp_pose.header
        self._matrix_to_pose(tool0_matrix, tool0_pose)
        return tool0_pose

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

    @staticmethod
    def _quaternion_from_rotation(rotation: np.ndarray) -> Tuple[float, float, float, float]:
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            diag = np.diagonal(rotation)
            axis = int(np.argmax(diag))
            if axis == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                qw = (rotation[2, 1] - rotation[1, 2]) / scale
                qx = 0.25 * scale
                qy = (rotation[0, 1] + rotation[1, 0]) / scale
                qz = (rotation[0, 2] + rotation[2, 0]) / scale
            elif axis == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                qw = (rotation[0, 2] - rotation[2, 0]) / scale
                qx = (rotation[0, 1] + rotation[1, 0]) / scale
                qy = 0.25 * scale
                qz = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                qw = (rotation[1, 0] - rotation[0, 1]) / scale
                qx = (rotation[0, 2] + rotation[2, 0]) / scale
                qy = (rotation[1, 2] + rotation[2, 1]) / scale
                qz = 0.25 * scale
        norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        return qx / norm, qy / norm, qz / norm, qw / norm


@dataclass
class DetectedTarget:
    block_id: int
    label: str
    confidence: float
    grasp_pose_tool0: PoseStamped


@dataclass(frozen=True)
class PickPlaceTask:
    block_id: int
    target_id: int


class InteractiveTaskCollector:
    def __init__(self, block_min: int = 1, block_max: int = 9, target_min: int = 1, target_max: int = 6):
        self.block_min = int(block_min)
        self.block_max = int(block_max)
        self.target_min = int(target_min)
        self.target_max = int(target_max)

    def collect(self) -> List[PickPlaceTask]:
        tasks: List[PickPlaceTask] = []
        while not rospy.is_shutdown():
            block_id = self._read_int("要抓取几号方块？请输入 1-9: ", self.block_min, self.block_max)
            target_id = self._read_int("要放置到几号格子？请输入 1-6: ", self.target_min, self.target_max)
            tasks.append(PickPlaceTask(block_id=block_id, target_id=target_id))

            if not self._read_continue():
                break
        return tasks

    @staticmethod
    def _input(prompt: str) -> str:
        return input(prompt).strip()

    def _read_int(self, prompt: str, minimum: int, maximum: int) -> int:
        while not rospy.is_shutdown():
            value = self._input(prompt)
            try:
                number = int(value)
            except ValueError:
                print("请输入整数。")
                continue
            if minimum <= number <= maximum:
                return number
            print("请输入 %d-%d 范围内的数字。" % (minimum, maximum))
        raise RuntimeError("ROS shutdown while reading task input")

    def _read_continue(self) -> bool:
        while not rospy.is_shutdown():
            value = self._input("是否还有任务？请输入 y 或 n: ").lower()
            if value == "y":
                return True
            if value == "n":
                return False
            print("请输入 y 或 n。")
        raise RuntimeError("ROS shutdown while reading task input")


class DetectionBuffer:
    def __init__(self, topic: str, target_frame: str):
        self.topic = topic
        self.target_frame = target_frame
        self._latest_payload: Optional[Dict] = None
        self._latest_payload_time: Optional[float] = None
        self._stable_counts: Dict[int, Tuple[Optional[float], int, Optional[DetectedTarget]]] = {}
        rospy.Subscriber(topic, String, self._callback, queue_size=5)

    def _callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except ValueError as exc:
            rospy.logwarn_throttle(2.0, "Could not parse detection payload: %s", exc)
            return
        self._latest_payload = payload
        self._latest_payload_time = time.monotonic()

    def wait_for_block(self, block_id: int, timeout: float) -> DetectedTarget:
        return self.wait_for_stable_block(block_id, timeout, stable_observations=1)

    def wait_for_stable_block(self, block_id: int, timeout: float, stable_observations: int = 1, poll_rate: float = 10.0) -> DetectedTarget:
        self.reset_stability(block_id)
        deadline = time.monotonic() + float(timeout)
        next_status_log = time.monotonic() if float(timeout) >= 2.0 else time.monotonic() + float(timeout) + 1.0
        rate = rospy.Rate(max(1.0, float(poll_rate)))
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            target = self.stable_block(block_id, stable_observations)
            if target is not None:
                return target
            now = time.monotonic()
            if now >= next_status_log:
                rospy.logwarn("%s", self._status_message(block_id))
                next_status_log = now + 3.0
            rate.sleep()
        raise RuntimeError(
            "Timed out waiting for block %d with a valid grasp_pose_tool0 on %s. Last status: %s"
            % (block_id, self.topic, self._status_message(block_id))
        )

    def reset_stability(self, block_id: int) -> None:
        # Do not count a cached image taken before the robot stopped.
        self._stable_counts[int(block_id)] = (self._latest_stamp(), 0, None)

    def stable_block(self, block_id: int, stable_observations: int = 1) -> Optional[DetectedTarget]:
        target = self._target_from_latest(block_id)
        stamp = self._latest_stamp()
        previous_stamp, count, previous_target = self._stable_counts.get(int(block_id), (None, 0, None))

        if target is None:
            if stamp != previous_stamp:
                self._stable_counts[int(block_id)] = (stamp, 0, None)
            return None

        if stamp != previous_stamp:
            count = count + 1 if previous_target is not None else 1
            self._stable_counts[int(block_id)] = (stamp, count, target)

        return target if count >= max(1, int(stable_observations)) else None

    def _latest_stamp(self) -> Optional[float]:
        if not self._latest_payload:
            return None
        try:
            return float(self._latest_payload.get("stamp"))
        except (TypeError, ValueError):
            return self._latest_payload_time

    def _target_from_latest(self, block_id: int) -> Optional[DetectedTarget]:
        if not self._latest_payload:
            return None

        best: Optional[DetectedTarget] = None
        for detection in self._latest_payload.get("detections", []):
            if not self._matches_block(detection, block_id):
                continue
            if detection.get("center_3d") is None:
                continue
            grasp_pose = detection.get("grasp_pose_tool0")
            if not grasp_pose:
                continue

            confidence = float(detection.get("confidence", 0.0))
            candidate = DetectedTarget(
                block_id=block_id,
                label=str(detection.get("label", block_id)),
                confidence=confidence,
                grasp_pose_tool0=pose_from_dict(grasp_pose, self.target_frame),
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate
        return best

    @staticmethod
    def _matches_block(detection: Dict, block_id: int) -> bool:
        label = str(detection.get("label", ""))
        if label == str(block_id):
            return True
        try:
            return int(detection.get("class_id")) == int(block_id)
        except (TypeError, ValueError):
            return False

    def _status_message(self, block_id: int) -> str:
        if not self._latest_payload:
            return "No messages received yet on %s" % self.topic

        detections = self._latest_payload.get("detections", [])
        debug = self._latest_payload.get("debug", {})
        age = "unknown"
        if self._latest_payload_time is not None:
            age = "%.1fs" % (time.monotonic() - self._latest_payload_time)
        if not detections:
            return (
                "Latest %s payload is %s old and contains no confirmed detections "
                "(contours=%s, candidates=%s, candidate_labels=%s, rejects=%s, camera_info_ready=%s, depth_ready=%s)"
                % (
                    self.topic,
                    age,
                    debug.get("contours", "?"),
                    debug.get("candidates", "?"),
                    debug.get("candidate_labels", []),
                    debug.get("rejects", []),
                    debug.get("camera_info_ready", "?"),
                    debug.get("depth_ready", "?"),
                )
            )

        labels = [str(item.get("label", "?")) for item in detections]
        matching = [item for item in detections if self._matches_block(item, block_id)]
        if not matching:
            return (
                "Latest %s payload is %s old; labels detected: [%s], but block %d is absent"
                % (self.topic, age, ", ".join(labels), block_id)
            )

        missing_reasons = []
        for item in matching:
            reasons = []
            if item.get("center_3d") is None:
                reasons.append("center_3d missing")
            if item.get("grasp_pose_tool0") is None:
                reasons.append("grasp_pose_tool0 missing")
            if item.get("center_camera_3d") is None:
                reasons.append("center_camera_3d missing")
            if reasons:
                missing_reasons.append("%s: %s" % (item.get("label", block_id), ", ".join(reasons)))
        if not missing_reasons:
            return "Latest %s payload has block %d but no valid target was selected" % (self.topic, block_id)
        return "Latest %s payload has block %d, but %s" % (self.topic, block_id, "; ".join(missing_reasons))


class SceneTargetResolver:
    def __init__(
        self,
        scene_config: str,
        base_frame: str,
        moveit_tip_frame: str,
        randomize_blocks: bool = False,
        random_seed: int = 0,
    ):
        self.scene_config = scene_config
        self.base_frame = base_frame
        self.moveit_tip_frame = moveit_tip_frame
        self.randomize_blocks = bool(randomize_blocks)
        self.scene = self._load_scene(scene_config)
        self.block_size = self._vector3("blocks", "size", [0.045, 0.045, 0.045])
        self.target_size = self._vector3("targets", "size", [0.090, 0.090, 0.012])
        self.target_board_size = self._vector3("target_board", "size", [0.35, 0.24, 0.015])
        self.blocks = self._block_map(bool(randomize_blocks), int(random_seed))
        self.targets = self._target_map()
        self.pose_converter = ToolPoseConverter()

    @staticmethod
    def _load_scene(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _vector3(self, section: str, key: str, default: List[float]) -> Tuple[float, float, float]:
        values = self.scene.get(section, {}).get(key, default)
        if len(values) != 3:
            raise ValueError("%s.%s must contain exactly 3 values" % (section, key))
        return float(values[0]), float(values[1]), float(values[2])

    def _target_map(self) -> Dict[int, Tuple[float, float, float]]:
        result: Dict[int, Tuple[float, float, float]] = {}
        for item in self.scene.get("targets", {}).get("items", []):
            if "id" not in item or "pose_xy_yaw" not in item:
                continue
            pose = item["pose_xy_yaw"]
            result[int(item["id"])] = (float(pose[0]), float(pose[1]), float(pose[2]))
        return result

    def _block_map(self, randomize_blocks: bool, random_seed: int) -> Dict[int, Tuple[float, float, float]]:
        items = self.scene.get("blocks", {}).get("items", [])
        if randomize_blocks:
            randomization = self.scene.get("randomization", {})
            poses = self._sample_random_poses(
                len(items),
                randomization.get("x_range", [0.45, 0.67]),
                randomization.get("y_range", [-0.20, 0.18]),
                randomization.get("yaw_range", [-np.pi, np.pi]),
                float(randomization.get("min_distance", 0.07)),
                int(randomization.get("max_attempts", 500)),
                random_seed,
            )
            return {int(item["id"]): pose for item, pose in zip(items, poses) if "id" in item}

        result: Dict[int, Tuple[float, float, float]] = {}
        for item in items:
            if "id" not in item or "pose_xy_yaw" not in item:
                continue
            pose = item["pose_xy_yaw"]
            result[int(item["id"])] = (float(pose[0]), float(pose[1]), float(pose[2]))
        return result

    @staticmethod
    def _sample_random_poses(count: int, x_range, y_range, yaw_range, min_distance: float, max_attempts: int, seed: int):
        rng = random.Random(seed)
        poses = []
        attempts = 0
        min_distance_sq = min_distance * min_distance
        while len(poses) < count and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(float(x_range[0]), float(x_range[1]))
            y = rng.uniform(float(y_range[0]), float(y_range[1]))
            yaw = rng.uniform(float(yaw_range[0]), float(yaw_range[1]))
            if all((x - px) ** 2 + (y - py) ** 2 >= min_distance_sq for px, py, _ in poses):
                poses.append((x, y, yaw))
        if len(poses) != count:
            raise RuntimeError("Could only sample %d non-overlapping poses for %d blocks" % (len(poses), count))
        return poses

    def block_search_bounds(self, margin: float = 0.04) -> Tuple[float, float, float, float]:
        if self.randomize_blocks:
            randomization = self.scene.get("randomization", {})
            x_range = randomization.get("x_range", [0.45, 0.67])
            y_range = randomization.get("y_range", [-0.20, 0.18])
            xmin, xmax = float(x_range[0]), float(x_range[1])
            ymin, ymax = float(y_range[0]), float(y_range[1])
        else:
            if not self.blocks:
                raise RuntimeError("No block poses are available for active search")
            xs = [pose[0] for pose in self.blocks.values()]
            ys = [pose[1] for pose in self.blocks.values()]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

        margin = max(0.0, float(margin))
        return xmin - margin, xmax + margin, ymin - margin, ymax + margin

    def narrow_opening_width(self, clearance: float) -> float:
        return float(self.block_size[0]) + float(clearance)

    def grasp_opening_width(self, compression: float) -> float:
        return max(0.0, float(self.block_size[0]) - float(compression))

    def block_grasp_pose_tool0(self, block_id: int, reference_orientation, tcp_center_z_offset: float = 0.0) -> PoseStamped:
        if block_id not in self.blocks:
            available = ", ".join(str(value) for value in sorted(self.blocks))
            raise ValueError("Block id %d is not configured. Available block ids: %s" % (block_id, available))

        x, y, _yaw = self.blocks[block_id]
        tcp_pose = PoseStamped()
        tcp_pose.header.frame_id = self.base_frame
        tcp_pose.header.stamp = rospy.Time.now()
        tcp_pose.pose.position.x = x
        tcp_pose.pose.position.y = y
        tcp_pose.pose.position.z = 0.5 * self.block_size[2] + float(tcp_center_z_offset)
        tcp_pose.pose.orientation = copy.deepcopy(reference_orientation)
        return self.pose_converter.tool0_pose_from_tcp(tcp_pose)

    def place_pose_tool0(self, target_id: int, reference_orientation) -> PoseStamped:
        if target_id not in self.targets:
            available = ", ".join(str(value) for value in sorted(self.targets))
            raise ValueError("Target id %d is not configured. Available target ids: %s" % (target_id, available))

        x, y, _yaw = self.targets[target_id]
        board_top_z = self.target_board_size[2]
        target_top_z = board_top_z + self.target_size[2]
        tcp_z = target_top_z + 0.5 * self.block_size[2]

        tcp_pose = PoseStamped()
        tcp_pose.header.frame_id = self.base_frame
        tcp_pose.header.stamp = rospy.Time.now()
        tcp_pose.pose.position.x = x
        tcp_pose.pose.position.y = y
        tcp_pose.pose.position.z = tcp_z
        tcp_pose.pose.orientation = copy.deepcopy(reference_orientation)
        return self.pose_converter.tool0_pose_from_tcp(tcp_pose)


class GripperCommander:
    def __init__(
        self,
        command_topic: str,
        max_gap: float,
        closed_position: float,
        command_repeats: int,
        settle_time: float,
    ):
        self.publisher = rospy.Publisher(command_topic, Float64, queue_size=1)
        self.max_gap = float(max_gap)
        self.closed_position = float(closed_position)
        self.command_repeats = max(1, int(command_repeats))
        self.settle_time = float(settle_time)
        self._joint = None
        self._contacts = {}
        rospy.Subscriber("/joint_states", JointState, self._joint_callback, queue_size=1)
        for side in ("left", "right"):
            rospy.Subscriber("/gripper/%s_contacts" % side, ContactsState,
                             self._contact_callback, callback_args=side, queue_size=1)

    def _joint_callback(self, msg) -> None:
        if "finger_joint" in msg.name:
            self._joint = (msg.position[msg.name.index("finger_joint")], time.monotonic())

    def _contact_callback(self, msg, side) -> None:
        self._contacts[side] = (msg.states, time.monotonic())

    def wait_for_grasp(self, block_id: int, timeout: float = 4.0) -> None:
        model = "digit_block_%d::" % block_id
        deadline = time.monotonic() + timeout
        stable_since = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            now = time.monotonic()
            both = all(side in self._contacts and now - self._contacts[side][1] < 0.2
                       and any(model in state.collision1_name or model in state.collision2_name
                               for state in self._contacts[side][0])
                       for side in ("left", "right"))
            if both:
                stable_since = now if stable_since is None else stable_since
                if now - stable_since >= 0.15:
                    return
            else:
                stable_since = None
            time.sleep(0.02)
        raise RuntimeError("grasp verification failed: both pads must contact block %d before attachment" % block_id)

    def position_for_width(self, opening: float) -> float:
        # From the inner-knuckle origins and pad collision mesh in this URDF.
        a, b, c = 0.03759940821, 0.04303959807, 0.0127 - 0.00780198
        angle = np.arccos(np.clip((opening / 2.0 - c) / np.hypot(a, b), -1.0, 1.0))
        return float(np.clip(angle - np.arctan2(b, a), 0.0, self.closed_position))

    def wait_for_controller(self, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.publisher.get_num_connections() > 0:
                return
            time.sleep(0.05)
        rospy.logwarn("No subscribers observed on gripper command topic; commands will still be published")

    def open_to_width(self, opening_width: float) -> None:
        opening = max(0.0, min(float(opening_width), self.max_gap))
        position = self.position_for_width(opening)
        rospy.loginfo("Opening gripper narrowly: width=%.3f m, joint=%.3f rad", opening, position)
        self._publish_position(position)
        deadline = time.monotonic() + 4.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self._joint and time.monotonic() - self._joint[1] < 0.5 and abs(self._joint[0] - position) < 0.025:
                return
            time.sleep(0.02)
        raise RuntimeError("gripper did not reach requested opening")

    def close_to_width(self, opening_width: float) -> None:
        opening = max(0.0, min(float(opening_width), self.max_gap))
        position = self.position_for_width(opening)
        rospy.loginfo("Closing gripper gently: width=%.3f m, joint=%.3f rad", opening, position)
        self._publish_position(position)

    def close(self) -> None:
        rospy.loginfo("Closing gripper: joint=%.3f rad", self.closed_position)
        self._publish_position(self.closed_position)

    def _publish_position(self, position: float) -> None:
        msg = Float64(data=max(0.0, min(self.closed_position, float(position))))
        for _ in range(self.command_repeats):
            if rospy.is_shutdown():
                return
            self.publisher.publish(msg)
            rospy.sleep(0.05)
        rospy.sleep(self.settle_time)


class GazeboGraspConstraint:
    def __init__(
        self,
        enabled: bool,
        reference_frame: str,
        gripper_frame: str,
        update_rate: float,
        service_timeout: float,
        tf_timeout: float,
    ):
        self.enabled = bool(enabled)
        self.reference_frame = reference_frame
        self.gripper_frame = gripper_frame
        self.update_rate = max(1.0, float(update_rate))
        self.tf_timeout = max(0.01, float(tf_timeout))
        self._lock = threading.Lock()
        self._attached_model: Optional[str] = None
        self._tcp_to_model: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available = False

        if not self.enabled:
            return

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        try:
            rospy.wait_for_service("/gazebo/get_model_state", timeout=float(service_timeout))
            rospy.wait_for_service("/gazebo/set_model_state", timeout=float(service_timeout))
        except rospy.ROSException as exc:
            rospy.logwarn("Gazebo grasp constraint disabled: %s", exc)
            return

        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self._available = True
        self._thread = threading.Thread(target=self._run, name="gazebo_grasp_constraint", daemon=True)
        self._thread.start()

    def attach(self, block_id: int) -> bool:
        if not self._available:
            return False

        model_name = "digit_block_%d" % int(block_id)
        try:
            tcp_matrix = self._tcp_matrix()
            response = self.get_model_state(model_name, self.reference_frame)
        except Exception as exc:
            rospy.logwarn("Could not attach %s to gripper: %s", model_name, exc)
            return False

        if not response.success:
            rospy.logwarn("Could not attach %s to gripper: %s", model_name, response.status_message)
            return False

        model_matrix = self._pose_matrix(response.pose)
        tcp_to_model = np.linalg.inv(tcp_matrix).dot(model_matrix)
        if np.any(np.abs(tcp_to_model[:3, 3]) > np.array([0.012, 0.008, 0.015])):
            raise RuntimeError("attachment rejected: block center is outside the gripper pads")
        with self._lock:
            self._attached_model = model_name
            self._tcp_to_model = tcp_to_model

        self._update_model_pose(model_name, tcp_to_model)
        rospy.loginfo("Attached %s to %s in Gazebo grasp constraint", model_name, self.gripper_frame)
        return True

    def compensated_place_pose(self, pose: PoseStamped) -> PoseStamped:
        with self._lock:
            relative = None if self._tcp_to_model is None else self._tcp_to_model.copy()
        if relative is None:
            return pose
        # place_pose specifies the desired block center. Preserve the measured
        # grasp offset so the carried block cannot hit the board prematurely.
        matrix = ToolPoseConverter._pose_to_matrix(pose)
        matrix[:3, 3] -= matrix[:3, :3].dot(relative[:3, 3])
        result = copy.deepcopy(pose)
        ToolPoseConverter._matrix_to_pose(matrix, result)
        return result

    def release(self) -> None:
        with self._lock:
            model_name = self._attached_model
            self._attached_model = None
            self._tcp_to_model = None
        if model_name:
            rospy.loginfo("Released %s from Gazebo grasp constraint", model_name)

    def stop(self) -> None:
        self.release()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        rate = rospy.Rate(self.update_rate)
        while not rospy.is_shutdown() and not self._stop.is_set():
            with self._lock:
                model_name = self._attached_model
                tcp_to_model = None if self._tcp_to_model is None else self._tcp_to_model.copy()
                if model_name and tcp_to_model is not None:
                    # release() waits for any in-flight update before opening.
                    self._update_model_pose(model_name, tcp_to_model)
            rate.sleep()

    def _update_model_pose(self, model_name: str, tcp_to_model: np.ndarray) -> None:
        try:
            tcp_matrix = self._tcp_matrix()
            model_matrix = tcp_matrix.dot(tcp_to_model)
            state = ModelState()
            state.model_name = model_name
            state.pose = self._matrix_pose(model_matrix)
            state.reference_frame = self.reference_frame
            self.set_model_state(state)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Could not update carried block pose: %s", exc)

    def _tcp_matrix(self) -> np.ndarray:
        transform = self.tf_buffer.lookup_transform(
            self.reference_frame,
            self.gripper_frame,
            rospy.Time(0),
            rospy.Duration(self.tf_timeout),
        )
        pose = PoseStamped()
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return ToolPoseConverter._pose_to_matrix(pose)

    @staticmethod
    def _pose_matrix(pose) -> np.ndarray:
        stamped = PoseStamped()
        stamped.pose = pose
        return ToolPoseConverter._pose_to_matrix(stamped)

    @staticmethod
    def _matrix_pose(matrix: np.ndarray):
        stamped = PoseStamped()
        ToolPoseConverter._matrix_to_pose(matrix, stamped)
        return stamped.pose


class MoveItGraspController:
    def __init__(
        self,
        planning_group: str,
        base_frame: str,
        moveit_tip_frame: str,
        cartesian_step: float,
        cartesian_fraction_threshold: float,
        velocity_scaling: float,
        acceleration_scaling: float,
        planning_time: float,
        planning_attempts: int,
        goal_position_tolerance: float,
        goal_orientation_tolerance: float,
        move_group_action: str,
        move_group_wait_timeout: float,
        auto_allow_accessory_collisions: bool,
        start_state_validity_timeout: float,
        allow_approximate_ik: bool,
        max_ik_joint_delta: float,
        max_ik_total_delta: float,
    ):
        self.base_frame = base_frame
        self.moveit_tip_frame = moveit_tip_frame
        self.velocity_scaling = float(velocity_scaling)
        self.acceleration_scaling = float(acceleration_scaling)
        self.cartesian_step = float(cartesian_step)
        self.cartesian_fraction_threshold = float(cartesian_fraction_threshold)
        self.auto_allow_accessory_collisions = bool(auto_allow_accessory_collisions)
        self.start_state_validity_timeout = float(start_state_validity_timeout)
        self.allow_approximate_ik = bool(allow_approximate_ik)
        self.max_ik_joint_delta = float(max_ik_joint_delta)
        self.max_ik_total_delta = float(max_ik_total_delta)
        self._allowed_contact_pairs = set()

        self._wait_for_move_group_action(move_group_action, move_group_wait_timeout)
        self.group = moveit_commander.MoveGroupCommander(
            planning_group,
            wait_for_servers=float(move_group_wait_timeout),
        )
        self.group.set_pose_reference_frame(base_frame)
        self.group.set_end_effector_link(moveit_tip_frame)
        self.group.set_max_velocity_scaling_factor(float(velocity_scaling))
        self.group.set_max_acceleration_scaling_factor(float(acceleration_scaling))
        self.group.set_planning_time(float(planning_time))
        self.group.set_num_planning_attempts(int(planning_attempts))
        self.group.set_goal_position_tolerance(float(goal_position_tolerance))
        self.group.set_goal_orientation_tolerance(float(goal_orientation_tolerance))

        self.home_joint_values = list(self.group.get_current_joint_values())
        self.state_validity = self._optional_service("/check_state_validity", GetStateValidity, 5.0)
        self.get_planning_scene = self._optional_service("/get_planning_scene", GetPlanningScene, 5.0)
        self.apply_planning_scene = self._optional_service("/apply_planning_scene", ApplyPlanningScene, 5.0)
        rospy.loginfo("Saved controller start joint state as return pose")

    @staticmethod
    def _wait_for_move_group_action(action_name: str, timeout: float) -> None:
        rospy.loginfo("Waiting up to %.1fs for MoveIt action server %s", timeout, action_name)
        client = actionlib.SimpleActionClient(action_name, MoveGroupAction)
        if client.wait_for_server(rospy.Duration(float(timeout))):
            return
        raise RuntimeError(
            "Unable to connect to MoveIt action server '%s' within %.1fs. "
            "Start the simulation with start_moveit:=true and check rostopic list | grep move_group."
            % (action_name, timeout)
        )

    @staticmethod
    def _optional_service(name: str, service_type, timeout: float):
        try:
            rospy.wait_for_service(name, timeout=float(timeout))
            return rospy.ServiceProxy(name, service_type)
        except Exception as exc:
            rospy.logwarn("MoveIt service %s is not available: %s", name, exc)
            return None

    def move_to_pose(self, stage: str, pose: PoseStamped) -> None:
        success, plan = self._plan_to_pose(stage, pose)
        if not success or not has_points(plan):
            raise RuntimeError("%s: MoveIt pose planning failed" % stage)
        self._execute(stage, plan)

    def move_to_pose_until(self, stage: str, pose: PoseStamped, should_stop, poll_rate: float = 10.0) -> bool:
        success, plan = self._plan_to_pose(stage, pose)
        if not success or not has_points(plan):
            raise RuntimeError("%s: MoveIt pose planning failed" % stage)
        return self._execute_until(stage, plan, should_stop, poll_rate)

    def current_tip_pose(self) -> PoseStamped:
        pose = self.group.get_current_pose(self.moveit_tip_frame)
        self._ensure_pose_frame(pose)
        return pose

    def move_cartesian(self, stage: str, target_pose: PoseStamped) -> None:
        self._ensure_pose_frame(target_pose)
        self._prepare_start_state(stage)
        self.group.set_start_state_to_current_state()
        waypoints = [copy.deepcopy(target_pose.pose)]
        rospy.loginfo("%s: computing Cartesian path", stage)
        plan, fraction = self._compute_cartesian_path(waypoints)
        if fraction < self.cartesian_fraction_threshold or not has_points(plan):
            raise RuntimeError(
                "%s: Cartesian path fraction %.3f below threshold %.3f"
                % (stage, fraction, self.cartesian_fraction_threshold)
            )
        plan = self.group.retime_trajectory(
            self.group.get_current_state(), plan,
            velocity_scaling_factor=min(self.velocity_scaling, 0.02),
            acceleration_scaling_factor=min(self.acceleration_scaling, 0.02),
            algorithm="iterative_time_parameterization")
        if not has_points(plan):
            raise RuntimeError("%s: Cartesian trajectory retiming failed" % stage)
        self._execute(stage, plan)

    def stop_motion(self) -> None:
        self.group.stop()
        self.group.clear_pose_targets()

    def _plan_to_pose(self, stage: str, pose: PoseStamped) -> Tuple[bool, RobotTrajectory]:
        self._ensure_pose_frame(pose)
        rospy.loginfo("%s: planning pose target in frame %s", stage, pose.header.frame_id)
        self._prepare_start_state(stage)
        success = False
        plan = RobotTrajectory()
        try:
            self.group.set_start_state_to_current_state()
            self.group.set_pose_target(pose, self.moveit_tip_frame)
            success, plan = self._plan()
        except Exception as exc:
            rospy.logwarn("%s: pose target setup failed; trying IK joint-target fallback: %s", stage, exc)
        finally:
            self.group.clear_pose_targets()
        if not success or not has_points(plan):
            rospy.logwarn("%s: pose target planning failed; trying IK joint-target fallback", stage)
            success, plan = self._plan_pose_as_joint_target(stage, pose)
        return success, plan

    def _compute_cartesian_path(self, waypoints):
        try:
            return self.group.compute_cartesian_path(waypoints, self.cartesian_step, True)
        except TypeError as noetic_exc:
            try:
                return self.group.compute_cartesian_path(waypoints, self.cartesian_step, 0.0, True)
            except TypeError:
                raise noetic_exc

    def return_home(self) -> None:
        rospy.loginfo("Returning to saved start joint state")
        self._prepare_start_state("return_home")
        self.group.set_start_state_to_current_state()
        try:
            self.group.set_joint_value_target(self.home_joint_values)
        except Exception as exc:
            bounded_home = self._normalize_joint_values_near_current(self.home_joint_values)
            rospy.logwarn(
                "return_home: saved joint target was rejected (%s); retrying with normalized target [%s]",
                exc,
                ", ".join("%.3f" % value for value in bounded_home),
            )
            self.group.set_joint_value_target(bounded_home)
        success, plan = self._plan()
        if not success or not has_points(plan):
            raise RuntimeError("return_home: MoveIt joint planning failed")
        self._execute("return_home", plan)

    def _plan(self) -> Tuple[bool, RobotTrajectory]:
        result = self.group.plan()
        if isinstance(result, tuple):
            if len(result) >= 2 and isinstance(result[0], bool):
                return bool(result[0]), result[1]
            if len(result) >= 1:
                plan = result[0]
                return has_points(plan), plan
        return has_points(result), result

    def _plan_pose_as_joint_target(self, stage: str, pose: PoseStamped) -> Tuple[bool, RobotTrajectory]:
        attempts = [False]
        if self.allow_approximate_ik:
            attempts.append(True)

        for approximate in attempts:
            self.group.clear_pose_targets()
            self.group.set_start_state_to_current_state()
            try:
                ik_ok = self.group.set_joint_value_target(pose, self.moveit_tip_frame, approximate)
            except Exception as exc:
                rospy.logwarn(
                    "%s: %s IK joint target setup failed: %s",
                    stage,
                    "approximate" if approximate else "exact",
                    exc,
                )
                continue
            if ik_ok is False:
                rospy.logwarn(
                    "%s: %s IK did not return a joint target",
                    stage,
                    "approximate" if approximate else "exact",
                    )
                continue

            target_joints = list(self.group.get_joint_value_target())
            if not self._joint_target_is_near_current(stage, target_joints, approximate):
                continue

            rospy.loginfo(
                "%s: planning with %s IK joint target",
                stage,
                "approximate" if approximate else "exact",
            )
            success, plan = self._plan()
            if success and has_points(plan):
                return success, plan
            rospy.logwarn(
                "%s: %s IK joint-target planning failed",
                stage,
                "approximate" if approximate else "exact",
            )

        return False, RobotTrajectory()

    def _joint_target_is_near_current(self, stage: str, target_joints: List[float], approximate: bool) -> bool:
        current_joints = list(self.group.get_current_joint_values())
        deltas = [
            self._shortest_angular_distance(float(current), float(target))
            for current, target in zip(current_joints, target_joints)
        ]
        abs_deltas = [abs(delta) for delta in deltas]
        max_delta = max(abs_deltas) if abs_deltas else 0.0
        total_delta = sum(abs_deltas)
        rospy.loginfo(
            "%s: %s IK joint deltas: [%s], max=%.3f, sum=%.3f",
            stage,
            "approximate" if approximate else "exact",
            ", ".join("%.3f" % delta for delta in deltas),
            max_delta,
            total_delta,
        )
        if max_delta > self.max_ik_joint_delta or total_delta > self.max_ik_total_delta:
            rospy.logwarn(
                "%s: rejecting %s IK target because it is too far from the current branch "
                "(max %.3f > %.3f or sum %.3f > %.3f)",
                stage,
                "approximate" if approximate else "exact",
                max_delta,
                self.max_ik_joint_delta,
                total_delta,
                self.max_ik_total_delta,
            )
            self.group.clear_pose_targets()
            return False
        return True

    def _normalize_joint_values_near_current(self, values: Sequence[float]) -> List[float]:
        current_joints = list(self.group.get_current_joint_values())
        normalized = []
        for index, value in enumerate(values):
            current = current_joints[index] if index < len(current_joints) else 0.0
            normalized.append(float(current) + self._shortest_angular_distance(float(current), float(value)))
        return normalized

    @staticmethod
    def _shortest_angular_distance(current: float, target: float) -> float:
        return float(np.arctan2(np.sin(target - current), np.cos(target - current)))

    def _execute(self, stage: str, plan: RobotTrajectory) -> None:
        rospy.loginfo("%s: executing trajectory with %d points", stage, len(plan.joint_trajectory.points))
        executed = self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        if executed is False:
            raise RuntimeError("%s: MoveIt execution failed" % stage)

    def _execute_until(self, stage: str, plan: RobotTrajectory, should_stop, poll_rate: float) -> bool:
        rospy.loginfo("%s: executing monitored trajectory with %d points", stage, len(plan.joint_trajectory.points))
        if should_stop():
            rospy.loginfo("%s: stop condition already satisfied before execution", stage)
            return True
        executed = self.group.execute(plan, wait=False)
        if executed is False:
            self.group.stop()
            self.group.clear_pose_targets()
            raise RuntimeError("%s: MoveIt async execution failed" % stage)

        deadline = time.monotonic() + max(0.5, self._trajectory_duration(plan) + 1.0)
        rate = rospy.Rate(max(1.0, float(poll_rate)))
        stopped_for_target = False
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if should_stop():
                stopped_for_target = True
                rospy.loginfo("%s: stop condition satisfied during execution", stage)
                break
            rate.sleep()

        self.group.stop()
        self.group.clear_pose_targets()
        return stopped_for_target

    @staticmethod
    def _trajectory_duration(plan: RobotTrajectory) -> float:
        if not has_points(plan):
            return 0.0
        return float(plan.joint_trajectory.points[-1].time_from_start.to_sec())

    def _prepare_start_state(self, stage: str) -> None:
        deadline = time.monotonic() + max(0.0, self.start_state_validity_timeout)
        last_contacts: List[Tuple[str, str]] = []

        while not rospy.is_shutdown():
            contacts = self._current_state_contacts()
            if not contacts:
                return

            last_contacts = contacts
            summary = ", ".join("%s-%s" % pair for pair in contacts[:12])
            rospy.logwarn_throttle(1.0, "%s: current start state is in collision: %s", stage, summary)

            if self.auto_allow_accessory_collisions:
                safe_pairs = [pair for pair in contacts if self._is_safe_start_state_pair(pair)]
                new_pairs = [pair for pair in safe_pairs if pair not in self._allowed_contact_pairs]
                if new_pairs and self._allow_collision_pairs(new_pairs):
                    self._allowed_contact_pairs.update(new_pairs)
                    rospy.logwarn(
                        "%s: allowed mounted-tool collision pairs: %s",
                        stage,
                        ", ".join("%s-%s" % pair for pair in new_pairs),
                    )
                    continue

            if time.monotonic() >= deadline:
                break
            rospy.sleep(0.1)

        if last_contacts:
            rospy.logerr(
                "%s: start state is still in collision; planning will probably fail. Contacts: %s",
                stage,
                ", ".join("%s-%s" % pair for pair in last_contacts[:20]),
            )

    @staticmethod
    def _is_safe_start_state_pair(pair: Tuple[str, str]) -> bool:
        link_a, link_b = pair
        if link_a in ACCESSORY_LINKS and link_b in ACCESSORY_LINKS:
            return True
        return (
            (link_a in ACCESSORY_LINKS and link_b in MOUNT_ADJACENT_LINKS)
            or (link_b in ACCESSORY_LINKS and link_a in MOUNT_ADJACENT_LINKS)
        )


    def _current_state_contacts(self) -> List[Tuple[str, str]]:
        if self.state_validity is None:
            return []
        try:
            response = self.state_validity(self.group.get_current_state(), self.group.get_name(), Constraints())
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Could not query /check_state_validity: %s", exc)
            return []
        if response.valid:
            return []
        pairs = []
        seen = set()
        for contact in response.contacts:
            pair = tuple(sorted((contact.contact_body_1, contact.contact_body_2)))
            if pair in seen:
                continue
            seen.add(pair)
            if contact.body_type_1 != contact.ROBOT_LINK or contact.body_type_2 != contact.ROBOT_LINK:
                rospy.logwarn(
                    "Current state contact involves non-robot body: %s(type=%d)-%s(type=%d)",
                    contact.contact_body_1,
                    contact.body_type_1,
                    contact.contact_body_2,
                    contact.body_type_2,
                )
            pairs.append(pair)
        return pairs

    def _allow_collision_pairs(self, pairs: List[Tuple[str, str]]) -> bool:
        if self.get_planning_scene is None or self.apply_planning_scene is None:
            return False
        try:
            components = PlanningSceneComponents(components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
            scene = self.get_planning_scene(components).scene
            acm = scene.allowed_collision_matrix
            self._ensure_acm_entries(acm, pairs)
            diff = PlanningScene(is_diff=True)
            diff.allowed_collision_matrix = acm
            return bool(self.apply_planning_scene(diff).success)
        except Exception as exc:
            rospy.logwarn("Could not update allowed collision matrix: %s", exc)
            return False

    @staticmethod
    def _ensure_acm_entries(acm, pairs: List[Tuple[str, str]]) -> None:
        for link_a, link_b in pairs:
            MoveItGraspController._ensure_acm_link(acm, link_a)
            MoveItGraspController._ensure_acm_link(acm, link_b)
            index_a = acm.entry_names.index(link_a)
            index_b = acm.entry_names.index(link_b)
            acm.entry_values[index_a].enabled[index_b] = True
            acm.entry_values[index_b].enabled[index_a] = True

    @staticmethod
    def _ensure_acm_link(acm, link: str) -> None:
        if link in acm.entry_names:
            return
        acm.entry_names.append(link)
        for entry in acm.entry_values:
            entry.enabled.append(False)
        acm.entry_values.append(AllowedCollisionEntry(enabled=[False] * len(acm.entry_names)))

    def _ensure_pose_frame(self, pose: PoseStamped) -> None:
        if not pose.header.frame_id:
            pose.header.frame_id = self.base_frame
        pose.header.stamp = rospy.Time.now()
        if pose.header.frame_id != self.base_frame:
            rospy.logwarn_once(
                "Controller expected poses in %s, got %s. MoveIt may need TF to transform it.",
                self.base_frame,
                pose.header.frame_id,
            )


class VisualServoGraspNode:
    def __init__(self):
        package_path = rospkg.RosPack().get_path("visual_servo_grasp_sim")
        default_scene_config = os.path.join(package_path, "config", "scene.yaml")

        self.interactive = self._default_interactive()
        self.block_id = int(rospy.get_param("~block_id", 1))
        self.target_id = int(rospy.get_param("~target_id", self.block_id))
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.moveit_tip_frame = rospy.get_param("~moveit_tip_frame", "tool0")
        self.pregrasp_lift = float(rospy.get_param("~pregrasp_lift", 0.10))
        self.place_lift = float(rospy.get_param("~place_lift", 0.10))
        self.detection_timeout = float(rospy.get_param("~detection_timeout", 30.0))
        self.detection_stable_observations = int(rospy.get_param("~detection_stable_observations", 3))
        self.active_search_enabled = self._parse_bool(rospy.get_param("~active_search_enabled", True))
        self.initial_detection_timeout = float(rospy.get_param("~initial_detection_timeout", 3.0))
        self.search_margin = float(rospy.get_param("~search_margin", 0.04))
        self.search_step_x = float(rospy.get_param("~search_step_x", 0.10))
        self.search_step_y = float(rospy.get_param("~search_step_y", 0.10))
        self.search_dwell_time = float(rospy.get_param("~search_dwell_time", 0.8))
        self.search_max_passes = int(rospy.get_param("~search_max_passes", 2))
        self.search_poll_rate = float(rospy.get_param("~search_poll_rate", 10.0))
        self.search_tool0_z = float(rospy.get_param("~search_tool0_z", "nan"))
        self.search_camera_frame = rospy.get_param("~search_camera_frame", "wrist_camera_optical_frame")
        self.gripper_clearance = float(rospy.get_param("~gripper_clearance", 0.020))
        self.gripper_grasp_compression = float(rospy.get_param("~gripper_grasp_compression", 0.004))
        self.use_scene_block_pose = self._parse_bool(rospy.get_param("~use_scene_block_pose", False))
        self.grasp_tcp_center_z_offset = float(rospy.get_param("~grasp_tcp_center_z_offset", 0.0))
        self.grasp_yaw_offsets = self._float_sequence_param(
            rospy.get_param("~grasp_yaw_offsets", [0.0, 1.57079632679, -1.57079632679, 3.14159265359])
        )

        self.scene = SceneTargetResolver(
            scene_config=rospy.get_param("~scene_config", default_scene_config),
            base_frame=self.base_frame,
            moveit_tip_frame=self.moveit_tip_frame,
            randomize_blocks=self._parse_bool(rospy.get_param("~randomize_blocks", False)),
            random_seed=int(rospy.get_param("~random_seed", 0)),
        )
        self.detections = DetectionBuffer(
            topic=rospy.get_param("~detections_topic", "/digit_detections"),
            target_frame=self.base_frame,
        )
        self.gripper = GripperCommander(
            command_topic=rospy.get_param("~gripper_command_topic", "/gripper_controller/command"),
            max_gap=float(rospy.get_param("~gripper_max_gap", 0.085)),
            closed_position=float(rospy.get_param("~closed_gripper_position", 0.78)),
            command_repeats=int(rospy.get_param("~gripper_command_repeats", 5)),
            settle_time=float(rospy.get_param("~gripper_settle_time", 0.5)),
        )
        self.grasp_constraint = GazeboGraspConstraint(
            enabled=self._parse_bool(rospy.get_param("~use_gazebo_grasp_constraint", True)),
            reference_frame=rospy.get_param("~gazebo_grasp_reference_frame", "world"),
            gripper_frame=rospy.get_param("~gazebo_grasp_gripper_frame", "gripper_tcp"),
            update_rate=float(rospy.get_param("~gazebo_grasp_update_rate", 50.0)),
            service_timeout=float(rospy.get_param("~gazebo_grasp_service_timeout", 5.0)),
            tf_timeout=float(rospy.get_param("~gazebo_grasp_tf_timeout", 0.2)),
        )
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.motion: Optional[MoveItGraspController] = None

    def run(self) -> None:
        try:
            tasks = self._tasks()
            if not tasks:
                rospy.loginfo("No pick-place tasks were provided")
                return

            rospy.loginfo("Queued %d pick-place task(s): %s", len(tasks), self._task_summary(tasks))
            self._wait_for_initial_pose()
            self.motion = self._make_motion_controller()
            self.gripper.wait_for_controller(timeout=5.0)
            narrow_width = self.scene.narrow_opening_width(self.gripper_clearance)
            grasp_width = self.scene.grasp_opening_width(self.gripper_grasp_compression)

            for index, task in enumerate(tasks, start=1):
                if rospy.is_shutdown():
                    return
                rospy.loginfo(
                    "Starting task %d/%d: block %d -> target cell %d",
                    index,
                    len(tasks),
                    task.block_id,
                    task.target_id,
                )
                self._execute_task(task, narrow_width, grasp_width)
            rospy.loginfo("All queued visual servo grasp tasks completed")
        finally:
            self.grasp_constraint.stop()

    def _execute_task(self, task: PickPlaceTask, narrow_width: float, grasp_width: float) -> None:
        rospy.loginfo("Locating block %d detection", task.block_id)
        self._locate_target(task.block_id)
        # Search can stop mid-trajectory; reacquire after the camera has settled.
        rospy.sleep(0.4)
        target = self.detections.wait_for_stable_block(
            task.block_id, self.detection_timeout,
            stable_observations=max(3, self.detection_stable_observations))
        rospy.loginfo(
            "Selected block %s with confidence %.3f; narrow opening is %.3f m; grasp opening is %.3f m",
            target.label,
            target.confidence,
            narrow_width,
            grasp_width,
        )

        if self.motion is None:
            raise RuntimeError("MoveIt controller is not initialized")
        self.gripper.open_to_width(narrow_width)

        grasp_pose_tool0 = self._calibrated_grasp_pose(task.block_id, target.grasp_pose_tool0)
        grasp_pose = self._move_to_reachable_pregrasp(grasp_pose_tool0)
        lift_pose = lifted_pose(grasp_pose, self.pregrasp_lift)
        place_pose = self.scene.place_pose_tool0(task.target_id, grasp_pose.pose.orientation)

        self.motion.move_cartesian("descend_to_grasp", grasp_pose)
        self.gripper.close_to_width(grasp_width)
        self.gripper.wait_for_grasp(task.block_id)
        attached = self.grasp_constraint.attach(task.block_id)
        place_pose = self.grasp_constraint.compensated_place_pose(place_pose)
        place_pose = lifted_pose(place_pose, 0.003)
        place_approach_pose = lifted_pose(place_pose, self.place_lift)
        try:
            self.motion.move_cartesian("lift_object", lift_pose)
            self.motion.move_to_pose("move_to_place_approach", place_approach_pose)
            self.motion.move_cartesian("descend_to_place", place_pose)
            self.grasp_constraint.release()
            attached = False
            self.gripper.open_to_width(narrow_width)
        finally:
            if attached:
                self.grasp_constraint.release()
        self.motion.move_cartesian("lift_after_place", place_approach_pose)
        self.motion.return_home()
        rospy.loginfo("Task completed: block %d -> target cell %d", task.block_id, task.target_id)

    def _locate_target(self, block_id: int) -> DetectedTarget:
        stable_observations = max(1, int(self.detection_stable_observations))
        if not self.active_search_enabled:
            rospy.loginfo(
                "Waiting up to %.1fs for block %d with %d stable detection observation(s)",
                self.detection_timeout,
                block_id,
                stable_observations,
            )
            return self.detections.wait_for_stable_block(
                block_id,
                self.detection_timeout,
                stable_observations=stable_observations,
                poll_rate=self.search_poll_rate,
            )

        initial_timeout = max(0.0, min(self.initial_detection_timeout, self.detection_timeout))
        if initial_timeout > 0.0:
            try:
                rospy.loginfo(
                    "Checking current view for block %d for %.1fs before active search",
                    block_id,
                    initial_timeout,
                )
                return self.detections.wait_for_stable_block(
                    block_id,
                    initial_timeout,
                    stable_observations=stable_observations,
                    poll_rate=self.search_poll_rate,
                )
            except RuntimeError:
                rospy.logwarn("Block %d is not stable in the current view; starting active search", block_id)

        return self._active_search_for_target(block_id, stable_observations)

    def _active_search_for_target(self, block_id: int, stable_observations: int) -> DetectedTarget:
        if self.motion is None:
            raise RuntimeError("MoveIt controller is not initialized")

        template_pose = self.motion.current_tip_pose()
        offset_x, offset_y = self._tool0_camera_xy_offset(template_pose)
        points = self._search_camera_points()
        bounds = self.scene.block_search_bounds(self.search_margin)
        rospy.loginfo(
            "Active search for block %d: bounds x=[%.3f, %.3f], y=[%.3f, %.3f], %d waypoint(s), stable=%d",
            block_id,
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            len(points),
            stable_observations,
        )

        self.detections.reset_stability(block_id)
        for pass_index in range(max(1, self.search_max_passes)):
            search_points = points if pass_index % 2 == 0 else list(reversed(points))
            rospy.loginfo("Active search pass %d/%d for block %d", pass_index + 1, self.search_max_passes, block_id)
            for point_index, (camera_x, camera_y) in enumerate(search_points, start=1):
                target = self.detections.stable_block(block_id, stable_observations)
                if target is not None:
                    rospy.loginfo("Block %d became stable before moving to next search waypoint", block_id)
                    return target

                search_pose = self._search_pose_from_camera_xy(template_pose, camera_x, camera_y, offset_x, offset_y)
                stage = "search_block_%d_%02d_%02d" % (block_id, pass_index + 1, point_index)
                stopped = self.motion.move_to_pose_until(
                    stage,
                    search_pose,
                    lambda: self.detections.stable_block(block_id, stable_observations) is not None,
                    poll_rate=self.search_poll_rate,
                )
                target = self.detections.stable_block(block_id, stable_observations)
                if target is not None:
                    rospy.loginfo(
                        "Active search stopped for block %d at waypoint %d/%d",
                        block_id,
                        point_index,
                        len(search_points),
                    )
                    return target
                if stopped:
                    rospy.logwarn("Search stop condition fired but stable target was not available after stop")

                if self.search_dwell_time <= 0.0:
                    continue
                try:
                    return self.detections.wait_for_stable_block(
                        block_id,
                        self.search_dwell_time,
                        stable_observations=stable_observations,
                        poll_rate=self.search_poll_rate,
                    )
                except RuntimeError:
                    pass

        raise RuntimeError(
            "Active search failed to find block %d after %d pass(es) over the configured block area"
            % (block_id, max(1, self.search_max_passes))
        )

    def _search_camera_points(self) -> List[Tuple[float, float]]:
        xmin, xmax, ymin, ymax = self.scene.block_search_bounds(self.search_margin)
        xs = self._axis_points(xmin, xmax, self.search_step_x)
        ys = self._axis_points(ymin, ymax, self.search_step_y)
        points: List[Tuple[float, float]] = []
        for row_index, y in enumerate(ys):
            row = xs if row_index % 2 == 0 else list(reversed(xs))
            points.extend((x, y) for x in row)
        center = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)
        return [center] + [point for point in points if point != center]

    @staticmethod
    def _axis_points(minimum: float, maximum: float, step: float) -> List[float]:
        minimum = float(minimum)
        maximum = float(maximum)
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        step = max(0.01, abs(float(step)))
        count = max(2, int(np.ceil((maximum - minimum) / step)) + 1)
        return [float(value) for value in np.linspace(minimum, maximum, count)]

    def _search_pose_from_camera_xy(
        self,
        template_pose: PoseStamped,
        camera_x: float,
        camera_y: float,
        offset_x: float,
        offset_y: float,
    ) -> PoseStamped:
        pose = copy.deepcopy(template_pose)
        pose.header.frame_id = self.base_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = float(camera_x) + float(offset_x)
        pose.pose.position.y = float(camera_y) + float(offset_y)
        if np.isfinite(self.search_tool0_z):
            pose.pose.position.z = self.search_tool0_z
        return pose

    def _tool0_camera_xy_offset(self, tool0_pose: PoseStamped) -> Tuple[float, float]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.search_camera_frame,
                rospy.Time(0),
                rospy.Duration(0.5),
            )
            camera_x = float(transform.transform.translation.x)
            camera_y = float(transform.transform.translation.y)
            return (
                float(tool0_pose.pose.position.x) - camera_x,
                float(tool0_pose.pose.position.y) - camera_y,
            )
        except Exception as exc:
            rospy.logwarn(
                "Could not compute %s -> %s XY offset for active search (%s); using tool0 XY directly",
                self.moveit_tip_frame,
                self.search_camera_frame,
                exc,
            )
            return 0.0, 0.0

    def _calibrated_grasp_pose(self, block_id: int, detected_pose: PoseStamped) -> PoseStamped:
        if not self.use_scene_block_pose:
            return detected_pose

        scene_pose = self.scene.block_grasp_pose_tool0(
            block_id,
            detected_pose.pose.orientation,
            tcp_center_z_offset=self.grasp_tcp_center_z_offset,
        )
        rospy.loginfo(
            "Using scene-calibrated grasp pose for block %d: detected xyz=(%.3f, %.3f, %.3f), "
            "calibrated xyz=(%.3f, %.3f, %.3f)",
            block_id,
            detected_pose.pose.position.x,
            detected_pose.pose.position.y,
            detected_pose.pose.position.z,
            scene_pose.pose.position.x,
            scene_pose.pose.position.y,
            scene_pose.pose.position.z,
        )
        return scene_pose

    def _move_to_reachable_pregrasp(self, base_grasp_pose: PoseStamped) -> PoseStamped:
        if self.motion is None:
            raise RuntimeError("MoveIt controller is not initialized")

        last_error: Optional[RuntimeError] = None
        for yaw_offset in self.grasp_yaw_offsets:
            grasp_pose = rotated_pose_about_base_z(base_grasp_pose, yaw_offset)
            pregrasp_pose = lifted_pose(grasp_pose, self.pregrasp_lift)
            rospy.loginfo(
                "Trying pregrasp yaw offset %.3f rad at xyz=(%.3f, %.3f, %.3f)",
                yaw_offset,
                pregrasp_pose.pose.position.x,
                pregrasp_pose.pose.position.y,
                pregrasp_pose.pose.position.z,
            )
            try:
                self.motion.move_to_pose("move_to_pregrasp", pregrasp_pose)
                # Pose planning (especially approximate IK) may leave position
                # and yaw error. Correct it above the block, before descending.
                self.motion.move_cartesian("align_pregrasp", pregrasp_pose)
                if abs(float(yaw_offset)) > 1e-9:
                    rospy.logwarn("Using alternate pregrasp yaw offset %.3f rad", yaw_offset)
                return grasp_pose
            except RuntimeError as exc:
                last_error = exc
                if "MoveIt pose planning failed" not in str(exc):
                    raise
                rospy.logwarn("Pregrasp yaw offset %.3f rad failed: %s", yaw_offset, exc)

        raise RuntimeError("move_to_pregrasp: all yaw-offset pose planning attempts failed: %s" % last_error)

    def _tasks(self) -> List[PickPlaceTask]:
        if self.interactive:
            tasks = InteractiveTaskCollector().collect()
            if not tasks:
                return []
            return tasks
        return [PickPlaceTask(block_id=self.block_id, target_id=self.target_id)]

    def _make_motion_controller(self) -> MoveItGraspController:
        return MoveItGraspController(
            planning_group=rospy.get_param("~planning_group", "manipulator"),
            base_frame=self.base_frame,
            moveit_tip_frame=self.moveit_tip_frame,
            cartesian_step=float(rospy.get_param("~cartesian_step", 0.002)),
            cartesian_fraction_threshold=float(rospy.get_param("~cartesian_fraction_threshold", 0.999)),
            velocity_scaling=float(rospy.get_param("~velocity_scaling", 0.10)),
            acceleration_scaling=float(rospy.get_param("~acceleration_scaling", 0.10)),
            planning_time=float(rospy.get_param("~planning_time", 10.0)),
            planning_attempts=int(rospy.get_param("~planning_attempts", 10)),
            goal_position_tolerance=float(rospy.get_param("~goal_position_tolerance", 0.001)),
            goal_orientation_tolerance=float(rospy.get_param("~goal_orientation_tolerance", 0.01)),
            move_group_action=rospy.get_param("~move_group_action", "move_group"),
            move_group_wait_timeout=float(rospy.get_param("~move_group_wait_timeout", 60.0)),
            auto_allow_accessory_collisions=self._parse_bool(
                rospy.get_param("~auto_allow_accessory_collisions", True)
            ),
            start_state_validity_timeout=float(rospy.get_param("~start_state_validity_timeout", 2.0)),
            allow_approximate_ik=self._parse_bool(rospy.get_param("~allow_approximate_ik", True)),
            max_ik_joint_delta=float(rospy.get_param("~max_ik_joint_delta", 1.8)),
            max_ik_total_delta=float(rospy.get_param("~max_ik_total_delta", 4.5)),
        )

    def _wait_for_initial_pose(self) -> None:
        if not self._parse_bool(rospy.get_param("~wait_for_initial_pose", False)):
            return

        ready_param = rospy.get_param("~initial_pose_ready_param", "/visual_servo_grasp_sim/initial_pose_ready")
        timeout = float(rospy.get_param("~initial_pose_wait_timeout", 150.0))
        deadline = time.monotonic() + max(0.0, timeout)
        rospy.loginfo("Waiting for initial top-down hover pose before initializing MoveIt")
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self._parse_bool(rospy.get_param(ready_param, False)):
                rospy.loginfo("Initial top-down hover pose is ready")
                return
            time.sleep(0.1)
        raise RuntimeError("Timed out waiting for initial top-down hover pose readiness at %s" % ready_param)

    def _default_interactive(self) -> bool:
        default = bool(sys.stdin.isatty())
        return self._parse_bool(rospy.get_param("~interactive", default))

    @staticmethod
    def _parse_bool(value) -> bool:
        return str(value).lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _float_sequence_param(value) -> Sequence[float]:
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return [float(part) for part in parts]
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        return [float(value)]

    @staticmethod
    def _task_summary(tasks: List[PickPlaceTask]) -> str:
        return ", ".join("%d->%d" % (task.block_id, task.target_id) for task in tasks)


def main() -> int:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("visual_servo_grasp_controller")
    try:
        VisualServoGraspNode().run()
        return 0
    except Exception as exc:
        rospy.logerr("Visual servo grasp controller failed: %s", exc)
        return 1
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    sys.exit(main())
