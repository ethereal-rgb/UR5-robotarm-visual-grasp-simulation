#!/usr/bin/env python3
import os
import random
import time

import numpy as np
import rospy
import yaml
from sensor_msgs.msg import JointState
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import euler_matrix, rotation_matrix, translation_matrix
from urdf_parser_py.urdf import URDF


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

TOPDOWN_HOVER = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]


def parse_bool(value) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def build_trajectory(positions) -> JointTrajectory:
    trajectory = JointTrajectory()
    trajectory.joint_names = JOINT_NAMES
    point = JointTrajectoryPoint()
    point.positions = [float(value) for value in positions]
    point.velocities = [0.0] * len(JOINT_NAMES)
    point.time_from_start = rospy.Duration(2.0)
    trajectory.points.append(point)
    return trajectory


def wait_for_robot_description(timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        description = rospy.get_param("/robot_description", "")
        if description:
            return description
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for /robot_description")


def block_center_from_scene(scene_config: str, randomize_blocks: bool, random_seed: int) -> np.ndarray:
    with open(scene_config, "r", encoding="utf-8") as handle:
        scene = yaml.safe_load(handle) or {}
    blocks = scene.get("blocks", {})
    items = blocks.get("items", [{"id": i} for i in range(1, 10)])

    if randomize_blocks:
        randomization = scene.get("randomization", {})
        poses = sample_random_poses(
            len(items),
            randomization.get("x_range", [0.45, 0.67]),
            randomization.get("y_range", [-0.20, 0.18]),
            randomization.get("yaw_range", [-np.pi, np.pi]),
            float(randomization.get("min_distance", 0.07)),
            int(randomization.get("max_attempts", 500)),
            random_seed,
        )
    else:
        poses = []
        for item in items:
            pose = item.get("pose_xy_yaw")
            if pose is None:
                raise ValueError(f"Block {item.get('id')} has no pose_xy_yaw and randomization is disabled")
            poses.append(pose)

    if not poses:
        raise ValueError("No block poses are configured")
    xy = np.array([[pose[0], pose[1]] for pose in poses], dtype=float)
    return xy.mean(axis=0)


def sample_random_poses(count: int, x_range, y_range, yaw_range, min_distance: float, max_attempts: int, seed: int):
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
        raise RuntimeError(f"Could only sample {len(poses)} non-overlapping poses for {count} blocks")
    return poses


def origin_matrix(origin) -> np.ndarray:
    if origin is None:
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
    else:
        xyz = origin.xyz or [0.0, 0.0, 0.0]
        rpy = origin.rpy or [0.0, 0.0, 0.0]
    return translation_matrix(xyz).dot(euler_matrix(*rpy, axes="sxyz"))


class ChainKinematics:
    def __init__(self, robot_description: str, root_link: str, tip_link: str):
        robot = URDF.from_xml_string(robot_description)
        joints_by_child = {joint.child: joint for joint in robot.joints}
        chain = []
        current = tip_link
        while current != root_link:
            joint = joints_by_child.get(current)
            if joint is None:
                raise ValueError(f"Could not build chain from {root_link} to {tip_link}")
            chain.append(joint)
            current = joint.parent
        self.chain = list(reversed(chain))

    def fk(self, positions) -> np.ndarray:
        joint_positions = dict(zip(JOINT_NAMES, positions))
        transform = np.eye(4)
        for joint in self.chain:
            transform = transform.dot(origin_matrix(joint.origin))
            if joint.type in ("revolute", "continuous"):
                axis = np.array(joint.axis or [1.0, 0.0, 0.0], dtype=float)
                transform = transform.dot(rotation_matrix(joint_positions.get(joint.name, 0.0), axis))
        return transform


def solve_topdown_hover(robot_description: str, block_center_xy: np.ndarray) -> np.ndarray:
    kinematics = ChainKinematics(robot_description, "base_link", "wrist_camera_optical_frame")
    seed = np.array(TOPDOWN_HOVER, dtype=float)
    seed_pose = kinematics.fk(seed)
    target_pose = seed_pose.copy()
    target_pose[0, 3] = block_center_xy[0]
    target_pose[1, 3] = block_center_xy[1]
    target_rotation = Rotation.from_matrix(target_pose[:3, :3])

    def residual(positions):
        pose = kinematics.fk(positions)
        position_error = pose[:3, 3] - target_pose[:3, 3]
        rotation_error = (target_rotation.inv() * Rotation.from_matrix(pose[:3, :3])).as_rotvec()
        return np.r_[position_error * 8.0, rotation_error * 2.0, (positions - seed) * 0.03]

    best = None
    starts = [seed]
    for shoulder_offset in np.linspace(-1.5, 1.5, 13):
        start = seed.copy()
        start[0] += shoulder_offset
        starts.append(start)

    for start in starts:
        result = least_squares(
            residual,
            start,
            bounds=(-2.0 * np.pi, 2.0 * np.pi),
            max_nfev=2000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        pose_error = np.linalg.norm(residual(result.x)[:6])
        closeness = np.linalg.norm(result.x - seed)
        key = (pose_error, closeness)
        if best is None or key < best[0]:
            best = (key, result)

    solution = best[1].x
    final_pose = kinematics.fk(solution)
    position_error = final_pose[:3, 3] - target_pose[:3, 3]
    rotation_error = (target_rotation.inv() * Rotation.from_matrix(final_pose[:3, :3])).as_rotvec()
    rospy.loginfo(
        "Solved initial hover: block center=(%.3f, %.3f), camera error xyz=(%.6f, %.6f, %.6f), "
        "orientation error norm=%.6f rad",
        block_center_xy[0],
        block_center_xy[1],
        position_error[0],
        position_error[1],
        position_error[2],
        np.linalg.norm(rotation_error),
    )
    return solution


def main() -> None:
    rospy.init_node("initial_pose_commander")
    ready_param = "/visual_servo_grasp_sim/initial_pose_ready"
    rospy.set_param(ready_param, False)
    default_scene_config = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "scene.yaml",
    )
    scene_config = rospy.get_param("~scene_config", default_scene_config)
    randomize_blocks = parse_bool(rospy.get_param("~randomize_blocks", False))
    random_seed = int(rospy.get_param("~random_seed", 0))

    block_center_xy = block_center_from_scene(scene_config, randomize_blocks, random_seed)
    positions = solve_topdown_hover(wait_for_robot_description(), block_center_xy)

    command_topic = "/eff_joint_traj_controller/command"
    publisher = rospy.Publisher(command_topic, JointTrajectory, queue_size=1, latch=True)
    rospy.loginfo("Waiting for %s subscribers", command_topic)

    deadline = time.monotonic() + 20.0
    while not rospy.is_shutdown() and publisher.get_num_connections() == 0 and time.monotonic() < deadline:
        time.sleep(0.1)

    trajectory = build_trajectory(positions)
    for _ in range(10):
        if rospy.is_shutdown():
            return
        trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.2)
        publisher.publish(trajectory)
        time.sleep(0.1)

    rospy.loginfo("Commanded UR5 top-down hover pose over block center")
    deadline = time.monotonic() + 120.0
    stable_since = None
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        try:
            state = rospy.wait_for_message("/joint_states", JointState, timeout=2.0)
        except rospy.ROSException:
            continue
        measured = dict(zip(state.name, state.position))
        errors = [abs(np.arctan2(np.sin(measured.get(name, float("nan")) - target),
                                np.cos(measured.get(name, float("nan")) - target)))
                  for name, target in zip(JOINT_NAMES, positions)]
        if all(error < 0.03 for error in errors):
            stable_since = time.monotonic() if stable_since is None else stable_since
            if time.monotonic() - stable_since >= 0.5:
                rospy.set_param(ready_param, True)
                rospy.loginfo("Initial hover joint state is stable")
                return
        else:
            stable_since = None
    raise RuntimeError("Initial hover did not reach a stable joint state")


if __name__ == "__main__":
    main()
