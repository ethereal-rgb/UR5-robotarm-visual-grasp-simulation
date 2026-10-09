#!/usr/bin/env python3
import argparse
import math
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import rospy
import yaml
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SpawnModel
from geometry_msgs.msg import Pose
from PIL import Image, ImageDraw, ImageFont


Vector3 = Tuple[float, float, float]
Rgba = Tuple[float, float, float, float]


def parse_bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def as_float_tuple(values: Sequence[float], length: int, name: str) -> Tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    return tuple(float(v) for v in values)


def pose_text(x: float, y: float, z: float, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> str:
    return f"{x:.6f} {y:.6f} {z:.6f} {roll:.6f} {pitch:.6f} {yaw:.6f}"


def gazebo_pose(x: float, y: float, z: float, yaw: float = 0.0) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.z = math.sin(yaw / 2.0)
    pose.orientation.w = math.cos(yaw / 2.0)
    return pose


def rgba_text(color: Rgba) -> str:
    return " ".join(f"{component:.4f}" for component in color)


def box_inertia(mass: float, size: Vector3) -> Dict[str, float]:
    sx, sy, sz = size
    return {
        "ixx": mass * (sy * sy + sz * sz) / 12.0,
        "iyy": mass * (sx * sx + sz * sz) / 12.0,
        "izz": mass * (sx * sx + sy * sy) / 12.0,
    }


@dataclass(frozen=True)
class PhysicsConfig:
    mu1: float
    mu2: float
    kp: float
    kd: float
    min_depth: float

    @classmethod
    def from_dict(cls, data: Dict) -> "PhysicsConfig":
        friction = data.get("friction", {})
        contact = data.get("contact", {})
        return cls(
            mu1=float(friction.get("mu1", 1.2)),
            mu2=float(friction.get("mu2", 1.2)),
            kp=float(contact.get("kp", 1000000.0)),
            kd=float(contact.get("kd", 1.0)),
            min_depth=float(contact.get("min_depth", 0.001)),
        )

    def surface_sdf(self) -> str:
        return f"""
          <surface>
            <friction>
              <ode>
                <mu>{self.mu1:.6f}</mu>
                <mu2>{self.mu2:.6f}</mu2>
              </ode>
            </friction>
            <contact>
              <ode>
                <kp>{self.kp:.6f}</kp>
                <kd>{self.kd:.6f}</kd>
                <min_depth>{self.min_depth:.6f}</min_depth>
              </ode>
            </contact>
          </surface>"""


@dataclass(frozen=True)
class TargetBoardConfig:
    name: str
    size: Vector3
    pose_xy_yaw: Tuple[float, float, float]
    color: Rgba

    @property
    def top_z(self) -> float:
        return self.size[2]

    @classmethod
    def from_dict(cls, data: Dict) -> "TargetBoardConfig":
        size = as_float_tuple(data.get("size", [0.42, 0.30, 0.015]), 3, "target_board.size")
        if size[2] <= 0.0:
            raise ValueError("target_board.size[2] must be positive because the bottom is fixed at z=0")
        return cls(
            name=str(data.get("name", "target_board")),
            size=size,
            pose_xy_yaw=as_float_tuple(data.get("pose_xy_yaw", [0.85, 0.18, 0.0]), 3, "target_board.pose_xy_yaw"),
            color=as_float_tuple(data.get("color", [0.3, 0.3, 0.3, 1.0]), 4, "target_board.color"),
        )


@dataclass(frozen=True)
class ObjectItem:
    item_id: int
    pose_xy_yaw: Optional[Tuple[float, float, float]]

    @classmethod
    def from_dict(cls, data: Dict) -> "ObjectItem":
        pose = data.get("pose_xy_yaw")
        return cls(
            item_id=int(data["id"]),
            pose_xy_yaw=None if pose is None else as_float_tuple(pose, 3, "pose_xy_yaw"),
        )


@dataclass(frozen=True)
class BlockConfig:
    size: Vector3
    mass: float
    color: Rgba
    items: Tuple[ObjectItem, ...]

    @classmethod
    def from_dict(cls, data: Dict) -> "BlockConfig":
        raw_items = data.get("items", [{"id": i} for i in range(1, 10)])
        return cls(
            size=as_float_tuple(data.get("size", [0.045, 0.045, 0.045]), 3, "blocks.size"),
            mass=float(data.get("mass", 0.05)),
            color=as_float_tuple(data.get("color", [0.1, 0.45, 0.85, 1.0]), 4, "blocks.color"),
            items=tuple(ObjectItem.from_dict(item) for item in raw_items),
        )


@dataclass(frozen=True)
class TargetConfig:
    size: Vector3
    color: Rgba
    items: Tuple[ObjectItem, ...]

    @classmethod
    def from_dict(cls, data: Dict) -> "TargetConfig":
        raw_items = data.get("items", [{"id": i} for i in range(1, 7)])
        return cls(
            size=as_float_tuple(data.get("size", [0.075, 0.075, 0.004]), 3, "targets.size"),
            color=as_float_tuple(data.get("color", [0.08, 0.55, 0.18, 1.0]), 4, "targets.color"),
            items=tuple(ObjectItem.from_dict(item) for item in raw_items),
        )


@dataclass(frozen=True)
class RandomizationConfig:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    yaw_range: Tuple[float, float]
    min_distance: float
    max_attempts: int

    @classmethod
    def from_dict(cls, data: Dict) -> "RandomizationConfig":
        return cls(
            x_range=as_float_tuple(data.get("x_range", [0.45, 0.67]), 2, "randomization.x_range"),
            y_range=as_float_tuple(data.get("y_range", [-0.20, 0.18]), 2, "randomization.y_range"),
            yaw_range=as_float_tuple(data.get("yaw_range", [-math.pi, math.pi]), 2, "randomization.yaw_range"),
            min_distance=float(data.get("min_distance", 0.07)),
            max_attempts=int(data.get("max_attempts", 500)),
        )


@dataclass(frozen=True)
class SceneConfig:
    target_board: TargetBoardConfig
    blocks: BlockConfig
    targets: TargetConfig
    randomization: RandomizationConfig
    physics: PhysicsConfig

    @classmethod
    def load(cls, path: str) -> "SceneConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(
            target_board=TargetBoardConfig.from_dict(data.get("target_board", {})),
            blocks=BlockConfig.from_dict(data.get("blocks", {})),
            targets=TargetConfig.from_dict(data.get("targets", {})),
            randomization=RandomizationConfig.from_dict(data.get("randomization", {})),
            physics=PhysicsConfig.from_dict(data.get("physics", {})),
        )


class RandomPoseSampler:
    def __init__(self, config: RandomizationConfig, seed: int):
        self.config = config
        self.random = random.Random(seed)

    def sample(self, count: int) -> List[Tuple[float, float, float]]:
        poses: List[Tuple[float, float, float]] = []
        attempts = 0
        while len(poses) < count and attempts < self.config.max_attempts:
            attempts += 1
            x = self.random.uniform(*self.config.x_range)
            y = self.random.uniform(*self.config.y_range)
            yaw = self.random.uniform(*self.config.yaw_range)
            if self._far_enough(x, y, poses):
                poses.append((x, y, yaw))
        if len(poses) != count:
            raise RuntimeError(
                f"Could only sample {len(poses)} non-overlapping poses for {count} blocks. "
                "Increase randomization area or reduce min_distance."
            )
        return poses

    def _far_enough(self, x: float, y: float, poses: Iterable[Tuple[float, float, float]]) -> bool:
        min_distance_sq = self.config.min_distance * self.config.min_distance
        return all((x - px) ** 2 + (y - py) ** 2 >= min_distance_sq for px, py, _ in poses)


class TextureGenerator:
    def __init__(self):
        self.root = os.path.join(tempfile.gettempdir(), "visual_servo_grasp_sim", "materials")
        self.script_dir = os.path.join(self.root, "scripts")
        self.texture_dir = os.path.join(self.root, "textures")
        self.material_path = os.path.join(self.script_dir, "DigitBlocks.material")

    @property
    def material_uri(self) -> str:
        return f"file://{self.script_dir}"

    def ensure(self) -> None:
        os.makedirs(self.script_dir, exist_ok=True)
        os.makedirs(self.texture_dir, exist_ok=True)
        self._write_textures()
        self._write_material_script()

    def _write_textures(self) -> None:
        for digit in range(1, 10):
            self._write_label_texture(f"digit_{digit}.png", str(digit), (245, 245, 235), (15, 15, 15))
        for target_id in range(1, 7):
            self._write_label_texture(f"target_{target_id}.png", str(target_id), (220, 255, 220), (5, 70, 20))

    def _write_label_texture(self, filename: str, label: str, background: Tuple[int, int, int], foreground: Tuple[int, int, int]) -> None:
        path = os.path.join(self.texture_dir, filename)
        image = Image.new("RGB", (256, 256), background)
        draw = ImageDraw.Draw(image)
        font = self._font(180)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(((256 - text_w) / 2 - bbox[0], (256 - text_h) / 2 - bbox[1] - 4), label, font=font, fill=foreground)
        if label in ("6", "9"):
            draw.line((78, 222, 178, 222), fill=foreground, width=16)
        image.save(path)

    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ):
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    def _write_material_script(self) -> None:
        lines = []
        for digit in range(1, 10):
            lines.append(self._material(f"DigitBlocks/Block{digit}", f"digit_{digit}.png"))
        for target_id in range(1, 7):
            lines.append(self._material(f"DigitBlocks/Target{target_id}", f"target_{target_id}.png"))
        with open(self.material_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    @staticmethod
    def _material(name: str, texture_name: str) -> str:
        return f"""material {name}
{{
  technique
  {{
    pass
    {{
      ambient 1 1 1 1
      diffuse 1 1 1 1
      texture_unit
      {{
        texture ../textures/{texture_name}
      }}
    }}
  }}
}}"""


class GazeboModelFactory:
    def __init__(self, scene: SceneConfig, material_uri: str):
        self.scene = scene
        self.material_uri = escape(material_uri)

    def target_board(self) -> Tuple[str, str, Pose]:
        board = self.scene.target_board
        sx, sy, sz = board.size
        x, y, yaw = board.pose_xy_yaw
        return board.name, f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{escape(board.name)}">
    <static>true</static>
    <link name="target_board_link">
      <collision name="target_board_collision">
        <geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry>
        {self.scene.physics.surface_sdf()}
      </collision>
      <visual name="target_board_visual">
        <geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry>
        <material>
          <ambient>{rgba_text(board.color)}</ambient>
          <diffuse>{rgba_text(board.color)}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>""", gazebo_pose(x, y, sz / 2.0, yaw)

    def block(self, item_id: int, xy_yaw: Tuple[float, float, float]) -> Tuple[str, str, Pose]:
        blocks = self.scene.blocks
        sx, sy, sz = blocks.size
        z = sz / 2.0
        x, y, yaw = xy_yaw
        inertia = box_inertia(blocks.mass, blocks.size)
        name = f"digit_block_{item_id}"
        return name, f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="block_link">
      <inertial>
        <mass>{blocks.mass:.6f}</mass>
        <inertia>
          <ixx>{inertia["ixx"]:.9f}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{inertia["iyy"]:.9f}</iyy><iyz>0</iyz>
          <izz>{inertia["izz"]:.9f}</izz>
        </inertia>
      </inertial>
      <collision name="block_collision">
        <geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry>
        {self.scene.physics.surface_sdf()}
      </collision>
      <visual name="block_visual">
        <geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry>
        <material>
          <ambient>{rgba_text(blocks.color)}</ambient>
          <diffuse>{rgba_text(blocks.color)}</diffuse>
        </material>
      </visual>
      <visual name="digit_top_visual">
        <pose>0 0 {sz / 2.0 + 0.0008:.6f} 0 0 0</pose>
        <geometry><box><size>{sx:.6f} {sy:.6f} 0.001000</size></box></geometry>
        <material>
          <script>
            <uri>{self.material_uri}</uri>
            <name>DigitBlocks/Block{item_id}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>""", gazebo_pose(x, y, z, yaw)

    def target(self, item_id: int, xy_yaw: Tuple[float, float, float]) -> Tuple[str, str, Pose]:
        targets = self.scene.targets
        sx, sy, sz = targets.size
        z = self.scene.target_board.top_z + sz / 2.0
        x, y, yaw = xy_yaw
        name = f"target_cell_{item_id}"
        return name, f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="target_link">
      <collision name="target_collision">
        <geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry>
      </collision>
      <visual name="target_visual">
        <geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry>
        <material>
          <ambient>{rgba_text(targets.color)}</ambient>
          <diffuse>{rgba_text(targets.color)}</diffuse>
        </material>
      </visual>
      <visual name="target_label_visual">
        <pose>0 0 {sz / 2.0 + 0.0006:.6f} 0 0 0</pose>
        <geometry><box><size>{sx:.6f} {sy:.6f} 0.001000</size></box></geometry>
        <material>
          <script>
            <uri>{self.material_uri}</uri>
            <name>DigitBlocks/Target{item_id}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>""", gazebo_pose(x, y, z, yaw)


class SceneSpawner:
    def __init__(self, scene: SceneConfig, randomize_blocks: bool, seed: int, material_uri: str):
        self.scene = scene
        self.randomize_blocks = randomize_blocks
        self.seed = seed
        self.factory = GazeboModelFactory(scene, material_uri)
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        rospy.wait_for_service("/gazebo/delete_model")
        rospy.wait_for_service("/gazebo/get_world_properties")
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        self.get_world_properties = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)

    def spawn(self) -> None:
        self._delete_legacy_models()
        models = [self.factory.target_board()]
        models.extend(self.factory.block(item.item_id, pose) for item, pose in self._block_poses())
        models.extend(self.factory.target(item.item_id, self._required_pose(item)) for item in self.scene.targets.items)
        for name, sdf, initial_pose in models:
            self._delete_if_present(name)
            response = self.spawn_model(name, sdf, "", initial_pose, "world")
            if not response.success:
                raise RuntimeError(f"Failed to spawn {name}: {response.status_message}")
            rospy.loginfo("Spawned %s", name)

    def _block_poses(self) -> List[Tuple[ObjectItem, Tuple[float, float, float]]]:
        if self.randomize_blocks:
            sampled = RandomPoseSampler(self.scene.randomization, self.seed).sample(len(self.scene.blocks.items))
            return list(zip(self.scene.blocks.items, sampled))
        return [(item, self._required_pose(item)) for item in self.scene.blocks.items]

    @staticmethod
    def _required_pose(item: ObjectItem) -> Tuple[float, float, float]:
        if item.pose_xy_yaw is None:
            raise ValueError(f"Object {item.item_id} has no pose_xy_yaw and randomization is disabled")
        return item.pose_xy_yaw

    def _delete_if_present(self, name: str) -> None:
        if name not in self.get_world_properties().model_names:
            return
        try:
            self.delete_model(name)
        except rospy.ServiceException:
            pass

    def _delete_legacy_models(self) -> None:
        for name in ("table", "digit_block_0"):
            self._delete_if_present(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spawn the visual servo grasping scene in Gazebo.")
    parser.add_argument("--config", required=True, help="Path to the scene YAML file.")
    parser.add_argument("--randomize", default="false", help="Whether to randomize block initial poses.")
    parser.add_argument("--seed", default="0", help="Random seed for block pose sampling.")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("visual_servo_scene_spawner")
    scene = SceneConfig.load(args.config)
    textures = TextureGenerator()
    textures.ensure()

    spawner = SceneSpawner(
        scene=scene,
        randomize_blocks=parse_bool(args.randomize),
        seed=int(args.seed),
        material_uri=textures.material_uri,
    )
    spawner.spawn()
    rospy.loginfo("Visual servo grasping scene is ready")


if __name__ == "__main__":
    main()
