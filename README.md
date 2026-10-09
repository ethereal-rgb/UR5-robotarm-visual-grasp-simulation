# UR5 数字方块视觉抓取仿真

基于 ROS Noetic、Gazebo Classic 和 MoveIt 的机器人视觉抓取仿真。UR5 搭载 Robotiq 85 夹爪和腕部 RGB-D 相机，识别数字方块、计算抓取位姿，并将方块放置到指定目标格。

使用者在终端中输入方块编号与目标格编号，可一次安排多个任务。仿真环境与任务控制器分别运行在两个终端中。
项目整体借助codex工具实现。

## 功能

- 生成 9 个数字方块和 6 个目标格，支持固定布局和指定种子的随机布局。
- 使用顶面分割、透视校正和数字模板匹配识别 1–9，无需训练权重。
- 结合深度图与 TF，将视觉检测结果转换为机器人基坐标系中的抓取位姿。
- 支持目标搜索、预抓取对准、接触检查、抓取、搬运和放置。
- 支持交互输入多个方块到目标格的任务，并按输入顺序执行。

## 环境要求

| 组件 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu 20.04 |
| ROS | Noetic |
| 仿真器 | Gazebo Classic 11 |
| 运动规划 | MoveIt 1 |
| Python | 3.8 |

以下步骤假设已安装 ROS Noetic，并且 `/opt/ros/noetic/setup.bash` 可用。其他环境尚未验证。

## 安装与编译

克隆仓库到 catkin 工作空间目录。将 `<owner>` 替换为本仓库在 GitHub 上的所有者：

```bash
git clone https://github.com/<owner>/robot_nocatch_ws.git ~/robot_nocatch_ws
cd ~/robot_nocatch_ws
```

安装依赖：

```bash
sudo apt update
sudo apt install python3-rosdep python3-numpy python3-opencv \
  python3-pil python3-yaml fonts-dejavu-core \
  ros-noetic-moveit ros-noetic-gazebo-ros-pkgs \
  ros-noetic-gazebo-ros-control ros-noetic-ros-controllers
```

首次使用 rosdep 时执行 `sudo rosdep init`；已初始化则跳过。随后解析包依赖并编译：

```bash
source /opt/ros/noetic/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro noetic
catkin_make -j2
```

项目已包含所需的 UR5 和 Robotiq 源码，无需另外克隆这两个依赖。首次编译会生成 `build/`、`devel/` 和 catkin 顶层 CMake 文件。

## 使用方法：两个终端

### 终端 1：启动仿真环境

```bash
cd ~/robot_nocatch_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch visual_servo_grasp_sim visual_servo_gazebo.launch \
  start_controller:=false
```

该命令启动 Gazebo、机器人控制器、场景生成、数字检测、MoveIt 和机械臂初始位姿设置。保持此终端运行。机械臂完成初始化后，日志会出现 `Initial hover joint state is stable`。

### 终端 2：交互输入并执行任务

在新的终端中执行：

```bash
cd ~/robot_nocatch_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun visual_servo_grasp_sim controller.py \
  _interactive:=true _wait_for_initial_pose:=true
```

按提示输入，例如将方块 1 放到格子 3，再将方块 2 放到格子 5：

```text
要抓取几号方块？请输入 1-9: 1
要放置到几号格子？请输入 1-6: 3
是否还有任务？请输入 y 或 n: y
要抓取几号方块？请输入 1-9: 2
要放置到几号格子？请输入 1-6: 5
是否还有任务？请输入 y 或 n: n
```

输入 `y` 继续添加任务，输入 `n` 结束任务录入。控制器收集完整任务列表后，按顺序执行；输入 `n` 之前不会开始抓取。方块编号为 **1–9**，目标格编号为 **1–6**。

`_interactive:=true` 显式启用交互输入，`_wait_for_initial_pose:=true` 使控制器在执行前等待初始位姿就绪。在普通交互终端中，直接执行 `rosrun visual_servo_grasp_sim controller.py` 也会启用交互输入；建议使用上面的完整命令。

全部任务完成后，控制器退出。若要继续提交任务，在终端 2 再次运行相同的 `rosrun` 命令即可，终端 1 的仿真场景会继续保留。选择尚未放置的方块及空闲目标格；控制器不会自动重置场景或分配空闲格子。使用 `Ctrl+C` 停止对应终端中的进程。

### 随机布局

在终端 1 用以下命令替换默认启动命令：

```bash
roslaunch visual_servo_grasp_sim visual_servo_gazebo.launch \
  start_controller:=false randomize_blocks:=true random_seed:=7
```

终端 2 同步使用相同的布局参数：

```bash
rosrun visual_servo_grasp_sim controller.py \
  _interactive:=true _wait_for_initial_pose:=true \
  _randomize_blocks:=true _random_seed:=7
```

如使用自定义场景文件，终端 1 的 `scene_config:=/绝对路径/scene.yaml` 与终端 2 的 `_scene_config:=/绝对路径/scene.yaml` 必须指向同一个文件。需要无界面运行时，在终端 1 的启动命令中添加 `gui:=false`。

## 配置

配置文件位于 `src/visual_servo_grasp_sim/config/`：

| 文件 | 内容 |
| --- | --- |
| `scene.yaml` | 方块尺寸、初始位置、目标格位置、随机布局范围和物理参数 |
| `detection.yaml` | 相机话题、顶面分割、模板匹配和坐标转换参数 |
| `controllers.yaml` | 关节控制器及 PID 参数 |

控制器参数可通过 `rosrun` 的私有参数传入，例如 `_active_search_enabled:=false`：

| 控制器参数 | 默认值 | 作用 |
| --- | --- | --- |
| `_interactive` | 交互终端中为 `true` | 交互录入任务 |
| `_wait_for_initial_pose` | `false` | 等待环境初始化；上述运行示例显式开启 |
| `_active_search_enabled` | `true` | 目标缺失时主动搜索 |
| `_use_scene_block_pose` | `false` | 使用场景坐标辅助抓取定位 |
| `_use_gazebo_grasp_constraint` | `true` | 双侧接触成立后辅助保持抓取 |

## 图像与检测结果

主要 ROS 话题：

| 话题 | 内容 |
| --- | --- |
| `/wrist_camera/color/image_raw` | 相机彩色图像 |
| `/wrist_camera/depth/image_raw` | 相机深度图像 |
| `/wrist_camera/camera_info` | 相机内参 |
| `/digit_detections` | JSON 检测结果，消息类型为 `std_msgs/String` |
| `/digit_detection_image` | 检测结果叠加图像 |
| `/digit_detection_markers` | RViz 可视化标记 |

可在额外的终端中加载工作空间环境后查看检测图像：

```bash
rosrun rqt_image_view rqt_image_view /digit_detection_image
```

如未安装该工具，执行 `sudo apt install ros-noetic-rqt-image-view`。

## 项目结构

```text
src/
├── visual_servo_grasp_sim/
│   ├── scripts/       # 场景生成、初始化、模板检测和任务控制
│   ├── config/        # 场景、视觉和控制器配置
│   ├── launch/        # 仿真环境入口
│   ├── urdf/          # 机器人、夹爪和相机组合模型
│   ├── worlds/        # Gazebo 世界
│   ├── docs/          # 坐标变换推导和仿真验证记录
│   └── tests/         # 坐标转换与抓取逻辑测试
├── ur5_robotiq85_camera_moveit_config/  # MoveIt 配置
├── universal_robot/                  # UR5 描述和 Gazebo 配置
└── robotiq/                          # Robotiq 85 描述和 Gazebo 插件
```

## 仿真说明

- 默认方块边长为 45 mm，夹爪开度与 TCP 标定针对该尺寸；改变方块尺寸后需要重新校准。
- 默认启用 Gazebo 抓取保持约束：双侧接触成立后，通过更新模型状态维持方块与夹爪的相对位姿。抓取结果包含这一辅助效果。
- 随机布局可能存在不可达位置；已有验证覆盖部分任务。
- 项目用于仿真实验，未提供实机控制接口。

详细说明见 [坐标转换推导](src/visual_servo_grasp_sim/docs/coordinate_transform_derivation.md) 和 [抓取验证记录](src/visual_servo_grasp_sim/docs/grasp_validation.md)。

## 第三方来源

机器人模型与插件基于 [Universal Robot](https://github.com/ros-industrial/universal_robot) 和 [Robotiq](https://github.com/cambel/robotiq)，包含针对本项目的裁剪和修改。来源版本及已有许可证信息见 [第三方来源说明](docs/third_party.md)。
