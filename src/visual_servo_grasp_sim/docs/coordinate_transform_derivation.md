# 视觉伺服抓取坐标转换推导

本文档说明当前 `visual_servo_grasp_sim` 仿真实验中，如何把图像检测结果转换为机械臂后续可执行的三维抓取位姿。推导基于当前 URDF、Gazebo RGB-D 相机参数和 `detection.py` 中的实现。

## 1. 坐标系定义

当前工程涉及以下坐标系：

| 坐标系 | 含义 | 来源 |
| --- | --- | --- |
| `world` | Gazebo 世界坐标系 | 仿真世界 |
| `base_link` | UR5 机械臂基坐标系 | URDF |
| `tool0` | UR 默认末端工具坐标系，也是当前 MoveIt 配置的 tip link | URDF |
| `gripper_tcp` | Robotiq 85 夹爪抓取中心点 | 自定义 URDF |
| `wrist_camera_link` | 腕部相机实体坐标系 | 自定义 URDF |
| `wrist_camera_optical_frame` | 相机光学坐标系，图像反投影使用该坐标系 | Gazebo 相机插件 |
| image pixel | 图像像素坐标系，检测输出 `(u, v)` | 数字面分割与模板匹配结果 |

当前 URDF 中有：

```text
world -> base_link = I
```

因此在当前仿真中，世界坐标系和机械臂基坐标系重合：

```text
T_world_base = I
T_base_world = I
```

坐标转换主链路为：

```text
image pixel -> wrist_camera_optical_frame -> base_link
```

后续生成抓取位姿时，再根据夹爪 TCP 和 MoveIt tip 的关系得到：

```text
gripper_tcp grasp pose -> tool0 grasp pose
```

## 2. 相机内参与像素反投影

Gazebo 深度相机配置为：

```xml
<horizontal_fov>1.0471975512</horizontal_fov>
<width>640</width>
<height>480</height>
```

实际运行时以内参话题 `/wrist_camera/camera_info` 为准。相机模型为针孔模型：

```text
K =
[ fx  0  cx
   0 fy  cy
   0  0   1 ]
```

若只根据当前 Gazebo 参数估算，水平视场角为 `hfov`，图像宽度为 `w`：

```text
fx = w / (2 * tan(hfov / 2))
```

代入：

```text
w = 640
hfov = 1.0471975512 rad

fx = 640 / (2 * tan(1.0471975512 / 2))
   ~= 554.256
```

Gazebo 相机通常给出近似方形像素，因此：

```text
fy ~= fx ~= 554.256
cx ~= 320.0 或 320.5
cy ~= 240.0 或 240.5
```

工程实现中不硬编码这些值，而是通过 `CameraInfo` 获取。

设检测框中心像素为：

```text
p = [u, v]^T
```

深度图中该点深度为：

```text
d = Zc
```

其中 `d` 表示目标点沿相机光轴的距离，单位为米。像素反投影到相机光学坐标系 `wrist_camera_optical_frame`：

```text
Xc = (u - cx) * d / fx
Yc = (v - cy) * d / fy
Zc = d
```

写成齐次坐标：

```text
Pc =
[ Xc
  Yc
  Zc
   1 ]
```

相机光学坐标系遵循 ROS optical frame 约定：

```text
x: 图像右方
y: 图像下方
z: 相机前方
```

## 3. 相机固定外参推导

URDF 中相机安装关系为：

```xml
<joint name="wrist_camera_joint" type="fixed">
  <parent link="tool0"/>
  <child link="wrist_camera_link"/>
  <origin xyz="0.02 0 0.02" rpy="0 ${-pi/2.0} 0"/>
</joint>

<joint name="wrist_camera_optical_joint" type="fixed">
  <parent link="wrist_camera_link"/>
  <child link="wrist_camera_optical_frame"/>
  <origin xyz="0 0 0" rpy="${-pi/2.0} 0 ${-pi/2.0}"/>
</joint>
```

URDF 的 `origin xyz rpy` 表示从父坐标系到子坐标系的固定变换：

```text
T_parent_child =
[ R_parent_child  t_parent_child
          0 0 0                 1 ]
```

其中旋转矩阵按 URDF `rpy` 约定：

```text
R = Rz(yaw) * Ry(pitch) * Rx(roll)
```

### 3.1 `tool0 -> wrist_camera_link`

平移：

```text
t_tool0_camera_link = [0.02, 0, 0.02]^T
```

旋转：

```text
rpy = [0, -pi/2, 0]
R = Ry(-pi/2)
```

展开：

```text
R_tool0_camera_link =
[ 0  0 -1
  0  1  0
  1  0  0 ]
```

因此：

```text
T_tool0_camera_link =
[ 0  0 -1  0.02
  0  1  0  0.00
  1  0  0  0.02
  0  0  0  1.00 ]
```

### 3.2 `wrist_camera_link -> wrist_camera_optical_frame`

平移为零：

```text
t_camera_link_optical = [0, 0, 0]^T
```

旋转：

```text
rpy = [-pi/2, 0, -pi/2]
R = Rz(-pi/2) * Ry(0) * Rx(-pi/2)
```

展开：

```text
R_camera_link_optical =
[  0  0  1
  -1  0  0
   0 -1  0 ]
```

因此：

```text
T_camera_link_optical =
[  0  0  1  0
  -1  0  0  0
   0 -1  0  0
   0  0  0  1 ]
```

### 3.3 合成 `tool0 -> wrist_camera_optical_frame`

固定外参合成：

```text
T_tool0_optical = T_tool0_camera_link * T_camera_link_optical
```

代入矩阵：

```text
T_tool0_optical =
[  0  1  0  0.02
  -1  0  0  0.00
   0  0  1  0.02
   0  0  0  1.00 ]
```

这表示相机光学坐标系相对 `tool0` 有 `x=0.02m, z=0.02m` 的安装偏移。

## 4. 相机点到机械臂基坐标系

若当前关节状态为 `q`，机械臂正运动学给出：

```text
T_base_tool0(q)
```

则目标点从相机光学坐标系转换到机械臂基坐标系：

```text
Pb = T_base_tool0(q) * T_tool0_optical * Pc
```

其中：

```text
Pc = [Xc, Yc, Zc, 1]^T
Pb = [Xb, Yb, Zb, 1]^T
```

在代码实现中，不手写实时 FK，而是使用 ROS TF：

```text
wrist_camera_optical_frame -> base_link
```

即：

```text
Pb = T_base_optical * Pc
```

其中：

```text
T_base_optical = T_base_tool0(q) * T_tool0_optical
```

这与上面的矩阵链路等价，并且可以自动跟随实时关节状态更新。

## 5. 物体点到抓取点

深度图检测点通常落在方块上表面中心附近。当前方块尺寸为：

```text
block_size = [0.045, 0.045, 0.045] m
```

方块高度：

```text
h = 0.045 m
```

为了让夹爪 TCP 对准方块几何中心，而不是上表面，默认使用：

```text
surface_to_grasp_z_offset = -h / 2 = -0.0225 m
```

设相机转换得到的物体表面点为：

```text
P_object =
[ Xb
  Yb
  Zb ]
```

则 TCP 抓取点位置为：

```text
P_tcp =
[ Xb
  Yb
  Zb + surface_to_grasp_z_offset ]
```

代入默认值：

```text
P_tcp =
[ Xb
  Yb
  Zb - 0.0225 ]
```

预抓取点沿基坐标系 `+Z` 方向抬高：

```text
pregrasp_lift = 0.10 m
```

因此：

```text
P_pregrasp =
[ Xtcp
  Ytcp
  Ztcp + 0.10 ]
```

## 6. 抓取姿态定义

当前默认使用固定垂直抓取策略：

```text
TCP +Z 轴指向 base_link 的 -Z 方向
```

令抓取 yaw 为 `psi`，默认：

```text
psi = 0
```

定义 TCP 坐标系三个单位轴在 `base_link` 下的方向：

```text
x_tcp = [cos(psi), sin(psi), 0]^T
z_tcp = [0, 0, -1]^T
y_tcp = z_tcp x x_tcp
```

其中 `x` 表示向量叉乘。将三个轴按列组成旋转矩阵：

```text
R_base_tcp =
[ x_tcp  y_tcp  z_tcp ]
```

当 `psi = 0` 时：

```text
x_tcp = [1, 0, 0]^T
y_tcp = [0, -1, 0]^T
z_tcp = [0, 0, -1]^T
```

所以：

```text
R_base_tcp =
[ 1   0   0
  0  -1   0
  0   0  -1 ]
```

最终 TCP 抓取位姿为：

```text
T_base_tcp =
[ R_base_tcp  P_tcp
       0 0 0      1 ]
```

预抓取位姿使用相同姿态，只改变位置：

```text
T_base_pregrasp_tcp =
[ R_base_tcp  P_pregrasp
       0 0 0           1 ]
```

## 7. `gripper_tcp` 到 MoveIt `tool0` 的换算

当前 MoveIt 默认规划链为：

```text
base_link -> tool0
```

但真实抓取更适合以夹爪 TCP 表达。自定义 URDF 中：

```xml
<joint name="gripper_tcp_joint" type="fixed">
  <parent link="robotiq_85_base_link"/>
  <child link="gripper_tcp"/>
  <origin xyz="0.142 0 0" rpy="0 ${pi/2.0} 0"/>
</joint>
```

同时 Robotiq 夹爪以如下方式安装到 `tool0`：

```xml
<xacro:robotiq_85_gripper prefix="" parent="tool0">
  <origin xyz="0 0 0" rpy="0 ${-pi/2.0} 0"/>
</xacro:robotiq_85_gripper>
```

两段固定变换合成后得到：

```text
T_tool0_tcp =
[ 1  0  0  0.00
  0  1  0  0.00
  0  0  1  0.142
  0  0  0  1.00 ]
```

即 `gripper_tcp` 位于 `tool0` 的 `+Z` 方向 `0.142m` 处，姿态与 `tool0` 相同。

如果已经得到期望 TCP 抓取位姿：

```text
T_base_tcp_desired
```

而 MoveIt 需要的是 `tool0` 目标位姿，则：

```text
T_base_tool0_desired = T_base_tcp_desired * inverse(T_tool0_tcp)
```

因为：

```text
T_base_tcp_desired = T_base_tool0_desired * T_tool0_tcp
```

整理得：

```text
T_base_tool0_desired = T_base_tcp_desired * T_tcp_tool0
```

其中：

```text
T_tcp_tool0 = inverse(T_tool0_tcp)
```

由于 `T_tool0_tcp` 只是沿 `+Z` 平移 `0.142m`，其逆矩阵为：

```text
T_tcp_tool0 =
[ 1  0  0   0.00
  0  1  0   0.00
  0  0  1  -0.142
  0  0  0   1.00 ]
```

因此 `tool0` 目标点会位于 TCP 目标点沿 TCP `-Z` 方向 `0.142m` 的位置。对于默认垂直抓取姿态，TCP 的 `-Z` 等价于基坐标系 `+Z`，所以 `tool0` 位于 TCP 上方 `0.142m`。

## 8. 输出数据含义

`/digit_detections` 中每个检测对象包含：

```text
center_3d
```

检测中心像素反投影后在 `base_link` 下的三维点，通常对应方块上表面点。

```text
object_pose
```

物体表面点位姿，位置来自 `center_3d`，姿态为单位四元数。

```text
grasp_pose_tcp
```

夹爪 TCP 抓取位姿，位置为物体表面点向下补偿半个方块高度，姿态为固定垂直抓取。

```text
pregrasp_pose_tcp
```

预抓取 TCP 位姿，在 `grasp_pose_tcp` 基础上沿 `base_link +Z` 抬高 `0.10m`。

```text
grasp_pose_tool0
```

由 `grasp_pose_tcp` 换算得到的 MoveIt 默认 `tool0` 目标位姿，可直接作为后续 MoveIt 规划目标。

## 9. 完整计算流程

整体流程可写为：

```text
1. 通过颜色分割得到浅色数字面区域，画框并取中心像素:
   (u, v)

2. 将数字面透视矫正后与标准数字模板匹配，得到方块编号:
   label = argmax(template_score)

3. 从深度图取深度:
   d = depth(u, v)

4. 像素反投影:
   Pc = [ (u - cx) * d / fx,
          (v - cy) * d / fy,
          d,
          1 ]^T

5. 使用 TF 转到机械臂基坐标系:
   Pb = T_base_optical * Pc

6. 得到物体表面点:
   P_object = [Xb, Yb, Zb]^T

7. 计算 TCP 抓取点:
   P_tcp = [Xb, Yb, Zb - 0.0225]^T

8. 构造固定垂直抓取姿态:
   R_base_tcp = [x_tcp, y_tcp, z_tcp]

9. 得到 TCP 抓取位姿:
   T_base_tcp = [R_base_tcp, P_tcp; 0 0 0 1]

10. 得到预抓取位姿:
   P_pregrasp = P_tcp + [0, 0, 0.10]^T

11. 换算 MoveIt tool0 位姿:
    T_base_tool0 = T_base_tcp * inverse(T_tool0_tcp)
```

## 10. 对应代码位置

当前实现位于：

```text
src/visual_servo_grasp_sim/scripts/detection.py
```

核心类：

```text
CameraBackProjector       # CameraInfo + depth 反投影
TfCoordinateTransformer   # 使用 tf2 做坐标系转换
GraspPoseEstimator        # 生成 TCP、预抓取和 tool0 位姿
DetectionPayloadBuilder   # 组织输出 JSON
TemplateDigitFaceDetector # 数字面颜色分割、画框取中心、模板匹配编号
```

相关配置位于：

```text
src/visual_servo_grasp_sim/config/detection.yaml
```

关键配置：

```yaml
detections_topic: /digit_detections
detection_image_topic: /digit_detection_image
debug_mask_topic: /digit_detection_mask
marker_topic: /digit_detection_markers
grasp_target_frame: base_link
primary_tcp_frame: gripper_tcp
moveit_tip_frame: tool0
block_height: 0.045
surface_to_grasp_z_offset: -0.0225
pregrasp_lift: 0.10
grasp_yaw: 0.0
```
