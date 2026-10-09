cmake_minimum_required(VERSION 3.0.2)
project(visual_servo_grasp_sim)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_geometry
  rospy
  sensor_msgs
  std_msgs
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS cv_bridge geometry_msgs image_geometry rospy sensor_msgs std_msgs tf2_geometry_msgs tf2_ros visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/controller.py
  scripts/detection.py
  scripts/initial_pose_commander.py
  scripts/scene_spawner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  urdf
  worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
