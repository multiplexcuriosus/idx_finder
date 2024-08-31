cmake_minimum_required(VERSION 3.0.2)
project(idx_finder)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_service_files(DIRECTORY srv FILES IDXAcquisition.srv)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs)


catkin_install_python(PROGRAMS 
  scripts/idx_finder.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/idx_finder
)
