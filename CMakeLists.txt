cmake_minimum_required(VERSION 3.0.2)
project(idx_finder)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

## Find Boost
find_package(Boost REQUIRED COMPONENTS system thread)

add_service_files(
  FILES IDXAcquisition.srv
)

generate_messages(DEPENDENCIES std_msgs sensor_msgs)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs roscpp sensor_msgs
  DEPENDS Boost
)

include_directories(include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

catkin_install_python(PROGRAMS 
  scripts/idx_finder_server.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/idx_finder
)
