cmake_minimum_required(VERSION 3.22)
project(ee484_projects)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(include)

add_executable(scan_monitor src/scan_monitor.cpp)
ament_target_dependencies(scan_monitor rclcpp sensor_msgs)



ament_python_install_package(${PROJECT_NAME})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    PATTERN "setup_assistant.launch" EXCLUDE)
endif()

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

file(GLOB PYTHON_SCRIPTS "ee484_projects/*.py")

install(PROGRAMS
  ${PYTHON_SCRIPTS}
  DESTINATION lib/${PROJECT_NAME}
)


install(DIRECTORY config DESTINATION share/${PROJECT_NAME})


ament_package()