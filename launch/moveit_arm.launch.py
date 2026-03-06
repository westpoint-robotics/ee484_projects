"""
A launch file for running the motion planning python api tutorial
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder(robot_name="turtlebot3_manipulation", package_name="turtlebot3_manipulation_moveit_config")
        .moveit_cpp(
            file_path=get_package_share_directory("ee484_projects")
            + "/config/moveit_arm_config.yaml"
        )
        .to_moveit_configs()
    )

    example_file = DeclareLaunchArgument(
        "script_name",
        default_value="moveit_arm.py",
        description="Python script that will be loaded.",
    )

    moveit_py_node = Node(
        name="cadetNode",
        package="ee484_projects",
        executable=LaunchConfiguration("script_name"),
        output="both",
        parameters=[moveit_config.to_dict()],
    )



    return LaunchDescription(
        [
            example_file,
            moveit_py_node,
        ]
    )