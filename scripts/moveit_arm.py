#!/usr/bin/env python3

"""
A script to outline the fundamentals of the moveit_py motion planning API.
"""

import time

# generic ros libraries
import rclpy
from rclpy.logging import get_logger
from geometry_msgs.msg import PoseStamped

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
    PlanRequestParameters,
)

###################################################################
# HELPER FUNCTIONS
###################################################################
def plan_and_execute(
    robot,
    planning_component,
    logger,
    single_plan_parameters=None,
    multi_plan_parameters=None,
    sleep_time=0.0,
):
    """Helper function to plan and execute a motion."""
    # plan to goal
    logger.info("Planning trajectory")
    if multi_plan_parameters is not None:
        plan_result = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters
        )
    elif single_plan_parameters is not None:
        plan_result = planning_component.plan(
            single_plan_parameters=single_plan_parameters
        )
    else:
        plan_result = planning_component.plan()

    # execute the plan
    if plan_result:
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
    else:
        logger.error("Planning failed")

    time.sleep(sleep_time)


def ik_checker(logger, robot_model, pose, planning_group, end_effector, timeout=0.5):
    """Helper function to check whether desired pose is reachable."""

    robot_state = RobotState(robot_model)

    ik_success = robot_state.set_from_ik(planning_group, pose, end_effector, timeout)

    if ik_success:
        logger.info("\n\t\t✅ IK SUCCESSFUL — pose reachable\n")
        return True
    else:
        logger.warning("\n\t\t⚠️ IK FAILED — pose unreachable.\n")
        return False

def main():

    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    logger = get_logger("moveit_py.pose_goal")

    # instantiate MoveItPy instance and get planning component
    tbot3 = MoveItPy(node_name="moveit_py")
    tbot3_arm = tbot3.get_planning_component("arm")
    tbot3_gripper = tbot3.get_planning_component("gripper")
    logger.info("MoveItPy instance created")

    ###########################################################################
    # Example #1 - Set goal state using group states (i.e. predefined strings) [These are defined in turtlebot3_manipulation.srdf]
    ###########################################################################

    #Set start state to current
    tbot3_arm.set_start_state_to_current_state()
      
    # set pose goal using predefined state
    tbot3_arm.set_goal_state(configuration_name="home")

    # plan to goal
    plan_and_execute(tbot3, tbot3_arm, logger, sleep_time=3.0) #Forward Kinematics are easy; therefore, default planner & parameters are sufficient - no need for IK solution checking

    ###########################################################################
    # Example #2: Set goal state with PoseStamped message (Inverse Kinematics)
    ###########################################################################
    
    # Set start state to current state
    tbot3_arm.set_start_state_to_current_state()

    #Create a PoseStamped message for the goal pose. Pose is relative to frame_id specified in header (e.g. base_link).
    pose_goal = PoseStamped()
    pose_goal.header.frame_id = "base_link"

    # Desired Position
    pose_goal.pose.position.x = 0.15
    pose_goal.pose.position.y = 0.0
    pose_goal.pose.position.z = 0.2

    # Desired Orientation (ignored when position_only_ik=True in kinematics.yaml)
    pose_goal.pose.orientation.x = 0.0
    pose_goal.pose.orientation.y = 0.0
    pose_goal.pose.orientation.z = 0.0
    pose_goal.pose.orientation.w = 1.0

    # Set the goal for the planning component
    tbot3_arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="link5") #pose_link should be the end-effector frame, which is the last frame on the arm kinematic chain link5 (see turtlebot3_manipulation.srdf and tbotomx.urdf)

    # ---------------------------------------------------------------------------
    # Recommendation: Test IK first to ensure that desired goal pose is reachable
    # ---------------------------------------------------------------------------

    if ik_checker(logger, tbot3.get_robot_model(), pose_goal.pose, "arm", "link5"):
        
        # -----------------------------------------
        # Goal pose is reachable - Plan and execute
        # -----------------------------------------

        params = PlanRequestParameters(tbot3, "ompl_cadet") #Parameters loaded from YAML File
        
        #You can override the motion planning components in this file by commenting out the line(s) below
        # params.planning_attempts = 5
        # params.planning_time = 3.0
        # params.max_velocity_scaling_factor = 0.8
        # params.max_acceleration_scaling_factor = 0.8

        plan_and_execute(tbot3, tbot3_arm, logger, single_plan_parameters=params)

      
    ###########################################################################
    # Example #3: Manipulate the gripper (Forward Kinematics) 
    ###########################################################################    

    # set constraints message (constraints are defined in joint_limits.yaml)
    from moveit.core.kinematic_constraints import construct_joint_constraint
    robot_state = RobotState(tbot3.get_robot_model())

    joint_values_open_gripper = {"gripper_left_joint": 1.0} #Open Gripper (m)
    joint_values_closed_gripper = {"gripper_left_joint": -1.0} #Closed Gripper (m)

    #Define command to open gripper
    robot_state.joint_positions = joint_values_open_gripper
    joint_constraint_open_gripper = construct_joint_constraint(
        robot_state=robot_state,
        joint_model_group=tbot3.get_robot_model().get_joint_model_group("gripper"),
    )

    #Define command to close gripper
    robot_state.joint_positions = joint_values_closed_gripper
    joint_constraint_closed_gripper = construct_joint_constraint(
        robot_state=robot_state,
        joint_model_group=tbot3.get_robot_model().get_joint_model_group("gripper"),
    )
    
    #Define planner parameters
    params = PlanRequestParameters(tbot3, "ompl_cadet") #Parameters loaded from YAML File
        
    # You can override the motion planning components in this file by commenting out the line(s) below
    params.max_velocity_scaling_factor = 0.1 #The scaling factors need to be reduced to ~0.1. Otherwise, the gripper trajectory is too short and MoveIt will time out before closing/opening the gripper.
    params.max_acceleration_scaling_factor = 0.1

    # Open the gripper
    tbot3_gripper.set_start_state_to_current_state()
    tbot3_gripper.set_goal_state(motion_plan_constraints=[joint_constraint_open_gripper])
    plan_and_execute(tbot3, tbot3_gripper, logger, sleep_time=3.0, single_plan_parameters=params)

    # Close the gripper
    tbot3_gripper.set_start_state_to_current_state()
    tbot3_gripper.set_goal_state(motion_plan_constraints=[joint_constraint_closed_gripper])
    plan_and_execute(tbot3, tbot3_gripper, logger, sleep_time=3.0, single_plan_parameters=params)

    ###########################################################################
    # Returning to Home position before program ends
    ########################################################################### 
    
    #Set start state to current
    tbot3_arm.set_start_state_to_current_state()
      
    # set pose goal using predefined state
    tbot3_arm.set_goal_state(configuration_name="home")

    # plan to goal
    plan_and_execute(tbot3, tbot3_arm, logger, sleep_time=3.0) #Forward Kinematics are easy; therefore, default planner & parameters are sufficient - no need for IK solution checking  

if __name__ == "__main__":
    main()