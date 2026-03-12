#!/usr/bin/env python3

"""
A script to outline the fundamentals of the moveit_py motion planning API.
"""

import time

# generic ros libraries
import rclpy
from rclpy.logging import get_logger
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient 
from control_msgs.action import ParallelGripperCommand

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.core.kinematic_constraints import construct_joint_constraint
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

def send_arm_goal(node, logger, client: ActionClient, position): 

    goal = ParallelGripperCommand.Goal()
    goal.command.position=[position]
    goal.command.name = ['gripper_left_joint']
    timeout_sec = 9.0

    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future, timeout_sec=3.0)
    goal_handle = send_future.result()
    if not goal_handle or not goal_handle.accepted:
        logger.error('Goal was rejected')
        return None

    logger.info('Goal accepted; waiting for result...')
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_sec)
    if not result_future.done():
        logger.error('Timed out waiting for gripper result')
        return None

    result_msg = result_future.result().result
    # Safely log common fields (implementations vary)
    fields_to_log = ['width', 'position', 'effort', 'force', 'stalled', 'reached_goal']
    summary = {f: getattr(result_msg, f) for f in fields_to_log if hasattr(result_msg, f)}
    logger.info(f'Result: {summary if summary else "(no standard fields present)"}')
    return result_msg

def main():

    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    logger = get_logger("moveit_py.pose_goal")

    # instantiate MoveItPy instance and get planning component
    tbot3 = MoveItPy(node_name="moveit_py")
    node = rclpy.create_node('moveit_py_helper_node')
    tbot3_arm = tbot3.get_planning_component("arm")
    tbot3_gripper = tbot3.get_planning_component("gripper")
    logger.info("MoveItPy instance created")

    action_name = '/gripper_controller/gripper_cmd'

    client = ActionClient(node, action_type=ParallelGripperCommand, action_name=action_name)

    try:
        if not client.wait_for_server(timeout_sec=5.0): 
            logger.warning(f'Gripper action server not available at {client._action_name}') 
            return False 
        logger.info(f'Connected to Gripper Server: {client._action_name}') 

        send_arm_goal(node, logger, client, position=0.0)

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
        # Example #3: Manipulate the gripper direct position commands
        ###########################################################################    
        
        # Open the gripper
        send_arm_goal(node, logger, client, position=0.015)

        # Sleep for few seconds
        time.sleep(2)

        ###########################################################################
        # Returning to Home position before program ends
        ########################################################################### 
        
        #Set start state to current
        tbot3_arm.set_start_state_to_current_state()
        
        # set pose goal using predefined state
        tbot3_arm.set_goal_state(configuration_name="home")

        # plan to goal
        plan_and_execute(tbot3, tbot3_arm, logger, sleep_time=3.0) #Forward Kinematics are easy; therefore, default planner & parameters are sufficient - no need for IK solution checking  
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()