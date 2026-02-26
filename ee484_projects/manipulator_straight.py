#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class ArmStraightUpController(Node):
    def __init__(self):
        super().__init__('arm_straight_up_controller')
        
        # Create action client for follow_joint_trajectory
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )
        
        self.get_logger().info('Waiting for action server...')
        server_available = self._action_client.wait_for_server(timeout_sec=10.0)
        
        if not server_available:
            self.get_logger().error('Action server not available after 10 seconds!')
            return
            
        self.get_logger().info('Action server connected!')
        
        # Move arm to straight up position
        self.move_arm_straight_up()


    def move_arm_stow(self):
        """Move the OpenManipulator arm to straight up position using action"""
        
        # Create goal message
        goal_msg = FollowJointTrajectory.Goal()
        
        # Create JointTrajectory
        trajectory = JointTrajectory()
        
        # Joint names for OpenManipulator
        trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        
        # Create trajectory point for straight up position
        point = JointTrajectoryPoint()
        
        # Joint positions (in radians) for straight up configuration
        # joint1: 0.0 (base - no rotation)
        # joint2: 0.0 (shoulder - straight up)
        # joint3: -1.57 (elbow - bent 90 degrees to make arm vertical)
        # joint4: 0.0 (wrist - straight)
        point.positions = [0.0, -1.05, 1.07, 0.0]
        # ['0.002', '-1.046', '1.077', '0.008']

        
        # Time to reach this position (2 seconds)
        point.time_from_start = Duration(sec=2, nanosec=0)
        
        # Add point to trajectory
        trajectory.points.append(point)
        
        # Set trajectory in goal
        goal_msg.trajectory = trajectory
        
        # Send goal
        self.get_logger().info('Sending goal to move arm straight up...')
        self.get_logger().info(f'Target positions: {point.positions}')
        
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        
    def move_arm_straight_up(self):
        """Move the OpenManipulator arm to straight up position using action"""
        
        # Create goal message
        goal_msg = FollowJointTrajectory.Goal()
        
        # Create JointTrajectory
        trajectory = JointTrajectory()
        
        # Joint names for OpenManipulator
        trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        
        # Create trajectory point for straight up position
        point = JointTrajectoryPoint()
        
        # Joint positions (in radians) for straight up configuration
        # joint1: 0.0 (base - no rotation)
        # joint2: 0.0 (shoulder - straight up)
        # joint3: -1.57 (elbow - bent 90 degrees to make arm vertical)
        # joint4: 0.0 (wrist - straight)
        point.positions = [0.0, 0.0, -1.57, 0.0]
        
        # Time to reach this position (2 seconds)
        point.time_from_start = Duration(sec=2, nanosec=0)
        
        # Add point to trajectory
        trajectory.points.append(point)
        
        # Set trajectory in goal
        goal_msg.trajectory = trajectory
        
        # Send goal
        self.get_logger().info('Sending goal to move arm straight up...')
        self.get_logger().info(f'Target positions: {point.positions}')
        
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        
    def goal_response_callback(self, future):
        """Callback for when goal is accepted or rejected"""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server')
            return
            
        self.get_logger().info('Goal accepted! Executing trajectory...')
        
        # Get result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
        
    def feedback_callback(self, feedback_msg):
        """Callback for trajectory execution feedback"""
        feedback = feedback_msg.feedback
        
        # Log current joint positions if available
        if hasattr(feedback, 'actual') and len(feedback.actual.positions) > 0:
            positions = [f"{p:.3f}" for p in feedback.actual.positions]
            self.get_logger().info(f'Current positions: {positions}', throttle_duration_sec=0.5)
        
    def get_result_callback(self, future):
        """Callback for final result"""
        result = future.result().result
        
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('✓ Arm successfully moved to straight up position!')
        else:
            self.get_logger().error(
                f'✗ Trajectory execution failed with error code: {result.error_code}'
            )
            if result.error_string:
                self.get_logger().error(f'Error string: {result.error_string}')


def main(args=None):
    rclpy.init(args=args)
    
    controller = ArmStraightUpController()
    
    try:
        # Keep node alive to receive callbacks
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.move_arm_stow()
        controller.get_logger().info('Shutting down...')
    finally:
        controller.move_arm_stow()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()