
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration

from rclpy.action import ActionClient
from control_msgs.action import GripperCommand

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
import traceback
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class PublisherJointTrajectory(Node):
    def __init__(self):
        super().__init__("publisher_position_trajectory_controller")
        # Declare all parameters
        self.declare_parameter("controller_name", "position_trajectory_controller")
        self.declare_parameter("wait_sec_between_publish", 6)
        self.declare_parameter("goal_names", ["pos1", "pos2"])
        self.declare_parameter("joints", [""])
        self.declare_parameter("check_starting_point", False)

        # Read parameters
        controller_name = self.get_parameter("controller_name").value
        wait_sec_between_publish = self.get_parameter("wait_sec_between_publish").value
        goal_names = self.get_parameter("goal_names").value
        self.joints = self.get_parameter("joints").value
        self.check_starting_point = self.get_parameter("check_starting_point").value
        self.starting_point = {}

        for joint in self.joints:
            self.get_logger().info(f'Joint Name is  "{joint}" ')

        if self.joints is None or len(self.joints) == 0:
            raise Exception('"joints" parameter is not set!')

        # starting point stuff
        if self.check_starting_point:
            # declare nested params
            for name in self.joints:
                param_name_tmp = "starting_point_limits" + "." + name
                self.declare_parameter(param_name_tmp, [-2 * 3.14159, 2 * 3.14159])
                self.starting_point[name] = self.get_parameter(param_name_tmp).value

            for name in self.joints:
                if len(self.starting_point[name]) != 2:
                    raise Exception('"starting_point" parameter is not set correctly!')
            self.joint_state_sub = self.create_subscription(
                #JointState, "joint_states", self.joint_state_callback, 10
                JointState, "/joint_states", self.joint_state_callback, 10
            )
        # initialize starting point status
        self.starting_point_ok = not self.check_starting_point

        self.joint_state_msg_received = False

        # Read all positions from parameters
        self.goals = []  # List of JointTrajectoryPoint
        for name in goal_names:
            self.declare_parameter(name, rclpy.Parameter.Type.DOUBLE_ARRAY)
            point = JointTrajectoryPoint()

            def get_sub_param(sub_param):
                param_name = name + "." + sub_param
                self.declare_parameter(param_name, [float()])
                param_value = self.get_parameter(param_name).value

                float_values = []

                if len(param_value) != len(self.joints):
                    return [False, float_values]

                float_values = [float(value) for value in param_value]

                return [True, float_values]

            one_ok = False

            [ok, values] = get_sub_param("positions")
            if ok:
                point.positions = values
                one_ok = True

            [ok, values] = get_sub_param("velocities")
            if ok:
                point.velocities = values
                one_ok = True

            [ok, values] = get_sub_param("accelerations")
            if ok:
                point.accelerations = values
                one_ok = True

            [ok, values] = get_sub_param("effort")
            if ok:
                point.effort = values
                one_ok = True

            if one_ok:
                point.time_from_start = Duration(sec=4)
                self.goals.append(point)
                self.get_logger().info(f'Goal "{name}" has definition {point}')

            else:
                self.get_logger().warn(
                    f'Goal "{name}" definition is wrong. This goal will not be used. '
                    "Use the following structure: \n<goal_name>:\n  "
                    "positions: [joint1, joint2, joint3, ...]\n  "
                    "velocities: [v_joint1, v_joint2, ...]\n  "
                    "accelerations: [a_joint1, a_joint2, ...]\n  "
                    "effort: [eff_joint1, eff_joint2, ...]"
                )

        if len(self.goals) < 1:
            self.get_logger().error("No valid goal found. Exiting...")
            exit(1)

        # publish_topic = "/" + controller_name + "/" + "joint_trajectory"
        publish_topic = "/arm_controller/joint_trajectory"

        self.get_logger().info(
            f"Publishing {len(goal_names)} goals on topic '{publish_topic}' every "
            f"{wait_sec_between_publish} s"
        )

        self.publisher_ = self.create_publisher(JointTrajectory, publish_topic, 1)
        
        # ADDED FOR GRIPPER
        # Create an ActionClient to communicate with the GripperActionController
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )
        
        self.cmd_vel = TwistStamped()
        cmd_vel_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.cmd_vel_pub_ = self.create_publisher(TwistStamped, '/diff_drive_controller/cmd_vel', qos_profile=cmd_vel_qos)

        self.timer = self.create_timer(wait_sec_between_publish, self.timer_callback)
        self.timer2 = self.create_timer(0.1, self.timer_cmd_vel_callback)
        self.i = 0

    def timer_cmd_vel_callback(self):
        self.pub_vel()
        pass

    def pub_vel(self, lin=None, ang=None):
        self.cmd_vel.header.stamp = self.get_clock().now().to_msg()
        self.cmd_vel.header.frame_id = 'Turtlebot3'
        if isinstance(lin, float):
            self.cmd_vel.twist.linear.x = lin
            if isinstance(ang, float):
                self.cmd_vel.twist.angular.z = ang
            else:
                self.cmd_vel.twist.angular.z = 0.0
        else:
            self.cmd_vel.twist.linear.x = 0.7
            self.cmd_vel.twist.angular.z = 1.0
        self.cmd_vel_pub_.publish(self.cmd_vel)


    def timer_callback(self):

        if self.starting_point_ok:

            self.get_logger().info(f"Sending goal {self.goals[self.i]}.")

            traj = JointTrajectory()
            traj.joint_names = self.joints
            traj.points.append(self.goals[self.i])
            self.publisher_.publish(traj)

            self.i += 1
            self.i %= len(self.goals)

        elif self.check_starting_point and not self.joint_state_msg_received:
            self.get_logger().warn(
                'Start configuration could not be checked! Check "joint_state" topic!'
            )
        else:
            self.get_logger().warn("Start configuration is not within configured limits!")

    def joint_state_callback(self, msg):

        if not self.joint_state_msg_received:

            # check start state
            limit_exceeded = [False] * len(msg.name)
            for idx, enum in enumerate(msg.name):
                try:
                    if (msg.position[idx] < self.starting_point[enum][0]) or (
                        msg.position[idx] > self.starting_point[enum][1]
                    ):
                        self.get_logger().warn(f"Starting point limits exceeded for joint {enum} !")
                        limit_exceeded[idx] = True
                except KeyError:
                    pass

            if any(limit_exceeded):
                self.starting_point_ok = False
            else:
                self.starting_point_ok = True

            self.joint_state_msg_received = True
        else:
            return
            
    def send_gripper_goal(self, position, max_effort=0.0):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self.gripper_client.wait_for_server()
        self._send_goal_future = self.gripper_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.get_logger().info(f"Sending gripper command: position={position}, max_effort={max_effort}")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Gripper goal rejected.')
            return

        self.get_logger().info('Gripper goal accepted.')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f"Gripper result: position={result.position:.4f}, "
            f"effort={result.effort}, "
            f"stalled={result.stalled}, "
            f"reached_goal={result.reached_goal}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = PublisherJointTrajectory()
    node.pub_vel(0.0,0.0)

    # For example, open the gripper right away:
    node.send_gripper_goal(position=0.03)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        print("Keyboard interrupt received. Shutting down node.")
        # Send stop command to diff drive
        node.pub_vel(0.0,0.0)

    except Exception as e:
        print(f"Unhandled exception: {traceback.format_exc()}")

    # Send stop command to diff drive
    rclpy.shutdown()

if __name__ == "__main__":
    main()
