#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from datetime import datetime

from geometry_msgs.msg import TwistStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

class TwistStampedCmdAndTF(Node):
    def __init__(self):
        super().__init__('twiststamped_cmd_and_tf')
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', qos)

        # TF buffer/listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Motion parameters
        self.period = 0.05  # 10 Hz
        self.duration_s = 8.0
        self.end_time = None
        self.sent_stop = False

        # Start timer after brief discovery wait
        self._wait_for_subscriber(timeout_s=2.0)
        self.timer = self.create_timer(self.period, self._tick)

        self.get_logger().info('Publishing TwistStamped to /cmd_vel at 10 Hz for 8s...')

    def _wait_for_subscriber(self, timeout_s: float = 2.0):
        start = time.time()
        while time.time() - start < timeout_s and rclpy.ok():
            if self.pub.get_subscription_count() > 0:
                return
            time.sleep(0.05)
        # Proceed even if none discovered; some systems connect late.

    def _make_msg(self, vx: float, wz: float) -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = vx
        msg.twist.angular.z = wz
        return msg

    def _tick(self):
        now = self.get_clock().now()
        if self.end_time is None:
            self.end_time = now + Duration(seconds=self.duration_s)
            # self.get_logger().info(f'Now {now} and stop time {self.end_time}')

        # 2. Get seconds and nanoseconds
        # The Time object's nanoseconds property gives total nanoseconds since epoch
        total_nanoseconds = now.nanoseconds

        # Convert to a standard Unix timestamp (float, in seconds)
        timestamp_seconds = total_nanoseconds / 1e9

        # 3. Convert to a standard Python datetime object (using local time for human readability)
        human_readable_time = datetime.fromtimestamp(timestamp_seconds)

        # 4. Format the time using strftime
        # Example format: "2024-05-21 15:30:45.123"
        formatted_time = human_readable_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Truncate microseconds to milliseconds for cleaner output

        self.get_logger().info(f"Current ROS time: {formatted_time}")

        if now < self.end_time:
            self.pub.publish(self._make_msg(0.15, 0.25))
        else:
            if not self.sent_stop:
                self.get_logger().info('Sending zero TwistStamped and stopping...')
                self.sent_stop = True
                zero = self._make_msg(0.0, 0.0)
                for _ in range(3):
                    self.pub.publish(zero)
                    time.sleep(0.05)
                # After sending stop, query TF then exit
                self._print_tf_odom_to_baselink()
                rclpy.shutdown()

    def _print_tf_odom_to_baselink(self, timeout_sec: float = 2.0):
        source = 'base_link'
        target = 'odom'

        try:
            # Prefer can_transform to avoid long exceptions
            if self.tf_buffer.can_transform(target, source, Time(), Duration(seconds=timeout_sec)):
                tf = self.tf_buffer.lookup_transform(target, source, Time())
            else:
                tf = self.tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=timeout_sec))

            t = tf.transform.translation
            q = tf.transform.rotation
            # Yaw from quaternion
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            print(
                f"Transform odom -> base_link\n"
                f"  Translation (m): x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f}\n"
                f"  Rotation (quat): x={q.x:.4f}, y={q.y:.4f}, z={q.z:.4f}, w={q.w:.4f}\n"
                f"  Yaw (rad): {yaw:.3f}\n"
                f"  Yaw (deg): {yaw*180/3.14:.3f}\n"
                f"  Stamp (sec.nanosec): {tf.header.stamp.sec}.{tf.header.stamp.nanosec:09d}\n"
                f"  Frame: {tf.header.frame_id} -> {tf.child_frame_id or 'base_link'}"
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Could not get TF odom -> base_link: {e}')

def main():
    rclpy.init()
    node = TwistStampedCmdAndTF()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()