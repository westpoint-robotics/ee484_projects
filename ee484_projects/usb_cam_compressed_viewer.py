#!/usr/bin/env python3
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CompressedImage

class ImageYUYVSubscriber(Node):
    def __init__(self):
        super().__init__('image_yuyv_subscriber')

        # Parameters
        self.declare_parameter('topic', '/image_raw')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('target_fps', 30.0)

        self.topic = self.get_parameter('topic').get_parameter_value().string_value
        self.use_compressed = self.get_parameter('use_compressed').get_parameter_value().bool_value
        self.width = int(self.get_parameter('width').get_parameter_value().integer_value or 640)
        self.height = int(self.get_parameter('height').get_parameter_value().integer_value or 480)
        self.target_fps = float(self.get_parameter('target_fps').get_parameter_value().double_value or 30.0)

        # Sensor-data QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE
        )

        if self.use_compressed:
            sub_topic = self.topic.rstrip('/') + '/compressed'
            self.sub = self.create_subscription(
                CompressedImage, sub_topic, self.compressed_cb, qos
            )
            self.get_logger().info(f'Subscribed to {sub_topic} (CompressedImage)')
        else:
            self.sub = self.create_subscription(
                Image, self.topic, self.image_cb, qos
            )
            self.get_logger().info(f'Subscribed to {self.topic} (Image)')

        cv2.namedWindow('Camera (YUYV -> BGR)', cv2.WINDOW_NORMAL)
        self.last_display_time = 0.0
        self.min_frame_interval = 1.0 / max(self.target_fps, 1e-3)

    def throttle_display(self):
        now = time.time()
        if now - self.last_display_time < self.min_frame_interval:
            return False
        self.last_display_time = now
        return True

    def show_frame(self, bgr):
        if not self.throttle_display():
            return
        cv2.imshow('Camera (YUYV -> BGR)', bgr)
        # 1 ms wait to keep UI responsive; adjust if you want key handling
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Quit requested (q). Shutting down...')
            rclpy.shutdown()

    def image_cb(self, msg: Image):
        # Expect encodings like 'yuyv', 'yuv422', 'yuy2', 'YUYV', etc.
        enc = (msg.encoding or '').lower()
        if not any(k in enc for k in ['yuyv', 'yuy2', 'yuv422']):
            self.get_logger().warn_once(f"Unexpected encoding '{msg.encoding}'. Trying to interpret as YUYV.")
        try:
            w, h = msg.width, msg.height
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            # YUYV is 2 bytes per pixel, interleaved Y0 U Y1 V ...
            expected = h * w * 2
            if arr.size != expected:
                # If image uses step (row stride), reshape using msg.step
                if msg.step and msg.step * h == arr.size:
                    arr = arr.reshape((h, msg.step))
                    arr = arr[:, :w * 2]  # crop to visible width
                    arr = arr.reshape((h, w, 2))
                else:
                    self.get_logger().error(f'Unexpected data size: got {arr.size}, expected {expected}')
                    return
            else:
                arr = arr.reshape((h, w, 2))

            # Convert YUYV (YUY2) to BGR
            bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)
            self.show_frame(bgr)
        except Exception as e:
            self.get_logger().error(f'Failed to process Image: {e}')

    def compressed_cb(self, msg: CompressedImage):
        fmt = (msg.format or '').lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            if 'jpeg' in fmt or 'jpg' in fmt or 'png' in fmt:
                # Standard compressed pipelines (e.g., image_transport/compressed)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError('cv2.imdecode returned None')
                self.show_frame(bgr)
                return

            # Some pipelines may stuff raw YUYV into CompressedImage with format like 'yuv422' or 'yuyv'
            if 'yuyv' in fmt or 'yuv422' in fmt or 'yuy2' in fmt:
                w, h = self.width, self.height
                expected = h * w * 2
                if data.size < expected:
                    self.get_logger().error(f'Compressed raw YUYV smaller than expected: {data.size} < {expected}')
                    return
                arr = data[:expected].reshape((h, w, 2))
                bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)
                self.show_frame(bgr)
                return

            self.get_logger().warn_once(f"Unknown compressed format '{msg.format}'. Attempting JPEG decode as fallback.")
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                self.get_logger().error('Fallback decode failed.')
                return
            self.show_frame(bgr)

        except Exception as e:
            self.get_logger().error(f'Failed to process CompressedImage: {e}')

def main():
    rclpy.init()
    node = ImageYUYVSubscriber()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()