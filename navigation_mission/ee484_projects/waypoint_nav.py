import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('waypoint_nav')
        self.get_logger().info('EE484 Waypoint Navigation Started')

def main(args=None):
    rclpy.init(args=args)
    my_node = MyNode()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()