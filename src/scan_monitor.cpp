#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

class ScanMonitor : public rclcpp::Node
{
public:
  ScanMonitor() : Node("scan_monitor")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", 
      10, 
      std::bind(&ScanMonitor::scan_callback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "Scan Monitor Node started. Searching for 3 consecutive points < 1m.");
  }

private:
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    int consecutive_points = 0;
    bool alert_triggered = false;
    float first_detected_range = 0.0f;

    // Iterate through the individual returns in the current scan
    for (size_t i = 0; i < msg->ranges.size(); ++i) {
      float range = msg->ranges[i];

      // Check if the point is valid and less than 1 meter
      if (range > msg->range_min && range < 1.0f) {
        consecutive_points++;
        
        // Track the first point in the sequence for logging
        if (consecutive_points == 1) first_detected_range = range;

        if (consecutive_points >= 3) {
          alert_triggered = true;
          break; // Found 3 consecutive points, no need to continue
        }
      } else {
        // Reset counter if point is >= 1m or invalid
        consecutive_points = 0;
      }
    }

    if (alert_triggered) {
      RCLCPP_WARN(
        this->get_logger(), 
        "WARNING: Small obstacle detected! 3 consecutive returns < 1m (starting at %.2fm)", 
        first_detected_range
      );
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ScanMonitor>());
  rclcpp::shutdown();
  return 0;
}