import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster


def quaternion_from_euler(roll, pitch, yaw):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


class DiffDriveOdometry(Node):
    def __init__(self):
        super().__init__('diff_drive_odometry')

        # Tham số hình học + cấu hình (mặc định giữ nguyên hành vi tuần 3)
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation', 0.35)
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')

        self.R = self.get_parameter('wheel_radius').value
        self.L = self.get_parameter('wheel_separation').value
        odom_topic = self.get_parameter('odom_topic').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_time = None
        self.last_left_pos = None
        self.last_right_pos = None

        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.get_logger().info(
            f'DiffDrive Odometry: R={self.R}, L={self.L}, topic=/{odom_topic}, '
            f'publish_tf={self.publish_tf}')

    def joint_state_callback(self, msg: JointState):
        try:
            left_idx = msg.name.index('left_wheel_joint')
            right_idx = msg.name.index('right_wheel_joint')
        except ValueError:
            return

        # Ưu tiên timestamp của message (chính là thời điểm "encoder" đo)
        stamp = Time.from_msg(msg.header.stamp)
        current_time = stamp if stamp.nanoseconds > 0 else self.get_clock().now()

        left_pos = msg.position[left_idx]
        right_pos = msg.position[right_idx]

        if self.last_left_pos is None:
            self.last_left_pos, self.last_right_pos = left_pos, right_pos
            self.last_time = current_time
            return

        dt = (current_time - self.last_time).nanoseconds / 1e9

        delta_s_left = (left_pos - self.last_left_pos) * self.R
        delta_s_right = (right_pos - self.last_right_pos) * self.R
        self.last_left_pos, self.last_right_pos = left_pos, right_pos

        delta_s = (delta_s_right + delta_s_left) / 2.0
        delta_theta = (delta_s_right - delta_s_left) / self.L

        # Midpoint integration (giữ nguyên như tuần 3)
        self.x += delta_s * math.cos(self.theta + delta_theta / 2.0)
        self.y += delta_s * math.sin(self.theta + delta_theta / 2.0)
        self.theta += delta_theta

        v = delta_s / dt if dt > 0 else 0.0
        w = delta_theta / dt if dt > 0 else 0.0
        q = quaternion_from_euler(0.0, 0.0, self.theta)

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = current_time.to_msg()
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation = q
            self.tf_broadcaster.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.odom_pub.publish(odom)

        self.last_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = DiffDriveOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
