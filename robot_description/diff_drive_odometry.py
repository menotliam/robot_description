import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster

def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q

class DiffDriveOdometry(Node):
    def __init__(self):
        super().__init__('diff_drive_odometry')

        # Khai báo thông số hình học xe
        self.R = 0.05       # Bán kính bánh xe (m)
        self.L = 0.35       # Khoảng cách 2 bánh xe (m)

        # Trạng thái odometry (x, y, theta)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_time = self.get_clock().now()
        self.last_left_pos = None
        self.last_right_pos = None

        # Khởi tạo Publisher và TF Broadcaster
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Đăng ký nhận dữ liệu joint_states
        self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.get_logger().info('DiffDrive Odometry Node Initialized.')

    def joint_state_callback(self, msg: JointState):
        try:
            left_idx = msg.name.index('left_wheel_joint')
            right_idx = msg.name.index('right_wheel_joint')
        except ValueError:
            return

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9

        left_pos = msg.position[left_idx]
        right_pos = msg.position[right_idx]

        if self.last_left_pos is None or self.last_right_pos is None:
            self.last_left_pos = left_pos
            self.last_right_pos = right_pos
            self.last_time = current_time
            return

        # Tính độ dịch chuyển góc quay của bánh (rad)
        delta_phi_left = left_pos - self.last_left_pos
        delta_phi_right = right_pos - self.last_right_pos

        self.last_left_pos = left_pos
        self.last_right_pos = right_pos

        # Tính quãng đường từng bánh di chuyển (m)
        delta_s_left = delta_phi_left * self.R
        delta_s_right = delta_phi_right * self.R

        # Tính chuyển động vi sai của tâm robot
        delta_s = (delta_s_right + delta_s_left) / 2.0
        delta_theta = (delta_s_right - delta_s_left) / self.L

        # Cập nhật vị trí toàn cục (Forward Euler Approximation)
        self.x += delta_s * math.cos(self.theta + delta_theta / 2.0)
        self.y += delta_s * math.sin(self.theta + delta_theta / 2.0)
        self.theta += delta_theta

        # Vận tốc tuyến tính và góc
        v = delta_s / dt if dt > 0 else 0.0
        w = delta_theta / dt if dt > 0 else 0.0

        q = quaternion_from_euler(0, 0, self.theta)

        # 1. Phát tán TF: odom -> base_footprint
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation = q
        self.tf_broadcaster.sendTransform(t)

        # 2. Phát tán Topic /odom
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
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
