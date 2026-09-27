import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_dir = get_package_share_directory('robot_description')
    robot_desc = xacro.process_file(
        os.path.join(pkg_dir, 'urdf', 'robot.urdf.xacro')).toxml()

    # Có file .rviz thì nạp, chưa có thì mở RViz trống (xử lý hạn chế 5.4 ở Phần 2)
    rviz_cfg = os.path.join(pkg_dir, 'rviz', 'sim.rviz')
    rviz_args = ['-d', rviz_cfg] if os.path.exists(rviz_cfg) else []

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'gui': LaunchConfiguration('gui')}.items())

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': True}],
        output='screen')

    spawn = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'diff_bot', '-z', '0.01'],
        output='screen')

    wheel_odom = Node(
        package='robot_description', executable='diff_drive_odometry',
        parameters=[{'use_sim_time': True,
                     'odom_topic': 'odom_wheel',
                     'publish_tf': False}],
        output='screen')

    rviz = Node(
        package='rviz2', executable='rviz2', arguments=rviz_args,
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')), output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        gazebo, rsp, spawn, wheel_odom, rviz,
    ])
