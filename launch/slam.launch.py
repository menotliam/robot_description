"""slam.launch.py - PSE_ROS_WT_05: Gazebo + robot + slam_toolbox (online async) + RViz2.

    ros2 launch robot_description slam.launch.py
    ros2 run teleop_twist_keyboard teleop_twist_keyboard        # terminal thu hai

Tuy chon:
    gui:=false rviz:=false        chay khong giao dien (nhu tuan 4)
    slam:=false                   chi mo Gazebo + robot (de kiem tra /scan truoc)
    wheel_collision:=cylinder     collision banh hinh tru cu -> odometry quay sai ~8,6%
                                  (thi nghiem: SLAM bu sai so odometry the nao)
    x:= y:= yaw:=                 vi tri spawn trong world (mac dinh khop tools/wt05_world.py)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, LogInfo,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_dir = get_package_share_directory('robot_description')
    xacro_file = os.path.join(pkg_dir, 'urdf', 'robot.urdf.xacro')

    # xacro chay luc launch (Command) thay vi xacro.process_file() luc doc file launch,
    # de doi so wheel_collision:=... truyen duoc vao xacro.
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file,
                 ' wheel_collision:=', LaunchConfiguration('wheel_collision')]),
        value_type=str)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': LaunchConfiguration('world'),
                          'gui': LaunchConfiguration('gui')}.items())

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen')

    spawn = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'diff_bot',
                   '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'),
                   '-z', '0.01', '-Y', LaunchConfiguration('yaw')],
        output='screen')

    # Dung launch file cua chinh slam_toolbox: no tu xu ly cac doi so rieng cua tung
    # phien ban (vd. lifecycle/autostart o cac ban Humble moi).
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')),
        launch_arguments={'use_sim_time': 'true',
                          'slam_params_file': LaunchConfiguration('slam_params_file')}.items(),
        condition=IfCondition(LaunchConfiguration('slam')))

    rviz = Node(
        package='rviz2', executable='rviz2',
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'slam.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')), output='screen')

    # slam_toolbox va RViz chi khoi dong SAU KHI robot da vao Gazebo (spawn_entity.py
    # thoat): luc do /clock, /scan va TF odom -> base_footprint da san sang.
    start_after_spawn = RegisterEventHandler(OnProcessExit(
        target_action=spawn,
        on_exit=[LogInfo(msg='[WT05] Robot da spawn -> khoi dong slam_toolbox va RViz2'),
                 slam, rviz]))

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=os.path.join(
            pkg_dir, 'worlds', 'wt05_room.world')),
        DeclareLaunchArgument('slam_params_file', default_value=os.path.join(
            pkg_dir, 'config', 'mapper_params_online_async.yaml')),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('slam', default_value='true'),
        DeclareLaunchArgument('wheel_collision', default_value='sphere'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='-1.8'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        gazebo, rsp, spawn, start_after_spawn,
    ])
