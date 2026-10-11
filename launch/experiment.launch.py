"""experiment.launch.py - Mot lan chay thi nghiem cua bao cao 05.2, tu dong tu dau den cuoi.

    ros2 launch robot_description experiment.launch.py run:=e1_default v:=0.5 w:=1.0

Lam lan luot:
  1. slam.launch.py (Gazebo + robot + slam_toolbox online async + RViz2), use_sim_time
  2. ros2 bag record ngay tu dau vao <bag_dir>/<run>  (them hau to _2, _3... neu da ton tai)
  3. ros2 param dump /slam_toolbox -> <bag_dir>/<run>_slam_params.yaml (ghi lai cau hinh thuc te)
  4. route_driver (neu driver:=true): cho TF map->odom, dung yen 3 s, lai theo lo trinh
  5. Khi route_driver ket thuc (sau idle_after giay dung yen) -> tat toan bo launch;
     ros2 bag nhan SIGINT va ghi metadata day du.
Voi driver:=false (lai tay bang teleop), bam Ctrl+C khi xong.

Doi so: run, v, w, laps, overshoot, idle_after, scenario (none|remove|move|moving), change_after_lap,
        mover_speed, mover_lane, wheel_collision (sphere|cylinder), driver, gui, rviz, bag_dir.
scenario khac none tu dung world wt05_room_mover.world (co vat 'mover' tai P1 va plugin
gazebo_ros_state cho dich vu /set_entity_state).
Kich ban C (vat chay qua lai lien tuc, dung + xoa sau vong change_after_lap):
    ros2 launch robot_description experiment.launch.py run:=e3c_m4 scenario:=moving \
        laps:=12 change_after_lap:=4 mover_speed:=0.5
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# /pose: slam_toolbox phat MOT lan cho moi scan duoc nhan lam nut (dau thoi gian = scan do).
# /exp/mover: vi tri vat o kich ban C.
TOPICS = ['/scan', '/tf', '/tf_static', '/odom', '/ground_truth', '/map',
          '/slam_toolbox/graph_visualization', '/pose', '/cmd_vel', '/exp/event', '/exp/mover']


def _setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    pkg_dir = get_package_share_directory('robot_description')
    run = arg('run')
    scenario = arg('scenario')
    bag_dir = os.path.expanduser(arg('bag_dir'))
    os.makedirs(bag_dir, exist_ok=True)
    bag_path = os.path.join(bag_dir, run)
    k = 2
    while os.path.exists(bag_path):
        bag_path = os.path.join(bag_dir, f'{run}_{k}')
        k += 1
    name = os.path.basename(bag_path)
    world = arg('world') or os.path.join(
        pkg_dir, 'worlds', 'wt05_room_mover.world' if scenario != 'none' else 'wt05_room.world')

    actions = [LogInfo(msg=f'[EXP] run={name} world={os.path.basename(world)} -> {bag_path}')]

    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_dir, 'launch', 'slam.launch.py')),
        launch_arguments={'world': world, 'gui': arg('gui'), 'rviz': arg('rviz'),
                          'wheel_collision': arg('wheel_collision')}.items()))

    actions.append(ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', bag_path] + TOPICS, output='screen'))

    params_file = os.path.join(bag_dir, f'{name}_slam_params.yaml')
    actions.append(ExecuteProcess(
        cmd=['bash', '-c', f'sleep 20; ros2 param dump /slam_toolbox > "{params_file}" 2>&1; '
                           f'echo "[EXP] Da ghi {params_file}"'],
        output='screen'))

    if arg('driver').lower() in ('true', '1'):
        driver = Node(
            package='robot_description', executable='route_driver', name='route_driver',
            output='screen',
            parameters=[{'use_sim_time': True,
                         'v_max': float(arg('v')), 'w_max': float(arg('w')),
                         'laps': int(arg('laps')), 'overshoot': float(arg('overshoot')),
                         'idle_after': float(arg('idle_after')), 'scenario': scenario,
                         'change_after_lap': int(arg('change_after_lap')),
                         'mover_speed': float(arg('mover_speed')),
                         'mover_lane': [float(v) for v in arg('mover_lane').split(',')]}])
        actions.append(TimerAction(period=8.0, actions=[driver]))
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=driver,
            on_exit=[LogInfo(msg='[EXP] route_driver xong -> tat launch, ros2 bag ghi metadata'),
                     EmitEvent(event=Shutdown(reason='experiment finished'))])))
    return actions


def generate_launch_description():
    args = [
        ('run', 'exp'), ('v', '0.5'), ('w', '1.0'), ('laps', '1'), ('overshoot', '2.0'),
        ('idle_after', '5.0'), ('scenario', 'none'), ('change_after_lap', '1'),
        ('mover_speed', '0.5'), ('mover_lane', '0.2,2.3,2.2,2.3'),
        ('wheel_collision', 'sphere'), ('driver', 'true'), ('gui', 'true'), ('rviz', 'true'),
        ('bag_dir', '~/ros2_ws/bags/wt052'), ('world', ''),
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d) for n, d in args] + [OpaqueFunction(function=_setup)])
