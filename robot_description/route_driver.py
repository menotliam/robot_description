#!/usr/bin/env python3
"""
route_driver.py - Tu lai robot theo lo trinh co dinh cho cac thi nghiem cua bao cao 05.2.

Muc dich: moi lan chay di DUNG CUNG MOT duong (lo trinh tuan 5, nguoc chieu kim dong ho
quanh khoi dao), chi khac toc do -> so sanh cong bang. Lai tay khong lap lai duoc.

Cach lai giong teleop tuan 5: di thang tung canh, dung han o goc, xoay tai cho, di tiep.
  - TURN : xoay tai cho toi huong cua canh tiep theo (omega <= w_max, gia toc goc <= a_ang)
  - DRIVE: di thang toi diem cuoi canh (v <= v_max, gia toc <= a_lin), sua lech ngang nhe
Bam duong bang /ground_truth (vi tri that tu Gazebo). Ground truth CHI dung de lai -
khong di vao slam_toolbox (slam_toolbox chi thay /scan va TF odometry nhu binh thuong).

Kich ban moi truong dong (scenario):
  none   : khong lam gi
  remove : sau vong `change_after_lap`, xoa vat 'mover'
  move   : sau vong `change_after_lap`, xoa 'mover' o P1 roi tao lai o P2
Moi su kien phat len /exp/event (std_msgs/String, JSON co thoi gian mo phong) de ghi vao bag.

Chay (thuong do experiment.launch.py goi):
  ros2 run robot_description route_driver --ros-args -p use_sim_time:=true \\
       -p v_max:=0.5 -p w_max:=1.0 -p laps:=1
"""

import json
import math

ROUTE = [(0.0, -1.8), (2.45, -1.8), (2.45, 1.3), (-2.75, 1.3), (-2.75, -1.8), (0.0, -1.8)]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ramp(target, prev, max_step):
    return max(prev - max_step, min(prev + max_step, target))


class RouteFollower:
    """Bo bam lo trinh thuan tuy (khong phu thuoc ROS) - kiem thu duoc ngoai ROS.

    step(x, y, th, dt) -> (v, w, events). events: list cac dict su kien phat sinh o buoc nay.
    Khi gap su kien 'hold' (het vong change_after_lap), follower dung yen cho toi khi resume().
    """

    TURN, DRIVE, HOLD, DONE = 'TURN', 'DRIVE', 'HOLD', 'DONE'

    def __init__(self, route=ROUTE, laps=1, overshoot=2.0, v_max=0.5, w_max=1.0,
                 a_lin=0.8, a_ang=3.0, hold_after_lap=None, pos_tol=0.01, ang_tol=0.015,
                 k_head=2.5, k_lat=2.0):
        self.v_max, self.w_max = v_max, w_max
        self.a_lin, self.a_ang = a_lin, a_ang
        self.pos_tol, self.ang_tol = pos_tol, ang_tol
        self.k_head, self.k_lat = k_head, k_lat
        self.hold_after_lap = hold_after_lap
        # Danh sach canh: (A, B, lap, la_canh_cuoi_vong)
        self.legs = []
        for lap in range(1, laps + 1):
            for i, (a, b) in enumerate(zip(route[:-1], route[1:])):
                self.legs.append((a, b, lap, i == len(route) - 2))
        if overshoot > 0:      # di tiep doc canh dau sau khi ve diem xuat phat
            a, b = route[0], route[1]
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            ov = min(overshoot, L - 0.3)
            ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
            self.legs.append((a, (a[0] + ux * ov, a[1] + uy * ov), laps + 1, False))
        self.i = 0
        self.state = self.TURN
        self.v = self.w = 0.0

    def resume(self):
        if self.state == self.HOLD:
            self.state = self.TURN if self.i < len(self.legs) else self.DONE

    @property
    def leg(self):
        return self.legs[self.i]

    def _leg_geom(self):
        (ax, ay), (bx, by), _, _ = self.leg
        L = math.hypot(bx - ax, by - ay)
        return ax, ay, (bx - ax) / L, (by - ay) / L, L, math.atan2(by - ay, bx - ax)

    def step(self, x, y, th, dt):
        events = []
        if self.state in (self.HOLD, self.DONE):
            self.v = ramp(0.0, self.v, self.a_lin * dt)
            self.w = ramp(0.0, self.w, self.a_ang * dt)
            return self.v, self.w, events

        ax, ay, ux, uy, L, th_leg = self._leg_geom()

        if self.state == self.TURN:
            e = wrap(th_leg - th)
            if abs(e) < self.ang_tol and abs(self.w) < 0.15:
                self.w = 0.0
                self.state = self.DRIVE
            else:
                w_t = math.copysign(min(self.w_max, math.sqrt(2 * 0.9 * self.a_ang * abs(e)),
                                        max(0.05, 3.0 * abs(e))), e)
                self.v = ramp(0.0, self.v, self.a_lin * dt)
                self.w = ramp(w_t, self.w, self.a_ang * dt)
                return self.v, self.w, events

        # DRIVE
        px, py = x - ax, y - ay
        along = px * ux + py * uy
        lat = ux * py - uy * px                 # > 0: robot nam ben trai duong
        d = L - along
        if d <= self.pos_tol:
            self.v, self.w = 0.0, 0.0
            _, _, lap, lap_end = self.leg
            self.i += 1
            if lap_end:
                events.append(dict(event='lap', lap=lap))
                if self.hold_after_lap is not None and lap == self.hold_after_lap:
                    events.append(dict(event='hold', lap=lap))
                    self.state = self.HOLD
                    return 0.0, 0.0, events
            if self.i >= len(self.legs):
                self.state = self.DONE
                events.append(dict(event='done'))
            else:
                self.state = self.TURN
            return 0.0, 0.0, events
        v_t = min(self.v_max, math.sqrt(2 * 0.9 * self.a_lin * d))
        e = wrap(th_leg - th)
        w_lim = min(self.w_max, 1.0)
        w_t = max(-w_lim, min(w_lim, self.k_head * e - self.k_lat * lat))
        self.v = ramp(v_t, self.v, self.a_lin * dt)
        self.w = ramp(w_t, self.w, self.a_ang * dt)
        return self.v, self.w, events


# ---------------------------------------------------------------- ROS node --

def mover_sdf(name, size, height, material):
    return f"""<?xml version="1.0"?>
<sdf version="1.6"><model name="{name}"><static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry><box><size>{size} {size} {height}</size></box></geometry></collision>
    <visual name="visual"><geometry><box><size>{size} {size} {height}</size></box></geometry>
      <material><script><uri>file://media/materials/scripts/gazebo.material</uri>
      <name>{material}</name></script></material></visual>
  </link></model></sdf>"""


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, Pose
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from gazebo_msgs.srv import DeleteEntity, SpawnEntity
    from tf2_ros import Buffer, TransformListener

    class RouteDriver(Node):
        def __init__(self):
            super().__init__('route_driver')
            p = self.declare_parameter
            self.v_max = p('v_max', 0.5).value
            self.w_max = p('w_max', 1.0).value
            self.laps = p('laps', 1).value
            self.overshoot = p('overshoot', 2.0).value
            self.wait_start = p('wait_start', 3.0).value
            self.idle_after = p('idle_after', 5.0).value
            self.scenario = p('scenario', 'none').value
            self.change_after_lap = p('change_after_lap', 1).value
            self.mover = dict(name=p('mover_name', 'mover').value,
                              size=p('mover_size', 0.5).value,
                              height=p('mover_height', 0.5).value,
                              material=p('mover_material', 'Gazebo/Purple').value,
                              p1=list(p('mover_p1', [1.2, 2.3]).value),
                              p2=list(p('mover_p2', [-1.5, -2.5]).value))
            if self.scenario not in ('none', 'remove', 'move'):
                raise ValueError(f'scenario khong hop le: {self.scenario}')

            hold = self.change_after_lap if self.scenario != 'none' else None
            self.f = RouteFollower(laps=self.laps, overshoot=self.overshoot, v_max=self.v_max,
                                   w_max=self.w_max, hold_after_lap=hold)
            self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
            self.ev_pub = self.create_publisher(String, 'exp/event', 10)
            self.create_subscription(Odometry, 'ground_truth', self.on_gt, 20)
            self.tf_buf = Buffer()
            self.tf_lis = TransformListener(self.tf_buf, self)
            self.del_cli = self.create_client(DeleteEntity, 'delete_entity')
            self.spawn_cli = self.create_client(SpawnEntity, 'spawn_entity')

            self.pose = None
            self.finished = False
            self.phase = 'WAIT'          # WAIT -> SETTLE -> RUN -> (CHANGE) -> IDLE -> END
            self.t_phase = None
            self.t_last = None
            self.futures = []
            self.change_steps = []
            self.create_timer(0.02, self.tick)
            self.get_logger().info(
                f'route_driver: v_max={self.v_max} w_max={self.w_max} laps={self.laps} '
                f'scenario={self.scenario} overshoot={self.overshoot} idle_after={self.idle_after}')

        # ------------------------------------------------------------ tien ich
        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def emit(self, **kw):
            kw['t'] = round(self.now(), 3)
            if self.pose is not None:
                kw.setdefault('x', round(self.pose[0], 4))
                kw.setdefault('y', round(self.pose[1], 4))
            self.ev_pub.publish(String(data=json.dumps(kw)))
            self.get_logger().info(f'[event] {kw}')

        def on_gt(self, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            th = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.pose = (p.x, p.y, th, self.now())

        def send(self, v, w):
            t = Twist()
            t.linear.x, t.angular.z = float(v), float(w)
            self.cmd_pub.publish(t)

        def set_phase(self, ph):
            self.phase, self.t_phase = ph, self.now()

        # ------------------------------------------------------------ kich ban
        def start_change(self):
            m = self.mover
            self.change_steps = [('delete', None)]
            if self.scenario == 'move':
                self.change_steps.append(('spawn', m['p2']))
            self.set_phase('CHANGE')
            self.futures = []

        def run_change(self):
            m = self.mover
            if self.futures:
                if not all(f.done() for f in self.futures):
                    return
                kind, arg = self.change_steps.pop(0)
                res = self.futures[0].result()
                ok = bool(getattr(res, 'success', False))
                if kind == 'delete':
                    self.emit(event='remove', obj=m['name'], ox=m['p1'][0], oy=m['p1'][1],
                              size=m['size'], ok=ok)
                else:
                    self.emit(event='spawn', obj=m['name'], ox=arg[0], oy=arg[1],
                              size=m['size'], ok=ok)
                if not ok:
                    self.get_logger().error(f'{kind} that bai: {getattr(res, "status_message", "")}')
                self.futures = []
                if not self.change_steps:
                    self.set_phase('RESUME_WAIT')
                return
            kind, arg = self.change_steps[0]
            if kind == 'delete':
                if not self.del_cli.service_is_ready():
                    return
                req = DeleteEntity.Request()
                req.name = m['name']
                self.futures = [self.del_cli.call_async(req)]
            else:
                if not self.spawn_cli.service_is_ready():
                    return
                req = SpawnEntity.Request()
                req.name = m['name']
                req.xml = mover_sdf(m['name'], m['size'], m['height'], m['material'])
                pose = Pose()
                pose.position.x, pose.position.y = float(arg[0]), float(arg[1])
                pose.position.z = m['height'] / 2.0
                req.initial_pose = pose
                req.reference_frame = 'world'
                self.futures = [self.spawn_cli.call_async(req)]

        # ------------------------------------------------------------ vong lap
        def tick(self):
            now = self.now()
            if self.phase == 'WAIT':
                ready = self.pose is not None and self.tf_buf.can_transform(
                    'map', 'odom', rclpy.time.Time())
                if ready:
                    self.set_phase('SETTLE')
                    self.get_logger().info('Da co ground truth va TF map->odom; dung yen '
                                           f'{self.wait_start} s roi bat dau')
                return
            if self.phase == 'SETTLE':
                if now - self.t_phase >= self.wait_start:
                    self.emit(event='start', v_max=self.v_max, w_max=self.w_max, laps=self.laps,
                              scenario=self.scenario)
                    self.t_last = now
                    self.set_phase('RUN')
                return
            if self.phase == 'CHANGE':
                self.send(0.0, 0.0)
                self.run_change()
                return
            if self.phase == 'RESUME_WAIT':
                self.send(0.0, 0.0)
                if now - self.t_phase >= 1.0:
                    self.f.resume()
                    self.t_last = now
                    self.set_phase('RUN')
                return
            if self.phase == 'IDLE':
                if now - self.t_phase >= self.idle_after:
                    self.emit(event='end')
                    self.set_phase('END')
                    self.finished = True
                return
            if self.phase != 'RUN':
                return

            if self.pose is None or now - self.pose[3] > 1.0:
                self.send(0.0, 0.0)            # mat ground truth -> dung
                return
            dt = max(1e-3, min(0.1, now - self.t_last))
            self.t_last = now
            v, w, events = self.f.step(self.pose[0], self.pose[1], self.pose[2], dt)
            self.send(v, w)
            for ev in events:
                if ev['event'] == 'hold':
                    self.start_change()
                elif ev['event'] == 'done':
                    self.send(0.0, 0.0)
                    self.emit(event='done', idle_after=self.idle_after)
                    self.set_phase('IDLE')
                else:
                    self.emit(**ev)

    rclpy.init(args=args)
    node = RouteDriver()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(5):                      # cho message 'end' ra khoi publisher
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.send(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
