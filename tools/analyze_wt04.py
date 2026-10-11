#!/usr/bin/env python3
"""
analyze_wt04.py - Phan tich rosbag cho PSE_ROS_WT_04 (Gazebo + teleop + odom/IMU).

So sanh 4 nguon du lieu trong moi bag:
  /odom          odometry cua plugin diff_drive (tinh tu goc banh, nhu encoder)
  /odom_wheel    odometry cua node diff_drive_odometry (code tuan 3)
  /ground_truth  pose/twist THAT cua robot trong Gazebo (plugin p3d)
  /imu           gyro + accelerometer
va lenh /cmd_vel tu teleop.

Dau ra (trong thu muc -o):
  <bag>.png       8 do thi cho moi kich ban (dung thang trong bao cao LaTeX)
  summary.csv     toan bo chi so
  summary.tex     3 bang LaTeX (can \\usepackage{booktabs,amsmath,graphicx})
  va bang tom tat in ra terminal.

Cach chay (tren may co ROS 2 Humble):
  source /opt/ros/humble/setup.bash
  python3 analyze_wt04.py ~/ros2_ws/bags/wt04_* -o ~/ros2_ws/bags/results

Neu khong co rosbag2_py (vd. may khong cai ROS), script tu dung thu vien
`rosbags` (pip install rosbags).
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

TOPICS = ('/odom', '/odom_wheel', '/ground_truth', '/imu', '/joint_states', '/cmd_vel')

LABELS = {
    'wt04_before_rotate': 'S0 Xoay trái (URDF cũ)',
    'wt04_forward': 'S1 Tiến',
    'wt04_backward': 'S2 Lùi',
    'wt04_rotate_left': 'S3 Xoay trái',
    'wt04_rotate_right': 'S4 Xoay phải',
    'wt04_arc': 'S5 Cung tròn',
}

# ---------------------------------------------------------------- doc bag --

def _storage_id(path):
    meta = Path(path) / 'metadata.yaml'
    if meta.exists():
        m = re.search(r'storage_identifier:\s*(\S+)', meta.read_text())
        if m:
            return m.group(1)
    return 'sqlite3'


def _iter_rosbag2_py(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id=_storage_id(path)),
        rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                    output_serialization_format='cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_cls = {name: get_message(types[name]) for name in TOPICS if name in types}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic in msg_cls:
            yield topic, t_ns, deserialize_message(data, msg_cls[topic])


def _iter_rosbags(path):
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(path) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        for conn, t_ns, raw in reader.messages(connections=conns):
            yield conn.topic, t_ns, typestore.deserialize_cdr(raw, conn.msgtype)


def iter_bag(path, backend):
    if backend == 'rosbag2_py':
        return _iter_rosbag2_py(path)
    if backend == 'rosbags':
        return _iter_rosbags(path)
    try:
        import rosbag2_py  # noqa: F401
        import rclpy  # noqa: F401
        return _iter_rosbag2_py(path)
    except ImportError:
        return _iter_rosbags(path)


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_bag(path, backend):
    rows = {k: [] for k in TOPICS}
    for topic, t_ns, msg in iter_bag(path, backend):
        tb = t_ns * 1e-9
        if topic in ('/odom', '/odom_wheel', '/ground_truth'):
            p, tw = msg.pose.pose, msg.twist.twist
            rows[topic].append((_stamp(msg), tb, p.position.x, p.position.y, _yaw(p.orientation),
                                tw.linear.x, tw.linear.y, tw.angular.z))
        elif topic == '/imu':
            w, a = msg.angular_velocity, msg.linear_acceleration
            rows[topic].append((_stamp(msg), tb, w.x, w.y, w.z, a.x, a.y, a.z))
        elif topic == '/joint_states':
            names = list(msg.name)
            if 'left_wheel_joint' not in names or 'right_wheel_joint' not in names:
                continue
            il, ir = names.index('left_wheel_joint'), names.index('right_wheel_joint')
            vel = list(msg.velocity)
            vl, vr = (vel[il], vel[ir]) if len(vel) > max(il, ir) else (np.nan, np.nan)
            rows[topic].append((_stamp(msg), tb, msg.position[il], msg.position[ir], vl, vr))
        elif topic == '/cmd_vel':
            rows[topic].append((tb, tb, msg.linear.x, msg.angular.z))

    data = {}
    for k, r in rows.items():
        a = np.array(r, dtype=float)
        if a.size == 0:
            data[k] = None
            continue
        a = a[np.argsort(a[:, 0], kind='stable')]
        if k != '/cmd_vel':
            _, keep = np.unique(a[:, 0], return_index=True)   # bo stamp trung lap
            a = a[keep]
        data[k] = a
    for k in ('/odom', '/ground_truth', '/imu'):
        if data[k] is None or len(data[k]) < 10:
            raise RuntimeError(f'{path}: thieu du lieu topic {k}')
    return data

# ----------------------------------------------------------- tien ich so --

def smooth(y, n):
    """Trung binh truot n mau, giu nguyen do dai (n chan hoac le)."""
    if n <= 1 or len(y) < n:
        return y.copy()
    k1 = n // 2
    k2 = n - 1 - k1
    yp = np.concatenate([np.full(k1, y[0]), y, np.full(k2, y[-1])])
    return np.convolve(yp, np.ones(n) / n, mode='valid')


def relative(x, y, yaw):
    """Bieu dien quy dao trong he toa do cua chinh pose dau tien."""
    c, s = math.cos(yaw[0]), math.sin(yaw[0])
    dx, dy = x - x[0], y - y[0]
    return c * dx + s * dy, -s * dx + c * dy, yaw - yaw[0]


def circle_fit(x, y):
    """Fit duong tron dai so (Kasa). Tra ve (cx, cy, R)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -D / 2, -E / 2
    return cx, cy, math.sqrt(max(cx ** 2 + cy ** 2 - F, 0.0))


def mean_in(t, y, t0, t1):
    m = (t >= t0) & (t <= t1)
    return float(np.mean(y[m])) if m.any() else np.nan


def phase_means(tg, gt_sig, ti, imu_raw, t0, t1):
    """Trung binh gia toc IMU va ground truth trong cung mot pha tang/giam toc.

    Pha duoc xac dinh tu ground truth: cac thoi diem trong [t0, t1] co |a_GT| > 50% dinh.
    Dung trung binh (khong dung dinh) vi plugin doi van toc banh theo bac 50 Hz con IMU
    lay mau 100 Hz, nen tung mau IMU dao dong manh quanh gia tri trung binh.
    """
    m = (tg >= t0) & (tg <= t1)
    if m.sum() < 3:
        return np.nan, np.nan
    seg, tm = gt_sig[m], tg[m]
    p = seg[np.argmax(np.abs(seg))]
    if abs(p) < 0.05:
        a, b = t0, t0 + 0.3
    else:
        idx = np.where(np.abs(seg) > 0.5 * abs(p))[0]
        a, b = tm[idx[0]], tm[idx[-1]]
        if b - a < 0.06:
            a, b = a - 0.03, b + 0.03
    return mean_in(ti, imu_raw, a, b), mean_in(tg, gt_sig, a, b)

# ------------------------------------------------------------- phan tich --

def analyze(data, wheel_radius, wheel_sep):
    od, ow, gt, imu = data['/odom'], data['/odom_wheel'], data['/ground_truth'], data['/imu']
    js, cmd = data['/joint_states'], data['/cmd_vel']

    # Thoi gian mo phong (header.stamp) lam truc chung; /cmd_vel khong co header nen
    # doi tu thoi diem ghi bag sang thoi gian mo phong qua do lech trung vi cua /odom.
    offset = float(np.median(od[:, 1] - od[:, 0]))
    t0 = od[0, 0]

    starts = [od[0, 0], gt[0, 0]] + ([ow[0, 0]] if ow is not None else [])
    ends = [od[-1, 0], gt[-1, 0]] + ([ow[-1, 0]] if ow is not None else [])
    g = od[(od[:, 0] >= max(starts)) & (od[:, 0] <= min(ends))]
    tg = g[:, 0] - t0

    def on_grid(a, col, unwrap=False):
        y = np.unwrap(a[:, col]) if unwrap else a[:, col]
        return np.interp(tg, a[:, 0] - t0, y)

    r = {'t': tg}
    # odom plugin
    r['od_x'], r['od_y'], r['od_th'] = relative(g[:, 2], g[:, 3], np.unwrap(g[:, 4]))
    r['od_v'], r['od_w'] = g[:, 5], g[:, 7]
    # odom tuan 3
    if ow is not None:
        r['ow_x'], r['ow_y'], r['ow_th'] = relative(on_grid(ow, 2), on_grid(ow, 3), on_grid(ow, 4, True))
        r['ow_v'], r['ow_w'] = on_grid(ow, 5), on_grid(ow, 7)
    # ground truth: twist cua p3d nam trong he world -> doi sang he than robot
    gt_th_abs = on_grid(gt, 4, True)
    gvx, gvy = on_grid(gt, 5), on_grid(gt, 6)
    r['gt_x'], r['gt_y'], r['gt_th'] = relative(on_grid(gt, 2), on_grid(gt, 3), gt_th_abs)
    r['gt_v'] = gvx * np.cos(gt_th_abs) + gvy * np.sin(gt_th_abs)
    r['gt_w'] = on_grid(gt, 7)
    # gia toc "that" tai goc base_footprint (trung vi tri x-y cua IMU), he than robot
    tgt = gt[:, 0] - t0
    th_gt = np.unwrap(gt[:, 4])
    axw = np.gradient(smooth(gt[:, 5], 5), tgt)
    ayw = np.gradient(smooth(gt[:, 6], 5), tgt)
    r['gt_ax'] = np.interp(tg, tgt, smooth(axw * np.cos(th_gt) + ayw * np.sin(th_gt), 5))
    r['gt_ay'] = np.interp(tg, tgt, smooth(-axw * np.sin(th_gt) + ayw * np.cos(th_gt), 5))
    # IMU
    ti = imu[:, 0] - t0
    r['imu_t'] = ti
    r['imu_wz'] = imu[:, 4]
    r['imu_ax_s'] = smooth(imu[:, 5], 10)          # 0.1 s @100 Hz (n chan: triet dao dong 50 Hz)
    r['imu_ay_s'] = smooth(imu[:, 6], 10)
    yaw_g = np.concatenate([[0.0], np.cumsum(0.5 * (imu[1:, 4] + imu[:-1, 4]) * np.diff(ti))])
    yg = np.interp(tg, ti, yaw_g)
    r['gyro_th'] = yg - yg[0]
    r['imu_wz_g'] = np.interp(tg, ti, imu[:, 4])

    # --- tach pha chuyen dong tu /cmd_vel ---
    notes = []
    if cmd is not None and len(cmd):
        ct = cmd[:, 0] - offset - t0
        moving = (np.abs(cmd[:, 2]) > 1e-6) | (np.abs(cmd[:, 3]) > 1e-6)
    else:
        ct, moving = np.array([]), np.array([], dtype=bool)
    if moving.any():
        i0 = int(np.argmax(moving))
        t_start = ct[i0]
        stops = np.where(~moving[i0:])[0]
        t_stop = ct[i0 + stops[0]] if len(stops) else tg[-1]
    else:
        notes.append('Khong thay lenh /cmd_vel khac 0 trong bag -> tach pha chuyen dong tu ground truth.')
        mv = (np.abs(r['gt_v']) > 0.05) | (np.abs(r['gt_w']) > 0.05)
        t_start = tg[np.argmax(mv)] if mv.any() else tg[0]
        t_stop = tg[len(mv) - 1 - np.argmax(mv[::-1])] if mv.any() else tg[-1]
    ws, we = t_start + 1.0, t_stop - 0.1
    if we - ws < 0.5:
        ws, we = t_start + 0.5 * (t_stop - t_start), t_stop
    mid = 0.5 * (ws + we)
    if moving.any():
        k = np.where(ct <= mid)[0]
        v_cmd, w_cmd = (cmd[k[-1], 2], cmd[k[-1], 3]) if len(k) else (np.nan, np.nan)
    else:
        v_cmd = w_cmd = np.nan
    r.update(t_start=t_start, t_stop=t_stop, ws=ws, we=we,
             cmd_t=ct, cmd_v=cmd[:, 2] if cmd is not None else np.array([]),
             cmd_w=cmd[:, 3] if cmd is not None else np.array([]))

    kind = 'straight' if abs(w_cmd) < 1e-6 else ('rotate' if abs(v_cmd) < 1e-6 else 'arc')

    # --- chi so ---
    m = {'kind': kind, 'v_cmd': v_cmd, 'w_cmd': w_cmd, 'T_motion': t_stop - t_start}
    sm = (tg >= ws) & (tg <= we)
    for key in ('od_v', 'ow_v', 'gt_v', 'od_w', 'ow_w', 'gt_w'):
        if key in r:
            m[key] = float(np.mean(r[key][sm]))
    m['imu_w'] = mean_in(ti, imu[:, 4], ws, we)

    m['dist_gt'] = float(np.sum(np.hypot(np.diff(r['gt_x']), np.diff(r['gt_y']))))
    m['dyaw_gt_deg'] = math.degrees(r['gt_th'][-1])
    e_pos = np.hypot(r['od_x'] - r['gt_x'], r['od_y'] - r['gt_y'])
    r['e_pos'] = e_pos
    m['e_pos_end_mm'] = 1000 * e_pos[-1]
    m['e_pos_max_mm'] = 1000 * e_pos.max()
    m['e_pos_end_pct'] = 100 * e_pos[-1] / m['dist_gt'] if m['dist_gt'] > 0.05 else np.nan
    m['e_yaw_end_deg'] = math.degrees(r['od_th'][-1] - r['gt_th'][-1])
    m['e_gyro_end_deg'] = math.degrees(r['gyro_th'][-1] - r['gt_th'][-1])
    mw = (tg >= t_start) & (tg <= t_stop + 0.5)
    m['rms_w_od_imu'] = float(np.sqrt(np.mean((r['od_w'][mw] - r['imu_wz_g'][mw]) ** 2)))
    if 'ow_x' in r:
        d = np.hypot(r['ow_x'] - r['od_x'], r['ow_y'] - r['od_y'])
        r['d_ow'] = d
        m['d_ow_max_mm'] = 1000 * d.max()
        m['d_ow_yaw_max_deg'] = math.degrees(np.max(np.abs(r['ow_th'] - r['od_th'])))

    m['az_mean'] = float(np.mean(imu[:, 7]))
    m['ax_imu_start'], m['ax_gt_start'] = phase_means(tg, r['gt_ax'], ti, imu[:, 5], t_start, t_start + 1.0)
    m['ax_imu_stop'], m['ax_gt_stop'] = phase_means(tg, r['gt_ax'], ti, imu[:, 5], t_stop, t_stop + 1.0)
    m['ay_imu_start'], m['ay_gt_start'] = phase_means(tg, r['gt_ay'], ti, imu[:, 6], t_start, t_start + 1.0)
    m['ax_imu_steady'] = mean_in(ti, imu[:, 5], ws, we)
    m['ay_imu_steady'] = mean_in(ti, imu[:, 6], ws, we)
    m['ay_gt_steady'] = float(np.mean(r['gt_ay'][sm]))
    m['vw_gt'] = m['gt_v'] * m['gt_w']

    rest_end = t_start - 0.2
    if rest_end - ti[0] >= 0.5:
        rm = ti <= rest_end
        for j, name in ((2, 'wx'), (3, 'wy'), (4, 'wz'), (5, 'ax')):
            m[f'sd_{name}_rest'] = float(np.std(imu[rm, j]))
    else:
        notes.append('Bag bat dau khi robot da chay -> khong co doan dung yen de do nhieu IMU.')

    if js is not None and not np.isnan(js[:, 4]).all():
        m['wl'] = mean_in(js[:, 0] - t0, js[:, 4], ws, we)
        m['wr'] = mean_in(js[:, 0] - t0, js[:, 5], ws, we)
        m['wl_exp'] = (v_cmd - w_cmd * wheel_sep / 2) / wheel_radius
        m['wr_exp'] = (v_cmd + w_cmd * wheel_sep / 2) / wheel_radius

    mm = (tg >= t_start) & (tg <= t_stop)
    if kind == 'rotate':
        disp = np.hypot(r['gt_x'][mm], r['gt_y'][mm])
        m['gt_disp_max_mm'] = 1000 * disp.max()
        if m['gt_disp_max_mm'] > 10:
            cx, cy, R = circle_fit(r['gt_x'][sm], r['gt_y'][sm])
            m['gt_circle_R_mm'], m['gt_circle_cx_mm'] = 1000 * R, 1000 * cx
            notes.append(f"Khi xoay tai cho, base_footprint THAT chay tren vong tron R = {1000*R:.1f} mm, "
                         f"tam cach diem xuat phat {1000*math.hypot(cx, cy):.1f} mm -> tam quay "
                         f"(truc banh) KHONG trung base_footprint, trong khi /odom bao dung yen.")
        else:
            notes.append(f"Khi xoay tai cho, base_footprint that chi dich toi da {m['gt_disp_max_mm']:.1f} mm "
                         f"-> tam quay trung base_footprint (URDF dung chuan).")
    if kind == 'arc':
        m['R_cmd'] = v_cmd / w_cmd
        vs = m.get('ow_v', m['gt_v'])
        m['R_odom'] = vs / m['od_w'] if abs(m['od_w']) > 1e-6 else np.nan
        _, _, m['R_gt_fit'] = circle_fit(r['gt_x'][sm], r['gt_y'][sm])

    if v_cmd < -1e-6 and m['od_v'] > 0:
        notes.append(f"/odom twist.linear.x = {m['od_v']:+.3f} DUONG trong khi lenh = {v_cmd:+.3f}: plugin "
                     f"gazebo_ros_diff_drive (che do ENCODER) tinh v = sqrt(dx^2+dy^2)/dt nen chi cho do lon. "
                     f"Pose /odom van dung chieu; /odom_wheel (tuan 3) cho v co dau = {m.get('ow_v', np.nan):+.3f}.")
    r['notes'] = notes
    return r, m

# -------------------------------------------------------------- do thi ----

C = dict(cmd='#898781', od='#2a78d6', gt='#eb6834', imu='#1baf7a', ow='#eda100')
INK, INK2, MUTED, GRID, AXIS, BAND = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#efeee9'


def plot(r, m, title, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7.5,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2,
        'axes.titlesize': 8, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'text.color': INK, 'lines.linewidth': 1.4, 'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round', 'legend.frameon': False,
    })
    t = r['t']
    # Bo cuc co dinh (khong dung constrained_layout) de chay giong nhau tren
    # matplotlib 3.5 (Ubuntu 22.04) va ban moi.
    fig, axs = plt.subplots(4, 2, figsize=(7.2, 9.6))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.05, hspace=0.62, wspace=0.3)

    def min_span(ax, span):
        lo, hi = ax.get_ylim()
        if hi - lo < span:
            ax.set_ylim(min(lo, -span / 2), max(hi, span / 2))

    def band(ax):
        ax.axvspan(r['t_start'], r['t_stop'], color=BAND, lw=0, zorder=0)

    def cmd_step(ax, vals):
        if len(r['cmd_t']):
            tt = np.concatenate([[t[0]], r['cmd_t'], [t[-1]]])
            vv = np.concatenate([[0.0], vals, [vals[-1]]])
            ax.step(tt, vv, where='post', color=C['cmd'], lw=1.1, zorder=2)

    # (a) quy dao
    ax = axs[0, 0]
    ax.plot(r['gt_x'], r['gt_y'], color=C['gt'], zorder=3)
    ax.plot(r['od_x'], r['od_y'], color=C['od'], zorder=4)
    if max(np.ptp(r['od_x']), np.ptp(r['od_y'])) < 0.005:
        ax.plot(r['od_x'][-1], r['od_y'][-1], 'x', ms=8, mew=2, color=C['od'], zorder=6)
        ax.annotate('odom đứng yên', (r['od_x'][-1], r['od_y'][-1]), xytext=(5, 5),
                    textcoords='offset points', color=INK2, fontsize=7)
    ax.plot([0], [0], 'o', ms=5, mfc=INK, mec='white', mew=1.2, zorder=5)
    ax.annotate('xuất phát', (0, 0), xytext=(5, -12), textcoords='offset points', color=INK2, fontsize=7)
    ax.set_aspect('equal', adjustable='box')
    span = max(np.ptp(r['gt_x']), np.ptp(r['gt_y']), np.ptp(r['od_x']), np.ptp(r['od_y']), 0.05)
    cx = 0.5 * (min(r['gt_x'].min(), r['od_x'].min()) + max(r['gt_x'].max(), r['od_x'].max()))
    cy = 0.5 * (min(r['gt_y'].min(), r['od_y'].min()) + max(r['gt_y'].max(), r['od_y'].max()))
    ax.set_xlim(cx - 0.65 * span, cx + 0.65 * span)
    ax.set_ylim(cy - 0.65 * span, cy + 0.65 * span)
    ax.set_title('(a) Quỹ đạo trong hệ của pose xuất phát')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # (b) sai so vi tri
    ax = axs[0, 1]
    band(ax)
    ax.plot(t, 1000 * r['e_pos'], color=C['od'], zorder=3, label='‖odom − GT‖')
    if 'd_ow' in r:
        ax.plot(t, 1000 * r['d_ow'], color=C['ow'], zorder=4, label='‖odom_wheel − odom‖')
    ax.legend(loc='upper left', fontsize=6.5, handlelength=1.4)
    ax.set_title('(b) Sai lệch vị trí')
    ax.set_ylabel('mm')
    ax.set_ylim(bottom=0)
    min_span(ax, 2.0)

    # (c) van toc dai
    ax = axs[1, 0]
    band(ax)
    cmd_step(ax, r['cmd_v'])
    ax.plot(t, r['gt_v'], color=C['gt'], zorder=3)
    ax.plot(t, r['od_v'], color=C['od'], zorder=4)
    if 'ow_v' in r:
        ax.plot(t, r['ow_v'], color=C['ow'], lw=1.0, zorder=5)
    ax.set_title('(c) Vận tốc dài v')
    ax.set_ylabel('m/s')
    min_span(ax, 0.2)

    # (d) van toc goc
    ax = axs[1, 1]
    band(ax)
    cmd_step(ax, r['cmd_w'])
    ax.plot(t, r['gt_w'], color=C['gt'], zorder=3)
    ax.plot(t, r['imu_wz_g'], color=C['imu'], zorder=4)
    ax.plot(t, r['od_w'], color=C['od'], lw=1.0, zorder=5)
    ax.set_title('(d) Vận tốc góc ω')
    ax.set_ylabel('rad/s')
    min_span(ax, 0.2)

    # (e) yaw
    ax = axs[2, 0]
    band(ax)
    ax.plot(t, np.degrees(r['gt_th']), color=C['gt'], zorder=3)
    ax.plot(t, np.degrees(r['gyro_th']), color=C['imu'], zorder=4)
    ax.plot(t, np.degrees(r['od_th']), color=C['od'], lw=1.0, zorder=5)
    ax.set_title('(e) Góc quay θ (so với lúc bắt đầu)')
    ax.set_ylabel('độ')
    min_span(ax, 2.0)

    # (f) sai so yaw
    ax = axs[2, 1]
    band(ax)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.plot(t, np.degrees(r['od_th'] - r['gt_th']), color=C['od'], zorder=3, label='odom − GT')
    ax.plot(t, np.degrees(r['gyro_th'] - r['gt_th']), color=C['imu'], zorder=4, label='tích phân gyro − GT')
    ax.legend(loc='upper left', fontsize=6.5, handlelength=1.4)
    ax.set_title('(f) Sai lệch θ so với ground truth')
    ax.set_ylabel('độ')
    min_span(ax, 0.5)

    # (g) ax, (h) ay
    for ax, key, lab in ((axs[3, 0], 'ax', '(g) Gia tốc dọc thân a_x'),
                         (axs[3, 1], 'ay', '(h) Gia tốc ngang thân a_y')):
        band(ax)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.plot(t, r[f'gt_{key}'], color=C['gt'], zorder=3)
        ax.plot(r['imu_t'], r[f'imu_{key}_s'], color=C['imu'], lw=1.1, zorder=4)
        ax.set_title(lab)
        ax.set_ylabel('m/s²')
        min_span(ax, 0.5)
        ax.set_xlabel('thời gian mô phỏng (s)')

    for ax in axs.flat[1:]:
        ax.set_xlim(t[0], t[-1])

    handles = [Line2D([], [], color=C['cmd'], lw=1.4, label='lệnh /cmd_vel'),
               Line2D([], [], color=C['od'], lw=2, label='/odom (plugin)'),
               Line2D([], [], color=C['ow'], lw=2, label='/odom_wheel (tuần 3)'),
               Line2D([], [], color=C['gt'], lw=2, label='ground truth'),
               Line2D([], [], color=C['imu'], lw=2, label='IMU (a: TB trượt 0,1 s)'),
               Patch(color=BAND, label='pha chuyển động')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.962), ncol=3, fontsize=7,
               handlelength=1.6, columnspacing=1.4)
    fig.suptitle(title, x=0.015, y=0.99, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.savefig(out_png, dpi=200, facecolor='white')
    plt.close(fig)

# ---------------------------------------------------------- bao cao -------

def f(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '--'
    if round(x, nd) == 0:
        return f'{0:.{nd}f}'
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


def write_outputs(results, outdir):
    keys = []
    for _, m, _ in results:
        for k in m:
            if k not in keys:
                keys.append(k)
    with open(outdir / 'summary.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['scenario'] + keys)
        for name, m, _ in results:
            w.writerow([name] + [m.get(k, '') for k in keys])

    def tex_row(cells):
        return ' & '.join(cells) + r' \\'

    lines = [r'% Sinh tu dong boi analyze_wt04.py -- can \usepackage{booktabs,amsmath,graphicx}', '']
    lines += [r'\begin{table}[htbp]', r'\centering\small',
              r'\caption{Giá trị xác lập trung bình trong pha chạy đều}',
              r'\label{tab:wt04_steady}',
              r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
              tex_row(['Kịch bản', r'$v_{\text{lệnh}}$', r'$v_{\text{odom}}$', r'$v_{\text{wheel}}$',
                       r'$v_{\text{GT}}$', r'$\omega_{\text{lệnh}}$', r'$\omega_{\text{odom}}$',
                       r'$\omega_{\text{IMU}}$', r'$\omega_{\text{GT}}$']),
              r'\midrule']
    for name, m, _ in results:
        lines.append(tex_row([name, f(m['v_cmd']), f(m['od_v']), f(m.get('ow_v', np.nan)), f(m['gt_v']),
                              f(m['w_cmd']), f(m['od_w']), f(m['imu_w']), f(m['gt_w'])]))
    lines += [r'\bottomrule', r'\end{tabular}}',
              r'\par\smallskip\footnotesize $v$ (m/s), $\omega$ (rad/s). GT = ground truth (plugin p3d).',
              r'\end{table}', '']

    lines += [r'\begin{table}[htbp]', r'\centering\small',
              r'\caption{Sai số odometry so với ground truth}', r'\label{tab:wt04_error}',
              r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{lrrrrrrr}', r'\toprule',
              tex_row(['Kịch bản', 'Quãng đường', r'$\Delta\theta_{\text{GT}}$', r'$e_{\text{pos}}$ cuối',
                       r'$e_{\text{pos}}$ max', r'$e_{\theta}$ cuối', r'RMS $\omega_{\text{odom}}-\omega_{\text{IMU}}$',
                       r'max $\lVert$wheel$-$odom$\rVert$']),
              tex_row(['', '(m)', '(độ)', '(mm)', '(mm)', '(độ)', '(rad/s)', '(mm)']),
              r'\midrule']
    for name, m, _ in results:
        lines.append(tex_row([name, f(m['dist_gt'], 3, False), f(m['dyaw_gt_deg'], 1), f(m['e_pos_end_mm'], 1, False),
                              f(m['e_pos_max_mm'], 1, False), f(m['e_yaw_end_deg'], 2), f(m['rms_w_od_imu'], 4, False),
                              f(m.get('d_ow_max_mm', np.nan), 2, False)]))
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']

    lines += [r'\begin{table}[htbp]', r'\centering\small',
              r'\caption{Gia tốc đo bởi IMU so với gia tốc suy ra từ ground truth}', r'\label{tab:wt04_imu}',
              r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{lrrrrrrr}', r'\toprule',
              tex_row(['Kịch bản', r'$a_x$ tăng tốc', r'$a_x$ phanh', r'$a_y$ đầu pha',
                       r'$a_y$ xác lập', r'$v\omega$ (GT)', r'$\bar a_z$', r'$\sigma_{\omega_z}$ đứng yên']),
              tex_row(['', 'IMU / GT', 'IMU / GT', 'IMU / GT', 'IMU / GT', r'(m/s$^2$)', r'(m/s$^2$)', '(rad/s)']),
              r'\midrule']
    for name, m, _ in results:
        lines.append(tex_row([name,
                              f"{f(m['ax_imu_start'], 2)} / {f(m['ax_gt_start'], 2)}",
                              f"{f(m['ax_imu_stop'], 2)} / {f(m['ax_gt_stop'], 2)}",
                              f"{f(m['ay_imu_start'], 2)} / {f(m['ay_gt_start'], 2)}",
                              f"{f(m['ay_imu_steady'], 2)} / {f(m['ay_gt_steady'], 2)}",
                              f(m['vw_gt'], 2), f(m['az_mean'], 2, False),
                              f(m.get('sd_wz_rest', np.nan), 4, False)]))
    lines += [r'\bottomrule', r'\end{tabular}}',
              r'\par\smallskip\footnotesize '
              r'Giá trị tăng tốc/phanh/đầu pha là trung bình trong pha có $|a_{\text{GT}}| > 50\%$ đỉnh.',
              r'\end{table}', '']
    (outdir / 'summary.tex').write_text('\n'.join(lines), encoding='utf-8')


def print_report(name, m, notes):
    p = print
    p(f'\n=== {name}  ({m["kind"]}) ===')
    p(f'  Lenh          : v = {f(m["v_cmd"])} m/s, w = {f(m["w_cmd"])} rad/s, '
      f'thoi gian chay {m["T_motion"]:.2f} s')
    p(f'  v xac lap     : odom {f(m["od_v"])} | odom_wheel {f(m.get("ow_v", np.nan))} | GT {f(m["gt_v"])} m/s')
    p(f'  w xac lap     : odom {f(m["od_w"])} | IMU {f(m["imu_w"])} | GT {f(m["gt_w"])} rad/s')
    p(f'  GT di duoc    : {m["dist_gt"]:.3f} m, quay {m["dyaw_gt_deg"]:+.1f} deg')
    p(f'  Sai so odom   : vi tri cuoi {m["e_pos_end_mm"]:.1f} mm '
      f'({f(m["e_pos_end_pct"], 2, False)} % quang duong), max {m["e_pos_max_mm"]:.1f} mm, '
      f'yaw cuoi {m["e_yaw_end_deg"]:+.2f} deg')
    p(f'  Gyro tich phan: sai yaw cuoi {m["e_gyro_end_deg"]:+.2f} deg | '
      f'RMS(w_odom - w_IMU) = {m["rms_w_od_imu"]:.4f} rad/s')
    if 'd_ow_max_mm' in m:
        p(f'  odom_wheel vs odom: lech vi tri max {m["d_ow_max_mm"]:.2f} mm, yaw max {m["d_ow_yaw_max_deg"]:.3f} deg')
    p(f'  IMU a_x TB pha: tang toc {f(m["ax_imu_start"], 2)} (GT {f(m["ax_gt_start"], 2)}), '
      f'phanh {f(m["ax_imu_stop"], 2)} (GT {f(m["ax_gt_stop"], 2)}) m/s^2')
    p(f'  IMU a_y       : dau pha {f(m["ay_imu_start"], 2)} (GT {f(m["ay_gt_start"], 2)}), '
      f'xac lap {f(m["ay_imu_steady"], 2)} (GT {f(m["ay_gt_steady"], 2)}; v*w = {f(m["vw_gt"], 2)}) m/s^2')
    p(f'  IMU a_z TB    : {m["az_mean"]:.3f} m/s^2')
    if 'sd_wz_rest' in m:
        p(f'  Nhieu dung yen: sd(wx, wy, wz) = {m["sd_wx_rest"]:.4f}, {m["sd_wy_rest"]:.4f}, '
          f'{m["sd_wz_rest"]:.4f} rad/s; sd(ax) = {m["sd_ax_rest"]:.4f} m/s^2')
    if 'wl' in m:
        p(f'  Banh (rad/s)  : trai {m["wl"]:+.2f} (ky vong {m["wl_exp"]:+.2f}), '
          f'phai {m["wr"]:+.2f} (ky vong {m["wr_exp"]:+.2f})')
    if 'R_cmd' in m:
        p(f'  Ban kinh cung : lenh {m["R_cmd"]:.3f} m | odom {m["R_odom"]:.3f} m | fit GT {m["R_gt_fit"]:.3f} m')
    for n in notes:
        p(f'  * {n}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+', help='thu muc bag (chua metadata.yaml)')
    ap.add_argument('-o', '--out', default='wt04_results', help='thu muc dau ra')
    ap.add_argument('--reader', choices=['auto', 'rosbag2_py', 'rosbags'], default='auto')
    ap.add_argument('--wheel-radius', type=float, default=0.05)
    ap.add_argument('--wheel-separation', type=float, default=0.35)
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args()

    outdir = Path(a.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    order = list(LABELS)
    bags = sorted((Path(b).expanduser() for b in a.bags if Path(b).expanduser().is_dir()),
                  key=lambda p: (order.index(p.name) if p.name in order else 99, p.name))
    if not bags:
        sys.exit('Khong tim thay thu muc bag nao.')

    results = []
    for bag in bags:
        name = LABELS.get(bag.name, bag.name)
        try:
            data = load_bag(bag, a.reader)
            r, m = analyze(data, a.wheel_radius, a.wheel_separation)
        except Exception as e:  # noqa: BLE001
            print(f'\n[LOI] {bag}: {e}')
            continue
        print_report(name, m, r['notes'])
        if not a.no_plot:
            plot(r, m, name, outdir / f'{bag.name}.png')
        results.append((name, m, r['notes']))

    if results:
        write_outputs(results, outdir)
        print(f'\nDa ghi: {outdir}/summary.csv, summary.tex' + ('' if a.no_plot else ' va cac file .png'))


if __name__ == '__main__':
    main()
