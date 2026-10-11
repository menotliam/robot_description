#!/usr/bin/env python3
"""
analyze_wt05.py - Phan tich rosbag ghi trong luc lap ban do (PSE_ROS_WT_05).

Tra loi bang so do cho muc "Verify data" cua de:
  /scan                  tan so, so tia, ti le tia hop le, frame_id
  TF map -> odom         slam_toolbox phat; so lan gia tri thay doi = so lan hieu chinh
  vi tri uoc luong       pose SLAM = (map -> odom) * (odom -> base_footprint)
                         so voi ground truth (plugin p3d) va so voi odometry thuan
  loop closure           cac buoc nhay lon nhat cua map -> odom, doi chieu voi
                         thoi diem robot quay lai gan diem xuat phat
  hai pha                tach pha DI CHUYEN va pha DUNG YEN (robot "bo" khi dung yen,
                         slam_toolbox khong them nut moi) - chi so chinh lay o pha di chuyen

Frame: frame map cua slam_toolbox bat dau trung frame odom (= tu the robot luc plugin
diff_drive khoi dong). Ground truth (frame world) duoc dua ve frame nay bang
    T_world_odom0 = GT(t0) * inverse(odom(t0))
nen ca ba quy dao (GT, odom, SLAM) cung nam trong mot he toa do va so sanh truc tiep.

Cach chay (sau khi source ROS 2 Humble):
  ros2 bag record -o ~/ros2_ws/bags/wt05_map /scan /tf /tf_static /odom /ground_truth /map
  python3 tools/analyze_wt05.py ~/ros2_ws/bags/wt05_map -o ~/ros2_ws/bags/results_wt05 \
          --map ~/ros2_ws/maps/wt05_room.yaml
Khong co rosbag2_py thi script tu dung thu vien `rosbags` (pip install rosbags).

Ve lai tu CSV quy dao da xuat (khong can bag; map->odom suy tu SLAM * odom^-1):
  python3 tools/analyze_wt05.py ~/ros2_ws/bags/results_wt05/wt05_map_traj.csv -o OUT --map MAP.yaml
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wt05_world as W  # noqa: E402

TOPICS = ('/odom', '/ground_truth', '/tf', '/scan')


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


def _t(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_bag(path, backend, map_frame='map', odom_frame='odom'):
    odom, gt, m2o, scans = [], [], [], []
    scan_info = None
    for topic, _t_ns, msg in iter_bag(path, backend):
        if topic in ('/odom', '/ground_truth'):
            p = msg.pose.pose
            (odom if topic == '/odom' else gt).append(
                (_t(msg.header.stamp), p.position.x, p.position.y, _yaw(p.orientation)))
        elif topic == '/tf':
            for tr in msg.transforms:
                if tr.header.frame_id.lstrip('/') == map_frame and \
                        tr.child_frame_id.lstrip('/') == odom_frame:
                    tt = tr.transform
                    m2o.append((_t(tr.header.stamp), tt.translation.x, tt.translation.y,
                                _yaw(tt.rotation)))
        elif topic == '/scan':
            r = np.asarray(msg.ranges, dtype=float)
            valid = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
            scans.append((_t(msg.header.stamp), len(r), int(valid.sum())))
            if scan_info is None:
                scan_info = dict(frame_id=msg.header.frame_id, angle_min=msg.angle_min,
                                 angle_max=msg.angle_max, angle_increment=msg.angle_increment,
                                 range_min=msg.range_min, range_max=msg.range_max, n=len(r))

    def arr(rows, name, need=10):
        a = np.array(rows, dtype=float)
        if len(a) < need:
            raise RuntimeError(f'{path}: thieu du lieu {name} ({len(a)} message)')
        a = a[np.argsort(a[:, 0], kind='stable')]
        return a

    odom = arr(odom, '/odom')
    gt = arr(gt, '/ground_truth')
    odom = odom[np.unique(odom[:, 0], return_index=True)[1]]
    gt = gt[np.unique(gt[:, 0], return_index=True)[1]]
    m2o = arr(m2o, 'TF map->odom (slam_toolbox co chay khong? co ghi /tf khong?)', need=1)
    scans = np.array(scans, dtype=float) if scans else None
    return dict(odom=odom, gt=gt, m2o=m2o, scans=scans, scan_info=scan_info)


# ----------------------------------------------------------- hinh hoc 2D --

def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    """a * b cho mang tu the (..., 3)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1],
                     a[..., 1] + s * b[..., 0] + c * b[..., 1],
                     wrap(a[..., 2] + b[..., 2])], axis=-1)


def inverse(a):
    a = np.asarray(a, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([-(c * a[..., 0] + s * a[..., 1]),
                     -(-s * a[..., 0] + c * a[..., 1]),
                     wrap(-a[..., 2])], axis=-1)


def interp_pose(t_src, pose_src, t):
    """Noi suy tuyen tinh tu the (yaw da unwrap)."""
    yaw = np.unwrap(pose_src[:, 2])
    return np.stack([np.interp(t, t_src, pose_src[:, 0]), np.interp(t, t_src, pose_src[:, 1]),
                     wrap(np.interp(t, t_src, yaw))], axis=-1)


# ----------------------------------------------------------- phan tich --

def motion_phase(t, P_odom, v_min=0.02, w_min=0.05, gap=3.0):
    """Khoang [t0, t1] robot duoc LAI (tu toc do cua odometry - tuc banh xe dang quay
    theo lenh). Bo qua cac lan dung ngan < gap giay (dung o goc de xoay)."""
    dt = np.diff(t)
    v = np.r_[0, np.hypot(np.diff(P_odom[:, 0]), np.diff(P_odom[:, 1])) / dt]
    w = np.r_[0, np.abs(np.diff(np.unwrap(P_odom[:, 2]))) / dt]
    idx = np.flatnonzero((v > v_min) | (w > w_min))
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1])


def prepare_bag(d, transform_timeout=0.2):
    odom, gt, m2o = d['odom'], d['gt'], d['m2o']
    t = odom[:, 0]
    keep = (t >= gt[0, 0]) & (t <= gt[-1, 0])
    t, P_odom = t[keep], odom[keep, 1:4]
    P_gt_w = interp_pose(gt[:, 0], gt[:, 1:4], t)

    # Frame "odom0" = goc odom trong world, suy tu mau dau tien
    T_w_o0 = compose(P_gt_w[0], inverse(P_odom[0]))
    P_gt = compose(inverse(T_w_o0)[None, :], P_gt_w)

    # map->odom: slam_toolbox dong dau = thoi diem scan + transform_timeout, va phat lai
    # 50 Hz voi cung dau thoi gian -> moi dau thoi gian phan biet ung voi mot scan
    # (10 Hz); gia tri hop le tu thoi diem scan (stamp - transform_timeout).
    _, idx = np.unique(m2o[:, 0], return_index=True)
    upd = m2o[np.sort(idx)]
    upd = upd[np.argsort(upd[:, 0], kind='stable')]
    t_upd = upd[:, 0] - transform_timeout
    k = np.searchsorted(t_upd, t, side='right') - 1
    has = k >= 0
    M = np.where(has[:, None], upd[np.clip(k, 0, None), 1:4], np.nan)
    return dict(t=t, P_gt=P_gt, P_odom=P_odom, M=M, t_upd=t_upd, upd=upd[:, 1:4],
                T_w_o0=T_w_o0, scan=scan_stats(d), source='bag')


def prepare_traj(csv_path, spawn=W.SPAWN):
    """Doc lai CSV quy dao (frame map). map->odom = SLAM * odom^-1 tai moi mau."""
    a = np.genfromtxt(csv_path, delimiter=',', names=True)
    t = a['t']
    P_gt = np.c_[a['gt_x'], a['gt_y'], a['gt_yaw']]
    P_odom = np.c_[a['odom_x'], a['odom_y'], a['odom_yaw']]
    P_slam = np.c_[a['slam_x'], a['slam_y'], a['slam_yaw']]
    M = compose(P_slam, inverse(P_odom))
    # CSV lam tron 1e-4 -> chi coi la thay doi khi lech > 5e-4 (0,5 mm / 0,03 do)
    ch = np.r_[True, (np.abs(np.diff(M, axis=0)) > 5e-4).any(axis=1)]
    return dict(t=t, P_gt=P_gt, P_odom=P_odom, M=M, t_upd=t[ch], upd=M[ch],
                T_w_o0=np.array(spawn, float), scan=None, source='traj')


def scan_stats(d):
    sc = d.get('scans')
    if sc is None or len(sc) <= 2:
        return None
    dt = np.diff(sc[:, 0])
    return dict(n_msgs=int(len(sc)), rate=float((len(sc) - 1) / (sc[-1, 0] - sc[0, 0])),
                dt_std_ms=float(dt.std() * 1000),
                valid_ratio=float(sc[:, 2].sum() / sc[:, 1].sum() * 100),
                **(d.get('scan_info') or {}))


def evaluate(p, jump_cm=2.0, jump_deg=0.15):
    t, P_gt, P_odom, M = p['t'], p['P_gt'], p['P_odom'], p['M']
    P_slam = compose(M, P_odom)

    def errs(P):
        e = np.hypot(P[:, 0] - P_gt[:, 0], P[:, 1] - P_gt[:, 1])
        h = np.degrees(wrap(P[:, 2] - P_gt[:, 2]))
        return e, h

    e_odom, h_odom = errs(P_odom)
    e_slam, h_slam = errs(P_slam)

    step = np.hypot(np.diff(P_gt[:, 0]), np.diff(P_gt[:, 1]))
    s = np.r_[0, np.cumsum(step)]
    d_start = np.hypot(P_gt[:, 0] - P_gt[0, 0], P_gt[:, 1] - P_gt[0, 1])
    returns = []
    near = (d_start < 0.5) & (s > 5.0)
    if near.any():
        edges = np.flatnonzero(np.diff(np.r_[0, near.astype(int)]) == 1)
        returns = [(float(t[i] - t[0]), float(s[i])) for i in edges]

    # Buoc nhay cua map->odom giua hai gia tri lien tiep
    t_upd, U = p['t_upd'], p['upd']
    dv = np.diff(U, axis=0)
    jump_t = np.hypot(dv[:, 0], dv[:, 1]) * 100
    jump_y = np.degrees(np.abs(wrap(dv[:, 2])))
    n_changes = int(((jump_t > 0.05) | (jump_y > 0.01)).sum())
    jumps = []
    for i in np.flatnonzero((jump_t > jump_cm) | (jump_y > jump_deg)):
        tj = t_upd[i + 1]
        j = min(np.searchsorted(t, tj), len(t) - 1)
        jumps.append(dict(t=float(tj - t[0]), dxy_cm=float(jump_t[i]), dyaw_deg=float(jump_y[i]),
                          d_start=float(d_start[j]), s=float(s[j]),
                          e_slam_before=float(e_slam[max(j - 1, 0)] * 100),
                          e_slam_after=float(e_slam[min(j + 25, len(t) - 1)] * 100)))
    jumps.sort(key=lambda j: -(j['dxy_cm'] + 5 * j['dyaw_deg']))   # lon nhat truoc

    valid = np.isfinite(e_slam)
    turned = np.degrees(np.abs(np.diff(np.unwrap(P_gt[:, 2]))))
    summ = dict(
        duration=float(t[-1] - t[0]), distance=float(s[-1]), turned_deg=float(turned.sum()),
        n_stamps=int(len(U)), n_changes=n_changes,
        first_m2o=tuple(float(v) for v in U[0]), final_m2o=tuple(float(v) for v in U[-1]),
        odom_final_cm=float(e_odom[-1] * 100), odom_final_deg=float(h_odom[-1]),
        slam_final_cm=float(e_slam[valid][-1] * 100), slam_final_deg=float(h_slam[valid][-1]),
        odom_max_cm=float(e_odom.max() * 100), slam_max_cm=float(e_slam[valid].max() * 100),
        odom_rms_cm=float(np.sqrt(np.mean(e_odom ** 2)) * 100),
        slam_rms_cm=float(np.sqrt(np.mean(e_slam[valid] ** 2)) * 100),
        odom_max_deg=float(np.abs(h_odom).max()), slam_max_deg=float(np.abs(h_slam[valid]).max()))

    # Tach pha di chuyen / dung yen
    ph = motion_phase(t, P_odom)
    if ph is not None:
        i0, i1 = ph
        m = np.zeros(len(t), bool)
        m[i0:i1 + 1] = True
        m &= valid
        summ.update(
            move_t0=float(t[i0] - t[0]), move_t1=float(t[i1] - t[0]),
            move_distance=float(s[i1] - s[i0]), move_turned_deg=float(turned[i0:i1].sum()),
            move_odom_end_cm=float(e_odom[i1] * 100), move_slam_end_cm=float(e_slam[i1] * 100),
            move_odom_end_deg=float(h_odom[i1]), move_slam_end_deg=float(h_slam[i1]),
            move_odom_max_cm=float(e_odom[m].max() * 100), move_slam_max_cm=float(e_slam[m].max() * 100),
            move_odom_rms_cm=float(np.sqrt(np.mean(e_odom[m] ** 2)) * 100),
            move_slam_rms_cm=float(np.sqrt(np.mean(e_slam[m] ** 2)) * 100),
            move_odom_max_deg=float(np.abs(h_odom[m]).max()),
            move_slam_max_deg=float(np.abs(h_slam[m]).max()),
            move_m2o=tuple(float(v) for v in M[i1]))
        if len(t) - 1 - i1 > 50:      # pha dung yen sau khi lai
            j = len(t) - 1
            summ.update(
                idle_s=float(t[j] - t[i1]),
                idle_gt_cm=float(np.hypot(*(P_gt[j, :2] - P_gt[i1, :2])) * 100),
                idle_gt_deg=float(np.degrees(wrap(P_gt[j, 2] - P_gt[i1, 2]))),
                idle_odom_cm=float(np.hypot(*(P_odom[j, :2] - P_odom[i1, :2])) * 100))
        rt = [r for r in returns if r[0] <= t[i1] - t[0]]
        if rt:
            k = int(np.searchsorted(t - t[0], rt[0][0]))
            summ.update(ret_t=rt[0][0], ret_s=rt[0][1], ret_odom_cm=float(e_odom[k] * 100),
                        ret_slam_cm=float(e_slam[k] * 100), ret_odom_deg=float(h_odom[k]),
                        ret_slam_deg=float(h_slam[k]))

    return dict(t=t - t[0], P_gt=P_gt, P_odom=P_odom, P_slam=P_slam, M=M, T_w_o0=p['T_w_o0'],
                e_odom=e_odom, h_odom=h_odom, e_slam=e_slam, h_slam=h_slam, s=s, d_start=d_start,
                t_upd=t_upd - t[0], upd=U, returns=returns, jumps=jumps, summary=summ,
                phase=ph, scan=p['scan'], source=p['source'])


def analyze(d, transform_timeout=0.2):
    return evaluate(prepare_bag(d, transform_timeout))


# ----------------------------------------------------------- do thi --

def plot(r, out_png, title, map_data=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    C = dict(gt='#e8901c', odom='#2060d0', slam='#1a9e5c')
    T = tuple(r['T_w_o0'])
    outl = [W.world_to_frame(pl, T) for pl in W.outline_polylines()]
    allp = np.vstack(outl)
    xlim = (allp[:, 0].min() - 0.15, allp[:, 0].max() + 0.15)
    ylim = (allp[:, 1].min() - 0.15, allp[:, 1].max() + 0.15)
    ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.5, wspace=0.16,
                          left=0.05, right=0.95, top=0.92, bottom=0.08)
    ax = fig.add_subplot(gs[:, 0])
    if map_data is not None:
        g, res = map_data['grid'], map_data['resolution']
        ox, oy, _ = map_data['origin']
        h, w = g.shape
        rgb = np.where(g == 100, 0, np.where(g == 0, 2, 1))
        ax.imshow(rgb, cmap=ListedColormap(['#222222', '#d0d0d0', '#ffffff']), vmin=0, vmax=2,
                  extent=(ox, ox + w * res, oy, oy + h * res), origin='upper',
                  interpolation='nearest', alpha=0.9)
    for q in outl:
        ax.plot(q[:, 0], q[:, 1], color='#e8501c', lw=0.7, alpha=0.8)
    ax.plot(r['P_gt'][:, 0], r['P_gt'][:, 1], color=C['gt'], lw=2.4, label='ground truth')
    ax.plot(r['P_odom'][:, 0], r['P_odom'][:, 1], color=C['odom'], lw=1.1, label='odometry (/odom)')
    ax.plot(r['P_slam'][:, 0], r['P_slam'][:, 1], color=C['slam'], lw=1.1, ls='--',
            label='SLAM: (map→odom)·(odom→base_footprint)')
    ax.plot(0, 0, '^', color='k', ms=8, label='xuất phát')
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)
    ax.set_xlabel('x frame map (m)', labelpad=1)
    ax.set_ylabel('y frame map (m)')
    ax.set_title('(a) Quỹ đạo trong frame map, nền là bản đồ đã lưu', fontsize=10)
    del ratio

    t = r['t']
    ph = r['phase']

    def marks(ax):
        if ph is not None:
            ax.axvspan(t[0], t[ph[0]], color='#e9e3f5', zorder=0)
            ax.axvspan(t[ph[1]], t[-1], color='#e9e3f5', zorder=0)
        for tr, _ in r['returns']:
            ax.axvline(tr, color='0.4', ls='--', lw=0.8)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, r['e_odom'] * 100, color=C['odom'], label='odometry')
    ax.plot(t, r['e_slam'] * 100, color=C['slam'], label='SLAM')
    marks(ax)
    if ph is not None and ph[1] < len(t) - 50:
        ax.text((t[ph[1]] + t[-1]) / 2, ax.get_ylim()[1] * 0.92, 'robot đứng yên',
                ha='center', va='top', fontsize=8, color='#5b3f99')
    ax.set_ylabel('sai lệch vị trí (cm)')
    ax.set_title('(b) Sai lệch vị trí so với ground truth (nét đứt: về gần điểm xuất phát)',
                 fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 1], sharex=ax)
    ax.plot(t, r['h_odom'], color=C['odom'])
    ax.plot(t, r['h_slam'], color=C['slam'])
    marks(ax)
    ax.set_ylabel('sai lệch hướng (độ)')
    ax.set_title('(c) Sai lệch hướng so với ground truth', fontsize=10)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[2, 1], sharex=ax)
    tu, U = r['t_upd'], r['upd']
    tu = np.r_[tu, t[-1]]
    U = np.vstack([U, U[-1:]])
    ax.step(tu, U[:, 0] * 100, where='post', color='#c0392b', label='x (cm)')
    ax.step(tu, U[:, 1] * 100, where='post', color='#8e44ad', label='y (cm)')
    ax2 = ax.twinx()
    ax2.step(tu, np.degrees(U[:, 2]), where='post', color='k', lw=1, label='yaw (độ)')
    ax2.set_ylabel('yaw (độ)')
    marks(ax)
    ax.set_ylabel('tịnh tiến (cm)')
    ax.set_xlabel('thời gian mô phỏng kể từ đầu bag (s)')
    ax.set_title('(d) TF map→odom do slam_toolbox phát', fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.grid(alpha=0.25)
    ax.set_xlim(t[0], t[-1])
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------- xuat ket qua --

def write_outputs(r, outdir, name):
    if r['source'] == 'bag':
        with open(outdir / f'{name}_traj.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['t', 'gt_x', 'gt_y', 'gt_yaw', 'odom_x', 'odom_y', 'odom_yaw',
                         'slam_x', 'slam_y', 'slam_yaw'])
            for i in range(len(r['t'])):
                wr.writerow([f"{r['t'][i]:.3f}", *(f'{v:.4f}' for v in r['P_gt'][i]),
                             *(f'{v:.4f}' for v in r['P_odom'][i]),
                             *(f'{v:.4f}' for v in r['P_slam'][i])])
    s = r['summary']
    with open(outdir / f'{name}_summary.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        for k, v in s.items():
            wr.writerow([k, v])
        for k, v in (r.get('scan') or {}).items():
            wr.writerow([f'scan_{k}', v])
        for i, j in enumerate(r['jumps']):
            wr.writerow([f'jump_{i}', j])

    def f(x, nd=1):
        return f'{x:.{nd}f}'.replace('.', '{,}').replace('-', '$-$')
    lines = [r'\begin{tabular}{llrr}', r'\toprule',
             r'Giai đoạn & Chỉ số & Odometry & SLAM \\', r'\midrule']
    if 'move_t1' in s:
        g = f"Di chuyển ({f(s['move_t0'], 0)}--{f(s['move_t1'], 0)} s, {f(s['move_distance'], 1)} m)"
        lines += [
            f"{g} & Vị trí, cuối pha (cm) & {f(s['move_odom_end_cm'])} & {f(s['move_slam_end_cm'])} \\\\",
            f" & Vị trí, lớn nhất (cm) & {f(s['move_odom_max_cm'])} & {f(s['move_slam_max_cm'])} \\\\",
            f" & Vị trí, RMS (cm) & {f(s['move_odom_rms_cm'])} & {f(s['move_slam_rms_cm'])} \\\\",
            f" & Hướng, cuối pha (độ) & {f(s['move_odom_end_deg'], 2)} & {f(s['move_slam_end_deg'], 2)} \\\\",
            f" & Hướng, lớn nhất (độ) & {f(s['move_odom_max_deg'], 2)} & {f(s['move_slam_max_deg'], 2)} \\\\",
            r'\midrule']
    lines += [f"Cuối bag ({f(s['duration'], 0)} s) & Vị trí (cm) & {f(s['odom_final_cm'])} & "
              f"{f(s['slam_final_cm'])} \\\\",
              f" & Hướng (độ) & {f(s['odom_final_deg'], 2)} & {f(s['slam_final_deg'], 2)} \\\\",
              r'\bottomrule', r'\end{tabular}']
    (outdir / f'{name}_summary.tex').write_text('\n'.join(lines) + '\n')


def print_report(name, r):
    s = r['summary']
    print(f'=== {name} ===')
    print(f"Thoi gian {s['duration']:.1f} s | quang duong {s['distance']:.2f} m | "
          f"tong goc quay {s['turned_deg']:.0f} do")
    c = r.get('scan')
    if c:
        print(f"/scan: {c['n_msgs']} msg, {c['rate']:.2f} Hz (std chu ky {c['dt_std_ms']:.2f} ms), "
              f"{c.get('n')} tia, buoc {math.degrees(c.get('angle_increment', 0)):.3f} do, "
              f"[{c.get('range_min'):.2f}, {c.get('range_max'):.1f}] m, frame '{c.get('frame_id')}', "
              f"tia hop le {c['valid_ratio']:.1f}%")
    print(f"map->odom: {s['n_stamps']} dau thoi gian, {s['n_changes']} lan doi gia tri | "
          f"cuoi x={s['final_m2o'][0]*100:+.2f} cm y={s['final_m2o'][1]*100:+.2f} cm "
          f"yaw={math.degrees(s['final_m2o'][2]):+.2f} do")
    if 'move_t1' in s:
        print(f"PHA DI CHUYEN t={s['move_t0']:.1f}-{s['move_t1']:.1f} s, {s['move_distance']:.2f} m, "
              f"quay {s['move_turned_deg']:.0f} do:")
        print(f"  Odometry: cuoi {s['move_odom_end_cm']:.2f} cm / {s['move_odom_end_deg']:+.2f} do | "
              f"max {s['move_odom_max_cm']:.2f} cm / {s['move_odom_max_deg']:.2f} do | "
              f"RMS {s['move_odom_rms_cm']:.2f} cm")
        print(f"  SLAM:     cuoi {s['move_slam_end_cm']:.2f} cm / {s['move_slam_end_deg']:+.2f} do | "
              f"max {s['move_slam_max_cm']:.2f} cm / {s['move_slam_max_deg']:.2f} do | "
              f"RMS {s['move_slam_rms_cm']:.2f} cm")
        if 'ret_t' in s:
            print(f"  Ve gan diem xuat phat t={s['ret_t']:.1f} s sau {s['ret_s']:.1f} m: odom "
                  f"{s['ret_odom_cm']:.2f} cm / {s['ret_odom_deg']:+.2f} do, SLAM {s['ret_slam_cm']:.2f} cm"
                  f" / {s['ret_slam_deg']:+.2f} do")
    if 'idle_s' in s:
        print(f"PHA DUNG YEN {s['idle_s']:.1f} s: ground truth troi {s['idle_gt_cm']:.2f} cm, "
              f"{s['idle_gt_deg']:+.2f} do; odometry troi {s['idle_odom_cm']:.2f} cm "
              f"(slam_toolbox khong them nut moi khi robot chua di du nguong)")
    print(f"CUOI BAG: odom {s['odom_final_cm']:.1f} cm / {s['odom_final_deg']:+.2f} do | "
          f"SLAM {s['slam_final_cm']:.1f} cm / {s['slam_final_deg']:+.2f} do")
    for j in r['jumps'][:5]:
        print(f"  Buoc nhay map->odom t={j['t']:.1f} s: {j['dxy_cm']:.1f} cm, {j['dyaw_deg']:.2f} do | "
              f"cach diem xuat phat {j['d_start']:.2f} m | sai lech SLAM {j['e_slam_before']:.1f} -> "
              f"{j['e_slam_after']:.1f} cm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='thu muc bag, hoac file *_traj.csv da xuat')
    ap.add_argument('-o', '--out', default='.')
    ap.add_argument('--map', help='ban do .yaml de ve nen o hinh (a)')
    ap.add_argument('--transform-timeout', type=float, default=0.2,
                    help='bang transform_timeout trong file cau hinh slam_toolbox')
    ap.add_argument('--reader', choices=['auto', 'rosbag2_py', 'rosbags'], default='auto')
    a = ap.parse_args()
    outdir = Path(a.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    map_data = None
    if a.map:
        from compare_map import load_map
        map_data = load_map(a.map)
    for inp in a.inputs:
        if inp.endswith('.csv'):
            name = Path(inp).stem.replace('_traj', '')
            r = evaluate(prepare_traj(inp))
        else:
            name = Path(inp).name
            r = analyze(load_bag(inp, a.reader), a.transform_timeout)
        plot(r, outdir / f'{name}.png', f'{name}: odometry, SLAM và ground truth', map_data)
        write_outputs(r, outdir, name)
        print_report(name, r)
        print(f'-> {outdir / (name + ".png")}, _summary.csv, _summary.tex')


if __name__ == '__main__':
    main()
