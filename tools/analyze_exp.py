#!/usr/bin/env python3
"""
analyze_exp.py - Phan tich cac lan chay thi nghiem cua bao cao 05.2 (bag do experiment.launch.py ghi).

Ba che do, moi che do tra loi mot cau hoi:

  speed   BAG [BAG ...]   Toc do lai anh huong toi ban do va dinh vi the nao?
            - ban do cuoi (lay thang tu /map trong bag) so voi hinh hoc that cua world
            - sai lech vi tri/huong cua SLAM va odometry trong pha di chuyen
            - do thi tu the: so nut, khoang cach nut, canh "xa" (noi hai nut cach nhau >= 10 id)
  idle    BAG             Khi robot "bo" luc dung yen, ban do / slam_toolbox co thay doi gi?
            - ground truth troi bao nhieu; odometry ghi nhan bao nhieu
            - so nut, TF map->odom, so o ban do thay doi theo thoi gian
            - quang duong odometry ke tu nut cuoi (so voi nguong tao nut)
  dynamic BAG             Vat da duoc quet roi bi lay di / doi cho thi ban do ra sao?
            - so o "tuong" o vi tri cu (bong ma) va vi tri moi theo tung vong chay
  moving  BAG [BAG ...]   (kich ban C) Vat chay qua lai lien tuc tren mot lan roi bi xoa?
            - so o "tuong" trong lan theo thoi gian / tung vong, tach mep lan quay ve robot
            - ty le co mat cua vat tai cac thoi diem tao nut (tu /pose) doc mep lan, so voi
              nguong cua Karto (o la tuong khi ty le > 1/9)
            - sai lech SLAM trong luc vat chay va sau khi vat bi xoa

Cach chay (sau khi source ROS 2; khong co rosbag2_py thi dung `pip install rosbags`):
  python3 tools/analyze_exp.py speed ~/ros2_ws/bags/wt052/e1_* -o ~/ros2_ws/bags/wt052/results
  python3 tools/analyze_exp.py idle ~/ros2_ws/bags/wt052/e2_idle -o ...
  python3 tools/analyze_exp.py dynamic ~/ros2_ws/bags/wt052/e3a_remove ~/ros2_ws/bags/wt052/e3b_move -o ...
  python3 tools/analyze_exp.py moving ~/ros2_ws/bags/wt052/e3c_* -o ...
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wt05_world as W                   # noqa: E402
import analyze_wt05 as A5                # noqa: E402
import compare_map as CM                 # noqa: E402

TOPICS = ('/odom', '/ground_truth', '/tf', '/tf_static', '/scan', '/map',
          '/slam_toolbox/graph_visualization', '/exp/event', '/cmd_vel', '/pose', '/exp/mover')
SPAWN = W.SPAWN
LONG_EDGE = 10          # canh noi hai nut cach nhau >= 10 id (~5 m duong di) = canh "xa"


# ================================================================ doc bag ==

def _storage_id(path):
    meta = Path(path) / 'metadata.yaml'
    if meta.exists():
        m = re.search(r'storage_identifier:\s*(\S+)', meta.read_text())
        if m:
            return m.group(1)
    return 'sqlite3'


def _iter_rosbag2_py(path, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=_storage_id(path)),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                            output_serialization_format='cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    cls = {n: get_message(types[n]) for n in topics if n in types}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic in cls:
            yield topic, t_ns, deserialize_message(data, cls[topic])


def _iter_rosbags(path, topics):
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(path) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, t_ns, raw in reader.messages(connections=conns):
            yield conn.topic, t_ns, ts.deserialize_cdr(raw, conn.msgtype)


def iter_bag(path, topics, backend='auto'):
    if backend == 'rosbags':
        return _iter_rosbags(path, topics)
    if backend == 'rosbag2_py':
        return _iter_rosbag2_py(path, topics)
    try:
        import rosbag2_py  # noqa: F401
        import rclpy  # noqa: F401
        return _iter_rosbag2_py(path, topics)
    except ImportError:
        return _iter_rosbags(path, topics)


def _t(st):
    return st.sec + st.nanosec * 1e-9


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_run(path, backend='auto'):
    """Doc bag MOT luot, gom moi thu can cho ca ba che do."""
    odom, gt, m2o, scans, maps, graphs, events, cmd = [], [], [], [], [], [], [], []
    poses, mover, tf_static = [], [], {}
    scan_info = None
    for topic, t_ns, msg in iter_bag(path, TOPICS, backend):
        if topic in ('/odom', '/ground_truth'):
            p, tw = msg.pose.pose, msg.twist.twist
            row = (_t(msg.header.stamp), p.position.x, p.position.y, _yaw(p.orientation),
                   tw.linear.x, tw.linear.y, tw.angular.z)
            (odom if topic == '/odom' else gt).append(row)
        elif topic == '/tf':
            for tr in msg.transforms:
                if tr.header.frame_id.lstrip('/') == 'map' and tr.child_frame_id.lstrip('/') == 'odom':
                    tt = tr.transform
                    m2o.append((_t(tr.header.stamp), tt.translation.x, tt.translation.y,
                                _yaw(tt.rotation)))
        elif topic == '/scan':
            r = np.asarray(msg.ranges, dtype=float)
            ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
            scans.append((_t(msg.header.stamp), len(r), int(ok.sum())))
            if scan_info is None:
                scan_info = dict(frame_id=msg.header.frame_id, angle_increment=msg.angle_increment,
                                 range_min=msg.range_min, range_max=msg.range_max, n=len(r))
        elif topic == '/map':
            info = msg.info
            data = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
            maps.append(dict(t=_t(msg.header.stamp), res=info.resolution,
                             ox=info.origin.position.x, oy=info.origin.position.y,
                             oyaw=_yaw(info.origin.orientation), data=data))
        elif topic == '/slam_toolbox/graph_visualization':
            nodes, seg_pts, stamp = {}, [], 0.0
            for mk in msg.markers:
                stamp = max(stamp, _t(mk.header.stamp))
                if mk.action != 0:
                    continue
                if mk.type == 2 and mk.ns == 'slam_toolbox':
                    nodes[int(mk.id)] = (mk.pose.position.x, mk.pose.position.y)
                elif mk.type == 5 and mk.ns == 'slam_toolbox_edges' and mk.id == 0:
                    seg_pts = [(p.x, p.y) for p in mk.points]
            graphs.append(dict(t=stamp, nodes=nodes, edges=_edges_to_ids(nodes, seg_pts)))
        elif topic == '/exp/event':
            try:
                ev = json.loads(msg.data)
                ev['t_bag'] = t_ns * 1e-9
                events.append(ev)
            except ValueError:
                pass
        elif topic == '/cmd_vel':
            cmd.append((t_ns * 1e-9, msg.linear.x, msg.angular.z))
        elif topic == '/pose':           # slam_toolbox: 1 message / nut, dau thoi gian = scan
            p = msg.pose.pose
            poses.append((_t(msg.header.stamp), p.position.x, p.position.y, _yaw(p.orientation)))
        elif topic == '/exp/mover':
            p = msg.pose
            mover.append((_t(msg.header.stamp), p.position.x, p.position.y))
        elif topic == '/tf_static':
            for tr in msg.transforms:
                tt = tr.transform
                tf_static[tr.child_frame_id.lstrip('/')] = (
                    tr.header.frame_id.lstrip('/'), tt.translation.x, tt.translation.y, _yaw(tt.rotation))

    def arr(rows):
        a = np.array(rows, dtype=float)
        if len(a) == 0:
            return a
        a = a[np.argsort(a[:, 0], kind='stable')]
        return a[np.unique(a[:, 0], return_index=True)[1]]

    d = dict(odom=arr(odom), gt=arr(gt), m2o=np.array(sorted(m2o), dtype=float),
             scans=np.array(scans, dtype=float) if scans else None, scan_info=scan_info,
             maps=sorted(maps, key=lambda m: m['t']), graphs=sorted(graphs, key=lambda g: g['t']),
             events=events, cmd=np.array(cmd, dtype=float), name=Path(path).name,
             poses=arr(poses) if poses else np.zeros((0, 4)), mover=arr(mover) if mover else np.zeros((0, 3)),
             tf_static=tf_static)
    for k in ('odom', 'gt'):
        if len(d[k]) < 10:
            raise RuntimeError(f'{path}: thieu du lieu {k}')
    if len(d['m2o']) == 0:
        raise RuntimeError(f'{path}: khong co TF map->odom (slam_toolbox co chay khong?)')
    return d


def _edges_to_ids(nodes, pts):
    if not nodes or len(pts) < 2:
        return []
    ids = np.array(list(nodes.keys()))
    xy = np.array([nodes[i] for i in ids])
    P = np.array(pts)
    d2 = ((P[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    k = d2.argmin(axis=1)
    ok = d2[np.arange(len(P)), k] < 1e-6
    out = []
    for a in range(0, len(P) - 1, 2):
        if ok[a] and ok[a + 1]:
            out.append((int(ids[k[a]]), int(ids[k[a + 1]])))
    return out


# ================================================================ tien ich ==

def grid_dict(m):
    """/map (hang 0 = y nho nhat) -> dict cung dang compare_map.load_map (hang 0 = tren cung)."""
    g = m['data'][::-1].astype(int)
    h, w = g.shape
    res = m['res']
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xl, yl = (cols + 0.5) * res, (h - 1 - rows + 0.5) * res
    c, s = math.cos(m['oyaw']), math.sin(m['oyaw'])
    centers = np.stack([m['ox'] + c * xl - s * yl, m['oy'] + s * xl + c * yl], axis=-1)
    return dict(grid=np.where(g < 0, -1, np.where(g >= 65, 100, np.where(g <= 25, 0, -1))),
                resolution=res, origin=(m['ox'], m['oy'], m['oyaw']), centers=centers, path='')


def map_at(maps, t):
    """Ban /map dau tien co dau thoi gian >= t (hoac ban cuoi)."""
    for m in maps:
        if m['t'] >= t:
            return m
    return maps[-1]


def pose_eval(d):
    p = A5.prepare_bag(d)
    r = A5.evaluate(p)
    r['t0'] = float(p['t'][0])
    return r


def graph_stats(g):
    nodes, edges = g['nodes'], g['edges']
    ids = sorted(nodes)
    xy = np.array([nodes[i] for i in ids]) if ids else np.zeros((0, 2))
    step = np.hypot(*np.diff(xy, axis=0).T) if len(xy) > 1 else np.zeros(0)
    gaps = [abs(a - b) for a, b in edges]
    return dict(n_nodes=len(ids), n_edges=len(edges),
                n_nonseq=int(sum(1 for x in gaps if x > 1)),
                n_long=int(sum(1 for x in gaps if x >= LONG_EDGE)),
                spacing_median=float(np.median(step)) if len(step) else float('nan'),
                spacing_mean=float(step.mean()) if len(step) else float('nan'))


def first_long_edge_time(graphs, t0):
    for g in graphs:
        if any(abs(a - b) >= LONG_EDGE for a, b in g['edges']):
            return g['t'] - t0
    return None


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '--'
    return f'{x:.{nd}f}'.replace('.', '{,}').replace('-', '$-$')


def write_table(path, header, rows):
    lines = [r'\begin{tabular}{l' + 'r' * (len(header) - 1) + '}', r'\toprule',
             ' & '.join(header) + r' \\', r'\midrule']
    lines += [' & '.join(r) + r' \\' for r in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    Path(path).write_text('\n'.join(lines) + '\n')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


# ================================================================ SPEED ==

def run_speed(paths, outdir, labels, backend):
    plt = _plt()
    runs = []
    for i, path in enumerate(paths):
        d = load_run(path, backend)
        name = labels[i] if labels and i < len(labels) else d['name']
        r = pose_eval(d)
        s = r['summary']
        fm = d['maps'][-1] if d['maps'] else None
        gm = grid_dict(fm) if fm else None
        cm = CM.analyze(gm, SPAWN) if gm else None
        gs = graph_stats(d['graphs'][-1]) if d['graphs'] else {}
        gt = d['gt']
        vmax = float(np.hypot(gt[:, 4], gt[:, 5]).max())
        wmax = float(np.abs(gt[:, 6]).max())
        t_long = first_long_edge_time(d['graphs'], r['t0'])
        start = next((e for e in d['events'] if e.get('event') == 'start'), {})
        runs.append(dict(name=name, d=d, r=r, s=s, gm=gm, cm=cm, gs=gs, vmax=vmax, wmax=wmax,
                         t_long=t_long, cmd=(start.get('v_max'), start.get('w_max'))))
        # hinh rieng tung lan chay (cung bo cuc tuan 5)
        A5.plot(r, outdir / f'{name}_pose.png', f'{name}: odometry, SLAM và ground truth', gm)
        if gm:
            traj = dict(gt_x=r['P_gt'][:, 0], gt_y=r['P_gt'][:, 1],
                        slam_x=r['P_slam'][:, 0], slam_y=r['P_slam'][:, 1])
            CM.plot(gm, cm, outdir / f'{name}_map.png', traj)
        print(f"=== {name} ===")
        print(f"  toc do lenh {start.get('v_max', '?')} m/s, {start.get('w_max', '?')} rad/s | "
              f"dat duoc max {vmax:.2f} m/s, {wmax:.2f} rad/s")
        if 'move_t1' in s:
            print(f"  pha di chuyen {s['move_t1'] - s['move_t0']:.1f} s, {s['move_distance']:.2f} m | "
                  f"SLAM RMS {s['move_slam_rms_cm']:.2f} cm, max {s['move_slam_max_cm']:.2f} cm, "
                  f"huong max {s['move_slam_max_deg']:.2f} do | odom cuoi {s['move_odom_end_cm']:.2f} cm, "
                  f"{s['move_odom_end_deg']:+.2f} do")
        if cm:
            nm, al = cm['nominal'], cm.get('aligned', cm['nominal'])
            print(f"  ban do: {nm['n_occ']} o occ | d<=1 o {nm['within_1cell']:.1f}% (ICP {al['within_1cell']:.1f}%)"
                  f" | p95 {nm['d_p95']*100:.1f} cm | o ma {nm['phantom']} | phu {nm['coverage']:.1f}% | "
                  f"free trong vat can {nm['free_in_solid']}")
        if gs:
            print(f"  do thi: {gs['n_nodes']} nut, khoang cach trung vi {gs['spacing_median']:.3f} m, "
                  f"{gs['n_edges']} canh, {gs['n_nonseq']} canh khong lien tiep, {gs['n_long']} canh xa"
                  f"{'' if t_long is None else f' (canh xa dau tien t={t_long:.1f} s)'}")

    # ---- bang so sanh
    hdr = ['Chỉ số'] + [ru['name'].replace('_', r'\_') for ru in runs]

    def row(label, fn):
        return [label] + [fn(ru) for ru in runs]

    def mv(ru, k, nd=1, sc=1.0):
        return fmt(ru['s'][k] * sc, nd) if k in ru['s'] else '--'

    rows = [
        row('Lệnh $v$ / $\\omega$', lambda ru: '--' if ru['cmd'][0] is None
            else f"{fmt(ru['cmd'][0], 2)} / {fmt(ru['cmd'][1], 2)}"),
        row('Đạt được $v_{max}$ (m/s)', lambda ru: fmt(ru['vmax'], 2)),
        row('Đạt được $|\\omega|_{max}$ (rad/s)', lambda ru: fmt(ru['wmax'], 2)),
        row('Thời gian di chuyển (s)', lambda ru: fmt(ru['s']['move_t1'] - ru['s']['move_t0'], 0)
            if 'move_t1' in ru['s'] else '--'),
        row('Số nút pose graph', lambda ru: str(ru['gs'].get('n_nodes', '--'))),
        row('Khoảng cách nút, trung vị (m)', lambda ru: fmt(ru['gs'].get('spacing_median'), 2)),
        row('Cạnh xa ($\\Delta$id $\\geq$ 10)', lambda ru: str(ru['gs'].get('n_long', '--'))),
        row('SLAM: vị trí RMS (cm)', lambda ru: mv(ru, 'move_slam_rms_cm')),
        row('SLAM: vị trí lớn nhất (cm)', lambda ru: mv(ru, 'move_slam_max_cm')),
        row('SLAM: hướng lớn nhất (độ)', lambda ru: mv(ru, 'move_slam_max_deg', 2)),
        row('Odometry: vị trí cuối pha (cm)', lambda ru: mv(ru, 'move_odom_end_cm')),
        row('Odometry: hướng cuối pha (độ)', lambda ru: mv(ru, 'move_odom_end_deg', 2)),
        row('Bản đồ: ô $d \\le$ 1 ô (\\%)', lambda ru: fmt(ru['cm']['nominal']['within_1cell']) if ru['cm'] else '--'),
        row('Bản đồ: $d$ phân vị 95\\% (cm)', lambda ru: fmt(ru['cm']['nominal']['d_p95'] * 100) if ru['cm'] else '--'),
        row("Bản đồ: ô ``ma''", lambda ru: str(ru['cm']['nominal']['phantom']) if ru['cm'] else '--'),
        row('Bản đồ: độ phủ (\\%)', lambda ru: fmt(ru['cm']['nominal']['coverage']) if ru['cm'] else '--'),
    ]
    write_table(outdir / 'speed_summary.tex', hdr, rows)
    with open(outdir / 'speed_summary.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        for ru in runs:
            flat = dict(name=ru['name'], vmax=ru['vmax'], wmax=ru['wmax'], **ru['gs'],
                        **{k: v for k, v in ru['s'].items() if not isinstance(v, tuple)})
            if ru['cm']:
                flat.update({f'map_{k}': v for k, v in ru['cm']['nominal'].items()})
                flat.update({f'map_icp_{k}': v for k, v in ru['cm'].get('aligned', {}).items()})
            for k, v in flat.items():
                wr.writerow([ru['name'], k, v])

    # ---- hinh tong hop
    n = len(runs)
    fig = plt.figure(figsize=(4.2 * n, 6.6))
    gsp = fig.add_gridspec(2, n, height_ratios=[1.0, 1], hspace=0.28, wspace=0.12)
    outl = [W.world_to_frame(pl, SPAWN) for pl in W.outline_polylines()]
    for i, ru in enumerate(runs):
        ax = fig.add_subplot(gsp[0, i])
        if ru['gm']:
            g = ru['gm']
            from matplotlib.colors import ListedColormap
            occ = g['centers'][g['grid'] == 100]
            d_occ = W.dist_to_surface(W.frame_to_world(occ, SPAWN)) if len(occ) else np.zeros(0)
            h, w = g['grid'].shape
            ox, oy, _ = g['origin']
            res = g['resolution']
            ax.imshow(np.where(g['grid'] == 0, 1, 0), cmap=ListedColormap(['#ffffff', '#e6e6e6']),
                      vmin=0, vmax=1, extent=(ox, ox + w * res, oy, oy + h * res), origin='upper',
                      interpolation='nearest')
            sc = ax.scatter(occ[:, 0], occ[:, 1], s=3, c=np.minimum(d_occ, 0.15) * 100, cmap='viridis',
                            vmin=0, vmax=15, marker='s', linewidths=0)
        for q in outl:
            ax.plot(q[:, 0], q[:, 1], color='#e8501c', lw=0.6)
        ax.plot(ru['r']['P_gt'][:, 0], ru['r']['P_gt'][:, 1], color='#f0a020', lw=0.9)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ru['name'], fontsize=10)
    if any(ru['gm'] for ru in runs):
        cax = fig.add_axes([0.92, 0.56, 0.008, 0.3])
        fig.colorbar(sc, cax=cax, label='d tới bề mặt thật (cm)')
    ax1 = fig.add_subplot(gsp[1, : max(1, n // 2)])
    ax2 = fig.add_subplot(gsp[1, max(1, n // 2):]) if n > 1 else ax1.twinx()
    cols = plt.cm.tab10(np.arange(n))
    for i, ru in enumerate(runs):
        r = ru['r']
        ph = r['phase']
        sl = slice(ph[0], ph[1] + 1) if ph else slice(None)
        ax1.plot(r['s'][sl], r['e_slam'][sl] * 100, color=cols[i], label=ru['name'])
        ax2.plot(r['s'][sl], r['e_odom'][sl] * 100, color=cols[i], label=ru['name'])
    ax1.set_title('SLAM: sai lệch vị trí theo quãng đường', fontsize=10)
    ax2.set_title('Odometry: sai lệch vị trí theo quãng đường', fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xlabel('quãng đường thật (m)')
        ax.set_ylabel('cm')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(outdir / 'speed_summary.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {outdir}/speed_summary.png, speed_summary.tex/.csv, <run>_pose.png, <run>_map.png')


# ================================================================ IDLE ==

def map_diff(a, b):
    """So o thay doi trang thai giua hai ban /map (so tai cung vi tri trong frame map)."""
    if a['data'].shape == b['data'].shape and abs(a['ox'] - b['ox']) < 1e-6 and abs(a['oy'] - b['oy']) < 1e-6:
        return int((a['data'] != b['data']).sum()), True
    res = a['res']
    h, w = b['data'].shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    x = b['ox'] + (jj + 0.5) * res
    y = b['oy'] + (ii + 0.5) * res
    ja = np.floor((x - a['ox']) / res).astype(int)
    ia = np.floor((y - a['oy']) / res).astype(int)
    ha, wa = a['data'].shape
    inside = (ja >= 0) & (ja < wa) & (ia >= 0) & (ia < ha)
    va = np.full(b['data'].shape, -1, dtype=np.int8)
    va[inside] = a['data'][ia[inside], ja[inside]]
    return int((va != b['data']).sum()), False


def run_idle(path, outdir, backend):
    plt = _plt()
    d = load_run(path, backend)
    name = d['name']
    r = pose_eval(d)
    t0 = r['t0']
    t = r['t']
    done = next((e for e in d['events'] if e.get('event') == 'done'), None)
    t_idle = (done['t'] - t0) if done else (t[r['phase'][1]] if r['phase'] else t[0])
    i0 = int(np.searchsorted(t, t_idle))
    m = slice(i0, None)
    tt = t[m] - t[i0]
    G, Od = r['P_gt'][m], r['P_odom'][m]
    gt_disp = np.hypot(G[:, 0] - G[0, 0], G[:, 1] - G[0, 1]) * 100
    gt_yaw = np.degrees(np.unwrap(G[:, 2]) - G[0, 2])
    od_disp = np.hypot(Od[:, 0] - Od[0, 0], Od[:, 1] - Od[0, 1]) * 100
    od_yaw = np.degrees(np.unwrap(Od[:, 2]) - Od[0, 2])

    # do thi tu the theo thoi gian: thoi diem co nut moi
    gtimes = np.array([g['t'] - t0 for g in d['graphs']])
    gn = np.array([len(g['nodes']) for g in d['graphs']])
    new_node_t = [float(gtimes[k]) for k in range(1, len(gn)) if gn[k] > gn[k - 1] and gtimes[k] >= t_idle]
    n_at_idle = int(gn[np.searchsorted(gtimes, t_idle) - 1]) if len(gn) and gtimes[0] <= t_idle else 0

    # Quang duong odometry ke tu nut moi nhat. Vi tri odometry cua nut = (map->odom)^-1 * vi tri nut
    # (vi tri nut lay tu pose graph; map->odom khong doi giua hai lan tao nut).
    last_xy = np.array([g['nodes'][max(g['nodes'])] if g['nodes'] else (np.nan, np.nan)
                        for g in d['graphs']])
    kg = np.clip(np.searchsorted(gtimes, t[m], side='right') - 1, 0, None)
    nxy = last_xy[kg]
    Mi = A5.inverse(r['M'][m])
    node_in_odom = A5.compose(Mi, np.c_[nxy, np.zeros(len(nxy))])
    od_since_node = np.hypot(Od[:, 0] - node_in_odom[:, 0], Od[:, 1] - node_in_odom[:, 1])

    # /map: so o khac so voi ban /map dau tien cua pha dung yen
    ref_map = map_at(d['maps'], t_idle + t0)
    idle_maps = [mp for mp in d['maps'] if mp['t'] - t0 >= t_idle]
    changes = [(mp['t'] - t0 - t_idle, *map_diff(ref_map, mp)) for mp in idle_maps]
    ch = np.array([c[1] for c in changes]) if changes else np.zeros(0)

    tu = r['t_upd']
    U = r['upd']
    sel = tu >= t_idle
    m2o_changes = int((np.abs(np.diff(U[sel], axis=0)) > 5e-4).any(axis=1).sum()) if sel.sum() > 1 else 0

    summ = dict(idle_s=float(tt[-1]), gt_disp_cm=float(gt_disp[-1]), gt_yaw_deg=float(gt_yaw[-1]),
                gt_rate_mm_s=float(gt_disp[-1] * 10 / max(tt[-1], 1e-6)),
                odom_disp_cm=float(od_disp[-1]), odom_yaw_deg=float(od_yaw[-1]),
                nodes_at_idle=n_at_idle, nodes_final=int(gn[-1]) if len(gn) else 0,
                new_nodes_idle=len(new_node_t), new_node_times=[round(x - t_idle, 1) for x in new_node_t],
                map_msgs_idle=len(idle_maps), map_rate_hz=float(len(idle_maps) / max(tt[-1], 1e-6)),
                map_cells_changed_max=int(ch.max()) if len(ch) else 0,
                map_cells_changed_final=int(ch[-1]) if len(ch) else 0,
                m2o_changes_idle=m2o_changes,
                slam_err_start_cm=float(r['e_slam'][i0] * 100), slam_err_end_cm=float(r['e_slam'][-1] * 100),
                slam_head_start_deg=float(r['h_slam'][i0]), slam_head_end_deg=float(r['h_slam'][-1]),
                odom_since_node_max_m=float(od_since_node.max()))
    with open(outdir / f'{name}_idle_summary.csv', 'w', newline='') as f:
        csv.writer(f).writerows(summ.items())
    rows = [
        ['Thời gian đứng yên (phút)', fmt(summ['idle_s'] / 60, 1)],
        ['Ground truth: trôi vị trí (cm)', fmt(summ['gt_disp_cm'])],
        ['Ground truth: trôi hướng (độ)', fmt(summ['gt_yaw_deg'], 2)],
        ['Odometry ghi nhận: vị trí (cm)', fmt(summ['odom_disp_cm'])],
        ['Odometry ghi nhận: hướng (độ)', fmt(summ['odom_yaw_deg'], 2)],
        ['Nút mới trong lúc đứng yên', str(summ['new_nodes_idle'])],
        ['TF map$\\to$odom đổi giá trị (lần)', str(summ['m2o_changes_idle'])],
        ['/map: số bản phát lại', str(summ['map_msgs_idle'])],
        ['/map: số ô khác bản đầu, lớn nhất', str(summ['map_cells_changed_max'])],
        ['SLAM: sai lệch vị trí đầu $\\to$ cuối (cm)',
         f"{fmt(summ['slam_err_start_cm'])} $\\to$ {fmt(summ['slam_err_end_cm'])}"],
        ['SLAM: sai lệch hướng đầu $\\to$ cuối (độ)',
         f"{fmt(summ['slam_head_start_deg'], 2)} $\\to$ {fmt(summ['slam_head_end_deg'], 2)}"],
    ]
    write_table(outdir / f'{name}_idle_summary.tex', ['Chỉ số', 'Giá trị'], rows)

    # ---- hinh
    fig, axs = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    tm = tt / 60
    ax = axs[0]
    ax.plot(tm, gt_disp, color='#e8901c', label='ground truth: dịch chuyển (cm)')
    ax.plot(tm, od_disp, color='#2060d0', label='odometry: dịch chuyển (cm)')
    ax2 = ax.twinx()
    ax2.plot(tm, gt_yaw, color='#e8901c', ls='--', label='ground truth: quay (độ)')
    ax2.plot(tm, od_yaw, color='#2060d0', ls='--', label='odometry: quay (độ)')
    ax2.set_ylabel('độ')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.set_ylabel('cm')
    ax.set_title('(a) Robot "bò": chuyển động thật và chuyển động odometry ghi nhận', fontsize=10)
    ax = axs[1]
    ax.plot(tm, od_since_node * 100, color='#2060d0')
    ax.axhline(30.0, color='0.4', ls='--', lw=1)
    ax.text(tm[-1], 30.0, '0,3 m (ngưỡng của Karto) ', va='bottom', ha='right', fontsize=8)
    ax.set_ylabel('cm')
    ax.set_title('(b) Khoảng cách odometry từ nút gần nhất. Nút mới cần: vượt 0,3 m (Karto) VÀ đã đi '
                 '≥ 0,268 m kể từ lần qua lọc gần nhất (slam_toolbox, không ghi trong bag)', fontsize=9)
    ax = axs[2]
    ax.plot(tm, r['e_slam'][m] * 100, color='#1a9e5c', label='SLAM: sai lệch vị trí (cm)')
    ax.plot(tm, r['e_odom'][m] * 100, color='#2060d0', label='odometry: sai lệch vị trí (cm)')
    ax2 = ax.twinx()
    ax2.plot(tm, r['h_slam'][m], color='#1a9e5c', ls='--', label='SLAM: sai lệch hướng (độ)')
    ax2.set_ylabel('độ')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.set_ylabel('cm')
    ax.set_title('(c) Sai lệch so với ground truth', fontsize=10)
    ax = axs[3]
    if len(changes):
        ax.plot(np.array([c[0] for c in changes]) / 60, ch, color='#8e44ad', label='/map: số ô khác bản đầu pha')
    ax2 = ax.twinx()
    ax2.step((gtimes - t_idle) / 60, gn, where='post', color='k', lw=1, label='số nút pose graph')
    ax2.set_ylabel('số nút')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.set_ylabel('số ô')
    ax.set_xlabel('thời gian kể từ khi robot dừng (phút)')
    ax.set_title('(d) Bản đồ và pose graph', fontsize=10)
    for ax in axs:
        ax.grid(alpha=0.25)
        for x in new_node_t:
            ax.axvline((x - t_idle) / 60, color='m', ls=':', lw=1)
    axs[0].set_xlim(0, tm[-1])
    fig.tight_layout()
    fig.savefig(outdir / f'{name}_idle.png', dpi=140)
    plt.close(fig)

    print(f'=== {name} (dung yen) ===')
    print(f"  dung yen {summ['idle_s'] / 60:.1f} phut | GT troi {summ['gt_disp_cm']:.2f} cm "
          f"({summ['gt_rate_mm_s']:.3f} mm/s), {summ['gt_yaw_deg']:+.2f} do | odom ghi nhan "
          f"{summ['odom_disp_cm']:.2f} cm, {summ['odom_yaw_deg']:+.2f} do")
    print(f"  nut: {summ['nodes_at_idle']} -> {summ['nodes_final']} (moi trong luc dung: "
          f"{summ['new_nodes_idle']} tai {summ['new_node_times']} s) | map->odom doi "
          f"{summ['m2o_changes_idle']} lan")
    print(f"  /map: {summ['map_msgs_idle']} ban ({summ['map_rate_hz']:.2f} Hz), so o khac ban dau pha: "
          f"max {summ['map_cells_changed_max']}, cuoi {summ['map_cells_changed_final']}")
    print(f"  SLAM sai lech {summ['slam_err_start_cm']:.1f} -> {summ['slam_err_end_cm']:.1f} cm, "
          f"{summ['slam_head_start_deg']:+.2f} -> {summ['slam_head_end_deg']:+.2f} do | odom ke tu nut cuoi "
          f"max {summ['odom_since_node_max_m']:.3f} m")
    print(f'-> {outdir}/{name}_idle.png, _idle_summary.tex/.csv')


# ================================================================ DYNAMIC ==

def box_sd(pts, cx, cy, half):
    """Khoang cach co dau toi bien hop vuong truc (am = ben trong)."""
    q = np.abs(pts - [cx, cy]) - half
    out = np.hypot(np.maximum(q[:, 0], 0), np.maximum(q[:, 1], 0))
    return out + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0)


def region_counts(m, cx, cy, size, band):
    g = grid_dict(m)
    c = g['centers'].reshape(-1, 2)
    v = g['grid'].ravel()
    sd = box_sd(c, cx, cy, size / 2)
    rim = np.abs(sd) <= band
    inner = sd < -band
    return dict(rim_occ=int((v[rim] == 100).sum()), rim_free=int((v[rim] == 0).sum()),
                rim_unk=int((v[rim] == -1).sum()), in_free=int((v[inner] == 0).sum()),
                in_occ=int((v[inner] == 100).sum()), in_unk=int((v[inner] == -1).sum()))


def run_dynamic(paths, outdir, backend):
    plt = _plt()
    for path in paths:
        d = load_run(path, backend)
        name = d['name']
        r = pose_eval(d)
        t0 = r['t0']
        ev = d['events']
        rem = next((e for e in ev if e.get('event') == 'remove'), None)
        spn = next((e for e in ev if e.get('event') == 'spawn'), None)
        laps = [(e['lap'], e['t']) for e in ev if e.get('event') == 'lap']
        if rem is None:
            print(f'{name}: khong co su kien remove -> bo qua')
            continue
        size = rem.get('size', W.MOVER['size'])
        p1 = W.world_to_frame(np.array([[rem['ox'], rem['oy']]]), SPAWN)[0]
        p2 = W.world_to_frame(np.array([[spn['ox'], spn['oy']]]), SPAWN)[0] if spn else None
        band = 1.5 * d['maps'][0]['res']
        t_ch = rem['t']
        rows = []
        for mp in d['maps']:
            c1 = region_counts(mp, p1[0], p1[1], size, band)
            c2 = region_counts(mp, p2[0], p2[1], size, band) if p2 is not None else {}
            rows.append(dict(t=mp['t'] - t0, **{f'p1_{k}': v for k, v in c1.items()},
                             **{f'p2_{k}': v for k, v in c2.items()}))
        T = np.array([x['t'] for x in rows])
        p1occ = np.array([x['p1_rim_occ'] for x in rows])
        p1in_free = np.array([x['p1_in_free'] for x in rows])
        p2occ = np.array([x.get('p2_rim_occ', 0) for x in rows])
        k_ch = max(0, int(np.searchsorted(T, t_ch - t0)) - 1)
        before = int(p1occ[k_ch])

        # gia tri tai cuoi moi vong (ban /map dau tien sau khi xong vong + 1,5 s)
        lap_rows = []
        for lap, tl in laps:
            k = min(len(T) - 1, int(np.searchsorted(T, tl - t0 + 1.5)))
            lap_rows.append(dict(lap=lap, t=float(tl - t0), p1_occ=int(p1occ[k]),
                                 p1_frac=float(p1occ[k] / max(before, 1)),
                                 p1_in_free=int(p1in_free[k]), p2_occ=int(p2occ[k])))
        lap_change = next((e.get('lap') for e in ev if e.get('event') == 'hold'), None)
        if lap_change is None:
            lap_change = next((lr['lap'] for lr in lap_rows if lr['t'] >= t_ch - t0 - 2.0), 1)
        cleared = next((lr['lap'] - lap_change for lr in lap_rows
                        if lr['t'] > t_ch - t0 and lr['p1_frac'] < 0.05), None)
        p2_final = int(p2occ[-1]) if p2 is not None else 0
        p2_half_t = None
        if p2 is not None and p2_final > 0:
            after = np.flatnonzero((T > t_ch - t0) & (p2occ >= 0.5 * p2_final))
            p2_half_t = float(T[after[0]] - (t_ch - t0)) if len(after) else None

        with open(outdir / f'{name}_dynamic.csv', 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        hdr = ['Vòng', 'P1: ô tường (\\% so với trước)', 'P1: ô trống bên trong']
        if p2 is not None:
            hdr.append('P2: ô tường')
        tl_rows = []
        for lr in lap_rows:
            rr = [f"{lr['lap']}" + (' (đổi)' if lr['lap'] == lap_change else ''),
                  f"{lr['p1_occ']} ({fmt(lr['p1_frac'] * 100, 0)})", str(lr['p1_in_free'])]
            if p2 is not None:
                rr.append(str(lr['p2_occ']))
            tl_rows.append(rr)
        write_table(outdir / f'{name}_dynamic_laps.tex', hdr, tl_rows)

        # ---- hinh: hang tren = phong to quanh P1 (va P2) o 4 thoi diem; hang duoi = so o theo thoi gian
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#111111', '#bdbdbd', '#ffffff'])
        snaps = [('trước khi đổi', t_ch - t0 - 0.5)]
        after_laps = [lr for lr in lap_rows if lr['t'] > t_ch - t0]
        if after_laps:
            snaps.append((f'sau {after_laps[0]["lap"] - lap_change} vòng', after_laps[0]['t'] + 1.5))
            mid = after_laps[min(len(after_laps) - 1, 2)]
            if mid is not after_laps[0]:
                snaps.append((f'sau {mid["lap"] - lap_change} vòng', mid['t'] + 1.5))
        snaps.append(('cuối', None))
        zones = [('P1', p1, '#8e44ad')] + ([('P2', p2, '#1a9e5c')] if p2 is not None else [])
        nz = len(zones)
        fig = plt.figure(figsize=(13, 4.0 + 2.6 * nz))
        gsp = fig.add_gridspec(nz + 1, len(snaps) + 1, height_ratios=[1] * nz + [1.5], hspace=0.35,
                               wspace=0.15)
        outl = [W.world_to_frame(pl, SPAWN) for pl in W.outline_polylines()]
        for zi, (zn, pc, col) in enumerate(zones):
            for j, (lab, tq) in enumerate(snaps):
                ax = fig.add_subplot(gsp[zi, j])
                mp = d['maps'][-1] if tq is None else map_at(d['maps'], tq + t0)
                g = grid_dict(mp)
                h, w = g['grid'].shape
                ext = (mp['ox'], mp['ox'] + w * mp['res'], mp['oy'], mp['oy'] + h * mp['res'])
                ax.imshow(np.where(g['grid'] == 100, 0, np.where(g['grid'] == 0, 2, 1)), cmap=cmap,
                          vmin=0, vmax=2, extent=ext, origin='upper', interpolation='nearest')
                for q in outl:
                    ax.plot(q[:, 0], q[:, 1], color='#e8501c', lw=0.6, alpha=0.7)
                s2 = size / 2
                ax.plot([pc[0] - s2, pc[0] + s2, pc[0] + s2, pc[0] - s2, pc[0] - s2],
                        [pc[1] - s2, pc[1] - s2, pc[1] + s2, pc[1] + s2, pc[1] - s2], color=col, lw=1.4,
                        ls='--')
                ax.set_xlim(pc[0] - 0.9, pc[0] + 0.9)
                ax.set_ylim(pc[1] - 0.75, pc[1] + 0.75)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                if zi == 0:
                    ax.set_title(lab, fontsize=10)
                if j == 0:
                    ax.set_ylabel(f'{zn} ({"vị trí cũ" if zn == "P1" else "vị trí mới"})', fontsize=10)
        ax = fig.add_subplot(gsp[:nz, len(snaps)])
        mp = d['maps'][-1]
        g = grid_dict(mp)
        h, w = g['grid'].shape
        ax.imshow(np.where(g['grid'] == 100, 0, np.where(g['grid'] == 0, 2, 1)), cmap=cmap, vmin=0, vmax=2,
                  extent=(mp['ox'], mp['ox'] + w * mp['res'], mp['oy'], mp['oy'] + h * mp['res']),
                  origin='upper', interpolation='nearest')
        for zn, pc, col in zones:
            ax.plot(*pc, 's', mfc='none', mec=col, ms=9, mew=1.5)
            ax.text(pc[0], pc[1] + 0.45, zn, color=col, ha='center', fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('toàn bản đồ (cuối)', fontsize=10)
        ax = fig.add_subplot(gsp[nz, :len(snaps) - 1])
        ax.plot(T, p1occ, color='#8e44ad', label='P1 (vị trí cũ): ô tường ở viền')
        ax.plot(T, p1in_free, color='#8e44ad', ls=':', label='P1: ô trống bên trong')
        if p2 is not None:
            ax.plot(T, p2occ, color='#1a9e5c', label='P2 (vị trí mới): ô tường ở viền')
        ax.axvline(t_ch - t0, color='k', lw=1.2)
        ax.text(t_ch - t0, ax.get_ylim()[1], ' đổi', va='top', fontsize=9)
        for lap, tl in laps:
            ax.axvline(tl - t0, color='0.6', ls='--', lw=0.7)
            ax.text(tl - t0, 0, f' {lap}', fontsize=7, color='0.4', va='bottom')
        ax.set_xlabel('thời gian mô phỏng (s); nét đứt xám: hết vòng')
        ax.set_ylabel('số ô')
        ax.set_title('Số ô theo thời gian', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax = fig.add_subplot(gsp[nz, len(snaps) - 1:])
        ax.plot(r['t'], r['e_slam'] * 100, color='#1a9e5c', lw=0.8)
        ax.axvline(t_ch - t0, color='k', lw=1.2)
        ax.set_title('SLAM: sai lệch vị trí (cm)', fontsize=10)
        ax.set_xlabel('thời gian (s)')
        ax.grid(alpha=0.25)
        fig.suptitle(f'{name}: vật {"bị lấy đi" if p2 is None else "dời từ P1 sang P2"} sau vòng {lap_change}',
                     fontsize=12)
        fig.savefig(outdir / f'{name}_dynamic.png', dpi=140, bbox_inches='tight')
        plt.close(fig)

        print(f'=== {name} ===')
        print(f'  doi sau vong {lap_change} (t={t_ch - t0:.1f} s) | P1 o tuong o vien truoc khi doi: {before}')
        for lr in lap_rows:
            print(f"  het vong {lr['lap']:2d} (t={lr['t']:6.1f} s): P1 o tuong {lr['p1_occ']:3d} "
                  f"({lr['p1_frac'] * 100:5.1f}%), P1 o trong ben trong {lr['p1_in_free']:3d}"
                  + (f", P2 o tuong {lr['p2_occ']:3d}" if p2 is not None else ''))
        p2_txt = ''
        if p2 is not None:
            p2_txt = f" | P2 dat 50% so o cuoi sau {'--' if p2_half_t is None else round(p2_half_t, 1)} s"
        print(f"  bong ma P1 xuong < 5%: {'chua' if cleared is None else f'sau {cleared} vong'}" + p2_txt)
        print(f'  SLAM sai lech: max {np.nanmax(r["e_slam"]) * 100:.1f} cm')
        print(f'-> {outdir}/{name}_dynamic.png, _dynamic.csv, _dynamic_laps.tex')


# ================================================================ MOVING (kich ban C) ==

def lidar_offset(d):
    """Tu the cua khung /scan trong khung robot (x, y, yaw), doc tu /tf_static."""
    target = (d['scan_info'] or {}).get('frame_id', 'lidar_link').lstrip('/') if d['scan_info'] else 'lidar_link'
    for base in ('base_footprint', 'base_link'):
        acc, f = np.zeros(3), target
        for _ in range(10):
            if f == base:
                return acc
            if f not in d['tf_static']:
                break
            par, x, y, yaw = d['tf_static'][f]
            acc = A5.compose(np.array([x, y, yaw]), acc)
            f = par
    print('  [canh bao] khong doc duoc vi tri lidar tu /tf_static -> dung (0,2; 0; 0)')
    return np.array([0.2, 0.0, 0.0])


def node_times(d, r):
    """Thoi diem cac scan duoc nhan lam nut. Uu tien /pose: slam_toolbox phat dung MOT lan cho moi scan
    duoc Karto nhan (dau thoi gian = scan do). Bag khong co /pose: lay lan dau nut xuat hien trong
    graph_visualization, roi chon thoi diem ground truth gan vi tri nut nhat trong ~2 s truoc do."""
    if len(d['poses']):
        return d['poses'][:, 0], '/pose'
    t_abs = r['t'] + r['t0']
    seen, out, t_prev = set(), [], t_abs[0]
    for g in d['graphs']:
        new = sorted(i for i in g['nodes'] if i not in seen)
        sel = np.flatnonzero((t_abs >= t_prev - 1.0) & (t_abs <= g['t']))
        for i in new:
            if len(sel):
                nx, ny = g['nodes'][i]
                k = sel[np.argmin(np.hypot(r['P_gt'][sel, 0] - nx, r['P_gt'][sel, 1] - ny))]
                out.append(t_abs[k])
        seen.update(new)
        t_prev = g['t']
    return np.array(sorted(out)), 'graph_visualization (uoc luong)'


def visible(sensors, pts, rmax):
    """(N,2) vi tri lidar, (K,2) diem (world) -> (N,K) True neu doan thang khong bi vat can tinh che
    va trong tam do. Vat chay KHONG tinh (diem xet nam ngoai lan, phia robot)."""
    seg = W.segments()
    S = sensors[:, None, None, :]
    P = pts[None, :, None, :]
    A, B = seg[None, None, :, 0:2], seg[None, None, :, 2:4]
    dd, e, w = P - S, B - A, A - S
    den = dd[..., 0] * e[..., 1] - dd[..., 1] * e[..., 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (w[..., 0] * e[..., 1] - w[..., 1] * e[..., 0]) / den
        u = (w[..., 0] * dd[..., 1] - w[..., 1] * dd[..., 0]) / den
    blocked = ((np.abs(den) > 1e-12) & (t > 1e-6) & (t < 1 - 1e-6) & (u >= 0) & (u <= 1)).any(-1)
    S2, D2 = sensors[:, None, :], pts[None, :, :] - sensors[:, None, :]
    for cx, cy, rad in W.circles():
        tc = np.clip(((cx - S2[..., 0]) * D2[..., 0] + (cy - S2[..., 1]) * D2[..., 1])
                     / np.maximum((D2 ** 2).sum(-1), 1e-12), 0, 1)
        blocked |= np.hypot(S2[..., 0] + tc * D2[..., 0] - cx, S2[..., 1] + tc * D2[..., 1] - cy) < rad
    return ~blocked & (np.hypot(D2[..., 0], D2[..., 1]) <= rmax)


def lane_geometry(ev):
    A = np.array([ev['ax'], ev['ay']], float)
    B = np.array([ev['bx'], ev['by']], float)
    L = float(np.hypot(*(B - A)))
    e = (B - A) / L
    n = np.array([-e[1], e[0]])
    centroid = np.array(W.ROUTE[:-1], float).mean(axis=0)
    sigma = 1.0 if (centroid - A) @ n > 0 else -1.0       # mep lan quay ve phia duong chay
    return dict(A=A, B=B, L=L, e=e, n=n, sigma=sigma, half=float(ev.get('size', W.MOVER['size'])) / 2)


def _raw_centers(m):
    h, w = m['data'].shape
    res = m['res']
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    xl, yl = (jj + 0.5) * res, (ii + 0.5) * res
    c, s = math.cos(m['oyaw']), math.sin(m['oyaw'])
    return np.stack([m['ox'] + c * xl - s * yl, m['oy'] + s * xl + c * yl], axis=-1).reshape(-1, 2)


class LaneCells:
    """Phan loai o trong lan (frame map) cho tung hinh hoc ban do (cache theo kich thuoc/goc).
    Vung lan = vet quet cua hop vuong truc chay doc lan (dung cho lan song song truc x hoac y)."""

    def __init__(self, lg, band):
        self.lg, self.band, self.cache = lg, band, {}

    def get(self, m):
        key = (m['data'].shape, round(m['ox'], 6), round(m['oy'], 6), m['res'])
        if key not in self.cache:
            lg, b = self.lg, self.band
            cw = W.frame_to_world(_raw_centers(m), SPAWN)
            rel = cw - lg['A']
            u, w = rel @ lg['e'], rel @ lg['n']
            h = lg['half']
            region = (u >= -h - b) & (u <= lg['L'] + h + b) & (np.abs(w) <= h + b)
            idx = np.flatnonzero(region)
            stat = np.zeros(len(cw), bool)
            stat[idx] = W.dist_to_surface(cw[idx]) <= b        # sat vat can tinh -> loai
            region &= ~stat
            near = region & (np.abs(w - lg['sigma'] * h) <= b)
            far = region & (np.abs(w + lg['sigma'] * h) <= b)
            self.cache[key] = dict(region=region, near=near, far=far, u=u)
        return self.cache[key]

    def counts(self, m):
        c = self.get(m)
        occ = m['data'].ravel() >= 65
        free = (m['data'].ravel() >= 0) & (m['data'].ravel() <= 25)
        tot = int((occ & c['region']).sum())
        near = int((occ & c['near']).sum())
        far = int((occ & c['far']).sum())
        return dict(total=tot, near=near, far=far, other=tot - near - far,
                    free=int((free & c['region']).sum()))


def run_moving(paths, outdir, backend):
    plt = _plt()
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#111111', '#bdbdbd', '#ffffff'])
    summary = []
    for path in paths:
        d = load_run(path, backend)
        name = d['name']
        ev = d['events']
        ms = next((e for e in ev if e.get('event') == 'move_start'), None)
        if ms is None:
            print(f'{name}: khong co su kien move_start (khong phai kich ban C) -> bo qua')
            continue
        r = pose_eval(d)
        t0 = r['t0']
        stop = next((e for e in ev if e.get('event') == 'move_stop'), None)
        rem = next((e for e in ev if e.get('event') == 'remove'), None)
        laps = [(e['lap'], e['t']) for e in ev if e.get('event') == 'lap']
        stop_lap = ms.get('stop_after_lap')
        t_stop = stop['t'] if stop else (rem['t'] if rem else np.inf)
        t_rm = rem['t'] if rem else np.inf
        lg = lane_geometry(ms)
        res = d['maps'][0]['res']
        band = 1.5 * res
        cells = LaneCells(lg, band)
        h, L = lg['half'], lg['L']

        # ---- vi tri vat theo thoi gian (world)
        mv = d['mover']

        def mover_xy(t):
            if t >= t_rm:
                return None
            if len(mv):
                k = int(np.searchsorted(mv[:, 0], t, side='right')) - 1
                return tuple(mv[max(k, 0), 1:3])
            s = ms.get('s0', L / 2) + ms['speed'] * max(0.0, min(t, t_stop) - ms['t'])
            s = s % (2 * L)
            s = s if s <= L else 2 * L - s
            return tuple(lg['A'] + lg['e'] * s)

        # ---- so o trong lan theo thoi gian
        rows = []
        for mp in d['maps']:
            rows.append(dict(t=mp['t'] - t0, **cells.counts(mp)))
        T = np.array([x['t'] for x in rows])
        tot = np.array([x['total'] for x in rows])
        near = np.array([x['near'] for x in rows])
        other = np.array([x['other'] + x['far'] for x in rows])

        def at(tq):
            return rows[min(len(rows) - 1, int(np.searchsorted(T, tq)))]

        lap_rows = []
        for lap, tl in laps:
            x = {k: v for k, v in at(tl - t0 + 1.5).items() if k != 't'}
            lap_rows.append(dict(lap=lap, t=float(tl - t0), moving=bool(tl < t_stop - 1e-3), **x))
        k_before = max(0, int(np.searchsorted(T, t_stop - t0)) - 1)
        before = rows[k_before]
        moving_laps = [lr for lr in lap_rows if lr['moving']]
        steady = float(np.mean([lr['total'] for lr in moving_laps])) if moving_laps else float('nan')
        peak = int(tot[T < t_stop - t0].max()) if (T < t_stop - t0).any() else 0
        thr = max(2, int(round(0.05 * peak)))
        cleared = None
        if stop_lap is not None:
            cleared = next((lr['lap'] - stop_lap for lr in lap_rows
                            if lr['t'] > t_stop - t0 and lr['total'] <= thr), None)

        # ---- ty le co mat cua vat tai cac thoi diem tao nut, doc mep lan quay ve robot
        tn, tsrc = node_times(d, r)
        seln = (tn >= ms['t']) & (tn < t_stop)
        tn_m = tn[seln]
        gtw = d['gt']
        Pn = A5.interp_pose(gtw[:, 0], gtw[:, 1:4], tn_m)
        off = lidar_offset(d)
        Sxy = A5.compose(Pn, off[None, :])[:, :2] if len(Pn) else np.zeros((0, 2))
        rmax = (d['scan_info'] or {}).get('range_max', 12.0) if d['scan_info'] else 12.0
        uk = np.arange(-h + res / 2, L + h, res)
        pts = lg['A'] + uk[:, None] * lg['e'] + (lg['sigma'] * (h + 0.6 * res)) * lg['n']
        vis = visible(Sxy, pts, rmax) if len(Sxy) else np.zeros((0, len(uk)), bool)
        mov = np.array([mover_xy(t) for t in tn_m], float).reshape(-1, 2)
        u_obj = (mov - lg['A']) @ lg['e']
        cov = np.abs(u_obj[:, None] - uk[None, :]) < h
        n_vis = vis.sum(0)
        f = np.where(n_vis > 0, (vis & cov).sum(0) / np.maximum(n_vis, 1), np.nan)
        m_end = map_at(d['maps'], t_stop - 0.05) if np.isfinite(t_stop) else d['maps'][-1]
        if m_end['t'] >= t_stop:            # ban /map cuoi cung TRUOC khi dung
            prev = [mp for mp in d['maps'] if mp['t'] < t_stop]
            m_end = prev[-1] if prev else m_end
        ce = cells.get(m_end)
        occ_e = (m_end['data'].ravel() >= 65) & ce['near']
        obs = np.array([bool((occ_e & (np.abs(ce['u'] - u) <= res / 2 + 1e-9)).any()) for u in uk])
        pred = f > 1.0 / 9.0
        ok = n_vis >= 5
        agree = float((pred[ok] == obs[ok]).mean()) if ok.any() else float('nan')
        n_seen = int(vis.any(axis=1).sum()) if len(vis) else 0

        # ---- SLAM trong luc vat chay / sau khi xoa
        tt = r['t'] + t0
        ph = r['phase']
        i1 = ph[1] if ph else len(tt) - 1
        e = r['e_slam']
        m1 = (tt >= ms['t']) & (tt < t_stop) & np.isfinite(e)
        m2 = (tt >= t_stop) & (np.arange(len(tt)) <= i1) & np.isfinite(e)

        def rms(x):
            return float(np.sqrt(np.mean(x ** 2)) * 100) if len(x) else float('nan')

        summ = dict(name=name, speed=float(ms['speed']), lane_len=L, stop_lap=stop_lap,
                    laps=len(laps), nodes_moving=int(len(tn_m)), nodes_seeing_lane=n_seen,
                    node_src=tsrc, steady=steady, peak=peak, before=before['total'],
                    before_near=before['near'], cleared_after=cleared, clear_thr=thr,
                    face_cols=int(len(uk)), face_occ_end=int(obs.sum()), face_pred=int(pred.sum()),
                    agree=agree, f_median=float(np.nanmedian(f)) if np.isfinite(f).any() else float('nan'),
                    slam_rms_moving=rms(e[m1]), slam_max_moving=float(np.max(e[m1]) * 100) if m1.any() else float('nan'),
                    slam_rms_after=rms(e[m2]), slam_max_after=float(np.max(e[m2]) * 100) if m2.any() else float('nan'),
                    tp_sent=stop.get('sent') if stop else None, tp_failed=stop.get('failed') if stop else None,
                    final_total=int(tot[-1]))
        summary.append(dict(s=summ, T=T, tot=tot, t_stop=t_stop - t0, lap_rows=lap_rows,
                            lap_t=np.array([tl - t0 for _, tl in laps])))

        # ---- file
        with open(outdir / f'{name}_moving.csv', 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        with open(outdir / f'{name}_moving_presence.csv', 'w', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['u_m', 'n_nodes_visible', 'presence_f', 'karto_ratio_f_over_1pf', 'pred_occ', 'map_occ'])
            for i in range(len(uk)):
                fi = f[i]
                wr.writerow([round(float(uk[i]), 3), int(n_vis[i]), '' if not np.isfinite(fi) else round(float(fi), 4),
                             '' if not np.isfinite(fi) else round(float(fi / (1 + fi)), 4), int(pred[i]), int(obs[i])])
        hdr = ['Vòng', 'Vật', 'Ô tường trong làn', 'mép phía robot', 'chỗ khác', 'ô trống trong làn']
        tl_rows = [[str(lr['lap']), 'chạy' if lr['moving'] else 'đã xóa', str(lr['total']), str(lr['near']),
                    str(lr['other'] + lr['far']), str(lr['free'])] for lr in lap_rows]
        write_table(outdir / f'{name}_moving_laps.tex', hdr, tl_rows)

        # ---- hinh
        snaps = []
        if lap_rows:
            snaps.append(('hết vòng 1', lap_rows[0]['t'] + 1.5))
        if np.isfinite(t_stop):
            snaps.append((f'trước khi xóa (vòng {stop_lap})', t_stop - t0 - 0.05))
            after = [lr for lr in lap_rows if lr['t'] > t_stop - t0]
            for k in (1, 3):
                if len(after) > k - 1 and after[k - 1] is not after[-1]:
                    snaps.append((f'+{after[k - 1]["lap"] - stop_lap} vòng sau khi xóa', after[k - 1]['t'] + 1.5))
        snaps.append(('cuối', None))
        fig = plt.figure(figsize=(15, 9.2))
        gsp = fig.add_gridspec(3, len(snaps), height_ratios=[0.62, 1.2, 0.75], hspace=0.42, wspace=0.08)
        Am = W.world_to_frame(lg['A'][None], SPAWN)[0]
        Bm = W.world_to_frame(lg['B'][None], SPAWN)[0]
        corners = W.world_to_frame(np.array([
            lg['A'] - h * lg['e'] - h * lg['n'], lg['B'] + h * lg['e'] - h * lg['n'],
            lg['B'] + h * lg['e'] + h * lg['n'], lg['A'] - h * lg['e'] + h * lg['n'],
            lg['A'] - h * lg['e'] - h * lg['n']]), SPAWN)
        outl = [W.world_to_frame(pl, SPAWN) for pl in W.outline_polylines()]
        lo, hi = np.minimum(Am, Bm) - h - 0.45, np.maximum(Am, Bm) + h + 0.45
        for j, (lab, tq) in enumerate(snaps):
            ax = fig.add_subplot(gsp[0, j])
            mp = d['maps'][-1] if tq is None else map_at(d['maps'], tq + t0)
            g = grid_dict(mp)
            hh, ww = g['grid'].shape
            ax.imshow(np.where(g['grid'] == 100, 0, np.where(g['grid'] == 0, 2, 1)), cmap=cmap, vmin=0, vmax=2,
                      extent=(mp['ox'], mp['ox'] + ww * mp['res'], mp['oy'], mp['oy'] + hh * mp['res']),
                      origin='upper', interpolation='nearest')
            for q in outl:
                ax.plot(q[:, 0], q[:, 1], color='#e8501c', lw=0.6, alpha=0.7)
            ax.plot(corners[:, 0], corners[:, 1], color='#8e44ad', lw=1.1, ls='--')
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            c = cells.counts(mp)
            ax.set_title(f'{lab}\n{c["total"]} ô tường ({c["near"]} ở mép)', fontsize=9)
        ax = fig.add_subplot(gsp[1, :len(snaps) // 2 + 1])
        if np.isfinite(t_stop):
            ax.axvspan(ms['t'] - t0, t_stop - t0, color='#8e44ad', alpha=0.08, lw=0)
            ax.axvline(t_rm - t0, color='k', lw=1.2)
            ax.text(t_rm - t0, 0.97, ' xóa vật', transform=ax.get_xaxis_transform(), va='top', fontsize=9)
        ax.plot(T, tot, color='#8e44ad', label='ô tường trong làn (tổng)')
        ax.plot(T, near, color='#1a9e5c', lw=1, label='… ở mép làn quay về robot')
        ax.plot(T, other, color='#e8901c', lw=1, label='… ở chỗ khác trong làn')
        for lap, tl in laps:
            ax.axvline(tl - t0, color='0.6', ls='--', lw=0.7)
            ax.text(tl - t0, 0.0, f' {lap}', fontsize=7, color='0.4', va='bottom',
                    transform=ax.get_xaxis_transform())
        ax.set_xlabel('thời gian mô phỏng (s); nền tím: vật đang chạy; nét đứt xám: hết vòng')
        ax.set_ylabel('số ô')
        ax.set_title('Ô "tường" trong làn của vật theo thời gian', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)
        ax = fig.add_subplot(gsp[1, len(snaps) // 2 + 1:])
        ax.bar(uk, np.nan_to_num(f), width=res * 0.9, color=np.where(pred, '#8e44ad', '#c9b3d9'))
        ax.axhline(1 / 9, color='k', ls='--', lw=1)
        ax.text(uk[0], 1 / 9, ' 1/9 (ngưỡng Karto)', va='bottom', fontsize=8)
        ax.plot(uk[obs], np.full(obs.sum(), -0.03), 's', color='#111111', ms=3.5,
                label='mép là tường trên bản đồ (trước khi xóa)')
        ax.set_ylim(-0.06, max(0.55, float(np.nanmax(f)) + 0.22) if np.isfinite(f).any() else 0.55)
        ax.set_xlabel('vị trí dọc làn (m, tính từ đầu A)')
        ax.set_ylabel('tỷ lệ có mặt f')
        ax.set_title(f'Tỷ lệ có mặt f của vật tại các nút, dọc mép làn (khớp bản đồ {agree * 100:.0f}% cột)',
                     fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.25, axis='y')
        ax = fig.add_subplot(gsp[2, :])
        ax.plot(r['t'], r['e_slam'] * 100, color='#1a9e5c', lw=0.8, label='SLAM: sai lệch vị trí (cm)')
        if np.isfinite(t_stop):
            ax.axvspan(ms['t'] - t0, t_stop - t0, color='#8e44ad', alpha=0.08, lw=0)
            ax.axvline(t_rm - t0, color='k', lw=1.2)
        ax.set_xlabel('thời gian mô phỏng (s)')
        ax.set_ylabel('cm')
        ax.set_title(f'SLAM: RMS {summ["slam_rms_moving"]:.1f} cm khi vật chạy, {summ["slam_rms_after"]:.1f} cm sau khi xóa',
                     fontsize=10)
        ax.grid(alpha=0.25)
        fig.suptitle(f'{name}: vật chạy qua lại {L:.1f} m, {ms["speed"]:g} m/s, xóa sau vòng {stop_lap}', fontsize=12)
        fig.savefig(outdir / f'{name}_moving.png', dpi=140, bbox_inches='tight')
        plt.close(fig)

        print(f'=== {name} (kich ban C) ===')
        print(f"  vat chay {lg['A'].tolist()} <-> {lg['B'].tolist()} ({L:.2f} m), {ms['speed']} m/s, "
              f"dung + xoa sau vong {stop_lap} (t={t_stop - t0:.1f} s)"
              + (f" | doi vat {summ['tp_sent']} lan, that bai {summ['tp_failed']}" if summ['tp_sent'] is not None else ''))
        print(f"  nut trong luc vat chay: {summ['nodes_moving']} (nguon: {tsrc}), {n_seen} nut thay mep lan")
        for lr in lap_rows:
            print(f"  het vong {lr['lap']:2d} ({'chay ' if lr['moving'] else 'da xoa'}, t={lr['t']:6.1f} s): "
                  f"o tuong trong lan {lr['total']:3d} (mep {lr['near']:3d}, cho khac {lr['other'] + lr['far']:3d}), "
                  f"o trong {lr['free']:3d}")
        print(f"  trong luc chay: trung binh het vong {steady:.1f} o, dinh {peak} | truoc khi xoa {before['total']} "
              f"(mep {before['near']}) | sach (<= {thr} o): "
              f"{'chua' if cleared is None else f'sau {cleared} vong'}")
        print(f"  mep lan: {summ['face_occ_end']}/{len(uk)} cot la tuong truoc khi xoa; du doan f > 1/9: "
              f"{summ['face_pred']} cot; khop {agree * 100:.0f}% | f trung vi {summ['f_median']:.3f}")
        print(f"  SLAM: RMS {summ['slam_rms_moving']:.2f} cm / max {summ['slam_max_moving']:.2f} cm khi vat chay; "
              f"{summ['slam_rms_after']:.2f} / {summ['slam_max_after']:.2f} cm sau khi xoa")
        print(f'-> {outdir}/{name}_moving.png, _moving.csv, _moving_presence.csv, _moving_laps.tex')

    if len(summary) > 1:
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, su in enumerate(summary):
            s, lt = su['s'], su['lap_t']
            if not np.isfinite(su['t_stop']) or len(lt) < 2:
                continue
            per = float(np.median(np.diff(lt)))
            x = (su['T'] - su['t_stop']) / per
            ax.plot(x, su['tot'], label=f"{s['name']} (chạy {s['stop_lap']} vòng, {s['speed']:g} m/s)")
        ax.axvline(0, color='k', lw=1.2)
        ax.set_xlabel('số vòng kể từ khi xóa vật (âm: vật đang chạy)')
        ax.set_ylabel('ô tường trong làn')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(outdir / 'moving_summary.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
    if summary:
        hdr = ['Chỉ số'] + [su['s']['name'].replace('_', r'\_') for su in summary]
        S_ = [su['s'] for su in summary]
        rows = [
            ['Tốc độ vật (m/s)'] + [fmt(s['speed'], 2) for s in S_],
            ['Số vòng vật chạy'] + [str(s['stop_lap']) for s in S_],
            ['Ô tường trong làn, TB hết vòng (khi chạy)'] + [fmt(s['steady'], 0) for s in S_],
            ['… ngay trước khi xóa (ở mép)'] + [f"{s['before']} ({s['before_near']})" for s in S_],
            ['Cột mép là tường / dự đoán $f>1/9$'] + [f"{s['face_occ_end']} / {s['face_pred']}" for s in S_],
            ['Số vòng để sạch sau khi xóa'] + ['--' if s['cleared_after'] is None else str(s['cleared_after'])
                                               for s in S_],
            ['SLAM RMS / max khi vật chạy (cm)'] + [f"{fmt(s['slam_rms_moving'])} / {fmt(s['slam_max_moving'])}"
                                                    for s in S_],
        ]
        write_table(outdir / 'moving_summary.tex', hdr, rows)
        with open(outdir / 'moving_summary.csv', 'w', newline='') as fh:
            wr = csv.writer(fh)
            for s in S_:
                for k, v in s.items():
                    wr.writerow([s['name'], k, v])
        print(f'-> {outdir}/moving_summary.tex/.csv' + (', moving_summary.png' if len(summary) > 1 else ''))


# ================================================================ main ==

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['speed', 'idle', 'dynamic', 'moving'])
    ap.add_argument('bags', nargs='+')
    ap.add_argument('-o', '--out', default='.')
    ap.add_argument('--labels', nargs='*', help='ten hien thi cho tung bag (che do speed)')
    ap.add_argument('--reader', choices=['auto', 'rosbag2_py', 'rosbags'], default='auto')
    a = ap.parse_args()
    bags = [b for b in a.bags if Path(b).expanduser().is_dir()]
    for b in a.bags:
        if b not in bags:
            print(f'bo qua {b}: khong phai thu muc bag (vd. file _slam_params.yaml)')
    a.bags = bags
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'speed':
        run_speed(a.bags, out, a.labels, a.reader)
    elif a.mode == 'idle':
        for b in a.bags:
            run_idle(b, out, a.reader)
    elif a.mode == 'dynamic':
        run_dynamic(a.bags, out, a.reader)
    else:
        run_moving(a.bags, out, a.reader)


if __name__ == '__main__':
    main()
