#!/usr/bin/env python3
"""
wt05_world.py - Hinh hoc cua world mo phong tuan 5 (nguon su that duy nhat).

Cung mot danh sach SHAPES duoc dung cho hai viec:
  1. Sinh file worlds/wt05_room.world cho Gazebo Classic   (chay script nay)
  2. Lam "ban do chuan" de so sanh voi ban do SLAM          (compare_map.py import)
Nho vay world trong Gazebo va ban do chuan khong bao gio lech nhau.

Bo cuc (he toa do world, don vi m; nhin tu tren xuong, +x sang phai, +y len tren):

  y= 3 +----------------------------------------------+
       |  [box_nw]        |stub                       |
       |   (xoay)         |              [box_ne]     |
       |                                               |
       |        +---------------------+                |
       |        |       island        |                |
       |        +---------------------+          +----|
       |                                         |alcv|
       |(cyl_sw)                (cyl_s)          +----|
  y=-3 +---------------- S ----------------------------+
      x=-4             spawn (0, -1.8)                x=4

Robot xuat phat o S, chay vong NGUOC chieu kim dong ho quanh "island"
-> quay lai diem xuat phat de kich hoat loop closure.

Cach chay:  python3 tools/wt05_world.py            (ghi worlds/wt05_room.world)
"""

import math
from pathlib import Path

import numpy as np

ROOM = (-4.0, 4.0, -3.0, 3.0)      # xmin, xmax, ymin, ymax: mat trong cua tuong bao
WALL_T = 0.2                       # be day tuong bao
SPAWN = (0.0, -1.8, 0.0)           # x, y, yaw cua robot khi spawn (trong world)

# (ten, loai, tham so, chieu cao, mau Gazebo)
#   box:      (cx, cy, sx, sy, yaw)
#   cylinder: (cx, cy, r)
SHAPES = [
    ('wall_north', 'box', (0.0, 3.0 + WALL_T / 2, 8.0 + 2 * WALL_T, WALL_T, 0.0), 1.0, 'Gazebo/Grey'),
    ('wall_south', 'box', (0.0, -3.0 - WALL_T / 2, 8.0 + 2 * WALL_T, WALL_T, 0.0), 1.0, 'Gazebo/Grey'),
    ('wall_east', 'box', (4.0 + WALL_T / 2, 0.0, WALL_T, 6.0, 0.0), 1.0, 'Gazebo/Grey'),
    ('wall_west', 'box', (-4.0 - WALL_T / 2, 0.0, WALL_T, 6.0, 0.0), 1.0, 'Gazebo/Grey'),
    ('island', 'box', (0.0, 0.0, 3.0, 1.2, 0.0), 0.8, 'Gazebo/Wood'),
    ('stub', 'box', (-0.5, 2.5, 0.15, 1.0, 0.0), 1.0, 'Gazebo/Grey'),
    ('alcove', 'box', (3.7, -1.0, 0.6, 1.0, 0.0), 1.0, 'Gazebo/Grey'),
    ('box_ne', 'box', (3.0, 2.2, 0.6, 0.6, 0.0), 0.5, 'Gazebo/Orange'),
    ('box_nw', 'box', (-3.0, 2.2, 0.8, 0.4, 0.5), 0.5, 'Gazebo/Orange'),
    ('cyl_sw', 'cylinder', (-3.3, -2.4, 0.3), 0.6, 'Gazebo/Green'),
    ('cyl_s', 'cylinder', (1.2, -2.6, 0.2), 0.6, 'Gazebo/Green'),
]

# Duong chay goi y (diem re), vong nguoc chieu kim dong ho, khep kin ve S
ROUTE = [(0.0, -1.8), (2.45, -1.8), (2.45, 1.3), (-2.75, 1.3), (-2.75, -1.8), (0.0, -1.8)]

# Vat "mover" cho thi nghiem moi truong dong (bao cao 05.2). KHONG thuoc SHAPES: ban do
# chuan cua compare_map van la phong tinh. Hop 0,5 x 0,5 m, cao 0,5 m (cat mat phang quet
# 0,17 m). P1: dai bac giua stub va box_ne, cach duong chay 0,75 m. P2: dai nam, vung
# trong da duoc quet o vong dau. route_driver.py dung dung cac gia tri nay (tham so mac dinh).
MOVER = dict(name='mover', size=0.5, height=0.5, material='Gazebo/Purple',
             P1=(1.2, 2.3), P2=(-1.5, -2.5))


# ------------------------------------------------------------ hinh hoc --

def box_corners(cx, cy, sx, sy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(+sx / 2, +sy / 2), (-sx / 2, +sy / 2), (-sx / 2, -sy / 2), (+sx / 2, -sy / 2)]
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts]


def segments():
    """Canh cua moi hinh hop: mang (N, 4) = x1, y1, x2, y2."""
    segs = []
    for _, kind, p, _, _ in SHAPES:
        if kind == 'box':
            cs = box_corners(*p)
            segs += [(*cs[i], *cs[(i + 1) % 4]) for i in range(4)]
    return np.array(segs, dtype=float)


def circles():
    return np.array([p for _, kind, p, _, _ in SHAPES if kind == 'cylinder'], dtype=float)


def closest_on_surface(pts):
    """Diem gan nhat tren be mat vat can va khoang cach (m) - chinh xac, giai tich.
    Tra ve (closest (N,2), dist (N,))."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    best = np.full(len(pts), np.inf)
    cp = np.zeros_like(pts)
    seg = segments()
    a, b = seg[None, :, 0:2], seg[None, :, 2:4]          # (1,m,2)
    ab = b - a
    for i in range(0, len(pts), 4096):
        p = pts[i:i + 4096, None, :]                     # (n,1,2)
        t = np.clip(((p - a) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
        q = a + t[..., None] * ab                        # (n,m,2)
        d = np.linalg.norm(p - q, axis=-1)
        k = d.argmin(axis=1)
        rows = np.arange(len(k))
        best[i:i + 4096] = d[rows, k]
        cp[i:i + 4096] = q[rows, k]
    for cx, cy, r in circles():
        v = pts - [cx, cy]
        n = np.linalg.norm(v, axis=1)
        d = np.abs(n - r)
        better = d < best
        u = v[better] / np.maximum(n[better], 1e-9)[:, None]
        cp[better] = [cx, cy] + r * u
        best[better] = d[better]
    return cp, best


def dist_to_surface(pts):
    """Khoang cach (m) tu moi diem (N,2) toi be mat vat can gan nhat."""
    return closest_on_surface(pts)[1]


def inside_solid(pts, margin=0.0):
    """True neu diem nam trong (sau hon `margin`) mot vat can."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    inside = np.zeros(len(pts), dtype=bool)
    for _, kind, p, _, _ in SHAPES:
        if kind == 'box':
            cx, cy, sx, sy, yaw = p
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            inside |= (np.abs(lx) < sx / 2 - margin) & (np.abs(ly) < sy / 2 - margin)
        else:
            cx, cy, r = p
            inside |= np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < r - margin
    return inside


def in_room(pts, margin=0.0):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    x0, x1, y0, y1 = ROOM
    return ((pts[:, 0] > x0 - margin) & (pts[:, 0] < x1 + margin)
            & (pts[:, 1] > y0 - margin) & (pts[:, 1] < y1 + margin))


def surface_samples(step=0.02):
    """Cac diem tren be mat vat can MA LIDAR CO THE NHIN THAY: nam trong phong va
    khong bi mot vat can khac che lap (vd. mat cua hop alcove ap vao tuong)."""
    pts = []
    for x1, y1, x2, y2 in segments():
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / step) + 1)
        t = np.linspace(0, 1, n)
        pts.append(np.c_[x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    for cx, cy, r in circles():
        n = max(8, int(2 * math.pi * r / step))
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        pts.append(np.c_[cx + r * np.cos(a), cy + r * np.sin(a)])
    pts = np.vstack(pts)
    # Day diem ra 1 cm theo moi huong roi kiem tra: neu MOI huong deu roi vao vat can
    # (hoac ra ngoai phong) thi mat do bi che, lidar khong the thay.
    keep = np.zeros(len(pts), dtype=bool)
    for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        q = pts + 0.01 * np.array([math.cos(ang), math.sin(ang)])
        keep |= in_room(q) & ~inside_solid(q)
    return pts[keep]


def world_to_frame(pts, pose):
    """Bieu dien diem world trong he toa do co goc tai pose = (x, y, yaw)."""
    x0, y0, th = pose
    c, s = math.cos(th), math.sin(th)
    d = np.asarray(pts, dtype=float) - [x0, y0]
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def frame_to_world(pts, pose):
    x0, y0, th = pose
    c, s = math.cos(th), math.sin(th)
    p = np.asarray(pts, dtype=float)
    return np.c_[x0 + c * p[:, 0] - s * p[:, 1], y0 + s * p[:, 0] + c * p[:, 1]]


def outline_polylines():
    """Duong vien tung vat can (de ve chong len ban do)."""
    out = []
    for _, kind, p, _, _ in SHAPES:
        if kind == 'box':
            cs = box_corners(*p)
            out.append(np.array(cs + cs[:1]))
        else:
            a = np.linspace(0, 2 * math.pi, 64)
            out.append(np.c_[p[0] + p[2] * np.cos(a), p[1] + p[2] * np.sin(a)])
    return out


# ------------------------------------------------------------ sinh SDF --

def _model_sdf(name, kind, p, h, mat):
    if kind == 'box':
        cx, cy, sx, sy, yaw = p
        geom = f'<box><size>{sx:.4f} {sy:.4f} {h:.3f}</size></box>'
        pose = f'{cx:.4f} {cy:.4f} {h / 2:.3f} 0 0 {yaw:.4f}'
    else:
        cx, cy, r = p
        geom = f'<cylinder><radius>{r:.4f}</radius><length>{h:.3f}</length></cylinder>'
        pose = f'{cx:.4f} {cy:.4f} {h / 2:.3f} 0 0 0'
    return f'''    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        <collision name="collision"><geometry>{geom}</geometry></collision>
        <visual name="visual">
          <geometry>{geom}</geometry>
          <material><script>
            <uri>file://media/materials/scripts/gazebo.material</uri><name>{mat}</name>
          </script></material>
        </visual>
      </link>
    </model>
'''


def mover_shape(at='P1'):
    m = MOVER
    x, y = m[at]
    return (m['name'], 'box', (x, y, m['size'], m['size'], 0.0), m['height'], m['material'])


def world_sdf(with_mover=False):
    shapes = list(SHAPES) + ([mover_shape('P1')] if with_mover else [])
    models = ''.join(_model_sdf(*s) for s in shapes)
    return f'''<?xml version="1.0"?>
<!-- SINH TU DONG boi tools/wt05_world.py - sua SHAPES trong script roi chay lai, khong sua tay. -->
<sdf version="1.6">
  <world name="wt05_room">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>

    <!-- Giong empty.world cua tuan 4: buoc 1 ms, 1000 Hz, RTF 1 -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Tat bong do de giam tai do hoa tren WSLg -->
    <scene>
      <shadows>false</shadows>
      <ambient>0.5 0.5 0.5 1</ambient>
      <background>0.75 0.75 0.75 1</background>
    </scene>

    <!-- Camera nhin gan nhu thang tu tren xuong, +x sang phai, +y len tren: cung
         huong voi ban do SLAM de chup anh so sanh -->
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>0 -1.2 12.5 0 1.45 1.5708</pose>
      </camera>
    </gui>

{models}  </world>
</sdf>
'''


def main():
    wdir = Path(__file__).resolve().parent.parent / 'worlds'
    wdir.mkdir(exist_ok=True)
    (wdir / 'wt05_room.world').write_text(world_sdf())
    (wdir / 'wt05_room_mover.world').write_text(world_sdf(with_mover=True))
    print(f'Da ghi {wdir}/wt05_room.world ({len(SHAPES)} vat the) va '
          f'wt05_room_mover.world (+ mover tai P1 {MOVER["P1"]})')


if __name__ == '__main__':
    main()
