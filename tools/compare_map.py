#!/usr/bin/env python3
"""
compare_map.py - So sanh ban do SLAM (map_saver_cli: .yaml + .pgm) voi hinh hoc THAT
cua world tuan 5 (tools/wt05_world.py). Tra loi expected result 1 bang so do:
"ban do sinh ra co khop moi truong mo phong khong, va lech bao nhieu?"

Quan he giua cac frame:
  frame map cua slam_toolbox bat dau trung frame odom = tu the robot luc spawn.
  => diem p trong frame map nam o  S * p  trong world, voi S = tu the spawn (x, y, yaw).
  Khong co buoc can chinh tay nao: neu ban do dung, cac o "occupied" phai nam
  ngay tren be mat vat can that sau phep bien doi S.

Chi so:
  d(o)          khoang cach tu tam moi o occupied toi be mat vat can that gan nhat
                (mot o 5 cm chua be mat that thi tam o cach be mat <= ~3,5 cm)
  phantom       o occupied cach moi vat can > 15 cm (vat can "ma")
  do phu        % diem tren be mat that (phan lidar nhin thay duoc) co o occupied trong 10 cm
  free-in-solid o free nam sau trong vat can that (> 1 o) - ban do "xuyen tuong"
  dien tich     dien tich free da kham pha / dien tich free that cua phong
  can chinh     ICP 2D tim phep bien doi nho nhat dua ban do khop world; do lon cua no
                = sai lech toan cuc cua frame map (goc + huong), phan du sau can chinh
                = sai so hinh dang cuc bo.

Cach chay:
  python3 tools/compare_map.py ~/ros2_ws/maps/wt05_room.yaml -o ~/ros2_ws/bags/results_wt05
  python3 tools/compare_map.py MAP.yaml --traj results_wt05/<bag>_traj.csv   (ve them quy dao)
Can: numpy, matplotlib, pyyaml (co san khi cai ROS 2). scipy neu co thi nhanh hon.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wt05_world as W  # noqa: E402

OCC, FREE, UNKNOWN = 100, 0, -1


# ------------------------------------------------------------ doc ban do --

def read_pgm(path):
    data = Path(path).read_bytes()
    tokens, i = [], 0
    while len(tokens) < 4:                       # magic, width, height, maxval
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b'#':
            while data[i:i + 1] not in (b'\n', b''):
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j].decode())
        i = j
    magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    if magic != 'P5':
        raise ValueError(f'{path}: chi ho tro PGM nhi phan (P5), gap {magic}')
    i += 1                                       # mot ky tu trang sau maxval
    dtype = np.uint8 if maxval < 256 else '>u2'
    img = np.frombuffer(data, dtype=dtype, count=w * h, offset=i).reshape(h, w)
    return img.astype(float) * (255.0 / maxval)


def load_map(yaml_path):
    """Doc ban do theo dung quy uoc cua nav2_map_server. Tra ve dict:
    grid (H,W) gia tri OCC/FREE/UNKNOWN (hang 0 = phia tren anh = y lon nhat),
    resolution, origin (x, y, yaw), centers (H,W,2) toa do tam o trong frame map."""
    import yaml
    yaml_path = Path(yaml_path)
    meta = yaml.safe_load(yaml_path.read_text())
    img_path = (yaml_path.parent / meta['image']).resolve()
    if img_path.suffix.lower() == '.pgm':
        img = read_pgm(img_path)
    else:
        import matplotlib.image as mpimg
        img = mpimg.imread(img_path).astype(float)
        if img.max() <= 1.0:
            img = img * 255.0
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
    negate = int(meta.get('negate', 0))
    p = img / 255.0 if negate else (255.0 - img) / 255.0
    occ_t = float(meta.get('occupied_thresh', 0.65))
    free_t = float(meta.get('free_thresh', 0.25))
    grid = np.full(img.shape, UNKNOWN, dtype=int)
    grid[p > occ_t] = OCC
    grid[p < free_t] = FREE
    if meta.get('mode', 'trinary') == 'trinary' and not negate:
        # map_saver (trinary) ghi dung 3 gia tri: 0 occupied, 254 free, 205 unknown.
        # 205 -> p = 0,196 < free_thresh 0,25 nen ap nguong thuan se doc NHAM unknown
        # thanh free; vi vay nhan dien 205 truc tiep.
        grid[np.abs(img - 205.0) < 0.5] = UNKNOWN
    res = float(meta['resolution'])
    ox, oy, oyaw = (list(meta['origin']) + [0.0])[:3]
    h, w = grid.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xl = (cols + 0.5) * res
    yl = (h - 1 - rows + 0.5) * res
    c, s = math.cos(oyaw), math.sin(oyaw)
    centers = np.stack([ox + c * xl - s * yl, oy + s * xl + c * yl], axis=-1)
    return dict(grid=grid, resolution=res, origin=(ox, oy, oyaw), centers=centers,
                img=img, path=str(yaml_path))


# ------------------------------------------------------------ tien ich --

def nn_dist(a, b):
    """Voi moi diem cua a: khoang cach toi diem gan nhat cua b."""
    if len(b) == 0:
        return np.full(len(a), np.inf)
    try:
        from scipy.spatial import cKDTree
        return cKDTree(b).query(a)[0]
    except ImportError:
        out = np.empty(len(a))
        for i in range(0, len(a), 1024):
            d = np.linalg.norm(a[i:i + 1024, None, :] - b[None, :, :], axis=-1)
            out[i:i + 1024] = d.min(axis=1)
        return out


def compose(a, b):
    """Hop hai tu the 2D (x, y, yaw): a * b."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1],
            math.atan2(math.sin(a[2] + b[2]), math.cos(a[2] + b[2])))


def icp_align(pts_w, iters=40):
    """ICP 2D (diem -> be mat that). Tra ve (R, t): p' = R p + t."""
    R, t = np.eye(2), np.zeros(2)
    gate = 0.30
    for k in range(iters):
        cur = pts_w @ R.T + t
        q, d = W.closest_on_surface(cur)
        m = d < gate
        if m.sum() < 10:
            break
        P, Q = cur[m], q[m]
        mp, mq = P.mean(0), Q.mean(0)
        H = (P - mp).T @ (Q - mq)
        U, _, Vt = np.linalg.svd(H)
        Ri = Vt.T @ U.T
        if np.linalg.det(Ri) < 0:
            Vt[1] *= -1
            Ri = Vt.T @ U.T
        ti = mq - Ri @ mp
        R, t = Ri @ R, Ri @ t + ti
        gate = max(0.10, gate * 0.8)
        if np.linalg.norm(ti) < 1e-5 and abs(math.atan2(Ri[1, 0], Ri[0, 0])) < 1e-6:
            break
    return R, t


def free_area_truth(step=0.02):
    x0, x1, y0, y1 = W.ROOM
    xs, ys = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step))
    pts = np.c_[xs.ravel(), ys.ravel()]
    return (~W.inside_solid(pts)).sum() * step * step


def metrics(occ_w, free_w, res, surf):
    d = W.dist_to_surface(occ_w)
    nd = nn_dist(surf, occ_w)
    free_in_solid = W.inside_solid(free_w, margin=res).sum() if len(free_w) else 0
    return dict(
        n_occ=len(occ_w),
        d_mean=float(d.mean()), d_median=float(np.median(d)),
        d_p95=float(np.percentile(d, 95)), d_max=float(d.max()),
        within_1cell=float((d <= res).mean() * 100), within_2cell=float((d <= 2 * res).mean() * 100),
        phantom=int((d > 0.15).sum()),
        coverage=float((nd <= 0.10).mean() * 100),
        free_in_solid=int(free_in_solid),
    ), d


# ------------------------------------------------------------ chinh --

def analyze(m, spawn, align=True):
    g, res = m['grid'], m['resolution']
    occ_m = m['centers'][g == OCC]
    free_m = m['centers'][g == FREE]
    surf = W.surface_samples(step=0.02)

    occ_w = W.frame_to_world(occ_m, spawn)
    free_w = W.frame_to_world(free_m, spawn)
    nominal, d_nom = metrics(occ_w, free_w, res, surf)

    out = dict(spawn=spawn, nominal=nominal, d_nominal=d_nom, occ_m=occ_m)
    known = (g != UNKNOWN)
    out['free_area'] = float((g == FREE).sum() * res * res)
    out['free_area_truth'] = float(free_area_truth())
    out['map_size'] = (g.shape[1] * res, g.shape[0] * res)
    out['known_cells'] = int(known.sum())

    if align:
        R, t = icp_align(occ_w)
        dth = math.atan2(R[1, 0], R[0, 0])
        so = np.array(spawn[:2])
        new_origin = R @ so + t
        corrected = (float(new_origin[0]), float(new_origin[1]), spawn[2] + dth)
        aligned, d_al = metrics(occ_w @ R.T + t, free_w @ R.T + t, res, surf)
        out.update(aligned=aligned, d_aligned=d_al, corrected=corrected,
                   delta=(corrected[0] - spawn[0], corrected[1] - spawn[1], dth))
    return out


def plot(m, r, out_png, traj=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    g, res = m['grid'], m['resolution']
    ox, oy, oyaw = m['origin']
    h, w = g.shape
    extent = (ox, ox + w * res, oy, oy + h * res)      # dung khi origin yaw = 0 (map_saver)
    spawn = r['spawn']
    outlines = [W.world_to_frame(pl, spawn) for pl in W.outline_polylines()]
    occ_m, d = r['occ_m'], r['d_nominal']

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 0.75]))
    rgb = np.where(g == OCC, 0, np.where(g == FREE, 2, 1))
    cmap = ListedColormap(['#111111', '#bdbdbd', '#ffffff'])
    known = np.argwhere(g != UNKNOWN)
    pad = 0.6
    if len(known):
        r0, c0 = known.min(0)
        r1, c1 = known.max(0)
        xlim = (ox + c0 * res - pad, ox + (c1 + 1) * res + pad)
        ylim = (oy + (h - 1 - r1) * res - pad, oy + (h - r0) * res + pad)
    else:
        xlim, ylim = extent[:2], extent[2:]

    ax = axes[0]
    ax.imshow(rgb, cmap=cmap, vmin=0, vmax=2, extent=extent, origin='upper',
              interpolation='nearest')
    for pl in outlines:
        ax.plot(pl[:, 0], pl[:, 1], color='#e8501c', lw=1.0)
    if traj is not None:
        ax.plot(traj['gt_x'], traj['gt_y'], color='#f0a020', lw=1.2, label='ground truth')
        ax.plot(traj['slam_x'], traj['slam_y'], color='#1a9e5c', lw=1.0, ls='--', label='SLAM (map)')
        ax.legend(loc='upper right', fontsize=8)
    ax.plot(0, 0, marker='^', color='#2060d0', ms=9)
    ax.set_title('(a) Bản đồ SLAM và viền vật cản thật (đỏ)', fontsize=10)

    ax = axes[1]
    ax.imshow(np.where(g == UNKNOWN, 1, 2), cmap=cmap, vmin=0, vmax=2, extent=extent,
              origin='upper', interpolation='nearest', alpha=0.5)
    for pl in outlines:
        ax.plot(pl[:, 0], pl[:, 1], color='0.35', lw=0.6)
    sc = ax.scatter(occ_m[:, 0], occ_m[:, 1], c=np.minimum(d, 0.15) * 100, s=4,
                    cmap='viridis', vmin=0, vmax=15, marker='s', linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label='d tới bề mặt thật (cm, cắt ở 15)')
    ax.set_title('(b) Sai lệch từng ô occupied', fontsize=10)

    for ax in axes[:2]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('x frame map (m)')
        ax.set_ylabel('y frame map (m)')
        ax.grid(alpha=0.25)

    ax = axes[2]
    bins = np.arange(0, 0.205, 0.005) * 100
    ax.hist(np.minimum(d, 0.2) * 100, bins=bins, color='#2060d0', alpha=0.8, label='theo tư thế spawn')
    if 'd_aligned' in r:
        ax.hist(np.minimum(r['d_aligned'], 0.2) * 100, bins=bins, histtype='step', color='#e8501c',
                lw=1.5, label='sau căn chỉnh ICP')
    ax.axvline(res * 100, color='0.3', ls=':', lw=1)
    ax.text(res * 100, ax.get_ylim()[1] * 0.95, ' 1 ô', fontsize=8, va='top')
    ax.set_xlabel('d (cm)')
    ax.set_ylabel('số ô occupied')
    ax.set_title('(c) Phân bố sai lệch', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_outputs(r, outdir, name):
    rows = []
    for key, label in [('nominal', 'theo spawn'), ('aligned', 'sau ICP')]:
        if key in r:
            rows.append(dict(frame=label, **r[key]))
    with open(outdir / f'{name}_compare.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    def fmt(x, nd=1):
        return f'{x:.{nd}f}'.replace('.', '{,}')
    lines = [r'\begin{tabular}{lrr}', r'\toprule',
             r'Chỉ số & Theo tư thế spawn & Sau căn chỉnh ICP \\', r'\midrule']
    nom, al = r['nominal'], r.get('aligned', r['nominal'])
    spec = [('Số ô occupied', 'n_occ', 0, 1),
            (r'$d$ trung bình (cm)', 'd_mean', 1, 100), (r'$d$ trung vị (cm)', 'd_median', 1, 100),
            (r'$d$ phân vị 95\% (cm)', 'd_p95', 1, 100), (r'$d$ lớn nhất (cm)', 'd_max', 1, 100),
            (r'Ô có $d \le 1$ ô (\%)', 'within_1cell', 1, 1), (r'Ô có $d \le 2$ ô (\%)', 'within_2cell', 1, 1),
            (r"Ô ``ma'' ($d > 15$ cm)", 'phantom', 0, 1), (r'Độ phủ bề mặt thật (\%)', 'coverage', 1, 1),
            ('Ô free nằm trong vật cản', 'free_in_solid', 0, 1)]
    for label, k, nd, sc in spec:
        lines.append(f'{label} & {fmt(nom[k] * sc, nd)} & {fmt(al[k] * sc, nd)} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (outdir / f'{name}_compare.tex').write_text('\n'.join(lines) + '\n')


def print_report(r):
    res_line = lambda m: (  # noqa: E731
        f"  o occupied: {m['n_occ']}\n"
        f"  d (cm): TB {m['d_mean']*100:.2f} | trung vi {m['d_median']*100:.2f} | "
        f"p95 {m['d_p95']*100:.2f} | max {m['d_max']*100:.2f}\n"
        f"  d <= 1 o: {m['within_1cell']:.1f}% | d <= 2 o: {m['within_2cell']:.1f}% | "
        f"o ma (>15 cm): {m['phantom']}\n"
        f"  do phu be mat that (<=10 cm): {m['coverage']:.1f}% | o free trong vat can: {m['free_in_solid']}")
    print(f"Tu the spawn (goc frame map trong world): {r['spawn']}")
    print('--- Theo tu the spawn (khong can chinh) ---')
    print(res_line(r['nominal']))
    if 'aligned' in r:
        dx, dy, dth = r['delta']
        print(f'--- Sau can chinh ICP: frame map lech dx={dx*100:+.2f} cm, dy={dy*100:+.2f} cm, '
              f'dyaw={math.degrees(dth):+.3f} do ---')
        print(res_line(r['aligned']))
    print(f"Dien tich free: ban do {r['free_area']:.2f} m2 / that {r['free_area_truth']:.2f} m2 "
          f"({r['free_area'] / r['free_area_truth'] * 100:.1f}%)")
    print(f"Kich thuoc anh ban do: {r['map_size'][0]:.2f} x {r['map_size'][1]:.2f} m")


def load_traj(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in rows]) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map_yaml')
    ap.add_argument('--spawn', nargs=3, type=float, default=list(W.SPAWN),
                    metavar=('X', 'Y', 'YAW'), help='tu the spawn trong world (mac dinh theo wt05_world)')
    ap.add_argument('--no-align', action='store_true')
    ap.add_argument('--traj', help='CSV quy dao do analyze_wt05.py xuat ra (ve chong len ban do)')
    ap.add_argument('-o', '--out', default='.')
    a = ap.parse_args()

    m = load_map(a.map_yaml)
    r = analyze(m, tuple(a.spawn), align=not a.no_align)
    outdir = Path(a.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    name = Path(a.map_yaml).stem
    traj = load_traj(a.traj) if a.traj else None
    plot(m, r, outdir / f'{name}_compare.png', traj)
    write_outputs(r, outdir, name)
    print_report(r)
    print(f'-> {outdir / (name + "_compare.png")}, _compare.csv, _compare.tex')


if __name__ == '__main__':
    main()
