import argparse
import csv
import sys

import numpy as np


def load_pairs(csv_path):
    pairs = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                s = int(row[0])
                t = int(row[1])
            except ValueError:
                continue
            pairs.append((s, t))
    return pairs


def compute_offset(xyz0, xyz1, user_offset=None):
    if user_offset is not None:
        return np.array([user_offset, 0.0, 0.0], dtype=np.float32)
    mins = np.minimum(xyz0.min(axis=0), xyz1.min(axis=0))
    maxs = np.maximum(xyz0.max(axis=0), xyz1.max(axis=0))
    extent = np.linalg.norm(maxs - mins)
    return np.array([extent * 1.2, 0.0, 0.0], dtype=np.float32)


def colors_for_pairs(num_pairs):
    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    n = max(1, num_pairs)
    vals = np.linspace(0.0, 1.0, n)
    colors = cmap(vals)[:, :3]
    return colors.astype(np.float32)


def visualize_open3d(xyz0, xyz1, pairs, offset, point_size, title=None):
    import open3d as o3d

    xyz0 = xyz0.astype(np.float32)
    xyz1 = xyz1.astype(np.float32) + offset[None, :]
    if len(pairs) == 0:
        print("No pairs to visualize.")
        return
    pair_colors = colors_for_pairs(len(pairs))
    src_points = []
    tgt_points = []
    src_colors = []
    tgt_colors = []
    for i, (s, t) in enumerate(pairs):
        if 0 <= s < xyz0.shape[0] and 0 <= t < xyz1.shape[0]:
            src_points.append(xyz0[s])
            tgt_points.append(xyz1[t])
            src_colors.append(pair_colors[i])
            tgt_colors.append(pair_colors[i])
    if not src_points:
        print("No valid pairs within bounds.")
        return
    src_points = np.asarray(src_points, dtype=np.float32)
    tgt_points = np.asarray(tgt_points, dtype=np.float32)
    src_colors = np.asarray(src_colors, dtype=np.float32)
    tgt_colors = np.asarray(tgt_colors, dtype=np.float32)

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(src_points)
    pcd0.colors = o3d.utility.Vector3dVector(src_colors)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(tgt_points)
    pcd1.colors = o3d.utility.Vector3dVector(tgt_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title or "matching_pairs")
    vis.add_geometry(pcd0)
    vis.add_geometry(pcd1)
    render = vis.get_render_option()
    render.point_size = point_size
    vis.run()
    vis.destroy_window()


def visualize_matplotlib(xyz0, xyz1, pairs, offset, title=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    xyz0 = xyz0.astype(np.float32)
    xyz1 = xyz1.astype(np.float32) + offset[None, :]
    if len(pairs) == 0:
        print("No pairs to visualize.")
        return
    pair_colors = colors_for_pairs(len(pairs))
    src_points = []
    tgt_points = []
    src_colors = []
    tgt_colors = []
    for i, (s, t) in enumerate(pairs):
        if 0 <= s < xyz0.shape[0] and 0 <= t < xyz1.shape[0]:
            src_points.append(xyz0[s])
            tgt_points.append(xyz1[t])
            src_colors.append(pair_colors[i])
            tgt_colors.append(pair_colors[i])
    if not src_points:
        print("No valid pairs within bounds.")
        return
    src_points = np.asarray(src_points, dtype=np.float32)
    tgt_points = np.asarray(tgt_points, dtype=np.float32)
    src_colors = np.asarray(src_colors, dtype=np.float32)
    tgt_colors = np.asarray(tgt_colors, dtype=np.float32)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)
    ax.scatter(src_points[:, 0], src_points[:, 1], src_points[:, 2], c=src_colors, s=8)
    ax.scatter(tgt_points[:, 0], tgt_points[:, 1], tgt_points[:, 2], c=tgt_colors, s=8)
    ax.set_axis_off()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with src_idx,tgt_idx pairs")
    parser.add_argument("--npz", required=True, help="NPZ file with schema in output_schema.md")
    parser.add_argument("--select_pairs", type=int, default=1000, help="Pairs to sample each round")
    parser.add_argument("--rounds", type=int, default=5, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--offset", type=float, default=None, help="Manual x-offset for target cloud")
    parser.add_argument("--point_size", type=float, default=2.0, help="Point size for Open3D viewer")
    args = parser.parse_args()

    data = np.load(args.npz)
    if "xyz0" not in data or "xyz1" not in data:
        raise ValueError("NPZ must contain xyz0 and xyz1")
    xyz0 = data["xyz0"]
    xyz1 = data["xyz1"]

    pairs = load_pairs(args.csv)
    if len(pairs) == 0:
        print("No pairs found in CSV.")
        return
    offset = compute_offset(xyz0, xyz1, args.offset)

    rng = np.random.default_rng(args.seed)
    rounds = max(1, args.rounds)
    select_pairs = max(1, args.select_pairs)
    if len(pairs) < select_pairs:
        select_pairs = max(1, int(len(pairs) * 0.4))
    for r in range(rounds):
        if len(pairs) <= select_pairs:
            sampled = pairs
        else:
            idx = rng.choice(len(pairs), size=select_pairs, replace=False)
            sampled = [pairs[i] for i in idx]
        title = f"sample {r + 1}/{rounds} ({len(sampled)} pairs)"
        try:
            visualize_open3d(xyz0, xyz1, sampled, offset, args.point_size, title=title)
        except Exception as exc:
            print(f"Open3D visualization failed ({exc}). Falling back to matplotlib.", file=sys.stderr)
            visualize_matplotlib(xyz0, xyz1, sampled, offset, title=title)


if __name__ == "__main__":
    main()
