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
    for s, t in pairs:
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


def visualize_matplotlib_projections(xyz0, xyz1, pairs, title=None):
    import matplotlib.pyplot as plt

    xyz0 = xyz0.astype(np.float32)
    xyz1 = xyz1.astype(np.float32)
    if len(pairs) == 0:
        print("No pairs to visualize.")
        return
    src_points = []
    tgt_points = []
    for i, (s, t) in enumerate(pairs):
        if 0 <= s < xyz0.shape[0] and 0 <= t < xyz1.shape[0]:
            src_points.append(xyz0[s])
            tgt_points.append(xyz1[t])
    if not src_points:
        print("No valid pairs within bounds.")
        return
    src_points = np.asarray(src_points, dtype=np.float32)
    tgt_points = np.asarray(tgt_points, dtype=np.float32)
    src_color = np.array([0.121, 0.466, 0.705], dtype=np.float32)
    tgt_color = np.array([1.0, 0.498, 0.054], dtype=np.float32)

    projections = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if title is not None:
        fig.suptitle(title)
    for ax, (name, d0, d1) in zip(axes, projections):
        ax.scatter(src_points[:, d0], src_points[:, d1], c=src_color[None, :], s=18, marker=".", label="src")
        ax.scatter(tgt_points[:, d0], tgt_points[:, d1], c=tgt_color[None, :], s=18, marker=".", label="tgt")
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
        all_x = np.concatenate([src_points[:, d0], tgt_points[:, d0]])
        all_y = np.concatenate([src_points[:, d1], tgt_points[:, d1]])
        x_min, x_max = float(all_x.min()), float(all_x.max())
        y_min, y_max = float(all_y.min()), float(all_y.max())
        x_pad = max((x_max - x_min) * 0.08, 1e-3)
        y_pad = max((y_max - y_min) * 0.08, 1e-3)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with src_idx,tgt_idx pairs")
    parser.add_argument("--npz", required=True, help="NPZ file with schema in output_schema.md")
    parser.add_argument("--select_pairs", type=int, default=25, help="Pairs to sample each round")
    parser.add_argument("--rounds", type=int, default=5, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--offset", type=float, default=None, help="Manual x-offset for target cloud")
    parser.add_argument("--point_size", type=float, default=2.0, help="Point size for Open3D viewer")
    parser.add_argument("--plot_mode", type=str, default="3d", choices=["3d", "proj"],
                        help="3d: interactive 3D view; proj: XY/XZ/YZ projections")
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
        if args.plot_mode == "proj":
            visualize_matplotlib_projections(xyz0, xyz1, sampled, title=title)
        else:
            try:
                visualize_open3d(xyz0, xyz1, sampled, offset, args.point_size, title=title)
            except Exception as exc:
                print(f"Open3D visualization failed ({exc}). Falling back to matplotlib.", file=sys.stderr)
                visualize_matplotlib(xyz0, xyz1, sampled, offset, title=title)


if __name__ == "__main__":
    main()
