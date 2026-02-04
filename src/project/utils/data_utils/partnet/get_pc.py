import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import sapien
from tqdm import tqdm


def load_scene_with_urdf(urdf_path: Path):
    # In SAPIEN 3, Scene is the entry point; no explicit renderer is needed.
    scene = sapien.Scene()
    urdf_loader = scene.create_urdf_loader()
    urdf_loader.fix_root_link = True

    articulation = None
    try:
        articulation = urdf_loader.load(str(urdf_path))
    except Exception as exc:
        # Fallback for URDFs with multiple objects or other load quirks.
        warnings.warn(f"Primary URDF load failed, falling back to load_multiple: {exc}")

    if articulation is None:
        # load_multiple returns (articulations, visual_bodies); either side may be empty
        warnings.warn("URDF load returned None; using load_multiple to populate render bodies.")
        urdf_loader.load_multiple(str(urdf_path))

    scene.update_render()
    return scene


def meshes_to_point_cloud(scene) -> np.ndarray:
    points = []

    for render_body in scene.render_system.render_bodies:
        body_pose = render_body.get_entity_pose()
        body_tf = body_pose.to_transformation_matrix()

        for shape in render_body.render_shapes:
            shape_verts = []

            if hasattr(shape, "get_vertices"):
                shape_verts.append(np.asarray(shape.get_vertices(), dtype=np.float32))
            elif hasattr(shape, "get_parts"):
                for part in shape.get_parts():
                    if hasattr(part, "get_vertices"):
                        shape_verts.append(np.asarray(part.get_vertices(), dtype=np.float32))

            if not shape_verts:
                continue

            local_tf = shape.get_local_pose().to_transformation_matrix()
            transform = body_tf @ local_tf

            for verts in shape_verts:
                if verts.size == 0:
                    continue
                homo = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=np.float32)], axis=1)
                world_vertices = (transform @ homo.T).T[:, :3]
                points.append(world_vertices)

    if not points:
        raise RuntimeError("No render meshes found to convert to a point cloud.")
    return np.concatenate(points, axis=0)


def extract_name(dataset_dir: Path) -> str:
    result_json = dataset_dir / "result.json"
    if not result_json.is_file():
        raise FileNotFoundError(f"result.json not found under {dataset_dir}")
    with result_json.open("r") as f:
        result = json.load(f)
    if not result or "name" not in result[0]:
        raise ValueError(f"result.json under {dataset_dir} missing name field")
    return result[0]["name"]


def process_index(idx: int, dataset_root: Path, out_root: Path) -> Path | None:
    dataset_dir = dataset_root / str(idx)
    if not dataset_dir.is_dir():
        warnings.warn(f"Skipping {idx}: directory not found under {dataset_root}")
        return None

    urdf_path = dataset_dir / "mobility.urdf"
    if not urdf_path.is_file():
        warnings.warn(f"Skipping {idx}: mobility.urdf missing in {dataset_dir}")
        return None

    try:
        name = extract_name(dataset_dir)
    except Exception as exc:
        warnings.warn(f"{idx}: failed to read name from {dataset_dir}/result.json ({exc})")
        name = f"part_{idx}"

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pc_{idx}.npy"

    try:
        scene = load_scene_with_urdf(urdf_path)
        cloud = meshes_to_point_cloud(scene)
        np.save(out_path, cloud)
        return out_path
    except Exception as exc:
        warnings.warn(f"{idx}: processing failed ({exc})")
        return None


def process_range(start: int | None, end: int | None, dataset_root: Path, out_root: Path):
    if start is None or end is None:
        indices = sorted(int(p.name) for p in dataset_root.iterdir() if p.is_dir() and p.name.isdigit())
        if not indices:
            raise ValueError(f"No numeric directories found under {dataset_root}")
    else:
        indices = list(range(start, end + 1))

    saved = []
    for idx in tqdm(indices, desc="Processing datasets", unit="item"):
        out_path = process_index(idx, dataset_root, out_root)
        if out_path:
            saved.append(out_path)
    print(f"Processed {len(saved)} items.")
    if saved:
        print("Saved files:")
        for p in saved:
            print(f" - {p}")


def main():
    parser = argparse.ArgumentParser(description="Process local PartNet-Mobility dataset into point clouds.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"), help="Path to local dataset root containing numeric directories")
    parser.add_argument("--start", type=int, help="Start index for processing (inclusive). If omitted, process all directories.")
    parser.add_argument("--end", type=int, help="End index for processing (inclusive). If omitted, process all directories.")
    parser.add_argument("--out-root", type=Path, default=Path("point_clouds"), help="Output root directory for saved point clouds")
    args = parser.parse_args()

    if (args.start is None) ^ (args.end is None):
        raise ValueError("Provide both --start and --end, or neither to process all datasets.")

    process_range(args.start, args.end, Path(args.dataset_root), Path(args.out_root))


if __name__ == "__main__":
    main()
