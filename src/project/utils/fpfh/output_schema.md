# get_fpfh Output Schema

This file documents the `.npz` format written by `src/project/utils/fpfh/get_fpfh.py`.

## File layout

Each output `.npz` represents one paired sample.

## Keys and shapes

All arrays are saved as `float32` unless noted otherwise.

- `xyz0`: `(N0, 3)`  
  Source point cloud after processing. When `--no-downsample` is set, this is the original `xyz0`. Otherwise, this is still the original `xyz0`, and features are interpolated back per point.
- `xyz1`: `(N1, 3)`  
  Target point cloud after processing. Same rules as `xyz0`.
- `features0`: `(N0, 33)`  
  FPFH features for `xyz0` (Open3D FPFH has 33 bins).
- `features1`: `(N1, 33)`  
  FPFH features for `xyz1`.
- `normal0`: `(N0, 3)`  
  Estimated normals for `xyz0` (either direct or interpolated from downsampled normals).
- `normal1`: `(N1, 3)`  
  Estimated normals for `xyz1`.
- `corres`: `(N0,)` `int32`  
  Index mapping from each source point in `xyz0` to its corresponding index in `xyz1`.

## Notes

- When downsampling is enabled, FPFH + normals are computed on the voxelized cloud and then interpolated to the original points via k-NN averaging (`--interp-k`).
- The output does **not** include `gt_trans`. If you need the ground-truth transform, keep it from the original dataset file.
