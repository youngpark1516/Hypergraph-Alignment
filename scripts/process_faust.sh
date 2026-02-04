#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="data/faust"
PC_OUT="data/faust/point_clouds"
CORRES_OUT="data/faust/corres"
MAPS_DIR="$CORRES_OUT/maps"
PAIRS_DIR="$CORRES_OUT/pairs"
FPFH_OUT="data/faust/fpfh"

echo "Processing FAUST data..."

# Generate point clouds for train split if not already present
if ls "$PC_OUT/train"/*.npy >/dev/null 2>&1; then
	echo "Point clouds for 'train' already exist at $PC_OUT/train — skipping get_pc for train"
else
	echo "Generating point clouds for train..."
	python src/project/utils/data_utils/faust/get_pc.py --raw-root "$RAW_ROOT" --out-root "$PC_OUT" --splits train
fi

# Generate point clouds for test split if not already present
if ls "$PC_OUT/test"/*.npy >/dev/null 2>&1; then
	echo "Point clouds for 'test' already exist at $PC_OUT/test — skipping get_pc for test"
else
	echo "Generating point clouds for test..."
	python src/project/utils/data_utils/faust/get_pc.py --raw-root "$RAW_ROOT" --out-root "$PC_OUT" --splits test
fi

# Build maps only if maps directory is missing or empty
if ls "$MAPS_DIR"/*.npy >/dev/null 2>&1; then
	echo "Maps already exist at $MAPS_DIR — skipping --build-maps"
else
	echo "Building maps into $MAPS_DIR..."
	python src/project/utils/data_utils/faust/build_correspondences.py --raw-root "$RAW_ROOT" --out-root "$CORRES_OUT" --build-maps
fi

# Build all pairs only if pairs dir is missing or empty
if ls "$PAIRS_DIR"/*.npz >/dev/null 2>&1; then
	echo "Pairs already exist at $PAIRS_DIR — skipping --build-all-pairs"
else
	echo "Building all pairs into $PAIRS_DIR..."
	python src/project/utils/data_utils/faust/build_correspondences.py --raw-root "$RAW_ROOT" --out-root "$CORRES_OUT" --build-all-pairs
fi

# Compute FPFH features only if output directory is missing or empty
if ls "$FPFH_OUT"/**/*.npz >/dev/null 2>&1 2>/dev/null || ls "$FPFH_OUT"/*.npz >/dev/null 2>&1; then
	echo "FPFH features already exist at $FPFH_OUT — skipping get_fpfh"
else
	echo "Computing FPFH features into $FPFH_OUT..."
	python src/project/utils/fpfh/get_fpfh.py --dataset faust --pair-root "$PAIRS_DIR" --save-root "$FPFH_OUT"
fi

echo "FAUST processing complete."