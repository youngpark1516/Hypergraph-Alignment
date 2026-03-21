#!/usr/bin/env bash
set -euo pipefail

PC_ROOT="data/3dmatch"
EVAL_ROOT="data/3dmatch"
PAIR_OUT="data/3dmatch/pairs"
FPFH_OUT="data/3dmatch/fpfh"

echo "Processing 3DMatch data..."

if ls "$PAIR_OUT"/**/*.npz >/dev/null 2>&1 || ls "$PAIR_OUT"/*.npz >/dev/null 2>&1; then
    echo "Pairs already exist at $PAIR_OUT - skipping pair generation"
else
    echo "Building pairs into $PAIR_OUT..."
    python src/project/utils/data_utils/3dmatch/build_correspondences.py \
        --pc-root "$PC_ROOT" \
        --eval-root "$EVAL_ROOT" \
        --out-root "$PAIR_OUT" \
        --max-frame-gap 0 \
        --match-radius 0.10 \
        --min-corr 256
fi

if ls "$FPFH_OUT"/**/*.npz >/dev/null 2>&1 || ls "$FPFH_OUT"/*.npz >/dev/null 2>&1; then
    echo "FPFH features already exist at $FPFH_OUT - skipping get_fpfh"
else
    echo "Computing FPFH features into $FPFH_OUT..."
    python src/project/utils/fpfh/get_fpfh.py \
        --dataset 3dmatch \
        --pair-root "$PAIR_OUT" \
        --save-root "$FPFH_OUT"
fi

echo "3DMatch processing complete."
