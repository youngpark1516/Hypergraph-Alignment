# Example generating rigid pairs, and affine pairs with different levels of noise
python src/project/utils/data_utils/partnet/augment_point_clouds.py --pc-root data/partnet/point_clouds --out-root data/partnet/rigid
python src/project/utils/data_utils/partnet/augment_point_clouds.py --pc-root data/partnet/point_clouds --affine --sigma-max 1.5 --out-root data/partnet/affine_1-5 
python src/project/utils/data_utils/partnet/augment_point_clouds.py --pc-root data/partnet/point_clouds --affine --sigma-max 2.0 --out-root data/partnet/affine_2-0

for ds in "rigid" "affine_1-5" "affine_2-0"; do
    python src/project/utils/fpfh/get_fpfh.py --dataset partnet --pair-root data/partnet/${ds} --save-root data/partnet/fpfh_${ds}
done