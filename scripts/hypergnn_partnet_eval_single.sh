# First export MODEL_PATH=snapshot/...pkl
# First export DATA_PATH=data/partnet/fpfh_affine_2-0/...npz
python -m src.project.hypergnn_main --use_features true --dataset partnet --root data/partnet/fpfh_affine_2-0 --eval_snapshot ${MODEL_PATH}  --full_data --config src/project/config/hypergnn_partnet_eval.yaml --eval_one_file ${DATA_PATH} --generation_method spectral-2