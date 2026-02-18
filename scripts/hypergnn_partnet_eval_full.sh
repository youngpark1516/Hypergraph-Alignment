# First export MODEL_PATH=snapshot/...pkl
python -m src.project.hypergnn_main --use_features true --dataset partnet --root data/partnet/fpfh_affine_2-0 --eval_snapshot ${MODEL_PATH} --full_data --config src/project/config/hypergnn_partnet_eval.yaml --generation_method spectral-2
