python -m src.project.hypergct_main --use_features true --dataset partnet --use_wandb --root ${DATA_PATH} --config src/project/config/hypergct_partnet.yaml
# The root data/partnet/fpfh_affine_2-0 is just an example.

# For no features: python -m src.project.hypergct_main --use_features false --dataset partnet --use_wandb --root data/partnet/fpfh_affine_2-0