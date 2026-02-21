import argparse
import os
import time
from pathlib import Path
from datetime import datetime

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


config_dir = Path(__file__).resolve().parent
dataset_configs = {
    'partnet': config_dir / 'hypergnn_partnet.yaml',
    'faust': config_dir / 'hypergnn_faust.yaml',
    '3DMatch': config_dir / 'hypergnn_3dmatch.yaml',
    'KITTI': config_dir / 'hypergnn_kitti.yaml',
}

# Loss configurations
loss_arg = add_argument_group('Loss')
loss_arg.add_argument('--evaluate_interval', type=int, default=1)
loss_arg.add_argument('--balanced', type=str2bool, default=False)
loss_arg.add_argument('--weight_classification', type=float, default=1.0)
loss_arg.add_argument('--weight_spectralmatching', type=float, default=1.0)
loss_arg.add_argument('--weight_hypergraph', type=float, default=1.0)
loss_arg.add_argument('--weight_transformation', type=float, default=0.0)
loss_arg.add_argument('--transformation_loss_start_epoch', type=int, default=0)

# Optimizer configurations
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'])
opt_arg.add_argument('--max_epoch', type=int, default=50)
opt_arg.add_argument('--training_max_iter', type=int, default=3500)
opt_arg.add_argument('--val_max_iter', type=int, default=1000)
opt_arg.add_argument('--lr', type=float, default=1e-4)
opt_arg.add_argument('--weight_decay', type=float, default=1e-6)
opt_arg.add_argument('--momentum', type=float, default=0.9)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--scheduler_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler_interval', type=int, default=1)
opt_arg.add_argument('--pooling_layer_idx', type=str, default='-1',
                     help='layer indices for wasserstein pooling (e.g., -1 or 1,3,5)')

# Dataset and dataloader configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, choices=sorted(dataset_configs.keys()))
data_arg.add_argument('--root', type=str, default='data/partnet/fpfh_rigid')
data_arg.add_argument('--descriptor', type=str, default='fpfh', choices=['fpfh', 'fcgf'])
data_arg.add_argument('--inlier_threshold', type=float, default=0.10)
data_arg.add_argument('--downsample', type=float, default=0.02)
data_arg.add_argument('--re_thre', type=float, default=15,
                      help='rotation error thrshold (deg)')
data_arg.add_argument('--te_thre', type=float, default=30,
                      help='translation error thrshold (cm)')

data_arg.add_argument('--val_ratio', type=float, default=0.1)
data_arg.add_argument('--use_features', type=str2bool, default=True)
data_arg.add_argument('--feature_dim', type=int, default=33)
data_arg.add_argument('--neg_ratio', type=float, default=2.0)
data_arg.add_argument('--full_data', action='store_true', default=False,
                      help='use FullAlignmentDataset (kNN feature-space pairing)')
data_arg.add_argument('--seed_ratio', type=float, default=0.2, help='max ratio of seeding points')
data_arg.add_argument('--num_node', type=int, default=512)
data_arg.add_argument('--augment_axis', type=int, default=3)
data_arg.add_argument('--augment_rotation', type=float, default=1.0, help='rotation angle = num * 2pi')
data_arg.add_argument('--augment_translation', type=float, default=0.5, help='translation = num (m)')
data_arg.add_argument('--batch_size', type=int, default=6)
data_arg.add_argument('--num_workers', type=int, default=12)

# Other configurations
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--gpu_mode', type=str2bool, default=True)
misc_arg.add_argument('--verbose', type=str2bool, default=True)
misc_arg.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
misc_arg.add_argument('--pretrain', type=str, default='')
misc_arg.add_argument('--weights_fixed', type=str2bool, default=False)
misc_arg.add_argument('--force_all_labels', type=str2bool, default=False)
misc_arg.add_argument('--skip_gt_trans', type=str2bool, default=True)
misc_arg.add_argument('--split_seed', type=int, default=0)
misc_arg.add_argument('--eval_snapshot', type=str, default='', help='run evaluation-only with this snapshot')
misc_arg.add_argument('--eval_num', type=int, default=0,
                      help='cap number of validation files evaluated; 0 uses all')
misc_arg.add_argument('--eval_seed', type=int, default=0,
                      help='random seed for eval_num file sampling')
misc_arg.add_argument('--generation_method', type=str, default='spectral-2', choices=['spectral-2', 'spectral', 'greedy'])
misc_arg.add_argument('--generation_min_score', type=float, default=None,
                      help='absolute minimum spectral score for spectral generation methods')
misc_arg.add_argument('--generation_min_confidence', type=float, default=None,
                      help='absolute minimum node confidence for greedy generation method')
misc_arg.add_argument('--use_wandb', action='store_true')
misc_arg.add_argument('--wandb_project', type=str, default='3d-alignment')
misc_arg.add_argument('--wandb_entity', type=str, default='')
misc_arg.add_argument('--wandb_run_name', type=str, default=None)
misc_arg.add_argument('--config', type=str, default='')

# snapshot configurations
snapshot_arg = add_argument_group('Snapshot')
snapshot_arg.add_argument('--exp_id', type=str, default=None)
snapshot_arg.add_argument('--snapshot_dir', type=str, default=None)
snapshot_arg.add_argument('--tboard_dir', type=str, default=None)
snapshot_arg.add_argument('--snapshot_interval', type=int, default=1)
snapshot_arg.add_argument('--save_dir', type=str, default=None)


def get_config():
    from project.utils.io import load_yaml

    initial_args, _ = parser.parse_known_args()
    config_path = initial_args.config or dataset_configs.get(initial_args.dataset)
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            yaml_cfg = load_yaml(config_path)
            if isinstance(yaml_cfg, dict) and yaml_cfg:
                parser.set_defaults(**yaml_cfg)
        else:
            raise FileNotFoundError(f"Config YAML not found: {config_path}")

    args = parser.parse_args()
    args.config = str(config_path) if config_path else ''
    experiment_id = f"HyperGNN_{args.dataset}_{time.strftime('%m%d%H%M')}"
    default_snapshot_dir = f'snapshot/{experiment_id}'
    default_tboard_dir = f'tensorboard/{experiment_id}'
    default_save_dir = os.path.join(default_snapshot_dir, 'models/')

    args.exp_id = args.exp_id or experiment_id
    args.snapshot_dir = args.snapshot_dir or default_snapshot_dir
    args.tboard_dir = args.tboard_dir or default_tboard_dir
    args.save_dir = args.save_dir or default_save_dir
    args.wandb_run_name = args.wandb_run_name or args.exp_id

    def _parse_pooling_layer_idx(value):
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        if isinstance(value, int):
            return [value]
        if value is None:
            return []
        text = str(value).strip()
        if text == "":
            return []
        return [int(v.strip()) for v in text.split(",") if v.strip() != ""]

    args.pooling_layer_idx = _parse_pooling_layer_idx(args.pooling_layer_idx)

    if args.dataset != args.dataset and args.exp_id == experiment_id:
        new_exp_id = f"HyperGNN_{args.dataset}_{datetime.now().strftime('%m%d%H%M%S_%f')}"
        args.exp_id = new_exp_id
        if args.snapshot_dir == default_snapshot_dir:
            args.snapshot_dir = f'snapshot/{new_exp_id}'
        if args.tboard_dir == default_tboard_dir:
            args.tboard_dir = f'tensorboard/{new_exp_id}'
        if args.save_dir == default_save_dir:
            args.save_dir = os.path.join(f'snapshot/{new_exp_id}', 'models/')
    return args
