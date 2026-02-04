import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from project.utils.timer import AverageMeter, Timer

def disjointness_from_w(W, eps=1e-8):
    """
    Compute per-batch disjointness from vertex-hyperedge membership scores.

    Args:
        W: Tensor of shape (B, V, E) or (V, E) with nonnegative memberships.
        eps: Small constant to avoid division by zero.

    Returns:
        Tensor of shape (B,) (or scalar if input is 2D) with mean disjointness.
    """
    if W.dim() == 2:
        W = W.unsqueeze(0)
    if W.dim() != 3:
        raise ValueError(f"W must have shape (B, V, E) or (V, E), got {W.shape}")
    W_sum = W.sum(dim=-1)
    W_max = W.max(dim=-1).values
    disjointness_per_vertex = W_max / (W_sum + eps)
    return disjointness_per_vertex.mean(dim=-1)

def weighted_pair_distance_var(W, src_pts, tgt_pts, eps=1e-8):
    """
    Compute weighted variance of pairwise src-tgt distances per hyperedge.

    Args:
        W: Tensor of shape (B, V, E) or (V, E) with nonnegative memberships.
        src_pts: Tensor of shape (B, V, 3) or (V, 3).
        tgt_pts: Tensor of shape (B, V, 3) or (V, 3).
        eps: Small constant to avoid division by zero.

    Returns:
        Tensor of shape (B,) (or scalar if inputs are 2D) with mean variance across hyperedges.
    """
    if W.dim() == 2:
        W = W.unsqueeze(0)
    if src_pts.dim() == 2:
        src_pts = src_pts.unsqueeze(0)
    if tgt_pts.dim() == 2:
        tgt_pts = tgt_pts.unsqueeze(0)
    if W.dim() != 3:
        raise ValueError(f"W must have shape (B, V, E) or (V, E), got {W.shape}")
    if src_pts.shape[:2] != W.shape[:2] or tgt_pts.shape[:2] != W.shape[:2]:
        raise ValueError("src_pts/tgt_pts must align with W on (B, V) dimensions")

    distances = torch.norm(src_pts - tgt_pts, dim=-1)  # (B, V)
    w_sum = W.sum(dim=1)  # (B, E)
    w_mean = (W * distances.unsqueeze(-1)).sum(dim=1) / (w_sum + eps)  # (B, E)
    centered = distances.unsqueeze(-1) - w_mean.unsqueeze(1)  # (B, V, E)
    w_var = (W * centered ** 2).sum(dim=1) / (w_sum + eps)  # (B, E)
    return w_var.mean(dim=-1)

def aggregatedness_from_w(W, eps=1e-8):
    """
    Entropy-based aggregatedness from dominant hyperedge assignments.

    Args:
        W: Tensor of shape (B, V, E) or (V, E) with nonnegative memberships.
        eps: Small constant to avoid division by zero/log(0).

    Returns:
        Tensor of shape (B,) (or scalar if input is 2D) in [0, 1] when E>1.
    """
    if W.dim() == 2:
        W = W.unsqueeze(0)
    if W.dim() != 3:
        raise ValueError(f"W must have shape (B, V, E) or (V, E), got {W.shape}")
    B, V, E = W.shape
    if E <= 1:
        return W.new_zeros((B,))

    dominant = torch.argmax(W, dim=-1)  # (B, V)
    counts = torch.zeros((B, E), device=W.device, dtype=W.dtype)
    counts.scatter_add_(1, dominant, torch.ones_like(dominant, dtype=W.dtype))
    p = counts / (counts.sum(dim=1, keepdim=True) + eps)
    entropy = -(p * torch.log(p + eps)).sum(dim=1)
    aggregatedness = 1.0 - entropy / np.log(E)
    return aggregatedness

class Trainer(object):
    def __init__(self, args):
        # parameters
        self.t = None
        self.max_epoch = args.max_epoch
        self.training_max_iter = args.training_max_iter
        self.val_max_iter = args.val_max_iter
        self.batch_size = args.batch_size
        self.snapshot_dir = args.snapshot_dir
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.verbose = args.verbose

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.evaluate_interval = args.evaluate_interval
        self.evaluate_metric = args.evaluate_metric
        self.metric_weight = args.metric_weight
        self.transformation_loss_start_epoch = args.transformation_loss_start_epoch
        self.force_all_labels = getattr(args, "force_all_labels", False)
        self.skip_gt_trans = getattr(args, "skip_gt_trans", False)
        self.use_wandb = getattr(args, "use_wandb", False)
        self.wandb = None
        self.writer = SummaryWriter(log_dir=args.tboard_dir)

        self.train_loader = args.train_loader
        self.val_loader = args.val_loader

        if self.gpu_mode:
            self.model = self.model.cuda(0)

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

        if self.use_wandb:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError("wandb is not installed. Install it or set --use_wandb false.") from exc
            self.wandb = wandb
            config = self._wandb_config(args)
            self.wandb.init(
                project=getattr(args, "wandb_project", "hypergnn"),
                entity=getattr(args, "wandb_entity", "") or None,
                name=getattr(args, "wandb_run_name", "") or None,
                config=config,
            )
            self.wandb.define_metric("train/step")
            self.wandb.define_metric("val/step")
            self.wandb.define_metric("train/*", step_metric="train/step")
            self.wandb.define_metric("val/*", step_metric="val/step")

    def _wandb_config(self, args):
        excluded = {
            "model",
            "optimizer",
            "scheduler",
            "evaluate_metric",
            "metric_weight",
            "train_loader",
            "val_loader",
        }
        return {key: value for key, value in vars(args).items() if key not in excluded}

    def train(self, resume, start_epoch, best_reg_recall, best_F1):
        # resume to train from given epoch
        if resume:
            print('Resuming from epoch {}'.format(start_epoch))
            # assert start_epoch != 0
            model_path = str(self.save_dir + '/model_{}.pkl'.format(start_epoch))
            #model_path = str(self.snapshot_dir + '/models/model_best.pkl'.format(start_epoch))
            print('Loading model parameters from {}'.format(model_path))
            self.model.load_state_dict(torch.load(model_path))
        else:
            start_epoch = 0
            best_reg_recall = 0
            best_F1 = 0

        start_time = time.time()
        self.model.train()
        res = self.evaluate(start_epoch)
        print(
            f'Evaluation: Epoch {start_epoch}: SM Loss {res["sm_loss"]:.2f} Class Loss {res["class_loss"]:.2f} Graph Loss {res["graph_loss"]:.2f} F1 {res["f1"]:.2f} Recall {res["reg_recall"]:.2f}')
        print('training start!!')
        self.t = tqdm(range(start_epoch, self.max_epoch), desc="Total Progress", ncols=2 * self.max_epoch)
        for epoch in self.t:
            self.train_epoch(epoch + 1)  # start from epoch 1
            if (epoch + 1) % self.evaluate_interval == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                self.t.write(
                    f'Evaluation: Epoch {epoch + 1}: SM Loss {res["sm_loss"]:.2f} Class Loss {res["class_loss"]:.2f} Graph Loss {res["graph_loss"]:.2f} F1 {res["f1"]:.2f} Recall {res["reg_recall"]:.2f}')
                if round(res['reg_recall'], 2) > best_reg_recall:  # reg_recall 相同时
                    if epoch < 10:
                        self.t.write('best model in 10 epoch will not be saved!')
                    else:
                        best_reg_recall = round(res['reg_recall'], 2)
                        best_F1 = res['f1']
                        self._snapshot('best')
                elif round(res['reg_recall'], 2) == best_reg_recall and res['f1'] > best_F1:
                    self.t.write(
                        f'previous best: RR {best_reg_recall:.2f} F1 {best_F1:.2f}, current: RR {res["reg_recall"]:.2f} F1 {res["f1"]:.2f}')
                    if epoch < 10:
                        self.t.write('best model in 10 epoch will not be saved!')
                    else:
                        best_F1 = res['f1']
                        self._snapshot('best')

            if (epoch + 1) % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

        # finish all epoch
        self.t.write("Training finish!... save training results")
        if self.use_wandb:
            self.wandb.finish()

    def train_epoch(self, epoch):
        # create meters and timers
        meter_list = ['class_loss', 'sm_loss', 'reg_recall', 'graph_loss', 're', 'te', 'precision', 'recall', 'f1']
        meter_dict = {}
        for key in meter_list:
            meter_dict[key] = AverageMeter()
        data_timer, model_timer = Timer(), Timer()

        num_iter = int(len(self.train_loader.dataset) / self.batch_size)
        num_iter = min(self.training_max_iter, num_iter)
        trainer_loader_iter = self.train_loader.__iter__()
        for iter in range(num_iter):
            data_timer.tic()
            batch = next(trainer_loader_iter)
            if len(batch) == 7:
                (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels) = batch
            elif len(batch) == 6:
                (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_labels) = batch
                gt_trans = None
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            if self.gpu_mode:
                corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_labels = \
                    corr_pos.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), src_normal.cuda(), tgt_normal.cuda(), gt_labels.cuda()
                if gt_trans is not None:
                    gt_trans = gt_trans.cuda()
            if self.force_all_labels:
                gt_labels = torch.ones_like(gt_labels)
            if self.skip_gt_trans:
                gt_trans = None

            # TODO 收敛更快
            if epoch <= 5:
                mask = gt_labels.mean(-1) > 0.2
                if mask.sum() > 0:
                    corr_pos = corr_pos[mask]
                    src_keypts = src_keypts[mask]
                    tgt_keypts = tgt_keypts[mask]
                    src_normal = src_normal[mask]
                    tgt_normal = tgt_normal[mask]
                    if gt_trans is not None:
                        gt_trans = gt_trans[mask]
                    gt_labels = gt_labels[mask]

            elif epoch <= 10:
                mask = gt_labels.mean(-1) > 0.1
                if mask.sum() > 0:
                    corr_pos = corr_pos[mask]
                    src_keypts = src_keypts[mask]
                    tgt_keypts = tgt_keypts[mask]
                    src_normal = src_normal[mask]
                    tgt_normal = tgt_normal[mask]
                    if gt_trans is not None:
                        gt_trans = gt_trans[mask]
                    gt_labels = gt_labels[mask]

            data = {
                'corr_pos': corr_pos,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
                'src_normal': src_normal,
                'tgt_normal': tgt_normal,
            }
            data_timer.toc()

            model_timer.tic()
            # forward
            self.optimizer.zero_grad(set_to_none=True)
            res = self.model(data)
            pred_trans, pred_labels = res['final_trans'], res['final_labels']
            # classification loss
            class_stats = self.evaluate_metric['ClassificationLoss'](pred_labels, gt_labels)
            class_loss = class_stats['loss']
            # spectral matching loss
            sm_loss = self.evaluate_metric['SpectralMatchingLoss'](res['M'], gt_labels)

            # hypergraph loss
            graph_loss = self.evaluate_metric['HypergraphLoss'](res['edge_score'], res['raw_H'], gt_labels)

            # transformation loss
            if gt_trans is not None:
                reg_recall, re, te, rmse = self.evaluate_metric['TransformationLoss'](
                    pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels
                )
            else:
                reg_recall, re, te, rmse = 0.0, 0.0, 0.0, 0.0

            loss = (self.metric_weight['ClassificationLoss'] * class_loss + self.metric_weight[
                'SpectralMatchingLoss'] * sm_loss
                    + self.metric_weight['HypergraphLoss'] * graph_loss)
            # if epoch > self.transformation_loss_start_epoch and self.metric_weight['TransformationLoss'] > 0.0:
            #     loss += self.metric_weight['TransformationLoss'] * trans_loss

            stats = {
                'class_loss': float(class_loss),
                'sm_loss': float(sm_loss),
                'graph_loss': float(graph_loss),
                'reg_recall': float(reg_recall),
                're': float(re),
                'te': float(te),
                'precision': class_stats['precision'],
                'recall': class_stats['recall'],
                'f1': class_stats['f1'],
            }

            # backward
            loss.backward()
            do_step = True
            for param in self.model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        break
            if do_step is True:
                self.optimizer.step()
            model_timer.toc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not np.isnan(float(loss)):
                for key in meter_list:
                    if not np.isnan(stats[key]):
                        meter_dict[key].update(stats[key])

            else:  # debug the loss calculation process.
                import pdb
                pdb.set_trace()

            if (iter + 1) % 100 == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + iter
                for key in meter_list:
                    self.writer.add_scalar(f"Train/{key}", meter_dict[key].avg, curr_iter)
                if self.use_wandb:
                    log_data = {f"train/{key}": meter_dict[key].avg for key in meter_list}
                    log_data["train/step"] = curr_iter
                    self.wandb.log(log_data)

                self.t.write(f"Epoch: {epoch} [{iter + 1:4d}/{num_iter}] "
                             f"sm_loss: {meter_dict['sm_loss'].avg:.2f} "
                             f"class_loss: {meter_dict['class_loss'].avg:.2f} "
                             f"graph_loss: {meter_dict['graph_loss'].avg:.2f} "
                             f"reg_recall: {meter_dict['reg_recall'].avg:.2f}% "
                             f"re: {meter_dict['re'].avg:.2f}° "
                             f"te: {meter_dict['te'].avg:.2f}cm "
                             f"data_time: {data_timer.avg:.2f}s "
                             f"model_time: {model_timer.avg:.2f}s "
                             )

    def evaluate(self, epoch):
        self.model.eval()

        # create meters and timers
        meter_list = ['class_loss', 'sm_loss', 'reg_recall', 'graph_loss', 're', 'te', 'precision', 'recall', 'f1']
        score_list = ['disjointness', 'edge_distance_var', 'aggregatedness']
        meter_dict = {}
        for key in meter_list + score_list:
            meter_dict[key] = AverageMeter()
        data_timer, model_timer = Timer(), Timer()

        num_iter = int(len(self.val_loader.dataset) / self.batch_size)
        num_iter = min(self.val_max_iter, num_iter)
        val_loader_iter = self.val_loader.__iter__()
        for iter in range(num_iter):
            data_timer.tic()
            batch = next(val_loader_iter)
            if len(batch) == 7:
                (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels) = batch
            elif len(batch) == 6:
                (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_labels) = batch
                gt_trans = None
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            if self.gpu_mode:
                corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_labels = \
                    corr_pos.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), src_normal.cuda(), tgt_normal.cuda(), gt_labels.cuda()
                if gt_trans is not None:
                    gt_trans = gt_trans.cuda()
            if self.force_all_labels:
                gt_labels = torch.ones_like(gt_labels)
            if self.skip_gt_trans:
                gt_trans = None
            data = {
                'corr_pos': corr_pos,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
                'src_normal': src_normal,
                'tgt_normal': tgt_normal,
            }
            data_timer.toc()

            model_timer.tic()
            # forward
            with torch.no_grad():
                res = self.model(data)
                pred_trans, pred_labels = res['final_trans'], res['final_labels']
                score_mat = res.get("edge_score")
                if score_mat is None:
                    score_mat = res.get("W")
                if score_mat is not None:
                    disjointness = disjointness_from_w(score_mat).mean().item()
                    edge_dist_var = weighted_pair_distance_var(
                        score_mat, src_keypts, tgt_keypts
                    ).mean().item()
                    aggregatedness = aggregatedness_from_w(score_mat).mean().item()
                    meter_dict['disjointness'].update(disjointness)
                    meter_dict['edge_distance_var'].update(edge_dist_var)
                    meter_dict['aggregatedness'].update(aggregatedness)
                # classification loss
                class_stats = self.evaluate_metric['ClassificationLoss'](pred_labels, gt_labels)
                class_loss = class_stats['loss']
                # spectral matching loss
                sm_loss = self.evaluate_metric['SpectralMatchingLoss'](res['M'], gt_labels)

                # hypergraph loss
                graph_loss = self.evaluate_metric['HypergraphLoss'](res['edge_score'], res['raw_H'], gt_labels)
                # transformation loss
                if gt_trans is not None:
                    reg_recall, re, te, rmse = self.evaluate_metric['TransformationLoss'](
                        pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels
                    )
                else:
                    reg_recall, re, te, rmse = 0.0, 0.0, 0.0, 0.0
            model_timer.toc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            stats = {
                'class_loss': float(class_loss),
                'sm_loss': float(sm_loss),
                'graph_loss': float(graph_loss),
                'reg_recall': float(reg_recall),
                're': float(re),
                'te': float(re),
                'precision': class_stats['precision'],
                'recall': class_stats['recall'],
                'f1': class_stats['f1'],
            }
            for key in meter_list:
                if not np.isnan(stats[key]):
                    meter_dict[key].update(stats[key])

        self.model.train()
        res = {
            'sm_loss': meter_dict['sm_loss'].avg,
            'class_loss': meter_dict['class_loss'].avg,
            'reg_recall': meter_dict['reg_recall'].avg,
            'graph_loss': meter_dict['graph_loss'].avg,
            'f1': meter_dict['f1'].avg,
        }
        for key in meter_list:
            self.writer.add_scalar(f"Val/{key}", meter_dict[key].avg, epoch)
        if self.use_wandb:
            log_dict = {f"val/{key}": meter_dict[key].avg for key in meter_list}
            for key in ['disjointness', 'edge_distance_var', 'aggregatedness']:
                log_dict[f"val/{key}"] = meter_dict[key].avg
            log_dict["val/step"] = epoch
            self.wandb.log(log_dict)

        return res

    def _snapshot(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{epoch}.pkl"))
        self.t.write(f"Save model to {self.save_dir}/model_{epoch}.pkl")

    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.t.write(f"Load model from {pretrain}.pkl")
