import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from project.utils.timer import AverageMeter, Timer
from project.trainers.generation_utils import run_generation_step


class Trainer(object):
    def __init__(self, args):
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
        self.force_all_labels = getattr(args, "force_all_labels", False)
        self.skip_gt_trans = getattr(args, "skip_gt_trans", False)
        self.use_wandb = getattr(args, "use_wandb", False)
        self.mode = args.mode
        self.generate = getattr(args, "generate", False)
        self.generation_method = getattr(args, "generation_method", "spectral-2")
        self.generation_min_score = getattr(args, "generation_min_score", None)
        self.generation_min_confidence = getattr(args, "generation_min_confidence", None)
        self.matching_dir = self._resolve_matching_dir(args)
        self.wandb = None
        self.writer = SummaryWriter(log_dir=args.tboard_dir)

        self.train_loader = args.train_loader
        self.val_loader = args.val_loader

        if self.gpu_mode:
            self.model = self.model.cuda(0)

        if args.pretrain != "":
            self._load_pretrain(args.pretrain)

        if self.use_wandb:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError("wandb is not installed. Install it or set --use_wandb false.") from exc
            self.wandb = wandb
            config = self._wandb_config(args)
            self.wandb.init(
                project=getattr(args, "wandb_project", "whnn"),
                entity=getattr(args, "wandb_entity", "") or None,
                name=getattr(args, "wandb_run_name", "") or None,
                config=config,
            )
            self.wandb.define_metric("train/step")
            self.wandb.define_metric("val/step")
            self.wandb.define_metric("train/*", step_metric="train/step")
            self.wandb.define_metric("val/*", step_metric="val/step")

    def _resolve_matching_dir(self, args):
        eval_snapshot = getattr(args, "eval_snapshot", "")
        model_path = eval_snapshot if isinstance(eval_snapshot, str) and eval_snapshot.strip() else getattr(args, "pretrain", "")
        if isinstance(model_path, str) and model_path.strip():
            model_path = os.path.abspath(model_path)
            model_parent = os.path.basename(os.path.dirname(model_path))
            if model_parent == "models":
                return os.path.join(os.path.dirname(os.path.dirname(model_path)), "matching")
        return os.path.join(self.snapshot_dir, "matching")

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

    def _unpack_batch(self, batch):
        file_name = None
        if len(batch) == 9:
            (
                corr_pos,
                src_keypts,
                tgt_keypts,
                src_normal,
                tgt_normal,
                src_indices,
                tgt_indices,
                gt_labels,
                file_name,
            ) = batch
            gt_trans = None
        elif len(batch) == 6:
            (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_labels) = batch
            src_indices = None
            tgt_indices = None
            gt_trans = None
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

        if self.gpu_mode:
            corr_pos = corr_pos.cuda()
            src_keypts = src_keypts.cuda()
            tgt_keypts = tgt_keypts.cuda()
            src_normal = src_normal.cuda()
            tgt_normal = tgt_normal.cuda()
            gt_labels = gt_labels.cuda()
            if gt_trans is not None:
                gt_trans = gt_trans.cuda()

        if self.force_all_labels:
            gt_labels = torch.ones_like(gt_labels)
        if self.skip_gt_trans:
            gt_trans = None

        data = {
            "corr_pos": corr_pos,
            "src_keypts": src_keypts,
            "tgt_keypts": tgt_keypts,
            "src_normal": src_normal,
            "tgt_normal": tgt_normal,
        }
        return data, gt_labels, gt_trans, src_indices, tgt_indices, file_name

    def train(self, resume, start_epoch, best_reg_recall, best_F1):
        if resume:
            print(f"Resuming from epoch {start_epoch}")
            model_path = str(self.save_dir + f"/model_{start_epoch}.pkl")
            print(f"Loading model parameters from {model_path}")
            self.model.load_state_dict(torch.load(model_path))
        else:
            start_epoch = 0
            best_reg_recall = 0
            best_F1 = 0

        self.model.train()
        res = self.evaluate(start_epoch)
        print(
            f'Evaluation: Epoch {start_epoch}: SM Loss {res["sm_loss"]:.2f} '
            f'Class Loss {res["class_loss"]:.2f} Graph Loss {res["graph_loss"]:.2f} '
            f'F1 {res["f1"]:.2f} Recall {res["reg_recall"]:.2f}'
        )
        print("training start!!")
        self.t = tqdm(range(start_epoch, self.max_epoch), desc="Total Progress", ncols=2 * self.max_epoch)
        for epoch in self.t:
            self.train_epoch(epoch + 1)
            if (epoch + 1) % self.evaluate_interval == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                self.t.write(
                    f'Evaluation: Epoch {epoch + 1}: SM Loss {res["sm_loss"]:.2f} '
                    f'Class Loss {res["class_loss"]:.2f} Graph Loss {res["graph_loss"]:.2f} '
                    f'F1 {res["f1"]:.2f} Recall {res["reg_recall"]:.2f}'
                )
                if res["f1"] > best_F1:
                    best_F1 = res["f1"]
                    if epoch >= 10:
                        self._snapshot("best")

            if (epoch + 1) % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

        self.t.write("Training finish!... save training results")
        if self.use_wandb:
            self.wandb.finish()

    def train_epoch(self, epoch):
        meter_list = ["class_loss", "sm_loss", "reg_recall", "graph_loss", "re", "te", "precision", "recall", "f1"]
        meter_dict = {key: AverageMeter() for key in meter_list}
        data_timer, model_timer = Timer(), Timer()

        num_iter = int(len(self.train_loader.dataset) / self.batch_size)
        num_iter = min(self.training_max_iter, num_iter)
        trainer_loader_iter = self.train_loader.__iter__()
        for iter in range(num_iter):
            data_timer.tic()
            batch = next(trainer_loader_iter)
            data, gt_labels, _, _, _, _ = self._unpack_batch(batch)
            data_timer.toc()

            model_timer.tic()
            self.optimizer.zero_grad(set_to_none=True)
            res = self.model(data)
            pred_labels = res["final_labels"]
            class_stats = self.evaluate_metric["ClassificationLoss"](pred_labels, gt_labels)
            class_loss = class_stats["loss"]
            class_loss.backward()

            do_step = True
            for param in self.model.parameters():
                if param.grad is not None and (1 - torch.isfinite(param.grad).long()).sum() > 0:
                    do_step = False
                    break
            if do_step:
                self.optimizer.step()
            model_timer.toc()

            stats = {
                "class_loss": float(class_loss),
                "sm_loss": 0.0,
                "graph_loss": 0.0,
                "reg_recall": 0.0,
                "re": 0.0,
                "te": 0.0,
                "precision": class_stats["precision"],
                "recall": class_stats["recall"],
                "f1": class_stats["f1"],
            }

            if not np.isnan(float(class_loss)):
                for key in meter_list:
                    if not np.isnan(stats[key]):
                        meter_dict[key].update(stats[key])

            if ((iter + 1) % 100 == 0 or iter == num_iter - 1) and self.verbose:
                curr_iter = num_iter * (epoch - 1) + iter
                for key in meter_list:
                    self.writer.add_scalar(f"Train/{key}", meter_dict[key].avg, curr_iter)
                if self.use_wandb:
                    log_data = {f"train/{key}": meter_dict[key].avg for key in meter_list}
                    log_data["train/step"] = curr_iter
                    self.wandb.log(log_data)

                self.t.write(
                    f"Epoch: {epoch} [{iter + 1:4d}/{num_iter}] "
                    f"class_loss: {meter_dict['class_loss'].avg:.2f} "
                    f"f1: {meter_dict['f1'].avg:.2f} "
                    f"precision: {meter_dict['precision'].avg:.2f} "
                    f"recall: {meter_dict['recall'].avg:.2f} "
                    f"data_time: {data_timer.avg:.2f}s "
                    f"model_time: {model_timer.avg:.2f}s "
                )

    def evaluate(self, epoch):
        self.model.eval()

        meter_list = ["class_loss", "sm_loss", "reg_recall", "graph_loss", "re", "te", "precision", "recall", "f1"]
        meter_dict = {key: AverageMeter() for key in meter_list}
        selected_abs_err_meter = AverageMeter()
        selected_l2_err_meter = AverageMeter()
        initial_abs_err_meter = AverageMeter()
        initial_l2_err_meter = AverageMeter()

        num_iter = int(len(self.val_loader.dataset) / self.batch_size)
        num_iter = min(self.val_max_iter, num_iter)
        val_loader_iter = self.val_loader.__iter__()
        for iter in range(num_iter):
            batch = next(val_loader_iter)
            data, gt_labels, _, src_indices, tgt_indices, file_name = self._unpack_batch(batch)

            with torch.no_grad():
                res = self.model(data)
                pred_labels = res["final_labels"]
                class_stats = self.evaluate_metric["ClassificationLoss"](pred_labels, gt_labels)
                class_loss = class_stats["loss"]
                sm_loss = 0.0
                if "M" in res:
                    sm_loss = float(self.evaluate_metric["SpectralMatchingLoss"](res["M"], gt_labels))

                if self.generate:
                    run_generation_step(
                        res=res,
                        gt_labels=gt_labels,
                        src_indices=src_indices,
                        tgt_indices=tgt_indices,
                        file_name=file_name,
                        epoch=epoch,
                        iter_idx=iter,
                        model=self.model,
                        generation_method=self.generation_method,
                        generation_min_score=self.generation_min_score,
                        generation_min_confidence=self.generation_min_confidence,
                        matching_dir=self.matching_dir,
                        selected_abs_err_meter=selected_abs_err_meter,
                        selected_l2_err_meter=selected_l2_err_meter,
                        initial_abs_err_meter=initial_abs_err_meter,
                        initial_l2_err_meter=initial_l2_err_meter,
                    )

            stats = {
                "class_loss": float(class_loss),
                "sm_loss": float(sm_loss),
                "graph_loss": 0.0,
                "reg_recall": 0.0,
                "re": 0.0,
                "te": 0.0,
                "precision": class_stats["precision"],
                "recall": class_stats["recall"],
                "f1": class_stats["f1"],
            }
            for key in meter_list:
                if not np.isnan(stats[key]):
                    meter_dict[key].update(stats[key])

        self.model.train()
        if self.generate and initial_abs_err_meter.count > 0:
            print(
                f"initial dataset target distance error (mean over {initial_abs_err_meter.count} files): "
                f"mean|y_pred-y_gt|={initial_abs_err_meter.avg:.6f}, "
                f"mean_l2={initial_l2_err_meter.avg:.6f}"
            )
        if self.generate and selected_abs_err_meter.count > 0:
            print(
                f"selected target distance error (mean over {selected_abs_err_meter.count} files): "
                f"mean|y_pred-y_gt|={selected_abs_err_meter.avg:.6f}, "
                f"mean_l2={selected_l2_err_meter.avg:.6f}"
            )
        res = {
            "sm_loss": meter_dict["sm_loss"].avg,
            "class_loss": meter_dict["class_loss"].avg,
            "reg_recall": meter_dict["reg_recall"].avg,
            "graph_loss": meter_dict["graph_loss"].avg,
            "f1": meter_dict["f1"].avg,
        }
        for key in meter_list:
            self.writer.add_scalar(f"Val/{key}", meter_dict[key].avg, epoch)
        if self.use_wandb:
            log_dict = {f"val/{key}": meter_dict[key].avg for key in meter_list}
            log_dict["val/step"] = epoch
            self.wandb.log(log_dict)
        return res

    def _snapshot(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{epoch}.pkl"))

    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location="cpu")
        self.model.load_state_dict(state_dict)
