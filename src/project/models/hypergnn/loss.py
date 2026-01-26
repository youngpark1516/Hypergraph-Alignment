import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from project.utils.SE3 import *
import warnings

warnings.filterwarnings('ignore')


def transformation_error(pred_trans, gt_trans):
    if len(pred_trans.shape) == 3:
        bs = pred_trans.shape[0]
        pred_Rs = pred_trans[:, :3, :3]
        gt_Rs = gt_trans[:, :3, :3]
        pred_ts = pred_trans[:, :3, 3:4]
        gt_ts = gt_trans[:, :3, 3:4]
        mat = torch.matmul(pred_Rs.transpose(-1, -2), gt_Rs)
        tr = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]
        RE = torch.acos(torch.clamp(0.5 * (tr - 1.0), min=-1, max=1)) * 180 / np.pi
        TE = torch.norm(pred_ts - gt_ts, dim=1) * 100
        RE = RE.reshape(bs)
        TE = TE.reshape(bs)

    else:
        pred_R = pred_trans[:3, :3]
        gt_R = gt_trans[:3, :3]
        pred_t = pred_trans[:3, 3:4]
        gt_t = gt_trans[:3, 3:4]
        tr = torch.trace(pred_R.T @ gt_R)
        RE = torch.acos(torch.clamp(0.5 * (tr - 1), min=-1, max=1)) * 180 / np.pi
        TE = torch.norm(pred_t - gt_t) * 100
    return RE, TE


class TransformationLoss(nn.Module):
    def __init__(self, re_thresh, te_thresh):
        super(TransformationLoss, self).__init__()
        self.re_thresh = re_thresh
        self.te_thresh = te_thresh

    def forward(self, pred_trans, gt_trans, src_kpts, tgt_kpts, inlier_probs):
        batch_size = pred_trans.shape[0]
        recall = 0
        RE = torch.tensor(0.0).to(pred_trans.device)
        TE = torch.tensor(0.0).to(pred_trans.device)
        RMSE = torch.tensor(0.0).to(pred_trans.device)
        RE, TE = transformation_error(pred_trans, gt_trans)

        trans_src = transform(src_kpts, pred_trans)
        RMSE = torch.norm(trans_src - tgt_kpts, dim=-1).mean(axis=1).reshape(batch_size)
        succ = torch.where(RE < self.re_thresh) and torch.where(TE < self.te_thresh)
        recall = len(succ[0])

        return recall * 100.0 / batch_size, RE.mean(), TE.mean(), RMSE.mean()


class ClassificationLoss(nn.Module):
    def __init__(self, balanced=True):
        super(ClassificationLoss, self).__init__()
        self.balanced = balanced

    def forward(self, pred, gt, weight=None):
        bs, num_corr = pred.shape
        num_pos = torch.relu(torch.sum(gt) - 1) + 1
        num_neg = torch.relu(torch.sum(1 - gt) - 1) + 1
        if weight is not None:
            loss = nn.BCEWithLogitsLoss(reduction='none')(pred, gt.float())
            loss = torch.mean(loss * weight)
        elif self.balanced is False:
            loss = nn.BCEWithLogitsLoss(reduction='mean')(pred, gt.float())
        else:
            loss = nn.BCEWithLogitsLoss(pos_weight=num_neg * 1.0 / num_pos, reduction='mean')(pred, gt.float())

        precision, recall, f1 = 0.0, 0.0, 0.0
        logit_true, logit_false = 0.0, 0.0
        pred_labels = pred > 0
        for i in range(bs):
            gt_np, pred_labels_np, pred_np = gt[i].detach().cpu().numpy(), pred_labels[i].detach().cpu().numpy(), pred[i].detach().cpu().numpy()
            precision += precision_score(gt_np, pred_labels_np)
            recall += recall_score(gt_np, pred_labels_np)
            f1 += f1_score(gt_np, pred_labels_np)
            logit_true += np.sum(pred_np * gt_np) / max(1, np.sum(gt_np))
            logit_false += np.sum(pred_np * (1 - gt_np)) / max(1, np.sum(1 - gt_np))

        precision /= bs
        recall /= bs
        f1 /= bs
        logit_true /= bs
        logit_false /= bs

        eval_stats = {
            "loss": loss,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "logit_true": float(logit_true),
            "logit_false": float(logit_false)
        }
        return eval_stats


class SpectralMatchingLoss(nn.Module):
    def __init__(self, balanced=True):
        super(SpectralMatchingLoss, self).__init__()
        self.balanced = balanced

    def forward(self, M, gt_labels):
        gt_M = ((gt_labels[:, None, :] + gt_labels[:, :, None]) == 2)
        for i in range(gt_M.shape[0]):
            gt_M[i].fill_diagonal_(0)
        if self.balanced:
            sm_loss_p = ((M - 1) ** 2 * gt_M).sum(-1).sum(-1) / (torch.relu((gt_M).sum(-1).sum(-1) - 1.0) + 1.0)
            sm_loss_n = ((M - 0) ** 2 * (1 - gt_M)).sum(-1).sum(-1) / (
                    torch.relu((1 - gt_M).sum(-1).sum(-1) - 1.0) + 1.0)
            loss = torch.mean(sm_loss_p * 0.5 + sm_loss_n * 0.5)
        else:
            loss = torch.nn.MSELoss(reduction='mean')(M, gt_M.float())
        return loss


class EdgeFeatureLoss(nn.Module):
    def __init__(self, transpose=False):
        super(EdgeFeatureLoss, self).__init__()
        self.transpose = transpose

    def forward(self, H, raw_H, gt_labels):
        bs, num_corr = gt_labels.shape
        gt_H = ((gt_labels[:, None, :] + gt_labels[:, :, None]) == 2).float()
        gt_H[:, torch.arange(num_corr), torch.arange(num_corr)] = 0
        gt_H = torch.matmul(gt_H, gt_H) * gt_H
        gt_H[gt_H > 0] = 1.0
        gt_H = gt_H * raw_H
        if self.transpose:
            H = H.mT

        degree_gt, degree_H = gt_H.sum(dim=1), H.sum(dim=1)  # [bs, num]
        valid_edge_gt = (degree_gt > 0).float().sum(dim=1)  # [bs]
        correct_num_H = torch.sum(H * gt_H, dim=1)  # [bs, num]

        D_H_1 = degree_H ** -1  # [bs, num]
        D_H_1[torch.isinf(D_H_1)] = 0
        D_gt_1 = degree_gt ** -1  # [bs, num]
        D_gt_1[torch.isinf(D_gt_1)] = 0
        D_gt_1[D_gt_1 > round(num_corr * 0.1)] = round(num_corr * 0.1)

        E_gt_1 = valid_edge_gt ** -1  # [bs]
        E_gt_1[torch.isinf(E_gt_1)] = 0

        precision = correct_num_H * D_H_1  # [bs, num]
        loss = torch.mul(torch.sum(precision, dim=1), E_gt_1).mean()
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.criterion = nn.BCELoss(reduction='none')

    def masked(self, edge_score, raw_H, gt_H):
        mask = torch.where(torch.greater(edge_score, 0), raw_H, torch.zeros_like(raw_H))
        masked_targets1 = gt_H * mask
        masked_outputs1 = edge_score * mask
        bce_loss = self.criterion(masked_outputs1, masked_targets1)  # [bs, num, num]

        valid_num = mask.sum(dim=-1)
        D_1 = valid_num ** -1  # [bs, num]
        D_1[torch.isinf(D_1)] = 0
        E_1 = (valid_num > 0).float().sum(dim=1) ** -1  # [bs]
        E_1[torch.isinf(E_1)] = 0

        loss = torch.mul((bce_loss.sum(dim=-1) * D_1).sum(dim=1), E_1).mean()
        return loss

    def forward(self, edge_score, raw_H, gt_labels):
        bs, num_corr = gt_labels.shape
        gt_H = ((gt_labels[:, None, :] + gt_labels[:, :, None]) == 2).float()
        gt_H[:, torch.arange(num_corr), torch.arange(num_corr)] = 0
        gt_H = torch.matmul(gt_H, gt_H) * gt_H
        gt_H[gt_H > 0] = 1.0
        loss = self.masked(edge_score, raw_H, gt_H)
        return loss


if __name__ == '__main__':
    batch_size = 4
    num_corr = 100
    edge_score_list = [torch.randn(batch_size, num_corr, num_corr)]
    raw_H = torch.randn(batch_size, num_corr, num_corr)
    gt_labels = torch.randint(0, 2, (batch_size, num_corr))

    loss_fn = EdgeLoss()
    loss = loss_fn(edge_score_list, raw_H, gt_labels)
    print("Loss computed")
