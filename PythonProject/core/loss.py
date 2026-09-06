import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import jaccard, point_form, center_size


class MultiBoxLoss(nn.Module):
    def __init__(self, priors, num_classes=21, overlap_thresh=0.4, neg_pos_ratio=2):
        """
        标准 SSD MultiBox Loss（交叉熵 + 难例挖掘）
        """
        super(MultiBoxLoss, self).__init__()
        self.priors = priors
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos_ratio
        self.variance = [0.1, 0.2]

    def forward(self, predictions, targets):
        loc_data, conf_data = predictions
        batch_size = loc_data.size(0)
        num_priors = self.priors.size(0)

        loc_t = torch.zeros(batch_size, num_priors, 4, device=loc_data.device)
        conf_t = torch.zeros(batch_size, num_priors, dtype=torch.long, device=loc_data.device)

        # ---------- 1. 锚框匹配 ----------
        for idx in range(batch_size):
            truths = targets[idx][0]
            labels = targets[idx][1]
            if truths.numel() == 0:
                continue

            overlaps = jaccard(self.priors, point_form(truths))
            best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
            best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

            for j in range(best_truth_idx.size(1)):
                anchor_idx = best_truth_idx[0, j]
                best_prior_overlap[anchor_idx] = 2.0

            best_prior_idx = best_prior_idx.squeeze(1)
            matches = truths[best_prior_idx]
            conf = labels[best_prior_idx]
            conf[best_prior_overlap.squeeze(1) < self.threshold] = 0

            loc = center_size(matches)
            loc_t[idx] = self._encode(loc)
            conf_t[idx] = conf

        pos = conf_t > 0
        N = max(pos.sum().float(), 1.0)

        # ---------- 2. 定位损失（Smooth L1） ----------
        pos_idx = pos.unsqueeze(2).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') / N

        # ---------- 3. 分类损失（交叉熵 + 难例挖掘） ----------
        conf_logits = conf_data.view(-1, self.num_classes)
        conf_target = conf_t.view(-1)

        # 计算所有锚框的交叉熵损失
        loss_c_all = F.cross_entropy(conf_logits, conf_target, reduction='none')
        loss_c_all = loss_c_all.view(batch_size, -1)

        # 难例挖掘：用副本排序，不修改原始 loss（保护正样本梯度）
        pos_mask = conf_t > 0
        loss_for_sorting = loss_c_all.detach().clone()
        loss_for_sorting[pos_mask] = 0  # 只对副本清零正样本，用于排序

        _, loss_idx = loss_for_sorting.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_pos = pos_mask.sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=num_priors)
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # 分类损失 = 选中的负样本 + 所有正样本（从原始 loss 中取，不是清零后的）
        loss_c = (loss_c_all[neg_mask].sum() + loss_c_all[pos_mask].sum()) / N

        return loss_l + loss_c

    def _encode(self, boxes):
        priors = self.priors
        return torch.cat([
            (boxes[:, :2] - priors[:, :2]) / (priors[:, 2:] * self.variance[0]),
            torch.log(boxes[:, 2:] / priors[:, 2:]) / self.variance[1]
        ], 1)
