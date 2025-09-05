import torch
import torch.nn as nn
import torch.nn.functional as F

class LESPLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(LESPLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: Tensor of shape (batch_size, num_labels)
        targets: Tensor of shape (batch_size, num_labels), 0/1 binary multi-label
        """
        batch_size, num_labels = logits.size()
        losses = []

        for i in range(batch_size):
            logit = logits[i]          # [num_labels]
            target = targets[i]        # [num_labels]

            pos_inds = target.nonzero(as_tuple=False).squeeze(1)
            neg_inds = (target == 0).nonzero(as_tuple=False).squeeze(1)

            if len(pos_inds) == 0 or len(neg_inds) == 0:
                continue  # skip samples with no valid pairs

            pos_scores = logit[pos_inds]  # [num_pos]
            neg_scores = logit[neg_inds]  # [num_neg]

            # pairwise diff: shape [num_pos, num_neg]
            pairwise_diff = neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)  # [num_pos, num_neg]

            loss = torch.log1p(torch.exp(pairwise_diff).sum())
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        losses = torch.stack(losses)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses  # no reduction


class DRLoss(nn.Module):
    def __init__(self,
                gamma1=1,
                gamma2=1):
        super(DRLoss, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        # self.db_loss = ResampleLoss(freq_file=freq_file,
        #     use_sigmoid=True,
        #     reweight_func='rebalance',
        #     focal=focal,
        #     logit_reg=logit_reg,
        #     map_param=map_param,
        #     loss_weight=loss_weight, num_classes = num_classes,
        #     class_split = class_split)
    def forward(self,cls_score,labels,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        cls_score0 = cls_score.clone()
        cls_score0 = (1 - 2 * labels) * cls_score0
        neg_score = cls_score0 - labels * 1e12
        pos_score = cls_score0 - (1 - labels) * 1e12

        ## positive scores and negative scores
        s_p0 = pos_score * self.gamma1
        s_n0 = self.gamma1 * neg_score

        ######### DR Loss
        loss_dr = (1 + torch.exp(torch.logsumexp(s_p0,dim=0)) * torch.exp(torch.logsumexp(s_n0,dim=0))  \
             + torch.exp(torch.logsumexp(neg_score * self.gamma2,dim=0))
             ).log()

        return loss_dr.mean()


class DRLossStable(nn.Module):
    def __init__(self, gamma1=5.0, gamma2=7.0, reduction="mean", eps=1e-12):
        super().__init__()
        self.g1 = gamma1
        self.g2 = gamma2
        self.reduction = reduction
        self.eps = eps

    @staticmethod
    def _lse(x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x, dim=0)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape
        device = logits.device
        very_neg = torch.finfo(logits.dtype).min

        pos_w_full = labels.clamp(0.0, 1.0)           # (B,C)
        neg_w_full = (1.0 - labels).clamp(0.0, 1.0)
        log_pos_w_full = torch.log(pos_w_full.clamp_min(self.eps))
        log_neg_w_full = torch.log(neg_w_full.clamp_min(self.eps))

        per_class = []

        for k in range(C):
            s = logits[:, k]                          # (B,)
            lpw, lnw = log_pos_w_full[:, k], log_neg_w_full[:, k]
            has_pos = torch.any(pos_w_full[:, k] > 0)
            has_neg = torch.any(neg_w_full[:, k] > 0)

            if not has_pos and not has_neg:
                continue

            if has_neg:
                A = self._lse(self.g1 * s + lnw)     # log sum_neg w_neg * exp(g1*s_neg)
            else:
                A = torch.tensor(very_neg, device=device, dtype=logits.dtype)

            if has_pos:
                B = self._lse(-self.g1 * s + lpw)    # log sum_pos w_pos * exp(-g1*s_pos)
            else:
                B = torch.tensor(very_neg, device=device, dtype=logits.dtype)

            lse_pair = A + B                         # == log ∑∑ exp(g1*(s_n - s_p))（带权）

            if has_neg:
                lse_ngc = self._lse(self.g2 * s + lnw)   # NGC：log sum_neg w_neg * exp(g2*s_neg)
            else:
                lse_ngc = torch.tensor(very_neg, device=device, dtype=logits.dtype)

            # 稳定合成 log(1 + e^{lse_pair} + e^{lse_ngc})
            m = torch.maximum(torch.maximum(lse_pair, lse_ngc), torch.zeros((), device=device, dtype=logits.dtype))
            loss_k = m + torch.log(torch.exp(-m) + torch.exp(lse_pair - m) + torch.exp(lse_ngc - m))

            if not has_pos and has_neg:
                # 只有负：log(1 + e^{lse_ngc})
                m2 = torch.maximum(lse_ngc, torch.zeros((), device=device, dtype=logits.dtype))
                loss_k = m2 + torch.log(torch.exp(-m2) + torch.exp(lse_ngc - m2))
            elif has_pos and not has_neg:
                # 只有正：退化为 0
                loss_k = torch.zeros((), device=device, dtype=logits.dtype)

            per_class.append(loss_k)

        if len(per_class) == 0:
            total = torch.zeros((), device=device, dtype=logits.dtype)
        else:
            total = torch.stack(per_class).mean()

        if self.reduction == "sum":
            return total * len(per_class)
        return total