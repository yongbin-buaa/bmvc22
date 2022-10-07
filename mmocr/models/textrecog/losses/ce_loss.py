import torch.nn as nn
import torch
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class CELoss(nn.Module):
    """Implementation of loss module for encoder-decoder based text recognition
    method with CrossEntropy loss.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
    """

    def __init__(self, ignore_index=-1, reduction='none'):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']

        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']

        return outputs.permute(0, 2, 1).contiguous(), targets

    def forward(self, outputs, targets_dict):
        outputs, targets = self.format(outputs, targets_dict)

        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        losses = dict(loss_ce=loss_ce)

        return losses


@LOSSES.register_module()
class SARLoss(CELoss):
    """Implementation of loss module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
    """

    def __init__(self, ignore_index=0, reduction='mean', **kwargs):
        super().__init__(ignore_index, reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        # targets[0, :], [start_idx, idx1, idx2, ..., end_idx, pad_idx...]
        # outputs[0, :, 0], [idx1, idx2, ..., end_idx, ...]

        # ignore first index of target in loss calculation
        targets = targets[:, 1:].contiguous()
        # ignore last index of outputs to be in same seq_len with targets
        outputs = outputs[:, :-1, :].permute(0, 2, 1).contiguous()

        return outputs, targets


@LOSSES.register_module()
class RadicalLoss(CELoss):
    def __init__(self, ignore_index=0, reduction='mean', lam=0.1, **kwargs):
        super().__init__(ignore_index, reduction)
        self.bce_loss = torch.nn.BCELoss(reduction='none')
        self.lam = lam

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        # targets[0, :], [start_idx, idx1, idx2, ..., end_idx, pad_idx...]
        # outputs[0, :, 0], [idx1, idx2, ..., end_idx, ...]

        # ignore first index of target in loss calculation
        targets = targets[:, 1:].contiguous()
        # ignore last index of outputs to be in same seq_len with targets
        class_prediction_logits = outputs['class_logits']
        class_prediction_logits = class_prediction_logits[:, :-1, :].permute(0, 2, 1).contiguous()

        radical_prediction = outputs['radical_logits'][:, :-1, :].contiguous()
        radical_gt = outputs['radical_gt'][:, 1:].contiguous()
        radical_mask = outputs['radical_mask'][:, 1:].contiguous()

        return class_prediction_logits, radical_prediction, targets, radical_gt, radical_mask

    def forward(self, outputs, targets_dict):
        class_logits, radical_prediction, targets, radical_gt, radical_mask = self.format(outputs, targets_dict)

        loss_radical = self.bce_loss(radical_prediction, radical_gt) * radical_mask.unsqueeze(-1)
        loss_radical = loss_radical.sum() / max(radical_mask.sum() * radical_prediction.shape[-1], 1e-5)
        loss_ce = self.loss_ce(class_logits, targets.to(class_logits.device))
        losses = dict(
            loss_ce=loss_ce,
            loss_radical=loss_radical * self.lam
        )

        return losses

@LOSSES.register_module()
class FRLoss(CELoss):
    def __init__(self, ignore_index=0, reduction='mean', lam=1.0, **kwargs):
        super().__init__(ignore_index, reduction)
        self.lam = lam

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        # targets[0, :], [start_idx, idx1, idx2, ..., end_idx, pad_idx...]
        # outputs[0, :, 0], [idx1, idx2, ..., end_idx, ...]

        # ignore first index of target in loss calculation
        targets = targets[:, 1:].contiguous()
        # ignore last index of outputs to be in same seq_len with targets
        outputs = outputs[:, :-1, :].permute(0, 2, 1).contiguous()

        return outputs, targets

    def forward(self, outputs, targets_dict, angle_var):
        outputs, targets = self.format(outputs, targets_dict)

        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        loss_ce += angle_var * self.lam
        losses = dict(loss_ce=loss_ce)

        return losses


@LOSSES.register_module()
class SARFocalLoss(SARLoss):
    def __init__(self, ignore_index=0, reduction='mean', gamma=2, **kwargs):
        super().__init__(ignore_index, reduction)
        self.gamma = gamma
        self.reduction = reduction
        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='none'
        )
        self.ignore_index = ignore_index

    def forward(self, outputs, targets_dict):
        outputs, targets = self.format(outputs, targets_dict)

        bz, cls_num, seq = outputs.shape

        targets = targets.to(outputs.device)
        loss_ce = self.loss_ce(outputs, targets)
        index = torch.clamp(targets.unsqueeze(1), 0, cls_num-1)
        sm = F.softmax(outputs, dim=1)
        pt = torch.gather(sm, 1, index).squeeze()
        loss_ce =  loss_ce * (1 - pt) ** self.gamma
        if self.reduction == 'mean':
            cnt = torch.sum(targets != self.ignore_index)
            loss_ce = loss_ce.sum() / cnt
        elif self.reduction == 'sum':
            loss_ce = loss_ce.sum()

        losses = dict(loss_ce=loss_ce)

        return losses


@LOSSES.register_module()
class TFLoss(CELoss):
    """Implementation of loss module for transformer."""

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 flatten=True,
                 **kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)

        self.flatten = flatten

    def format(self, outputs, targets_dict):
        outputs = outputs[:, :-1, :].contiguous()
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()

        return outputs, targets
