import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    ce_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return ce_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)

    if alpha is not None:
        logpt *= alpha

    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def generate_boundary(mask, num_classes=21, ignore_index=None, boundary_width=1, soft_boundary=True):
    """
    从多类别分割标签生成边界 GT。
    mask: [B, H, W]
    return: [B, 1, H, W]
    """
    if mask.dim() != 3:
        raise ValueError("mask must have shape [B, H, W]")

    boundary = torch.zeros_like(mask, dtype=torch.float32)

    valid = torch.ones_like(mask, dtype=torch.bool)
    if ignore_index is not None:
        valid = mask != ignore_index

    center = mask[:, 1:-1, 1:-1]
    center_valid = valid[:, 1:-1, 1:-1]

    neighbor_diffs = [
        center != mask[:, :-2, 1:-1],
        center != mask[:, 2:, 1:-1],
        center != mask[:, 1:-1, :-2],
        center != mask[:, 1:-1, 2:],
        center != mask[:, :-2, :-2],
        center != mask[:, :-2, 2:],
        center != mask[:, 2:, :-2],
        center != mask[:, 2:, 2:],
    ]
    neighbor_valids = [
        valid[:, :-2, 1:-1],
        valid[:, 2:, 1:-1],
        valid[:, 1:-1, :-2],
        valid[:, 1:-1, 2:],
        valid[:, :-2, :-2],
        valid[:, :-2, 2:],
        valid[:, 2:, :-2],
        valid[:, 2:, 2:],
    ]

    edge_map = torch.zeros_like(center, dtype=torch.bool)
    for diff, neighbor_valid in zip(neighbor_diffs, neighbor_valids):
        edge_map |= diff & center_valid & neighbor_valid

    boundary[:, 1:-1, 1:-1] = edge_map.float()
    boundary = boundary.unsqueeze(1)
    hard_boundary = boundary

    if boundary_width > 1:
        if boundary_width % 2 == 0:
            raise ValueError("boundary_width should be odd, such as 3 or 5")
        padding = boundary_width // 2
        dilated_boundary = F.max_pool2d(boundary, kernel_size=boundary_width, stride=1, padding=padding)
        if soft_boundary:
            boundary = torch.maximum(hard_boundary, dilated_boundary * 0.5)
        else:
            boundary = dilated_boundary

    if ignore_index is not None:
        boundary = boundary * valid.unsqueeze(1).float()

    return boundary


def binary_dice_loss(pred_logits, target, smooth=1e-5):
    pred_prob = torch.sigmoid(pred_logits)
    pred_prob = pred_prob.view(pred_prob.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred_prob * target).sum(dim=1)
    union = pred_prob.sum(dim=1) + target.sum(dim=1)
    dice = 1 - (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def compute_boundary_loss(pred_boundary, seg_mask, num_classes=21, ignore_index=None, pos_weight=2.0,
                          bce_weight=0.5, dice_weight=0.5, smooth=1e-5,
                          boundary_width=1, soft_boundary=True):
    """
    边界损失函数（Weighted BCE + Dice）
    pred_boundary: [B, 1, H, W] logits
    seg_mask: [B, H, W] 分割标签
    """
    boundary_gt = generate_boundary(
        seg_mask,
        num_classes=num_classes,
        ignore_index=ignore_index,
        boundary_width=boundary_width,
        soft_boundary=soft_boundary,
    )

    valid_mask = torch.ones_like(boundary_gt)
    if ignore_index is not None:
        valid_mask = (seg_mask != ignore_index).unsqueeze(1).float()

    bce = F.binary_cross_entropy_with_logits(
        pred_boundary,
        boundary_gt,
        reduction='none',
        pos_weight=torch.as_tensor(pos_weight, device=pred_boundary.device, dtype=pred_boundary.dtype),
    )
    bce = (bce * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    pred_boundary = pred_boundary * valid_mask + (1.0 - valid_mask) * (-20.0)
    boundary_gt = boundary_gt * valid_mask
    dice = binary_dice_loss(pred_boundary, boundary_gt, smooth=smooth)

    return bce_weight * bce + dice_weight * dice


def segmentation_boundary_loss(outputs, seg_target, seg_onehot_target, cls_weights,
                               num_classes=21, focal_loss=True, dice_loss=True,
                               boundary_loss_weight=0.4, boundary_pos_weight=2.0,
                               boundary_width=1, soft_boundary=True):
    seg_outputs = outputs["seg"] if isinstance(outputs, dict) else outputs
    boundary_outputs = outputs.get("boundary", None)

    def compute_seg_loss(pred):
        if focal_loss:
            current_seg_loss = Focal_Loss(pred, seg_target, cls_weights, num_classes=num_classes)
        else:
            current_seg_loss = CE_Loss(pred, seg_target, cls_weights, num_classes=num_classes)

        if dice_loss:
            current_seg_loss = current_seg_loss + Dice_loss(pred, seg_onehot_target)
        return current_seg_loss

    seg_loss = compute_seg_loss(seg_outputs)

    boundary_loss = 0.0
    if boundary_outputs is not None:
        boundary_loss = compute_boundary_loss(
            boundary_outputs,
            seg_target,
            num_classes=num_classes,
            ignore_index=num_classes,
            pos_weight=boundary_pos_weight,
            boundary_width=boundary_width,
            soft_boundary=soft_boundary,
        )

    total_loss = seg_loss + boundary_loss_weight * boundary_loss
    return total_loss


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
