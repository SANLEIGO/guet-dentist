import os

import torch
import numpy as np
import torch.nn.functional as F
from model.unet_training import segmentation_boundary_loss



from utils.utils import get_lr
from torch.cuda.amp import autocast
import time


class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"


def unpack_seg_output(output):
    if isinstance(output, dict):
        return output["seg"]
    return output


def tooth_stats(output, target, num_classes):
    output = unpack_seg_output(output)
    with torch.no_grad():
        _, predicted = torch.max(output, dim=1)

        total_intersection = 0
        total_union = 0
        total_pred = 0
        total_target = 0

        for i in range(1, num_classes):
            target_mask = (target == i)
            pred_mask = (predicted == i)

            if target_mask.sum().item() > 0:
                intersection = torch.logical_and(target_mask, pred_mask).sum().item()
                union = torch.logical_or(target_mask, pred_mask).sum().item()
                pred_sum = pred_mask.sum().item()
                target_sum = target_mask.sum().item()

                total_intersection += intersection
                total_union += union
                total_pred += pred_sum
                total_target += target_sum

        return total_intersection, total_union, total_pred, total_target


def train_one_epoch(model, optimizer, train_loader, device, dice_loss, focal_loss,
                    gpu_used, num_classes, scaler, epoch, train_epoch):

    cls_weights = np.ones([num_classes], np.float32)
    epoch_loss = 0.0

    model_train = model.train()
    model_train = model_train.cuda()

    for iteration, batch in enumerate(train_loader):
        imgs, pngs, labels = batch

        weights = torch.tensor(cls_weights).to(device)
        imgs = imgs.to(device)
        pngs = pngs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is None:
            outputs = model_train(imgs)
            loss= segmentation_boundary_loss(
                outputs, pngs, labels, weights,
                num_classes=num_classes,
                focal_loss=focal_loss,
                dice_loss=dice_loss,
            )
            loss.backward()
            optimizer.step()
        else:
            with torch.amp.autocast('cuda'):
                outputs = model_train(imgs)
                loss = segmentation_boundary_loss(
                    outputs, pngs, labels, weights,
                    num_classes=num_classes,
                    focal_loss=focal_loss,
                    dice_loss=dice_loss,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += loss.item()

        if iteration == 0:
            print(f"{LogColor.GREEN}Epoch{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}data_num{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}GPU Mem{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Loss{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}LR{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Image_size{LogColor.RESET}{' ' * 12}")

        a = 1
        Epoch_len = len("Epoch") + 12 - len(str(f"{epoch + 1}/{train_epoch}"))
        batch_len = len("data_num") + 12 - len(str(f"{iteration + a}/{len(train_loader)}"))
        GPU_len = len("GPU Mem") + 12 - len(str(f"{gpu_used:.2f} MB"))
        Loss_len = len("Loss") + 12 - len(str(f"{loss.item():.8f}"))
        LR_len = len("LR") + 12 - len(str(f"{get_lr(optimizer):.8f}"))

        print(f"\r{epoch + 1}/{train_epoch}{' ' * max(1, Epoch_len)}"
              f"{iteration + a}/{len(train_loader)}{' ' * max(1, batch_len)}"
              f"{gpu_used:.2f} MB{' ' * max(1, GPU_len)}"
              f"{loss.item():.8f}{' ' * max(1, Loss_len)}"
              f"{get_lr(optimizer):.8f}{' ' * max(1, LR_len)}"
              f"{imgs.shape[2]}", end='', flush=True)

    print(f"{LogColor.GREEN}")
    time.sleep(1)
    return epoch_loss / len(train_loader)


def _print_eval_header(prefix_width=0):
    print(
        f"{' ' * prefix_width}"
        f"{LogColor.RED}data_num{LogColor.RESET}{' ' * 12}"
        f"{LogColor.RED}Tooth IoU{LogColor.RESET}{' ' * 10}"
        f"{LogColor.RED}Tooth Dice{LogColor.RESET}{' ' * 8}"
        f"{LogColor.RED}Precision{LogColor.RESET}{' ' * 8}"  # 新增
        f"{LogColor.RED}Recall{LogColor.RESET}{' ' * 12}"     # 新增
    )


def _print_eval_metrics(progress_text, tooth_iou, tooth_dice, precision, recall, prefix_width=0):
    print(
        f"\r{' ' * prefix_width}"
        f"{progress_text}{' ' * max(2, 20 - len(str(progress_text)))}"
        f"{tooth_iou:.4f}{' ' * max(2, 18 - len(f'{tooth_iou:.4f}'))}"
        f"{tooth_dice:.4f}{' ' * max(2, 18 - len(f'{tooth_dice:.4f}'))}"
        f"{precision:.4f}{' ' * max(2, 18 - len(f'{precision:.4f}'))}"  # 新增
        f"{recall:.4f}{' ' * max(2, 18 - len(f'{recall:.4f}'))}"        # 新增
        , end='', flush=True
    )


def evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes):

    cls_weights = np.ones([num_classes], np.float32)
    val_loss = 0

    model_eval = model.eval()
    model_eval = model_eval.cuda()

    total_intersection = 0
    total_union = 0
    total_pred = 0
    total_target = 0
    num_batches = len(val_loader)
    epoch_len = len("Epoch") + 12

    with torch.no_grad():
        for iteration, batch in enumerate(val_loader):
            imgs, pngs, labels= batch

            weights = torch.tensor(cls_weights).to(device)
            imgs = imgs.to(device)
            pngs = pngs.to(device)
            labels = labels.to(device)
            outputs = model_eval(imgs)

            loss = segmentation_boundary_loss(
                outputs, pngs, labels, weights,
                num_classes=num_classes,
                focal_loss=focal_loss,
                dice_loss=dice_loss,
            )

            intersection, union, pred_sum, target_sum = tooth_stats(outputs, pngs, num_classes)

            total_intersection += intersection
            total_union += union
            total_pred += pred_sum
            total_target += target_sum
            val_loss += loss.item()

            running_tooth_iou = total_intersection / total_union if total_union > 0 else 0.0
            running_tooth_dice = (2.0 * total_intersection) / (total_pred + total_target) if (total_pred + total_target) > 0 else 0.0
            running_precision = intersection / pred_sum if pred_sum > 0 else 0.0
            running_recall = intersection / target_sum if target_sum > 0 else 0.0

            if iteration == 0:
                _print_eval_header(prefix_width=epoch_len)

            progress_text = f"{iteration + 1}/{len(val_loader)}"
            _print_eval_metrics(
                progress_text, running_tooth_iou, running_tooth_dice,running_precision,running_recall,
                prefix_width=epoch_len
            )

    
    avg_tooth_iou = total_intersection / total_union if total_union > 0 else 0.0
    avg_tooth_dice = (2.0 * total_intersection) / (total_pred + total_target) if (total_pred + total_target) > 0 else 0.0
    precision = total_intersection / total_pred if total_pred > 0 else 0.0
    recall = total_intersection / total_target if total_target > 0 else 0.0
    avg_loss = val_loss / num_batches

    metrics = {
        'Tooth IoU': avg_tooth_iou,
        'Tooth Dice': avg_tooth_dice,
        'Precision': precision,        # 新增
        'Recall': recall,              # 新增
        'Loss': avg_loss
    }
    print(f"\n{LogColor.GREEN}")
    time.sleep(1)

    return metrics
