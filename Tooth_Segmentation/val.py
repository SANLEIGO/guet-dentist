# 导入标准库和第三方库
import torch
from torch.utils.data import DataLoader
import time

# 导入自定义模块和模型
from model import AttentionUNet, ResNet50UNet, StandardUNet, DualBranchCrossAttnUNet, deeplabv3_resnet50
from model.vit_seg_modeling import VisionTransformer as ViT_seg
from model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.train_and_eval import tooth_stats


class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"


def _print_header():
    data_num_len = 12
    tooth_iou_len = len("LR") - len("Tooth IoU") + 12
    tooth_dice_len = 12
    precision_len = 12  # 新增
    recall_len = 12     # 新增

    print(
        f"{LogColor.RED}data_num{LogColor.RESET}{' ' * data_num_len}"
        f"{LogColor.RED}Tooth IoU{LogColor.RESET}{' ' * tooth_iou_len}"
        f"{LogColor.RED}Tooth Dice{LogColor.RESET}{' ' * tooth_dice_len}"
        f"{LogColor.RED}Precision{LogColor.RESET}{' ' * precision_len}"  # 新增
        f"{LogColor.RED}Recall{LogColor.RESET}{' ' * recall_len}"        # 新增
    )


def _print_metrics(progress_text, tooth_iou, tooth_dice, precision, recall):
    data_num_len = max(2, 12 + len("data_num") - len(str(progress_text)))
    tooth_iou_len = max(2, len("LR") + 12 - len(f"{tooth_iou:.4f}"))
    tooth_dice_len = max(2, 12 + len("Tooth Dice") - len(f"{tooth_dice:.4f}"))
    precision_len = max(2, 12 + len("Precision") - len(f"{precision:.4f}"))  # 新增
    recall_len = max(2, 12 + len("Recall") - len(f"{recall:.4f}"))            # 新增

    print(
        f"{progress_text}{' ' * data_num_len}"
        f"{tooth_iou:.4f}{' ' * tooth_iou_len}"
        f"{tooth_dice:.4f}{' ' * tooth_dice_len}"
        f"{precision:.4f}{' ' * precision_len}"  # 新增
        f"{recall:.4f}{' ' * recall_len}",        # 新增
        flush=True
    )


def evaluate(model, val_loader, device, num_classes):
    model_eval = model.eval()
    model_eval = model_eval.cuda()

    total_intersection = 0
    total_union = 0
    total_pred = 0
    total_target = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for iteration, batch in enumerate(val_loader):
            imgs, pngs, labels= batch

            imgs = imgs.to(device)
            pngs = pngs.to(device)
            outputs = model_eval(imgs)

            intersection, union, pred_sum, target_sum = tooth_stats(outputs, pngs, num_classes)

            total_intersection += intersection
            total_union += union
            total_pred += pred_sum
            total_target += target_sum

            running_tooth_iou = total_intersection / total_union if total_union > 0 else 0.0
            running_tooth_dice = (2.0 * total_intersection) / (total_pred + total_target) if (total_pred + total_target) > 0 else 0.0
            running_precision = intersection / pred_sum if pred_sum > 0 else 0.0
            running_recall = intersection / target_sum if target_sum > 0 else 0.0

            if iteration == 0:
                _print_header()

            _print_metrics(f"{iteration + 1}/{len(val_loader)}", running_tooth_iou, running_tooth_dice,running_precision,running_recall)

    avg_tooth_iou = total_intersection / total_union if total_union > 0 else 0.0
    avg_tooth_dice = (2.0 * total_intersection) / (total_pred + total_target) if (total_pred + total_target) > 0 else 0.0
    precision = total_intersection / total_pred if total_pred > 0 else 0.0
    recall = total_intersection / total_target if total_target > 0 else 0.0

    _print_metrics("avg", avg_tooth_iou, avg_tooth_dice,precision,recall)
    print(f"\n{LogColor.GREEN}")
    time.sleep(1)


def create_model(num_classes, model_name="dual_branch_cross_attn_unet"):
    model_name = model_name.lower()
    if model_name == "standard_unet":
        return StandardUNet(num_classes=num_classes)
    if model_name == "attention_unet":
        return AttentionUNet(num_classes=num_classes)
    if model_name == "resnet50_unet":
        return ResNet50UNet(num_classes=num_classes)
    if model_name == "dual_branch_cross_attn_unet":
        return DualBranchCrossAttnUNet(num_classes=num_classes)
    elif model_name == "deeplabv3":
        return deeplabv3_resnet50(num_classes=num_classes)
    elif model_name == "transunet":
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(512 / 16), int(512 / 16))
        return ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes)

def val(args):
    num_classes = args.num_classes + 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    input_shape = [512, 512]

    val_dataset = UnetDataset(
        args.data_path,
        input_shape,
        num_classes,
        augmentation=False,
        txt_name="test.txt"
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=unet_dataset_collate,
        sampler=None,
    )

    model = create_model(num_classes=num_classes, model_name=args.model_name)
    weights_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights_dict)
    model.to(device)

    evaluate(model, val_loader, device, num_classes)



def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch fcn validation")
    parser.add_argument("--data-path", default="VOCdevkit", help="VOCdevkit root")
    parser.add_argument("--weights", default="run/train/dual_branch_cross_attn_unet/exp16/weights/best_model.pth")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="validation device")
    parser.add_argument("--model-name", default="dual_branch_cross_attn_unet",
                        choices=["resnet50_unet", "standard_unet", "attention_unet","dual_branch_cross_attn_unet","deeplabv3","transunet"],
                        help="choose segmentation model")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    val(args)
