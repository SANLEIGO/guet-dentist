# 导入标准库和第三方库
import datetime
import os
import subprocess
import time
from functools import partial

# 导入Numpy和PyTorch相关库
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入自定义模块和模型
from model import AttentionUNet, DualBranchCrossAttnUNet,ResNet50UNet, StandardUNet, deeplabv3_resnet50
from model.vit_seg_modeling import VisionTransformer as ViT_seg
from model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.create_exp_folder import create_exp_folder
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.plot_results import plot_training_curves
from utils.train_and_eval import train_one_epoch, evaluate
from utils.utils import seed_everything, worker_init_fn


def get_gpu_usage():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    used, total = map(int, result.strip().split(','))
    return used


def create_model(num_classes, weights, model_name="improved_resnet_unet"):
    model_name = model_name.lower()
    if model_name == "standard_unet":
        model = StandardUNet(num_classes=num_classes)
    elif model_name == "attention_unet":
        model = AttentionUNet(num_classes=num_classes)
    elif model_name == "resnet50_unet":
        model = ResNet50UNet(num_classes=num_classes)
    elif model_name == "dual_branch_cross_attn_unet":
        model = DualBranchCrossAttnUNet(num_classes=num_classes)
    elif model_name == "deeplabv3":
        model = deeplabv3_resnet50(num_classes=num_classes)
    elif model_name == "transunet":
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(512 / 16), int(512 / 16))
        model = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes)

    weights_init(model)

    if weights:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights, map_location='cpu')

        temp_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    return model


def get_optimizer_and_lr(model, batch_size, train_epoch, momentum, weight_decay):
    init_lr = 1e-4
    min_lr = init_lr * 0.01
    lr_decay_type = 'cos'

    nbs = 16
    lr_limit_max = 1e-4
    lr_limit_min = 1e-4

    init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = optim.Adam(model.parameters(), init_lr_fit, betas=(momentum, 0.999),
                           weight_decay=weight_decay)
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, train_epoch)
    return optimizer, lr_scheduler_func

init_fn = partial(worker_init_fn, seed=11)

def train(args):
    seed_everything(11)

    num_classes = args.num_classes + 1
    train_epoch = args.epochs
    batch_size = args.batch_size
    num_workers = args.workers

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    exp_folder, weights_folder = create_exp_folder(args.model_name)
    input_shape = [512, 512]

    train_dataset = UnetDataset(
        args.data_path,
        input_shape,
        num_classes,
        augmentation=True,
        txt_name="train.txt",
        gaussian_blur_prob=args.gaussian_blur_prob,
        motion_blur_prob=args.motion_blur_prob,
        gaussian_noise_prob=args.gaussian_noise_prob,
        saliva_prob=args.saliva_prob,
        gaussian_blur_kernel=(args.gaussian_blur_kernel_min, args.gaussian_blur_kernel_max),
        gaussian_blur_sigma=(args.gaussian_blur_sigma_min, args.gaussian_blur_sigma_max),
        motion_blur_kernel=(args.motion_blur_kernel_min, args.motion_blur_kernel_max),
        gaussian_noise_std=(args.gaussian_noise_std_min, args.gaussian_noise_std_max),
        saliva_alpha=(args.saliva_alpha_min, args.saliva_alpha_max),
        saliva_count=(args.saliva_count_min, args.saliva_count_max),
        bubble_count=(args.bubble_count_min, args.bubble_count_max),
    )
    val_dataset = UnetDataset(
        args.data_path,
        input_shape,
        num_classes,
        augmentation=False,
        txt_name="val.txt",
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=unet_dataset_collate,
        sampler=None,
        worker_init_fn=init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=unet_dataset_collate,
        sampler=None,
        worker_init_fn=init_fn
    )

    model = create_model(num_classes=num_classes, weights=args.weights, model_name=args.model_name)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())
    optimizer, lr_scheduler_func = get_optimizer_and_lr(model, batch_size, train_epoch, args.momentum, args.weight_decay)

    start_time = time.time()

    best_dice = 0.0
    best_model_path = os.path.join(weights_folder, f"best_model.pth")
    last_model_path = os.path.join(weights_folder, f"last_model.pth")

    train_losses = []
    val_losses = []
    val_metrics_history = []
    metrics_log_path = os.path.join(exp_folder, f"epoch_metrics_{args.model_name}.txt")

    with open(metrics_log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,  val_loss,    tooth_iou,   tooth_dice,   precision,   recall\n")

    focal_loss = True
    dice_loss = True

    for epoch in range(train_epoch):
        gpu_used = get_gpu_usage()
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        loss = train_one_epoch(
            model, optimizer, train_loader, device, dice_loss, focal_loss,
            gpu_used, num_classes, scaler, epoch, train_epoch
        )
        train_losses.append(loss)
        metrics = evaluate(
            model, val_loader, device, dice_loss, focal_loss, num_classes
        )

        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        current_dice = float(metrics["Tooth Dice"])
        if current_dice > best_dice:
            best_dice = current_dice
            torch.save(model.state_dict(), best_model_path)

        with open(metrics_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch + 1:3d},{loss:12.8f},{metrics['Loss']:12.8f},"
                f"{metrics['Tooth IoU']:12.8f},"
                f"{metrics['Tooth Dice']:12.8f},"
                f"{metrics['Precision']:12.8f},"
                f"{metrics['Recall']:12.8f}\n"
            )


        torch.save(model.state_dict(), last_model_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--weights", default="",
                        help="Path to the directory containing model weights")
    parser.add_argument("--data-path", default="VOCdevkit", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--model-name", default="dual_branch_cross_attn_unet",
                        choices=["resnet50_unet", "standard_unet", "attention_unet","dual_branch_cross_attn_unet","deeplabv3","transunet"],
                        help="choose segmentation model")
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument("--workers", default=6, type=int, metavar="N",
                        help="number of data loading workers (default: 0, meaning data loading runs in main process)")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument("--gaussian-blur-prob", default=0.35, type=float)
    parser.add_argument("--motion-blur-prob", default=0.25, type=float)
    parser.add_argument("--gaussian-noise-prob", default=0.3, type=float)
    parser.add_argument("--saliva-prob", default=0.2, type=float)
    parser.add_argument("--gaussian-blur-kernel-min", default=3, type=int)
    parser.add_argument("--gaussian-blur-kernel-max", default=9, type=int)
    parser.add_argument("--gaussian-blur-sigma-min", default=0.6, type=float)
    parser.add_argument("--gaussian-blur-sigma-max", default=1.8, type=float)
    parser.add_argument("--motion-blur-kernel-min", default=5, type=int)
    parser.add_argument("--motion-blur-kernel-max", default=17, type=int)
    parser.add_argument("--gaussian-noise-std-min", default=6.0, type=float)
    parser.add_argument("--gaussian-noise-std-max", default=18.0, type=float)
    parser.add_argument("--saliva-alpha-min", default=0.18, type=float)
    parser.add_argument("--saliva-alpha-max", default=0.40, type=float)
    parser.add_argument("--saliva-count-min", default=1, type=int)
    parser.add_argument("--saliva-count-max", default=3, type=int)
    parser.add_argument("--bubble-count-min", default=4, type=int)
    parser.add_argument("--bubble-count-max", default=11, type=int)

    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
