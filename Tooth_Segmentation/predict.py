import colorsys
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import AttentionUNet, ResNet50UNet, StandardUNet, DualBranchCrossAttnUNet
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.create_exp_folder import create_val_exp_folder


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def create_model(num_classes, model_name="improved_resnet_unet"):
    model_name = model_name.lower()
    if model_name == "standard_unet":
        return StandardUNet(num_classes=num_classes)
    if model_name == "attention_unet":
        return AttentionUNet(num_classes=num_classes)
    if model_name == "resnet50_unet":
        return ResNet50UNet(num_classes=num_classes)
    if model_name == "dual_branch_cross_attn_unet":
        return DualBranchCrossAttnUNet(num_classes=num_classes)


def load_model(model_path, num_classes, device, model_name="improved_resnet_unet"):
    net = create_model(num_classes, model_name=model_name)
    weights = torch.load(model_path, map_location=device)
    net.load_state_dict(weights)
    net.eval()
    net.to(device)
    return net


def detect_image(file_path, model, num_classes, exp_folder, mix_type=True):
    try:
        image = Image.open(file_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error opening image: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = cvtColor(image)
    old_img = image.copy()

    input_shape = [512, 512]
    original_h, original_w = np.array(image).shape[:2]
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))

    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
        0
    )

    if num_classes <= 21:
        colors = [(0, 0, 0), (0, 128, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                  (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                  (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128)]
    else:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    with torch.no_grad():
        images = torch.from_numpy(image_data).to(device)
        outputs = model(images)

        # --------------------------
        # 统一处理分割输出
        # --------------------------
        if isinstance(outputs, dict):
            seg_logits = outputs["seg"]
            # 判断是否有 boundary 输出
            has_boundary = "boundary" in outputs
            boundary_pred = outputs["boundary"] if has_boundary else None
        else:
            seg_logits = outputs
            has_boundary = False
            boundary_pred = None

        # 处理分割结果
        seg_logits = seg_logits[0]  # [1, C, H, W] -> [C, H, W]
        pr = F.softmax(seg_logits.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr = pr[
            int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh),
            int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)
        ]
        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)

    # --------------------------
    # 保存分割彩色图
    # --------------------------
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])
    if mix_type:
        old_img_np = np.array(old_img)
        alpha = 0.4
        blended_img = cv2.addWeighted(old_img_np, 1 - alpha, seg_img, alpha, 0)
        image = Image.fromarray(blended_img)
    else:
        image = Image.fromarray(np.uint8(seg_img))

    img_name = os.path.basename(file_path)
    base_name = os.path.splitext(img_name)[0]
    mask_filename = f"{base_name}_mask.png"
    save_path = os.path.join(exp_folder, mask_filename)
    image.save(save_path)
    print(f"Segmentation mask saved: {save_path}")

    # --------------------------
    # ✅ 自动保存边界图（如果有）
    # --------------------------
    if has_boundary and boundary_pred is not None:
        with torch.no_grad():
            # 处理 boundary [1,1,H,W]
            boundary = boundary_pred[0]  # 取 batch 第 0 张
            boundary = torch.sigmoid(boundary)  # 转 0~1 概率图
            boundary = boundary.cpu().numpy().squeeze(0)  # [H,W]

            # 裁剪 + resize 回原图大小
            boundary = boundary[
                int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh),
                int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)
            ]
            boundary = cv2.resize(boundary, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            # 转 0~255 可视化
            boundary = (boundary * 255).astype(np.uint8)

        # 保存边界图
        boundary_filename = f"{base_name}_boundary.png"
        boundary_save_path = os.path.join(exp_folder, boundary_filename)
        cv2.imwrite(boundary_save_path, boundary)
        print(f"Boundary map saved: {boundary_save_path}")


def predict(args):
    exp_folder = create_val_exp_folder(args.model_name)
    num_classes = args.num_classes + 1

    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights, num_classes, device, model_name=args.model_name)
    model.model_name = args.model_name

    if os.path.isdir(args.data_path):
        file_paths = [str(p) for p in Path(args.data_path).rglob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    elif os.path.isfile(args.data_path):
        file_paths = [args.data_path]
    else:
        raise ValueError(f"Unsupported input path: {args.data_path}")

    t_start = time_synchronized()

    for file_path in file_paths:
        if file_path.lower().endswith((".jpg", ".png", ".jpeg")):
            detect_image(file_path, model, num_classes, exp_folder, mix_type=args.mix_type)

    t_end = time_synchronized()
    print(f"inference time for: {t_end - t_start}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch unet predict")
    parser.add_argument("--data_path", default="test/testimage1", help="data root")
    parser.add_argument("--weights", default="run/train/dual_branch_cross_attn_unet/exp17/weights/best_model.pth")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--model-name", default="dual_branch_cross_attn_unet",
                        choices=["resnet50_unet", "standard_unet", "attention_unet", "dual_branch_cross_attn_unet"],
                        help="choose segmentation model")
    parser.add_argument("--mix_type", default=True, action='store_true',
                        help="Save original and segmentation result side by side")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    predict(args)
