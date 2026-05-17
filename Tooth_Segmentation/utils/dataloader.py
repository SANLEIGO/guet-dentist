import torch
from utils.utils import cvtColor, preprocess_input
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2


class UnetDataset(Dataset):
    def __init__(
        self,
        data_path,
        input_shape,
        num_classes,
        augmentation=True,
        txt_name: str = "train.txt",
        gaussian_blur_prob: float = 0.35,
        motion_blur_prob: float = 0.25,
        gaussian_noise_prob: float = 0.35,
        saliva_prob: float = 0.30,
        gaussian_blur_kernel=(3, 9),
        gaussian_blur_sigma=(0.6, 1.8),
        motion_blur_kernel=(5, 17),
        gaussian_noise_std=(6.0, 18.0),
        saliva_alpha=(0.18, 0.40),
        saliva_count=(1, 3),
        bubble_count=(4, 11),
    ):
        with open(os.path.join(data_path, "VOC2012/ImageSets/Segmentation", txt_name), "r") as f:
            self.annotation_lines = f.readlines()

        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.data_path = data_path

        self.gaussian_blur_prob = gaussian_blur_prob
        self.motion_blur_prob = motion_blur_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.saliva_prob = saliva_prob

        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.motion_blur_kernel = motion_blur_kernel
        self.gaussian_noise_std = gaussian_noise_std
        self.saliva_alpha = saliva_alpha
        self.saliva_count = saliva_count
        self.bubble_count = bubble_count

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        jpg = Image.open(os.path.join(self.data_path, "VOC2012/JPEGImages", name + ".jpg"))
        png = Image.open(os.path.join(self.data_path, "VOC2012/SegmentationClass", name + ".png"))

        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.augmentation)

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png, dtype=np.uint8)
        png[png >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def apply_gaussian_blur(self, image):
        k_min, k_max = self.gaussian_blur_kernel
        k = int(self.rand(k_min, k_max))
        if k % 2 == 0:
            k += 1
        sigma_min, sigma_max = self.gaussian_blur_sigma
        sigma = self.rand(sigma_min, sigma_max)
        return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)

    def apply_motion_blur(self, image):
        k_min, k_max = self.motion_blur_kernel
        k = int(self.rand(k_min, k_max))
        if k % 2 == 0:
            k += 1
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        center = (k / 2 - 0.5, k / 2 - 0.5)
        angle = self.rand(0, 180)
        rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotate_matrix, (k, k))
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        return cv2.filter2D(image, -1, kernel)

    def apply_gaussian_noise(self, image):
        std_min, std_max = self.gaussian_noise_std
        noise = np.random.normal(0, self.rand(std_min, std_max), image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_saliva_simulation(self, image):
        h, w = image.shape[:2]
        overlay = image.astype(np.float32).copy()

        saliva_min, saliva_max = self.saliva_count
        for _ in range(np.random.randint(saliva_min, saliva_max + 1)):
            center = (int(self.rand(0, w)), int(self.rand(0, h)))
            axes = (
                int(self.rand(max(12, w * 0.04), max(20, w * 0.16))),
                int(self.rand(max(8, h * 0.03), max(16, h * 0.12)))
            )
            angle = self.rand(0, 180)
            alpha_min, alpha_max = self.saliva_alpha
            alpha = self.rand(alpha_min, alpha_max)
            color = np.array([
                self.rand(210, 255),
                self.rand(210, 255),
                self.rand(210, 255)
            ], dtype=np.float32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=self.rand(4, 10), sigmaY=self.rand(4, 10))
            mask_f = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
            overlay = overlay * (1.0 - mask_f) + color * mask_f

        bubble_min, bubble_max = self.bubble_count
        for _ in range(np.random.randint(bubble_min, bubble_max + 1)):
            center = (int(self.rand(0, w)), int(self.rand(0, h)))
            radius = int(self.rand(4, max(6, min(h, w) * 0.03)))
            thickness = max(1, int(radius * self.rand(0.15, 0.35)))
            bubble_color = (
                int(self.rand(210, 255)),
                int(self.rand(210, 255)),
                int(self.rand(210, 255))
            )
            cv2.circle(overlay, center, radius, bubble_color, thickness)
            highlight_center = (max(0, center[0] - radius // 3), max(0, center[1] - radius // 3))
            cv2.circle(overlay, highlight_center, max(1, radius // 5), (255, 255, 255), -1)

        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=self.rand(0.6, 1.8), sigmaY=self.rand(0.6, 1.8))
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def apply_image_degradations(self, image):
        if self.rand() < self.gaussian_blur_prob:
            image = self.apply_gaussian_blur(image)
        if self.rand() < self.motion_blur_prob:
            image = self.apply_motion_blur(image)
        if self.rand() < self.gaussian_noise_prob:
            image = self.apply_gaussian_noise(image)
        if self.rand() < self.saliva_prob:
            image = self.apply_saliva_simulation(image)
        return image

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, max(w - nw, 1)))
        dy = int(self.rand(0, max(h - nh, 1)))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        if self.rand() < 0.5:
            crop_scale = self.rand(0.7, 1.0)
            crop_h = int(h * crop_scale)
            crop_w = int(w * crop_scale)
            top = int(self.rand(0, h - crop_h + 1))
            left = int(self.rand(0, w - crop_w + 1))
            image_crop = image_data[top:top + crop_h, left:left + crop_w]
            label_crop = np.array(label)[top:top + crop_h, left:left + crop_w]
            image_data = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST))

        image_data = self.apply_image_degradations(image_data)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_c, sat_c, val_c = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue_c, lut_hue), cv2.LUT(sat_c, lut_sat), cv2.LUT(val_c, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    # edges = []

    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
        # edges.append(edge)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    # edges = torch.from_numpy(np.array(edges)).type(torch.FloatTensor)

    return images, pngs, seg_labels
