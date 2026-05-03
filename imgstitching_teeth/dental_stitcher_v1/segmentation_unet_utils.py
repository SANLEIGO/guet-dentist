"""
U-Net 分割工具函数
从之前的项目中移植
"""
import numpy as np
from PIL import Image


def cvtColor(image):
    """转换图像为 RGB"""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        return image.convert('RGB')


def preprocess_input(image):
    """预处理图像"""
    image /= 255.0
    return image


def resize_image(image, size):
    """
    调整图像大小，保持宽高比
    返回: (image_data, new_width, new_height)
    """
    iw, ih = image.size
    w, h = size

    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
