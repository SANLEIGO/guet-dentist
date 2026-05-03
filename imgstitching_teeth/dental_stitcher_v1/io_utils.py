from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImagePacket:
    def __init__(
        self,
        image: np.ndarray,
        name: str,
        timestamp: Optional[str] = None,
        arch: Optional[str] = None,  # "lower" or "upper"
    ):
        self.image = image
        self.name = name
        self.timestamp = timestamp
        self.arch = arch


def load_uploaded_images(files: Iterable) -> list[ImagePacket]:
    packets: list[ImagePacket] = []
    for uploaded in files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            continue
        packets.append(ImagePacket(image=image, name=uploaded.name))
    return packets


def resize_for_display(image: np.ndarray, max_width: int = 900, max_height: int = 700) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_image_metrics(image: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_32F).var()
    mean_intensity = float(gray.mean())
    exposure_score = 1.0 - min(abs(mean_intensity - 127.5) / 127.5, 1.0)
    return float(sharpness), float(exposure_score)


def normalize_image(image: np.ndarray, max_size: int = 1200) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_size / max(height, width), 1.0)
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
