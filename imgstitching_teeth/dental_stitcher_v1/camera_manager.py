"""USB 摄像头管理模块 — 使用 OpenCV 管理 UVC 标准摄像头。"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


class CameraManager:
    """管理单个 UVC 摄像头的打开、预览和拍照。

    Streamlit 每次 rerun 都在新线程执行，因此内部使用线程锁保护
    cv2.VideoCapture 的读写操作。
    """

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._device_index: int = 0
        self._lock = threading.Lock()
        self._target_width: int = 1920
        self._target_height: int = 1080

    # ── 生命周期 ──────────────────────────────────────────

    def open(self, device_index: int = 0, width: int = 1920, height: int = 1080) -> bool:
        """打开指定设备索引的摄像头。

        Args:
            device_index: 摄像头设备索引（默认 0）
            width: 目标分辨率宽度（默认 1920）
            height: 目标分辨率高度（默认 1080）

        Returns:
            是否成功打开
        """
        self.close()  # 先关闭已有的

        with self._lock:
            # macOS 上优先使用 AVFoundation 后端
            cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                # 尝试默认后端
                cap = cv2.VideoCapture(device_index)
                if not cap.isOpened():
                    return False

            # 先读取一帧让摄像头初始化
            cap.read()

            # 设置 MJPEG 格式（高分辨率通常需要）
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # 读取实际分辨率
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._cap = cap
            self._device_index = device_index
            self._target_width = actual_width
            self._target_height = actual_height
            return True

    def close(self) -> None:
        """释放摄像头资源。"""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def is_opened(self) -> bool:
        """摄像头是否已打开。"""
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    # ── 帧获取 ──────────────────────────────────────────

    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧（用于实时预览）。

        Returns:
            BGR 格式 numpy 数组，失败返回 None
        """
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
            return frame

    def capture(self) -> Optional[np.ndarray]:
        """拍一张照片（连续读两帧取第二帧，提高成功率）。

        第一帧可能是缓存中的旧帧，丢弃后取第二帧确保是当前画面。

        Returns:
            BGR 格式 numpy 数组，失败返回 None
        """
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            # 丢弃可能过时的缓存帧
            self._cap.read()
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
            return frame.copy()

    # ── 属性 ──────────────────────────────────────────

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def frame_size(self) -> tuple[int, int]:
        """当前摄像头的分辨率 (width, height)。"""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return (0, 0)
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )

    # ── 工具 ──────────────────────────────────────────

    def get_supported_resolutions(self) -> list[tuple[int, int]]:
        """获取摄像头支持的所有分辨率。

        注意：这个函数可能会暂时改变分辨率，但会恢复。

        Returns:
            支持的分辨率列表 [(width, height), ...]
        """
        if not self.is_opened():
            return []

        resolutions = [
            (1920, 1080),
            (1280, 720),
            (640, 480),
            (320, 240),
        ]

        supported = []

        with self._lock:
            if self._cap is None:
                return []

            original_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            for w, h in resolutions:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_w == w and actual_h == h:
                    supported.append((w, h))

            # 恢复原分辨率
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_h)

        return supported

    @staticmethod
    def list_available_devices(max_check: int = 5) -> list[int]:
        """枚举系统中可用的摄像头设备索引。

        尝试打开 0 ~ max_check-1，返回能成功打开的索引列表。

        Args:
            max_check: 最多检查几个设备

        Returns:
            可用设备索引列表
        """
        available: list[int] = []
        # 临时抑制 OpenCV 警告
        old_level = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

        for idx in range(max_check):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available.append(idx)
                cap.release()

        cv2.utils.logging.setLogLevel(old_level)
        return available
