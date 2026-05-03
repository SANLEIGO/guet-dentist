# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'src\\image_acquisition_widget.py'
# Bytecode version: 3.10.b1 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
class CameraStreamThread(QThread):
    """\n    相机流线程\n    用于在后台线程中持续获取相机图像\n    """
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    def __init__(self, camera_manager, parent=None):
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._running = False
        self._current_frame = None
    def run(self):
        self._running = True
        while self._running:
            try:
                frame = self._camera_manager.capture_frame()
                if frame is not None:
                    self._current_frame = frame.copy()
                    self.frame_ready.emit(self._current_frame)
                self.msleep(33)
            except Exception as e:
                self.error_occurred.emit(f'相机流错误: {str(e)}')
                self._running = False
    def stop(self):
        self._running = False
        self.wait()
    def get_current_frame(self):
        return self._current_frame
class ImageAcquisitionWidget(QWidget):
    """\n    图像采集窗口类\n    用于实时从摄像头获取图像\n    """
    image_acquired = pyqtSignal(np.ndarray)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._init_components()
        self._init_layout()
        self._init_signals_slots()
        self.camera_manager = None
        self.is_acquiring = False
        self.camera_thread = None
        self.current_frame = None
    def _init_ui(self):
        """\n        初始化窗口基本属性\n        """
        self.setWindowTitle('Image Acquisition')
        self.setMinimumSize(400, 300)
    def _init_components(self):
        """\n        初始化各个组件\n        """
        self.image_label = QLabel('Camera View')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet('background-color: #2a2a2a; color: white; border: 1px solid #444;')
        self.start_button = QPushButton('Start')
        self.start_button.setFixedSize(100, 30)
        self.stop_button = QPushButton('Stop')
        self.stop_button.setFixedSize(100, 30)
        self.capture_button = QPushButton('Capture')
        self.capture_button.setFixedSize(100, 30)
    def _init_layout(self):
        """\n        初始化布局\n        """
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.image_label, 1)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.capture_button)
        button_layout.setSpacing(10)
        main_layout.addLayout(button_layout)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
    def _init_signals_slots(self):
        """\n        初始化信号和槽连接\n        注：按钮信号现在通过InteractionController连接到相应的处理方法\n        """
        return None
    def set_camera_manager(self, camera_manager):
        """\n        设置相机管理器\n        参数:\n            camera_manager: EndoscopeCameraManager实例\n        """
        self.camera_manager = camera_manager
        if camera_manager:
            camera_manager.camera_connected.connect(self._on_camera_connected)
            camera_manager.camera_disconnected.connect(self._on_camera_disconnected)
            cameras = camera_manager.get_available_cameras()
            if cameras:
                self._update_camera_info(cameras)
    def _on_camera_connected(self, camera_info):
        """\n        相机连接成功的回调\n        """
        camera_text = f"相机信息: {camera_info['name']} (ID: {camera_info['id']})\n"
        camera_text += f"分辨率: {camera_info['width']}x{camera_info['height']}\n"
        camera_text += f"FPS: {camera_info['fps']:.1f}\n"
        if camera_info['is_endoscope']:
            camera_text += '设备类型: 内窥镜相机'
        else:
            camera_text += '设备类型: 普通相机'
    def _on_camera_disconnected(self):
        """\n        相机关闭的回调\n        """
        return None
    def _update_camera_info(self, cameras):
        """\n        更新相机信息显示\n        """
        if not cameras:
            self.camera_info_label.setText('相机信息: 未检测到相机')
    def start_acquisition(self):
        """\n        开始图像采集\n        """
        print('acquisition start image_acquisition_widget')
        if not self.is_acquiring:
            if self.camera_manager:
                if not self.camera_manager.is_camera_connected():
                    available_cameras = self.camera_manager.get_available_cameras()
                    if available_cameras:
                        endoscope_cameras = self.camera_manager.get_endoscope_cameras()
                        camera_to_use = endoscope_cameras[0] if endoscope_cameras else available_cameras[0]
                        self.camera_manager.connect_camera(camera_to_use['id'])
                    else:
                        self.camera_manager.detect_cameras()
                        return None
                self.camera_thread = CameraStreamThread(self.camera_manager, self)
                self.camera_thread.frame_ready.connect(self._on_new_frame)
                self.camera_thread.error_occurred.connect(self._on_stream_error)
                self.camera_thread.start()
                self.is_acquiring = True
    def stop_acquisition(self):
        """\n        停止图像采集\n        """
        if self.is_acquiring:
            self.is_acquiring = False
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread = None
    def capture_image(self):
        """\n        捕获当前图像\n        """
        if self.current_frame is not None:
            self.image_acquired.emit(self.current_frame.copy())
        else:
            if self.camera_manager:
                frame = self.camera_manager.capture_frame()
                if frame is not None:
                    self.image_acquired.emit(frame.copy())
    def _on_new_frame(self, frame):
        """\n        收到新帧的回调\n        """
        self.current_frame = frame
        self.update_display(frame)
    def _on_stream_error(self, error_message):
        """\n        相机流错误的回调\n        """
        print(f'相机流错误: {error_message}')
        self.stop_acquisition()
    def update_display(self, frame):
        """\n        更新显示的图像\n        参数:\n            frame: OpenCV格式的图像 (numpy.ndarray)\n        """
        if frame is not None:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_image = q_image.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(QPixmap.fromImage(scaled_image))
    def closeEvent(self, event):
        """\n        关闭窗口时的清理工作\n        """
        self.stop_acquisition()
        event.accept()