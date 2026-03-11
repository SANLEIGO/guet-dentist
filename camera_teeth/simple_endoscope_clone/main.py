import os
import sys
import cv2
import subprocess
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class CameraApp(QMainWindow):
    """
    最小化摄像头桌面程序：
    - 开始/停止实时预览
    - 拍照（抓取当前帧）
    - 默认保存到本地，并支持“另存为”
    """

    def __init__(self):
        super().__init__()

        # ---------------------------------------------------------------------
        # 第1部分：主窗口配置
        # 固定窗口大小，避免布局和画面区域在运行时被拉伸。
        # ---------------------------------------------------------------------
        self.setWindowTitle('Simple Dental Camera')
        self.setFixedSize(1000, 650)

        # ---------------------------------------------------------------------
        # 第2部分：运行时状态
        # cap：OpenCV 相机句柄
        # current_frame：当前最新一帧（BGR）
        # capture_dir：拍照默认保存目录
        # ---------------------------------------------------------------------
        self.cap = None
        self.current_frame = None
        self.capture_dir = os.path.join(os.path.dirname(__file__), 'captures')
        os.makedirs(self.capture_dir, exist_ok=True)

        # ---------------------------------------------------------------------
        # 第3部分：定时器驱动取帧
        # 每33ms读取一帧，约30FPS。
        # ---------------------------------------------------------------------
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_frame)

        # ---------------------------------------------------------------------
        # 第4部分：图像显示区域
        # video_label：实时预览画面
        # captured_label：最近一次拍照结果
        # QSizePolicy.Ignored：避免 pixmap 反向撑大布局
        # ---------------------------------------------------------------------
        self.video_label = QLabel('Camera Preview')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet('background-color:#1f1f1f;color:#e8e8e8;border:1px solid #555;')
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.captured_label = QLabel('Captured Image')
        self.captured_label.setAlignment(Qt.AlignCenter)
        self.captured_label.setStyleSheet('background-color:#2b2b2b;color:#e8e8e8;border:1px solid #555;')
        self.captured_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # ---------------------------------------------------------------------
        # 第5部分：控制按钮
        # Start：打开摄像头并开始预览
        # Stop：停止预览并释放摄像头
        # Capture：拍照（抓取当前帧）
        # Save As：把最近一次拍照保存到指定路径
        # ---------------------------------------------------------------------
        self.start_btn = QPushButton('Start')
        self.stop_btn = QPushButton('Stop')
        self.capture_btn = QPushButton('Capture')
        self.save_as_btn = QPushButton('Save As...')
        self.refresh_btn = QPushButton('Refresh Cameras')
        self.camera_selector = QComboBox()
        self.camera_selector.setMinimumWidth(180)

        # 初始按钮状态：
        # - 未启动相机前，Stop/Capture 禁用
        # - 未拍照前，Save As 禁用
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.save_as_btn.setEnabled(False)

        # 绑定按钮点击事件到对应处理函数。
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.capture_btn.clicked.connect(self.capture_image)
        self.save_as_btn.clicked.connect(self.save_as)
        self.refresh_btn.clicked.connect(self.refresh_camera_list)

        # ---------------------------------------------------------------------
        # 第6部分：界面布局
        # 上方：实时预览 + 拍照结果
        # 下方：操作按钮
        # ---------------------------------------------------------------------
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Camera:'))
        controls.addWidget(self.camera_selector)
        controls.addWidget(self.refresh_btn)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.capture_btn)
        controls.addWidget(self.save_as_btn)

        images = QHBoxLayout()
        images.addWidget(self.video_label, 2)
        images.addWidget(self.captured_label, 1)

        root = QVBoxLayout()
        root.addLayout(images, 1)
        root.addLayout(controls)

        wrapper = QWidget()
        wrapper.setLayout(root)
        self.setCentralWidget(wrapper)

        # 首次启动时检测可用相机，填充选择器。
        self.refresh_camera_list()

    def start_camera(self):
        """
        打开相机并启动定时取帧。
        """
        # 已经打开时直接返回，避免重复打开设备。
        if self.cap is not None:
            return

        # 打开默认相机（索引0）。
        selected_camera_id = self.camera_selector.currentData()
        if selected_camera_id is None:
            QMessageBox.warning(self, 'Warning', 'No camera selected.')
            return

        self.cap = cv2.VideoCapture(int(selected_camera_id), cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            QMessageBox.critical(self, 'Error', f'Cannot open camera (index {selected_camera_id}).')
            return

        # 根据你这类高像素镜头，优先尝试更高分辨率；驱动不支持时会自动回退。
        # 依次尝试：4K -> 2K -> 1080p。
        candidate_resolutions = [(3840, 2160), (2560, 1440), (1920, 1080)]
        for width, height in candidate_resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if real_w >= width and real_h >= height:
                break

        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 启动实时更新，并切换按钮状态。
        self.timer.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self.camera_selector.setEnabled(False)
        self.refresh_btn.setEnabled(False)

    def stop_camera(self):
        """
        停止预览并安全释放相机资源。
        """
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # 恢复初始按钮状态。
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.camera_selector.setEnabled(True)
        self.refresh_btn.setEnabled(True)

    def update_frame(self):
        """
        从相机读取一帧并刷新实时预览。
        """
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        # 保存当前帧副本，供拍照功能直接使用。
        self.current_frame = frame.copy()
        self.show_image(self.video_label, self.current_frame)

    def capture_image(self):
        """
        拍照流程：
        1）将当前帧显示到“拍照结果”区域
        2）按时间戳文件名保存到 captures/ 目录
        """
        if self.current_frame is None:
            QMessageBox.warning(self, 'Warning', 'No frame available to capture.')
            return

        captured = self.current_frame.copy()
        self.show_image(self.captured_label, captured)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out_path = os.path.join(self.capture_dir, f'capture_{timestamp}.jpg')
        ok = cv2.imwrite(out_path, captured)
        if ok:
            self.last_saved_path = out_path
            self.save_as_btn.setEnabled(True)
            self.statusBar().showMessage(f'Captured and saved: {out_path}', 5000)
        else:
            QMessageBox.critical(self, 'Error', 'Failed to save captured image.')

    def save_as(self):
        """
        把最近一次拍照结果保存到用户指定路径。
        """
        # 先确认已有可保存的拍照文件。
        if not hasattr(self, 'last_saved_path') or not os.path.exists(self.last_saved_path):
            QMessageBox.warning(self, 'Warning', 'No captured image to save.')
            return

        # 弹出“另存为”对话框，让用户选择路径与格式。
        default_name = os.path.basename(self.last_saved_path)
        target, _ = QFileDialog.getSaveFileName(
            self,
            'Save Captured Image',
            default_name,
            'Images (*.jpg *.jpeg *.png *.bmp)'
        )
        if not target:
            return

        image = cv2.imread(self.last_saved_path)
        if image is None:
            QMessageBox.critical(self, 'Error', 'Cannot read temporary captured image.')
            return

        if not cv2.imwrite(target, image):
            QMessageBox.critical(self, 'Error', 'Save As failed.')
            return

        self.statusBar().showMessage(f'Saved as: {target}', 5000)

    def show_image(self, label: QLabel, bgr_image):
        """
        将 OpenCV 的 BGR 图像转换为 Qt 图像，并显示到 QLabel。
        """
        # OpenCV 默认是 BGR；QImage.Format_RGB888 需要 RGB。
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 缩放到标签大小，同时保持宽高比，避免画面变形。
        pix = QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    def detect_cameras(self, max_index=10):
        """
        扫描可用摄像头索引。
        通过尝试打开设备判断可用性，返回可用索引列表。
        """
        available = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    available.append(idx)
            if cap is not None:
                cap.release()
        return available

    def get_camera_names(self):
        """
        获取摄像头名称列表（尽力而为）：
        1）优先尝试 pygrabber（DirectShow，名称通常最准确）
        2）失败则返回空列表，界面回退显示 Camera N
        """
        # 优先走 pygrabber（如果环境中已安装）。
        try:
            from pygrabber.dshow_graph import FilterGraph

            graph = FilterGraph()
            devices = graph.get_input_devices()
            if isinstance(devices, list):
                return devices
        except Exception:
            pass

        # 可选回退：尝试从 PowerShell 读取系统摄像头名称。
        # 注意：这里无法稳定映射到 OpenCV 的索引，仅作为候选名称来源。
        try:
            ps_cmd = (
                "Get-PnpDevice -Class Camera | "
                "Where-Object {$_.FriendlyName -ne $null} | "
                "Select-Object -ExpandProperty FriendlyName"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if names:
                    return names
        except Exception:
            pass

        return []

    def refresh_camera_list(self):
        """
        刷新摄像头下拉框：
        - 扫描本机可用设备
        - 写入下拉框
        - 无设备时给出提示并禁用开始按钮
        """
        self.camera_selector.clear()
        cameras = self.detect_cameras(max_index=10)
        camera_names = self.get_camera_names()

        if not cameras:
            self.camera_selector.addItem('No camera found', None)
            self.start_btn.setEnabled(False)
            self.statusBar().showMessage('未检测到可用摄像头。', 4000)
            return

        for idx, cam_id in enumerate(cameras):
            if idx < len(camera_names):
                show_name = f'{camera_names[idx]} (ID: {cam_id})'
            else:
                show_name = f'Camera {cam_id}'
            self.camera_selector.addItem(show_name, cam_id)

        self.start_btn.setEnabled(True)
        self.statusBar().showMessage(f'检测到 {len(cameras)} 个摄像头。', 3000)

    def resizeEvent(self, event):
        """
        窗口尺寸变化时重绘预览，保证显示位置和清晰度一致。
        """
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.show_image(self.video_label, self.current_frame)

    def closeEvent(self, event):
        """
        关闭窗口时确保释放相机资源。
        """
        self.stop_camera()
        super().closeEvent(event)


def main():
    """
    程序入口。
    """
    app = QApplication(sys.argv)
    w = CameraApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
