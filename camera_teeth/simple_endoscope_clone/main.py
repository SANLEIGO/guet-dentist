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
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
        self.last_saved_path = None
        self.current_image_path = None
        self.image_paths = []
        self.current_image_index = -1
        self.caries_model = None
        self.caries_model_path = None
        self.is_detecting = False
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
        self.import_images_btn = QPushButton('Import Images...')
        self.prev_image_btn = QPushButton('Prev')
        self.next_image_btn = QPushButton('Next')
        self.import_model_btn = QPushButton('Import Model...')
        self.detect_caries_btn = QPushButton('Detect Caries')
        self.refresh_btn = QPushButton('Refresh Cameras')
        self.camera_selector = QComboBox()
        self.camera_selector.setMinimumWidth(180)

        # 初始按钮状态：
        # - 未启动相机前，Stop/Capture 禁用
        # - 未拍照前，Save As 禁用
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.save_as_btn.setEnabled(False)
        self.prev_image_btn.setEnabled(False)
        self.next_image_btn.setEnabled(False)
        self.detect_caries_btn.setEnabled(False)

        # 绑定按钮点击事件到对应处理函数。
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.capture_btn.clicked.connect(self.capture_image)
        self.save_as_btn.clicked.connect(self.save_as)
        self.import_images_btn.clicked.connect(self.import_images)
        self.prev_image_btn.clicked.connect(self.show_prev_image)
        self.next_image_btn.clicked.connect(self.show_next_image)
        self.import_model_btn.clicked.connect(self.import_caries_model)
        self.detect_caries_btn.clicked.connect(self.detect_caries)
        self.refresh_btn.clicked.connect(self.refresh_camera_list)

        # ---------------------------------------------------------------------
        # 第6部分：界面布局
        # 上方：实时预览 + 拍照结果
        # 下方：操作按钮
        # ---------------------------------------------------------------------
        controls_top = QHBoxLayout()
        controls_top.addWidget(QLabel('Camera:'))
        controls_top.addWidget(self.camera_selector)
        controls_top.addWidget(self.refresh_btn)
        controls_top.addWidget(self.start_btn)
        controls_top.addWidget(self.stop_btn)
        controls_top.addWidget(self.capture_btn)
        controls_top.addWidget(self.save_as_btn)

        controls_bottom = QHBoxLayout()
        controls_bottom.addWidget(self.import_images_btn)
        controls_bottom.addWidget(self.import_model_btn)
        controls_bottom.addWidget(self.detect_caries_btn)

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.captured_label, 1)

        image_nav = QHBoxLayout()
        image_nav.addWidget(self.prev_image_btn)
        image_nav.addWidget(self.next_image_btn)
        right_panel.addLayout(image_nav)

        right_wrapper = QWidget()
        right_wrapper.setLayout(right_panel)

        images = QHBoxLayout()
        images.addWidget(self.video_label, 2)
        images.addWidget(right_wrapper, 1)

        root = QVBoxLayout()
        root.addLayout(images, 1)
        root.addLayout(controls_top)
        root.addLayout(controls_bottom)

        wrapper = QWidget()
        wrapper.setLayout(root)
        self.setCentralWidget(wrapper)

        # 首次启动时检测可用相机，填充选择器。
        self.refresh_camera_list()
        self.update_detection_controls()

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
            self.current_image_path = out_path
            self.image_paths = []
            self.current_image_index = -1
            self.save_as_btn.setEnabled(True)
            self.update_detection_controls()
            self.statusBar().showMessage(f'Captured and saved: {out_path}', 5000)
        else:
            QMessageBox.critical(self, 'Error', 'Failed to save captured image.')

    def set_current_image(self, image_path: str):
        """
        设置当前操作图片（检测/另存为），并显示到右侧区域。
        """
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, 'Warning', '图片文件不存在。')
            return

        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.warning(self, 'Warning', f'无法读取图片：{image_path}')
            return

        self.current_image_path = image_path
        self.last_saved_path = image_path

        if self.image_paths and image_path in self.image_paths:
            self.current_image_index = self.image_paths.index(image_path)
        else:
            self.current_image_index = -1

        self.show_image(self.captured_label, image)
        self.update_detection_controls()

    def import_images(self):
        """
        统一导入入口：支持单张或批量导入牙齿图片。
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            'Import Dental Images',
            '',
            'Images (*.jpg *.jpeg *.png *.bmp)'
        )
        if not paths:
            return

        valid_paths = []
        invalid_count = 0
        for path in paths:
            if not os.path.exists(path):
                invalid_count += 1
                continue
            image = cv2.imread(path)
            if image is None:
                invalid_count += 1
                continue
            valid_paths.append(path)

        if not valid_paths:
            QMessageBox.warning(self, 'Warning', '导入失败：没有可读取的图片。')
            return

        self.image_paths = valid_paths
        self.current_image_index = 0 if len(valid_paths) > 1 else -1
        self.set_current_image(valid_paths[0])

        if len(valid_paths) == 1:
            self.statusBar().showMessage(f'已导入图片: {valid_paths[0]}', 4000)
        else:
            self.statusBar().showMessage(
                f'批量导入完成：{len(valid_paths)} 张可用，{invalid_count} 张不可读。',
                5000,
            )

    def show_prev_image(self):
        """
        在批量导入列表中显示上一张图片。
        """
        if not self.image_paths or self.current_image_index <= 0:
            return

        self.current_image_index -= 1
        self.set_current_image(self.image_paths[self.current_image_index])
        self.statusBar().showMessage(
            f'当前批量图片：{self.current_image_index + 1}/{len(self.image_paths)}',
            3000,
        )

    def show_next_image(self):
        """
        在批量导入列表中显示下一张图片。
        """
        if not self.image_paths or self.current_image_index >= len(self.image_paths) - 1:
            return

        self.current_image_index += 1
        self.set_current_image(self.image_paths[self.current_image_index])
        self.statusBar().showMessage(
            f'当前批量图片：{self.current_image_index + 1}/{len(self.image_paths)}',
            3000,
        )

    def import_caries_model(self):
        """
        手动导入 YOLO .pt 龋齿检测模型。
        """
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            'Import Caries Model',
            '',
            'YOLO Model (*.pt)'
        )
        if not model_path:
            return

        if self.load_caries_model(model_path):
            self.statusBar().showMessage(f'已加载龋齿检测模型: {model_path}', 5000)
        self.update_detection_controls()

    def load_caries_model(self, model_path: str) -> bool:
        """
        加载 YOLO .pt 模型。
        """
        if not model_path.lower().endswith('.pt'):
            QMessageBox.warning(self, 'Warning', '当前仅支持 YOLO .pt 模型。')
            return False

        if not os.path.exists(model_path):
            QMessageBox.critical(self, 'Error', '模型文件不存在。')
            return False

        try:
            from ultralytics import YOLO
        except Exception as exc:
            QMessageBox.critical(self, 'Error', f'未安装 ultralytics，请先安装：pip install ultralytics\n\n{exc}')
            return False

        try:
            self.caries_model = YOLO(model_path)
            self.caries_model_path = model_path
            return True
        except Exception as exc:
            self.caries_model = None
            self.caries_model_path = None
            QMessageBox.critical(self, 'Error', f'模型加载失败：{exc}')
            return False

    def run_caries_inference(self, image_path: str):
        """
        对单张图片执行龋齿检测，返回叠加图与检出数量。
        """
        image = cv2.imread(image_path)
        if image is None:
            return None, 0, '无法读取图片。'

        try:
            results = self.caries_model.predict(image, conf=0.10, verbose=False)
        except Exception as exc:
            return None, 0, f'龋齿检测失败：{exc}'

        overlay = image.copy()
        det_count = 0
        names = getattr(self.caries_model, 'names', {})

        if results:
            boxes = getattr(results[0], 'boxes', None)
            if boxes is not None and boxes.data is not None:
                xyxy_arr = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls

                for idx in range(len(xyxy_arr)):
                    x1, y1, x2, y2 = [int(v) for v in xyxy_arr[idx]]
                    conf = float(conf_arr[idx])
                    cls_id = int(cls_arr[idx])

                    if isinstance(names, dict):
                        label_name = names.get(cls_id, f'class_{cls_id}')
                    elif isinstance(names, list) and 0 <= cls_id < len(names):
                        label_name = names[cls_id]
                    else:
                        label_name = f'class_{cls_id}'

                    label_text = f'{label_name} {conf:.2f}'
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        overlay,
                        label_text,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    det_count += 1

        return overlay, det_count, None

    def show_single_result_dialog(self, image_path: str, overlay, det_count: int):
        """
        显示单张推理结果弹窗。
        """
        dialog = QDialog(self)
        dialog.setWindowTitle('Caries Detection Result')
        dialog.resize(900, 700)

        layout = QVBoxLayout(dialog)
        info = QLabel(f'{os.path.basename(image_path)} | 检出: {det_count}')
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(840, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(pix)
        layout.addWidget(image_label)

        buttons = QHBoxLayout()

        save_btn = QPushButton('Download Result')

        def save_single_result():
            default_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_detected.jpg"
            target, _ = QFileDialog.getSaveFileName(
                dialog,
                'Save Detection Result',
                default_name,
                'Images (*.jpg *.jpeg *.png *.bmp)'
            )
            if not target:
                return
            if not cv2.imwrite(target, overlay):
                QMessageBox.critical(dialog, 'Error', '保存推理结果失败。')
                return
            self.statusBar().showMessage(f'已保存推理结果: {target}', 5000)

        save_btn.clicked.connect(save_single_result)

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dialog.accept)

        buttons.addWidget(save_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        dialog.exec_()

    def show_batch_results_dialog(self, result_items):
        """
        显示批量推理结果弹窗（网格布局）。
        """
        dialog = QDialog(self)
        dialog.setWindowTitle('Batch Caries Detection Results')
        dialog.resize(1200, 800)

        root = QVBoxLayout(dialog)

        summary = QLabel(f'批量推理完成：{len(result_items)} 张检测到疑似龋齿')
        summary.setAlignment(Qt.AlignCenter)
        root.addWidget(summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)

        for idx, item in enumerate(result_items):
            path = item['path']
            overlay = item['overlay']
            det_count = item['det_count']

            cell = QWidget()
            cell_layout = QVBoxLayout(cell)

            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(340, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setPixmap(pix)

            text = QLabel(f"{os.path.basename(path)}\n检出: {det_count}")
            text.setAlignment(Qt.AlignCenter)

            cell_layout.addWidget(img_label)
            cell_layout.addWidget(text)

            row = idx // 3
            col = idx % 3
            grid.addWidget(cell, row, col)

        scroll.setWidget(container)
        root.addWidget(scroll)

        buttons = QHBoxLayout()

        download_all_btn = QPushButton('Download All Results')

        def download_all_results():
            target_dir = QFileDialog.getExistingDirectory(dialog, 'Select Directory to Save Results')
            if not target_dir:
                return

            success = 0
            failed = 0
            for item in result_items:
                src_name = os.path.splitext(os.path.basename(item['path']))[0]
                out_path = os.path.join(target_dir, f'{src_name}_detected.jpg')
                if cv2.imwrite(out_path, item['overlay']):
                    success += 1
                else:
                    failed += 1

            if failed == 0:
                self.statusBar().showMessage(f'已导出 {success} 张推理结果到: {target_dir}', 5000)
            else:
                QMessageBox.warning(dialog, 'Warning', f'导出完成：成功 {success} 张，失败 {failed} 张。')

        download_all_btn.clicked.connect(download_all_results)

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dialog.accept)

        buttons.addWidget(download_all_btn)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

        dialog.exec_()

    def detect_caries(self):
        """
        对当前图片或批量图片执行龋齿检测，并以弹窗展示结果。
        """
        if self.is_detecting:
            return

        if self.caries_model is None:
            QMessageBox.warning(self, 'Warning', '请先导入 YOLO .pt 模型。')
            return

        if not self.current_image_path or not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, 'Warning', '请先拍照或导入图片，再执行龋齿检测。')
            self.update_detection_controls()
            return

        self.is_detecting = True
        self.update_detection_controls()
        self.import_model_btn.setEnabled(False)

        try:
            if len(self.image_paths) > 1:
                self.statusBar().showMessage('批量龋齿检测中...', 2000)
                result_items = []

                for path in self.image_paths:
                    overlay, det_count, err = self.run_caries_inference(path)
                    if err is not None:
                        continue
                    if det_count > 0:
                        result_items.append({'path': path, 'overlay': overlay, 'det_count': det_count})

                if not result_items:
                    QMessageBox.warning(self, 'Warning', '批量图片中未检测到疑似龋齿。')
                else:
                    self.show_batch_results_dialog(result_items)
                    self.statusBar().showMessage(f'批量检测完成：{len(result_items)} 张有疑似龋齿。', 5000)
            else:
                self.statusBar().showMessage('龋齿检测中...', 2000)
                overlay, det_count, err = self.run_caries_inference(self.current_image_path)
                if err is not None:
                    QMessageBox.critical(self, 'Error', err)
                elif det_count == 0:
                    QMessageBox.warning(self, 'Warning', '未检测到疑似龋齿。')
                else:
                    self.show_single_result_dialog(self.current_image_path, overlay, det_count)
                    self.statusBar().showMessage(f'龋齿检测完成：检测到 {det_count} 处疑似目标。', 5000)
        finally:
            self.is_detecting = False
            self.import_model_btn.setEnabled(True)
            self.update_detection_controls()

    def update_detection_controls(self):
        """
        根据模型与当前图片状态更新按钮可用性。
        """
        has_model = self.caries_model is not None
        has_image = bool(self.current_image_path and os.path.exists(self.current_image_path))
        self.detect_caries_btn.setEnabled(has_model and has_image and not self.is_detecting)
        self.save_as_btn.setEnabled(has_image)

        has_batch = len(self.image_paths) > 0 and self.current_image_index >= 0
        self.prev_image_btn.setEnabled(has_batch and self.current_image_index > 0)
        self.next_image_btn.setEnabled(has_batch and self.current_image_index < len(self.image_paths) - 1)

    def save_as(self):
        """
        把当前图片保存到用户指定路径。
        """
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, 'Warning', 'No image to save.')
            return

        default_name = os.path.basename(self.current_image_path)
        target, _ = QFileDialog.getSaveFileName(
            self,
            'Save Captured Image',
            default_name,
            'Images (*.jpg *.jpeg *.png *.bmp)'
        )
        if not target:
            return

        image = cv2.imread(self.current_image_path)
        if image is None:
            QMessageBox.critical(self, 'Error', 'Cannot read current image.')
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
