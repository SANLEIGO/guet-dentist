# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'src\\interaction_controller.py'
# Bytecode version: 3.10.b1 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import cv2
try:
    from .endoscope_camera_manager import EndoscopeCameraManager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.endoscope_camera_manager import EndoscopeCameraManager
class InteractionController(QObject):
    """\n    交互控制器类\n    负责协调各个组件之间的交互\n    作为系统的中央协调器\n    """
    acquisition_started = pyqtSignal()
    acquisition_stopped = pyqtSignal()
    image_captured = pyqtSignal(np.ndarray)
    detection_started = pyqtSignal()
    detection_completed = pyqtSignal(object)
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._is_acquiring = False
        self._current_image = None
        self._detection_result = None
        print('初始化内窥镜相机管理器')
        self._camera_manager = EndoscopeCameraManager(self)
        self._camera_manager.cameras_detected.connect(self._on_cameras_detected)
        self._camera_manager.camera_error.connect(self._on_camera_error)
        self._camera_manager.no_camera_available.connect(self._on_no_camera_available)
        print('交互控制器初始化完成')
        self._init_connections()
        self._camera_manager.detect_cameras()
    def _on_cameras_detected(self, cameras):
        """\n        相机检测完成的回调\n        参数:\n            cameras: 检测到的相机列表\n        """
        print('[调试] 交互控制器收到cameras_detected信号')
        print(f'[调试] 检测到的相机数量: {len(cameras)}')
        if self.main_window is not None and hasattr(self.main_window, 'tool_bar'):
                toolbar = self.main_window.tool_bar
                if hasattr(toolbar, 'camera_selector'):
                    toolbar.camera_selector.clear()
                    print('检测到的相机列表:')
                    for i, camera in enumerate(cameras):
                        print(f'相机 {i + 1}:')
                        print(f"  ID: {camera['id']}")
                        print(f"  名称: {camera['name']}")
                        print(f"  类型: {('内窥镜' if camera['is_endoscope'] else '普通相机')}")
                        for key, value in camera.items():
                            if key not in ['id', 'name', 'is_endoscope']:
                                print(f'  {key}: {value}')
                    print('-----------------------------------')
                    for camera in cameras:
                        camera_name = camera['name']
                        if camera['is_endoscope']:
                            camera_name += ' (内窥镜)'
                        toolbar.camera_selector.addItem(camera_name, camera['id'])
    def _init_connections(self):
        """\n        初始化各个组件之间的信号连接\n        """
        if self.main_window is not None:
            toolbar = self.main_window.tool_bar
            toolbar.start_acquisition.connect(self.start_acquisition)
            toolbar.stop_acquisition.connect(self.stop_acquisition)
            toolbar.capture_image.connect(self.capture_image)
            toolbar.run_detection.connect(self.run_detection)
            toolbar.reset_view.connect(self.reset_view)
            toolbar.camera_changed.connect(self.change_camera)
            toolbar.load_image.connect(self.load_image)
            menu_bar = self.main_window.menu_bar
            menu_bar.start_acquisition.connect(self.start_acquisition)
            menu_bar.stop_acquisition.connect(self.stop_acquisition)
            menu_bar.capture_image.connect(self.capture_image)
            menu_bar.run_detection.connect(self.run_detection)
            menu_bar.show_acquisition.connect(self.toggle_acquisition_visibility)
            menu_bar.show_display.connect(self.toggle_display_visibility)
            menu_bar.show_toolbar.connect(self.toggle_toolbar_visibility)
            menu_bar.exit_app.connect(self.exit_application)
            acquisition_widget = self.main_window.acquisition_widget
            acquisition_widget.image_acquired.connect(self.on_image_acquired)
            acquisition_widget.start_button.clicked.connect(self.start_acquisition)
            acquisition_widget.stop_button.clicked.connect(self.stop_acquisition)
            acquisition_widget.capture_button.clicked.connect(self.capture_image)
    @pyqtSlot()
    def start_acquisition(self):
        """\n        开始图像采集\n        """
        print('交互控制器 start_acquisition')
        if not self._is_acquiring:
            self._is_acquiring = True
            self._update_ui_state()
            if self.main_window is not None:
                acquisition_widget = self.main_window.acquisition_widget
                acquisition_widget.set_camera_manager(self._camera_manager)
                acquisition_widget.start_acquisition()
            self.acquisition_started.emit()
    @pyqtSlot()
    def stop_acquisition(self):
        """\n        停止图像采集\n        """
        if self._is_acquiring:
            self._is_acquiring = False
            self._update_ui_state()
            if self.main_window is not None:
                self.main_window.acquisition_widget.stop_acquisition()
            self.acquisition_stopped.emit()
    @pyqtSlot()
    def capture_image(self):
        """\n        捕获当前图像\n        """
        if self.main_window is not None:
            self.main_window.acquisition_widget.capture_image()
    @pyqtSlot(np.ndarray)
    def on_image_acquired(self, image):
        """\n        当新图像被采集到时的处理\n        参数:\n            image: 采集到的图像\n        """
        print('交互控制器 on_image_acquired')
        self._current_image = image.copy()
        if self.main_window is not None:
            self.main_window.display_widget.display_image(image, is_original=True)
        self.image_captured.emit(image)
    @pyqtSlot()
    def run_detection(self):
        """\n        运行龋病检测算法\n        使用MainWindow的yolo_algorithm检测_current_image\n        检测完成后触发Yolo_detection_completed信号\n        """
        # ***<module>.InteractionController.run_detection: Failure: Different control flow
        if self._current_image is not None:
            self.detection_started.emit()
        else:
            print('没有可用的图像进行检测，请先捕获图像')
        try:
            if self.main_window is not None and hasattr(self.main_window, 'get_yolo_algorithm'):
                yolo_algorithm = self.main_window.get_yolo_algorithm()
                if yolo_algorithm is not None and hasattr(yolo_algorithm, 'detect'):
                    print('开始运行YOLOv11检测算法...')
                    detection_result = yolo_algorithm.detect(self._current_image)
                    self._detection_result = detection_result
                    self.detection_completed.emit(detection_result)
                    if hasattr(self.main_window, 'Yolo_detection_completed'):
                        print('触发Yolo_detection_completed信号...')
                        self.main_window.Yolo_detection_completed.emit(detection_result)
                    print('检测完成！')
                else:
                    print('YOLO检测算法未正确初始化或缺少detect方法')
            else:
                print('MainWindow未初始化或缺少get_yolo_algorithm方法')
        except Exception as e:
            print(f'运行检测算法时出错: {str(e)}')
            import traceback
            traceback.print_exc()
    @pyqtSlot()
    def reset_view(self):
        """\n        重置视图\n        """
        if self.main_window is not None:
            pass
    @pyqtSlot(int)
    def change_camera(self, camera_index):
        """\n        切换相机\n        参数:\n            camera_index: 相机索引\n        """
        if camera_index < 0:
            self._show_error_message('无效的相机索引，请选择有效的相机')
            return None
        else:
            success = self._camera_manager.connect_camera(camera_index)
            if success and self.main_window is not None:
                    pass
    def _show_error_message(self, message):
        """\n        显示错误消息\n        参数:\n            message: 错误消息内容\n        """
        print(f'错误: {message}')
    def _on_cameras_detected(self, cameras):
        """\n        相机检测完成的回调\n        参数:\n            cameras: 检测到的相机列表\n        """
        if self.main_window is not None and hasattr(self.main_window, 'tool_bar'):
                toolbar = self.main_window.tool_bar
                if hasattr(toolbar, 'camera_selector'):
                    toolbar.camera_selector.clear()
                    for camera in cameras:
                        camera_name = camera['name']
                        if camera['is_endoscope']:
                            camera_name += ' (内窥镜)'
                        toolbar.camera_selector.addItem(camera_name, camera['id'])
    def _on_camera_error(self, error_message):
        """\n        相机错误回调\n        参数:\n            error_message: 错误信息\n        """
        print(f'相机错误: {error_message}')
        self._show_error_message(error_message)
    def _on_no_camera_available(self):
        """\n        处理没有可用相机的情况\n        """
        print('===================================')
        print('未检测到可用相机')
        print('相机检测状态: 完成')
        print('可用相机数量: 0')
        print('请检查:')
        print('1. 相机是否正确连接')
        print('2. 相机驱动是否已安装')
        print('3. 相机是否被其他应用占用')
        print('===================================')
        self._show_error_message('未检测到可用相机，请检查相机连接')
    def refresh_cameras(self):
        """\n        刷新相机列表\n        """
        self._camera_manager.detect_cameras()
    def get_camera_manager(self):
        """\n        获取相机管理器实例\n        """
        return self._camera_manager
    @pyqtSlot(bool)
    def toggle_acquisition_visibility(self, visible):
        """\n        切换图像采集窗口的可见性\n        参数:\n            visible: 是否可见\n        """
        if self.main_window is not None:
            self.main_window.acquisition_widget.setVisible(visible)
    @pyqtSlot(bool)
    def toggle_display_visibility(self, visible):
        """\n        切换图像显示窗口的可见性\n        参数:\n            visible: 是否可见\n        """
        if self.main_window is not None:
            self.main_window.display_widget.setVisible(visible)
    @pyqtSlot(bool)
    def toggle_toolbar_visibility(self, visible):
        """\n        切换工具栏的可见性\n        参数:\n            visible: 是否可见\n        """
        if self.main_window is not None:
            self.main_window.toolBar.setVisible(visible)
    @pyqtSlot()
    def exit_application(self):
        """\n        退出应用程序\n        """
        if self._is_acquiring:
            self.stop_acquisition()
        if self.main_window is not None:
            self.main_window.close()
    def _update_ui_state(self):
        """\n        更新UI组件的状态\n        """
        if self.main_window is not None:
            self.main_window.tool_bar.update_toolbar_state(self._is_acquiring)
            self.main_window.menu_bar.update_menu_state(self._is_acquiring)
    def get_current_image(self):
        """\n        获取当前图像\n        返回:\n            当前图像或None\n        """
        return self._current_image
    def get_detection_result(self):
        """\n        获取检测结果\n        返回:\n            检测结果或None\n        """
        return self._detection_result
    def is_acquiring(self):
        """\n        获取当前是否正在采集图像\n        返回:\n            是否正在采集\n        """
        return self._is_acquiring
    @pyqtSlot()
    def load_image(self):
        # irreducible cflow, using cdg fallback
        """\n        加载图像文件\n        打开文件选择对话框，选择并加载图像文件，然后将图像显示在界面上\n        """
        # ***<module>.InteractionController.load_image: Failure: Compilation Error
        supported_formats = 'Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif);;Yolo (*.pt)'
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, 'Open Image File', '', supported_formats)
        if file_path:
            file_ext = file_path.lower().split('.')[(-1)]
            if file_ext == 'pt':
                pass
            else:
                image = cv2.imread(file_path)
                if image is not None:
                    print(f'成功加载图像: {file_path}')
                    print(f'图像尺寸: {image.shape[1]}x{image.shape[0]}')
                    self._current_image = image.copy()
                    if self.main_window is not None:
                        self.main_window.display_widget.display_image(image, is_original=True)
                    self.image_captured.emit(image)
                else:
                    self._show_error_message(f'无法读取图像文件: {file_path}')
            if self.main_window is not None and hasattr(self.main_window, 'yolo_algorithm'):
                self.main_window.yolo_algorithm.load_model(file_path)
                print(f'成功加载YOLO模型: {file_path}')
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self.main_window, '成功', f'YOLO模型加载成功: {file_path}')
                self._show_error_message('主窗口中没有yolo_algorithm实例')
                    except Exception as e:
                            print(f'加载YOLO模型时出错: {str(e)}')
                            self._show_error_message(f'加载YOLO模型失败: {str(e)}')
            except Exception as e:
                print(f'加载图像时出错: {str(e)}')
                self._show_error_message(f'加载图像失败: {str(e)}')