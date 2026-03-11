# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'src\\ParaSetting.py'
# Bytecode version: 3.10.b1 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

from PyQt5.QtWidgets import QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QComboBox
from PyQt5.QtCore import Qt, pyqtSignal
class CameraParaPanel(QWidget):
    """相机参数设置面板"""
    exposure_changed = pyqtSignal(int)
    brightness_changed = pyqtSignal(int)
    contrast_changed = pyqtSignal(int)
    gain_changed = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.exposure_layout = QHBoxLayout()
        self.exposure_label = QLabel('Exposure: 0')
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(0, 100)
        self.exposure_slider.setValue(50)
        self.exposure_slider.valueChanged.connect(self.on_exposure_changed)
        self.exposure_layout.addWidget(self.exposure_label)
        self.exposure_layout.addWidget(self.exposure_slider)
        self.brightness_layout = QHBoxLayout()
        self.brightness_label = QLabel('Brightness: 50')
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(50)
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        self.brightness_layout.addWidget(self.brightness_label)
        self.brightness_layout.addWidget(self.brightness_slider)
        self.contrast_layout = QHBoxLayout()
        self.contrast_label = QLabel('Contrast: 50')
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 100)
        self.contrast_slider.setValue(50)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        self.contrast_layout.addWidget(self.contrast_label)
        self.contrast_layout.addWidget(self.contrast_slider)
        self.gain_layout = QHBoxLayout()
        self.gain_label = QLabel('Gain: 50')
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 100)
        self.gain_slider.setValue(50)
        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        self.gain_layout.addWidget(self.gain_label)
        self.gain_layout.addWidget(self.gain_slider)
        self.button_layout = QHBoxLayout()
        self.apply_button = QPushButton('CamApply')
        self.cancel_button = QPushButton('CamCancel')
        self.button_layout.addWidget(self.apply_button)
        self.button_layout.addWidget(self.cancel_button)
        layout.addLayout(self.exposure_layout)
        layout.addLayout(self.brightness_layout)
        layout.addLayout(self.contrast_layout)
        layout.addLayout(self.gain_layout)
        layout.addLayout(self.button_layout)
        self.initial_values = {'exposure': 0, 'brightness': 2, 'contrast': 7, 'gain': 1}
        self.current_values = self.initial_values.copy()
    def on_exposure_changed(self, value):
        """曝光值变化处理"""
        self.exposure_label.setText(f'Exposure: {value}')
        self.exposure_changed.emit(value)
        self.current_values['exposure'] = value
    def on_brightness_changed(self, value):
        """亮度值变化处理"""
        self.brightness_label.setText(f'Brightness: {value}')
        self.brightness_changed.emit(value)
        self.current_values['brightness'] = value
    def on_contrast_changed(self, value):
        """对比度值变化处理"""
        self.contrast_label.setText(f'Contrast: {value}')
        self.contrast_changed.emit(value)
        self.current_values['contrast'] = value
    def on_gain_changed(self, value):
        """增益值变化处理"""
        self.gain_label.setText(f'Gain: {value}')
        self.gain_changed.emit(value)
        self.current_values['gain'] = value
    def reset_to_initial(self):
        """重置到初始值"""
        self.exposure_slider.setValue(self.initial_values['exposure'])
        self.brightness_slider.setValue(self.initial_values['brightness'])
        self.contrast_slider.setValue(self.initial_values['contrast'])
        self.gain_slider.setValue(self.initial_values['gain'])
        self.exposure_label.setText(f"Exposure: {self.initial_values['exposure']}")
        self.brightness_label.setText(f"Brightness: {self.initial_values['brightness']}")
        self.contrast_label.setText(f"Contrast: {self.initial_values['contrast']}")
        self.gain_label.setText(f"Gain: {self.initial_values['gain']}")
        self.current_values = self.initial_values.copy()
class YoloParaPanel(QWidget):
    """Yolo参数设置面板"""
    confidence_threshold_changed = pyqtSignal(float)
    iou_threshold_changed = pyqtSignal(float)
    device_changed = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.confidence_layout = QHBoxLayout()
        self.confidence_label = QLabel('Confidence Threshold: 0.5')
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        self.confidence_layout.addWidget(self.confidence_label)
        self.confidence_layout.addWidget(self.confidence_slider)
        self.iou_layout = QHBoxLayout()
        self.iou_label = QLabel('IOU Threshold: 0.5')
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(50)
        self.iou_slider.valueChanged.connect(self.on_iou_changed)
        self.iou_layout.addWidget(self.iou_label)
        self.iou_layout.addWidget(self.iou_slider)
        self.device_layout = QHBoxLayout()
        self.device_label = QLabel('Device:')
        self.device_combo = QComboBox()
        self.device_combo.addItems(['CPU', 'GPU'])
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        self.device_layout.addWidget(self.device_label)
        self.device_layout.addWidget(self.device_combo)
        self.button_layout = QHBoxLayout()
        self.apply_button = QPushButton('YoloApply')
        self.cancel_button = QPushButton('YoloCancel')
        self.button_layout.addWidget(self.apply_button)
        self.button_layout.addWidget(self.cancel_button)
        layout.addLayout(self.confidence_layout)
        layout.addLayout(self.iou_layout)
        layout.addLayout(self.device_layout)
        layout.addLayout(self.button_layout)
        self.initial_values = {'confidence_threshold': 0.5, 'iou_threshold': 0.45, 'device': 'CPU'}
        self.current_values = self.initial_values.copy()
    def on_confidence_changed(self, value):
        """置信度阈值变化处理"""
        threshold = value / 100.0
        self.confidence_label.setText(f'Confidence Threshold: {threshold:.2f}')
        self.confidence_threshold_changed.emit(threshold)
        self.current_values['confidence_threshold'] = value
    def on_iou_changed(self, value):
        """IOU阈值变化处理"""
        threshold = value / 100.0
        self.iou_label.setText(f'IOU Threshold: {threshold:.2f}')
        self.iou_threshold_changed.emit(threshold)
        self.current_values['iou_threshold'] = value
    def on_device_changed(self, device):
        """设备变化处理"""
        self.device_changed.emit(device)
        self.current_values['device'] = device
    def reset_to_initial(self):
        """重置到初始值"""
        self.confidence_slider.setValue(self.initial_values['confidence_threshold'])
        self.iou_slider.setValue(self.initial_values['iou_threshold'])
        self.device_combo.setCurrentText(self.initial_values['device'])
        confidence_threshold = self.initial_values['confidence_threshold'] / 100.0
        iou_threshold = self.initial_values['iou_threshold'] / 100.0
        self.confidence_label.setText(f'Confidence Threshold: {confidence_threshold:.2f}')
        self.iou_label.setText(f'IOU Threshold: {iou_threshold:.2f}')
        self.current_values = self.initial_values.copy()
class ParaSettingDialog(QDialog):
    """\n    参数设置对话框主类\n    包含CameraPara和YoloPara两个面板\n    """
    def __init__(self, parent=None, camera_manager=None, yolo_algorithm=None):
        super().__init__(parent)
        self.setWindowTitle('参数设置')
        self.resize(600, 400)
        self.setWindowModality(Qt.NonModal)
        self.camera_manager = camera_manager
        self.yolo_algorithm = yolo_algorithm
        print('Camera Manager参数信息:')
        print(f'  类型: {type(camera_manager).__name__}')
        if camera_manager:
            print(f"  相机参数: {camera_manager.get_camera_parameter('exposure')}, {camera_manager.get_camera_parameter('brightness')}, {camera_manager.get_camera_parameter('contrast')}, {camera_manager.get_camera_parameter('gain')}")
        else:
            print('  camera_manager为None')
        print('Yolo Algorithm参数信息:')
        print(f"  类型: {(type(yolo_algorithm).__name__ if yolo_algorithm else 'None')}")
        if yolo_algorithm:
            print(f"  Yolo参数: 置信度阈值={yolo_algorithm.get_parameter('confidence_threshold')}, IOU阈值={yolo_algorithm.get_parameter('iou_threshold')}, 设备={yolo_algorithm.get_parameter('device')}")
        else:
            print('  yolo_algorithm为None')
        self.init_ui()
        self.initialize_parameters()
        self.connect_signals()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        self.camera_para_panel = CameraParaPanel(self)
        self.tab_widget.addTab(self.camera_para_panel, '相机参数')
        self.yolo_para_panel = YoloParaPanel(self)
        self.tab_widget.addTab(self.yolo_para_panel, 'YOLO参数')
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    def initialize_parameters(self):
        """\n        从相机管理器和Yolo算法初始化参数面板的值\n        """
        if self.camera_manager:
            exposure = self.camera_manager.get_camera_parameter('exposure') or 50
            brightness = self.camera_manager.get_camera_parameter('brightness') or 50
            contrast = self.camera_manager.get_camera_parameter('contrast') or 50
            gain = self.camera_manager.get_camera_parameter('gain') or 50
            self.camera_para_panel.exposure_slider.setValue(int(exposure))
            self.camera_para_panel.brightness_slider.setValue(int(brightness))
            self.camera_para_panel.contrast_slider.setValue(int(contrast))
            self.camera_para_panel.gain_slider.setValue(int(gain))
            self.camera_para_panel.initial_values = {'exposure': exposure, 'brightness': brightness, 'contrast': contrast, 'gain': gain}
            self.camera_para_panel.current_values = self.camera_para_panel.initial_values.copy()
        if self.yolo_algorithm:
            confidence_threshold = self.yolo_algorithm.get_parameter('confidence_threshold') or 0.5
            iou_threshold = self.yolo_algorithm.get_parameter('iou_threshold') or 0.5
            device = self.yolo_algorithm.get_parameter('device') or 'cpu'
            self.yolo_para_panel.confidence_slider.setValue(int(confidence_threshold * 100))
            self.yolo_para_panel.iou_slider.setValue(int(iou_threshold * 100))
            self.yolo_para_panel.device_combo.setCurrentText(device.upper())
            self.yolo_para_panel.initial_values = {'confidence_threshold': int(confidence_threshold * 100), 'iou_threshold': int(iou_threshold * 100), 'device': device.upper()}
            self.yolo_para_panel.current_values = self.yolo_para_panel.initial_values.copy()
    def connect_signals(self):
        """\n        连接参数变化信号到相应的设置函数\n        """
        if self.camera_manager:
            self.camera_para_panel.exposure_changed.connect(lambda value: self.camera_manager.set_camera_parameter('exposure', value))
            self.camera_para_panel.brightness_changed.connect(lambda value: self.camera_manager.set_camera_parameter('brightness', value))
            self.camera_para_panel.contrast_changed.connect(lambda value: self.camera_manager.set_camera_parameter('contrast', value))
            self.camera_para_panel.gain_changed.connect(lambda value: self.camera_manager.set_camera_parameter('gain', value))
        if self.yolo_algorithm:
            self.yolo_para_panel.confidence_threshold_changed.connect(lambda value: self.yolo_algorithm.set_parameter('confidence_threshold', value))
            self.yolo_para_panel.iou_threshold_changed.connect(lambda value: self.yolo_algorithm.set_parameter('iou_threshold', value))
            self.yolo_para_panel.device_changed.connect(lambda value: self.yolo_algorithm.set_parameter('device', value.lower()))
        self.camera_para_panel.apply_button.clicked.connect(self.on_camera_apply_clicked)
        self.camera_para_panel.cancel_button.clicked.connect(self.on_camera_cancel_clicked)
        self.yolo_para_panel.apply_button.clicked.connect(self.on_yolo_apply_clicked)
        self.yolo_para_panel.cancel_button.clicked.connect(self.on_yolo_cancel_clicked)
    def on_camera_apply_clicked(self):
        """\n        相机参数应用按钮点击处理\n        """
        print('相机参数已应用')
    def on_camera_cancel_clicked(self):
        """\n        相机参数取消按钮点击处理\n        """
        self.camera_para_panel.reset_to_initial()
        print('相机参数已重置')
    def on_yolo_apply_clicked(self):
        """\n        Yolo参数应用按钮点击处理\n        执行工具栏的Detect命令\n        """
        print('Yolo参数已应用，执行Detect命令')
        menu_bar = self.parent()
        if menu_bar:
            main_window = menu_bar.parent()
            if main_window:
                if hasattr(main_window, 'tool_bar') and hasattr(main_window.tool_bar, 'detect_action'):
                    main_window.tool_bar.detect_action.trigger()
                    print('Detect命令已触发')
                else:
                    print('无法访问工具栏的Detect命令')
                    if hasattr(menu_bar, 'run_detection'):
                        menu_bar.run_detection.emit()
                        print('使用备选方案：直接触发MenuBar中的run_detection信号')
            else:
                print('无法获取主窗口引用（通过MenuBar的parent()）')
        else:
            print('无法获取MenuBar引用')
    def on_yolo_cancel_clicked(self):
        """\n        Yolo参数取消按钮点击处理\n        """
        self.yolo_para_panel.reset_to_initial()
        print('Yolo参数已重置')