# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'src\\main_window.py'
# Bytecode version: 3.10.b1 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from src.menu_bar import MenuBar
from src.tool_bar import ToolBar
from src.image_acquisition_widget import ImageAcquisitionWidget
from src.image_display_widget import ImageDisplayWidget
from src.interaction_controller import InteractionController
from src.algorithm_interface import YOLOv11DetectionAlgorithm
import os
class MainWindow(QMainWindow):
    """\n    主窗口类，作为整个应用程序的主界面\n    包含菜单栏、工具栏和两个子窗口\n    """
    Yolo_detection_completed = pyqtSignal(object)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._init_components()
        self._init_layout()
        self._init_signals_slots()
    def _init_ui(self):
        """\n        初始化窗口基本属性\n        """
        self.setWindowTitle('DentalCam Caries Detection')
        self.resize(1200, 800)
        self.setMinimumSize(800, 600)
    def _init_components(self):
        """\n        初始化各个组件\n        """
        self.menu_bar = MenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.tool_bar = ToolBar(self)
        self.addToolBar(self.tool_bar)
        self.acquisition_widget = ImageAcquisitionWidget(self)
        self.display_widget = ImageDisplayWidget(self)
        self.interaction_controller = InteractionController(self)
        self.yolo_algorithm = YOLOv11DetectionAlgorithm()
        model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        if os.path.exists(model_path):
            print(f'正在初始化YOLOv11模型: {model_path}')
            success = self.yolo_algorithm.initialize(model_path)
            if success:
                print('YOLOv11模型初始化成功')
            else:
                print('YOLOv11模型初始化失败')
        else:
            print(f'模型文件不存在: {model_path}')
    def _init_layout(self):
        """\n        初始化布局\n        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(self.acquisition_widget, 1)
        main_layout.addWidget(self.display_widget, 1)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
    def _init_signals_slots(self):
        """\n        初始化信号和槽连接\n        """
        self.Yolo_detection_completed.connect(self._on_yolo_detection_completed)
    def get_acquisition_widget(self):
        """\n        获取图像采集窗口\n        """
        return self.acquisition_widget
    def get_display_widget(self):
        """\n        获取图像显示窗口\n        """
        return self.display_widget
    def get_interaction_controller(self):
        """\n        获取交互控制器\n        """
        return self.interaction_controller
    def get_yolo_algorithm(self):
        """\n        获取YOLOv11检测算法实例\n        """
        return self.yolo_algorithm
    def _on_yolo_detection_completed(self, detection_result):
        """\n        处理YOLO检测完成信号\n        参数:\n            detection_result: 检测结果对象，包含.result_image属性\n        """
        if detection_result and hasattr(detection_result, 'result_image'):
            self.display_widget.display_image(detection_result.result_image, is_original=False)
        else:
            print('检测结果格式不正确，缺少result_image属性')