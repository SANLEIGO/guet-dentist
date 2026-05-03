# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'src\\tool_bar.py'
# Bytecode version: 3.10.b1 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

from PyQt5.QtWidgets import QToolBar, QAction, QComboBox, QPushButton, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QKeySequence
class ToolBar(QToolBar):
    """\n    工具栏类\n    提供常用操作的快捷按钮\n    """
    start_acquisition = pyqtSignal()
    stop_acquisition = pyqtSignal()
    capture_image = pyqtSignal()
    save_image = pyqtSignal()
    load_image = pyqtSignal()
    run_detection = pyqtSignal()
    reset_view = pyqtSignal()
    camera_changed = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__('Main Toolbar', parent)
        self._init_ui()
        self._init_actions()
        self._init_camera_selector()
        self._init_signals_slots()
    def _init_ui(self):
        """\n        初始化工具栏基本属性\n        """
        self.setMovable(True)
        self.setFloatable(True)
        self.setIconSize(QSize(24, 24))
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    def _init_actions(self):
        """\n        初始化工具栏中的各个动作\n        """
        self.start_action = QAction('Start', self)
        self.start_action.setShortcut(QKeySequence('Ctrl+S'))
        self.start_action.setToolTip('Start image acquisition (Ctrl+S)')
        self.stop_action = QAction('Stop', self)
        self.stop_action.setShortcut(QKeySequence('Ctrl+T'))
        self.stop_action.setToolTip('Stop image acquisition (Ctrl+T)')
        self.stop_action.setEnabled(False)
        self.capture_action = QAction('Capture', self)
        self.capture_action.setShortcut(QKeySequence('Ctrl+Space'))
        self.capture_action.setToolTip('Capture current image (Ctrl+Space)')
        self.load_action = QAction('Load', self)
        self.load_action.setShortcut(QKeySequence('Ctrl+O'))
        self.load_action.setToolTip('Load image from file (Ctrl+O)')
        self.save_action = QAction('Save', self)
        self.save_action.setShortcut(QKeySequence('Ctrl+Shift+S'))
        self.save_action.setToolTip('Save result to file (Ctrl+Shift+S)')
        self.detect_action = QAction('Detect', self)
        self.detect_action.setShortcut(QKeySequence('Ctrl+D'))
        self.detect_action.setToolTip('Run caries detection (Ctrl+D)')
        self.reset_view_action = QAction('Reset View', self)
        self.reset_view_action.setShortcut(QKeySequence('Ctrl+R'))
        self.reset_view_action.setToolTip('Reset view (Ctrl+R)')
        self.addAction(self.start_action)
        self.addAction(self.stop_action)
        self.addAction(self.capture_action)
        self.addSeparator()
        self.addAction(self.load_action)
        self.addAction(self.save_action)
        self.addSeparator()
        self.addAction(self.detect_action)
        self.addSeparator()
        self.addAction(self.reset_view_action)
    def _init_camera_selector(self):
        """\n        初始化相机选择器\n        """
        self.addSeparator()
        self.addWidget(QLabel(' Camera: '))
        self.camera_selector = QComboBox()
        self.camera_selector.addItem('Camera 0')
        self.camera_selector.addItem('Camera 1')
        self.camera_selector.addItem('Camera 2')
        self.camera_selector.setMinimumWidth(120)
        self.addWidget(self.camera_selector)
    def _init_signals_slots(self):
        """\n        初始化信号和槽连接\n        """
        self.start_action.triggered.connect(self._on_start_clicked)
        self.stop_action.triggered.connect(self._on_stop_clicked)
        self.capture_action.triggered.connect(self._on_capture_clicked)
        self.load_action.triggered.connect(self._on_load_clicked)
        self.save_action.triggered.connect(self._on_save_clicked)
        self.detect_action.triggered.connect(self._on_detect_clicked)
        self.reset_view_action.triggered.connect(self._on_reset_view_clicked)
        self.camera_selector.currentIndexChanged.connect(self._on_camera_changed)
    def _on_start_clicked(self):
        """\n        开始采集按钮点击处理\n        """
        print('toolbar start_acquisition')
        self.start_acquisition.emit()
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(True)
    def _on_stop_clicked(self):
        """\n        停止采集按钮点击处理\n        """
        self.stop_acquisition.emit()
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
    def _on_capture_clicked(self):
        """\n        捕获图像按钮点击处理\n        """
        self.capture_image.emit()
    def _on_load_clicked(self):
        """\n        加载图像按钮点击处理\n        """
        self.load_image.emit()
    def _on_save_clicked(self):
        """\n        保存结果按钮点击处理\n        """
        self.save_image.emit()
    def _on_detect_clicked(self):
        """\n        运行检测按钮点击处理\n        """
        self.run_detection.emit()
    def _on_reset_view_clicked(self):
        """\n        重置视图按钮点击处理\n        """
        self.reset_view.emit()
    def _on_camera_changed(self, index):
        """\n        相机选择变化处理\n        """
        self.camera_changed.emit(index)
    def update_toolbar_state(self, is_acquiring):
        """\n        更新工具栏状态\n        参数:\n            is_acquiring: 当前是否正在采集图像\n        """
        self.start_action.setEnabled(not is_acquiring)
        self.stop_action.setEnabled(is_acquiring)