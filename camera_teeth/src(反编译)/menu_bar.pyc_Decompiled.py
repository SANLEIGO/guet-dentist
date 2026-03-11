# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'src\\menu_bar.py'
# Bytecode version: 3.10.b1 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

from PyQt5.QtWidgets import QMenuBar, QMenu, QAction, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from src.ParaSetting import ParaSettingDialog
class MenuBar(QMenuBar):
    """\n    菜单栏类\n    提供应用程序的各种菜单功能\n    """
    new_project = pyqtSignal()
    open_project = pyqtSignal()
    save_project = pyqtSignal()
    save_project_as = pyqtSignal()
    export_results = pyqtSignal()
    exit_app = pyqtSignal()
    undo_action = pyqtSignal()
    redo_action = pyqtSignal()
    copy_action = pyqtSignal()
    paste_action = pyqtSignal()
    show_acquisition = pyqtSignal(bool)
    show_display = pyqtSignal(bool)
    show_toolbar = pyqtSignal(bool)
    start_acquisition = pyqtSignal()
    stop_acquisition = pyqtSignal()
    capture_image = pyqtSignal()
    run_detection = pyqtSignal()
    configure_settings = pyqtSignal()
    about_app = pyqtSignal()
    help_contents = pyqtSignal()
    def __init__(self, parent=None, camera_manager=None, yolo_algorithm=None):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.yolo_algorithm = yolo_algorithm
        self._init_menus()
        self._init_actions()
        self._init_signals_slots()
    def _init_menus(self):
        """\n        初始化各个菜单\n        """
        self.file_menu = self.addMenu('&File')
        self.edit_menu = self.addMenu('&Edit')
        self.view_menu = self.addMenu('&View')
        self.tools_menu = self.addMenu('&Tools')
        self.help_menu = self.addMenu('&Help')
    def _init_actions(self):
        """\n        初始化各个菜单项\n        """
        self.new_project_action = QAction('&New Project', self)
        self.new_project_action.setShortcut(QKeySequence.New)
        self.open_project_action = QAction('&Open Project', self)
        self.open_project_action.setShortcut(QKeySequence.Open)
        self.save_project_action = QAction('&Save Project', self)
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_as_action = QAction('Save Project &As...', self)
        self.save_project_as_action.setShortcut(QKeySequence('Ctrl+Shift+S'))
        self.export_results_action = QAction('&Export Results...', self)
        self.export_results_action.setShortcut(QKeySequence('Ctrl+E'))
        self.file_menu.addSeparator()
        self.exit_action = QAction('E&xit', self)
        self.exit_action.setShortcut(QKeySequence('Ctrl+Q'))
        self.file_menu.addAction(self.new_project_action)
        self.file_menu.addAction(self.open_project_action)
        self.file_menu.addAction(self.save_project_action)
        self.file_menu.addAction(self.save_project_as_action)
        self.file_menu.addAction(self.export_results_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)
        self.undo_action_obj = QAction('&Undo', self)
        self.undo_action_obj.setShortcut(QKeySequence.Undo)
        self.undo_action_obj.setEnabled(False)
        self.redo_action_obj = QAction('&Redo', self)
        self.redo_action_obj.setShortcut(QKeySequence.Redo)
        self.redo_action_obj.setEnabled(False)
        self.edit_menu.addSeparator()
        self.copy_action_obj = QAction('&Copy', self)
        self.copy_action_obj.setShortcut(QKeySequence.Copy)
        self.paste_action_obj = QAction('&Paste', self)
        self.paste_action_obj.setShortcut(QKeySequence.Paste)
        self.edit_menu.addAction(self.undo_action_obj)
        self.edit_menu.addAction(self.redo_action_obj)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.copy_action_obj)
        self.edit_menu.addAction(self.paste_action_obj)
        self.show_acquisition_action = QAction('Show Acquisition Window', self)
        self.show_acquisition_action.setCheckable(True)
        self.show_acquisition_action.setChecked(True)
        self.show_display_action = QAction('Show Display Window', self)
        self.show_display_action.setCheckable(True)
        self.show_display_action.setChecked(True)
        self.show_toolbar_action = QAction('Show Toolbar', self)
        self.show_toolbar_action.setCheckable(True)
        self.show_toolbar_action.setChecked(True)
        self.view_menu.addAction(self.show_acquisition_action)
        self.view_menu.addAction(self.show_display_action)
        self.view_menu.addAction(self.show_toolbar_action)
        self.start_acquisition_action = QAction('Start Acquisition', self)
        self.stop_acquisition_action = QAction('Stop Acquisition', self)
        self.stop_acquisition_action.setEnabled(False)
        self.capture_image_action = QAction('Capture Image', self)
        self.capture_image_action.setShortcut(QKeySequence('Ctrl+Space'))
        self.run_detection_action = QAction('Run Detection', self)
        self.run_detection_action.setShortcut(QKeySequence('Ctrl+D'))
        self.tools_menu.addSeparator()
        self.settings_action = QAction('Settings...', self)
        self.tools_menu.addAction(self.start_acquisition_action)
        self.tools_menu.addAction(self.stop_acquisition_action)
        self.tools_menu.addAction(self.capture_image_action)
        self.tools_menu.addAction(self.run_detection_action)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.settings_action)
        self.help_contents_action = QAction('&Help Contents', self)
        self.help_contents_action.setShortcut(QKeySequence.HelpContents)
        self.about_action = QAction('&About', self)
        self.help_menu.addAction(self.help_contents_action)
        self.help_menu.addAction(self.about_action)
    def _init_signals_slots(self):
        """\n        初始化信号和槽连接\n        """
        self.new_project_action.triggered.connect(self._on_new_project)
        self.open_project_action.triggered.connect(self._on_open_project)
        self.save_project_action.triggered.connect(self._on_save_project)
        self.save_project_as_action.triggered.connect(self._on_save_project_as)
        self.export_results_action.triggered.connect(self._on_export_results)
        self.exit_action.triggered.connect(self._on_exit)
        self.undo_action_obj.triggered.connect(self._on_undo)
        self.redo_action_obj.triggered.connect(self._on_redo)
        self.copy_action_obj.triggered.connect(self._on_copy)
        self.paste_action_obj.triggered.connect(self._on_paste)
        self.show_acquisition_action.toggled.connect(self._on_show_acquisition_toggled)
        self.show_display_action.toggled.connect(self._on_show_display_toggled)
        self.show_toolbar_action.toggled.connect(self._on_show_toolbar_toggled)
        self.start_acquisition_action.triggered.connect(self._on_start_acquisition)
        self.stop_acquisition_action.triggered.connect(self._on_stop_acquisition)
        self.capture_image_action.triggered.connect(self._on_capture_image)
        self.run_detection_action.triggered.connect(self._on_run_detection)
        self.settings_action.triggered.connect(self._on_settings)
        self.help_contents_action.triggered.connect(self._on_help_contents)
        self.about_action.triggered.connect(self._on_about)
    def _on_new_project(self):
        self.new_project.emit()
    def _on_open_project(self):
        self.open_project.emit()
    def _on_save_project(self):
        self.save_project.emit()
    def _on_save_project_as(self):
        self.save_project_as.emit()
    def _on_export_results(self):
        self.export_results.emit()
    def _on_exit(self):
        self.exit_app.emit()
    def _on_undo(self):
        self.undo_action.emit()
    def _on_redo(self):
        self.redo_action.emit()
    def _on_copy(self):
        self.copy_action.emit()
    def _on_paste(self):
        self.paste_action.emit()
    def _on_show_acquisition_toggled(self, checked):
        self.show_acquisition.emit(checked)
    def _on_show_display_toggled(self, checked):
        self.show_display.emit(checked)
    def _on_show_toolbar_toggled(self, checked):
        self.show_toolbar.emit(checked)
    def _on_start_acquisition(self):
        self.start_acquisition.emit()
        self.start_acquisition_action.setEnabled(False)
        self.stop_acquisition_action.setEnabled(True)
    def _on_stop_acquisition(self):
        self.stop_acquisition.emit()
        self.start_acquisition_action.setEnabled(True)
        self.stop_acquisition_action.setEnabled(False)
    def _on_capture_image(self):
        self.capture_image.emit()
    def _on_run_detection(self):
        self.run_detection.emit()
    def _on_settings(self):
        main_window = self.parent()
        if main_window:
            self.camera_manager = main_window.interaction_controller._camera_manager
            self.yolo_algorithm = main_window.yolo_algorithm
            dialog = ParaSettingDialog(self, self.camera_manager, self.yolo_algorithm)
            dialog.show()
        else:
            print('无法获取主窗口引用')
    def _on_help_contents(self):
        self.help_contents.emit()
    def _on_about(self):
        self.about_app.emit()
    def update_menu_state(self, is_acquiring):
        """\n        更新菜单状态\n        参数:\n            is_acquiring: 当前是否正在采集图像\n        """
        self.start_acquisition_action.setEnabled(not is_acquiring)
        self.stop_acquisition_action.setEnabled(is_acquiring)