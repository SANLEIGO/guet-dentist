# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'src\\image_display_widget.py'
# Bytecode version: 3.10.b1 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QGroupBox, QLabel, QSizePolicy
from PyQt5.QtCore import Qt
import vtk
import vtkmodules
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
import cv2
class ImageDisplayWidget(QWidget):
    """\n    图像显示窗口类\n    用于显示采集的图像和检测结果\n    集成VTK进行3D可视化\n    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._init_components()
        self._init_layout()
        self._init_vtk_renderer()
        self._init_2d_vtk_renderers()
    def _init_ui(self):
        """\n        初始化窗口基本属性\n        """
        self.setWindowTitle('Image Display')
        self.setMinimumSize(400, 300)
    def _init_components(self):
        """\n        初始化各个组件\n        """
        self.tab_widget = QTabWidget()
        self.tab_2d = QWidget()
        self._init_2d_tab()
        self.tab_3d = QWidget()
        self._init_3d_tab()
        self.tab_widget.addTab(self.tab_2d, '2D View')
        self.tab_widget.addTab(self.tab_3d, '3D View')
    def _init_2d_tab(self):
        """\n        初始化2D显示标签页\n        """
        layout = QVBoxLayout(self.tab_2d)
        self.original_vtk_widget = QVTKRenderWindowInteractor(self.tab_2d)
        self.original_vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_vtk_widget = QVTKRenderWindowInteractor(self.tab_2d)
        self.result_vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.original_vtk_widget, 1)
        layout.addWidget(self.result_vtk_widget, 1)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        self._init_2d_vtk_renderers()
    def _init_3d_tab(self):
        """\n        初始化3D显示标签页\n        """
        layout = QVBoxLayout(self.tab_3d)
        self.vtk_widget = QVTKRenderWindowInteractor(self.tab_3d)
        layout.addWidget(self.vtk_widget)
        layout.setContentsMargins(10, 10, 10, 10)
    def _init_layout(self):
        """\n        初始化主布局\n        """
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget, 1)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
    def _init_vtk_renderer(self):
        """\n        初始化VTK渲染器\n        """
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.Initialize()
    def _init_2d_vtk_renderers(self):
        """\n        初始化2D视图的VTK渲染器\n        """
        try:
            self.original_renderer = vtk.vtkRenderer()
            self.original_renderer.SetBackground(0.1, 0.1, 0.1)
            self.original_render_window = self.original_vtk_widget.GetRenderWindow()
            self.original_render_window.AddRenderer(self.original_renderer)
            self.original_interactor = self.original_render_window.GetInteractor()
            self.original_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.result_renderer = vtk.vtkRenderer()
            self.result_renderer.SetBackground(0.1, 0.1, 0.1)
            self.result_render_window = self.result_vtk_widget.GetRenderWindow()
            self.result_render_window.AddRenderer(self.result_renderer)
            self.result_interactor = self.result_render_window.GetInteractor()
            self.result_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.original_interactor.Initialize()
            self.result_interactor.Initialize()
            self.current_original_image = None
            self.current_result_image = None
        except Exception as e:
            print(f'初始化2D渲染器出错: {str(e)}')
            import traceback
            traceback.print_exc()
    def display_image(self, image, is_original=True):
        """\n        显示图像\n        参数:\n            image: 要显示的图像 (OpenCV BGR格式)\n            is_original: 是否是原始图像\n        """
        try:
            if is_original:
                self.current_original_image = image.copy()
                self._display_image_in_vtk(image, self.original_renderer, self.original_render_window, self.original_interactor)
            else:
                self.current_result_image = image.copy()
                self._display_image_in_vtk(image, self.result_renderer, self.result_render_window, self.result_interactor)
        except Exception as e:
            print(f'显示图像出错: {str(e)}')
            import traceback
            traceback.print_exc()
    def create_texture_from_image(self, img):
        """\n        将OpenCV图像转换为VTK纹理\n        参数:\n            img: OpenCV图像 (BGR格式)\n        返回:\n            vtkTexture: 创建的VTK纹理\n        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img_rgb.shape
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(width, height, 1)
        image_data.SetSpacing(1, 1, 1)
        image_data.SetOrigin(0, 0, 0)
        image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, channels)
        img_array = np.array(img_rgb.flatten(), dtype=np.uint8)
        scalars = vtk.vtkUnsignedCharArray()
        scalars.SetNumberOfComponents(channels)
        scalars.SetNumberOfTuples(width * height)
        for i in range(len(img_array)):
            scalars.SetValue(i, img_array[i])
        image_data.GetPointData().SetScalars(scalars)
        texture = vtk.vtkTexture()
        texture.SetInputData(image_data)
        texture.InterpolateOn()
        return texture
    def create_image_plane(self, texture, img_shape):
        """\n        创建图像平面\n        参数:\n            texture: VTK纹理\n            img_shape: 图像形状 (height, width, channels)\n        返回:\n            vtkActor: 创建的VTK演员\n        """
        height, width = img_shape[:2]
        plane_source = vtk.vtkPlaneSource()
        plane_source.SetOrigin(0, 0, 0)
        plane_source.SetPoint1(width, 0, 0)
        plane_source.SetPoint2(0, height, 0)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane_source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetTexture(texture)
        return actor
    def _display_image_in_vtk(self, numpy_image, renderer, render_window, interactor):
        # irreducible cflow, using cdg fallback
        """\n        使用VTK渲染器显示图像\n        参数:\n            numpy_image: numpy数组格式的图像 (OpenCV BGR格式)\n            renderer: VTK渲染器\n            render_window: VTK渲染窗口\n            interactor: VTK交互器\n        """
        # ***<module>.ImageDisplayWidget._display_image_in_vtk: Failure: Compilation Error
        renderer.RemoveAllViewProps()
        height, width, channels = numpy_image.shape
        print(f'重采样后的图像尺寸: {width}x{height}')
        window_width, window_height = render_window.GetSize()
        print(f'渲染窗口尺寸: {window_width}x{window_height}')
        texture = self.create_texture_from_image(numpy_image)
        actor = self.create_image_plane(texture, numpy_image.shape)
        renderer.AddActor(actor)
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput('原始图像' if renderer == self.original_renderer else '检测结果')
        text_actor.GetTextProperty().SetFontSize(16)
        text_actor.GetTextProperty().SetColor(1, 1, 1)
        text_actor.SetPosition(10, window_height - 30)
        renderer.AddActor2D(text_actor)
        renderer.ResetCamera()
        renderer.ResetCameraScreenSpace()
            render_window.Render()
            if not hasattr(interactor, '_initialized'):
                interactor.Initialize()
                interactor._initialized = True
            interactor.Render()
                except Exception as render_error:
                        print(f'渲染过程出错: {str(render_error)}')
            except Exception as e:
                print(f'VTK图像显示出错: {str(e)}')
                import traceback
                traceback.print_exc()
    def display_3d_model(self, model_data):
        """\n        显示3D模型\n        参数:\n            model_data: 3D模型数据\n        """
        return None
    def clear_display(self):
        """\n        清除显示内容\n        """
        return None
    def update_detection_result(self, result):
        """\n        更新检测结果显示\n        参数:\n            result: 检测结果数据\n        """
        return None
    def closeEvent(self, event):
        # irreducible cflow, using cdg fallback
        """\n        关闭窗口时的清理工作\n        """
        # ***<module>.ImageDisplayWidget.closeEvent: Failure: Compilation Error
        try:
            pass
        finally:
            if hasattr(self, 'interactor'):
                try:
                    self.interactor.TerminateApp()
                except:
                    pass
            if hasattr(self, 'original_interactor'):
                try:
                    self.original_interactor.TerminateApp()
                except:
                    pass
            if hasattr(self, 'result_interactor'):
                try:
                    self.result_interactor.TerminateApp()
                except:
                    pass
        except Exception as e:
            pass
        print(f'关闭窗口时出错: {str(e)}')
        event.accept()