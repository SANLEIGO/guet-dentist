# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'src\\algorithm_interface.py'
# Bytecode version: 3.10.b1 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

from abc import ABC, abstractmethod
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import cv2
import time
import ultralytics
import ultralytics.cfg
import ultralytics.models
import ultralytics.data
import ultralytics.nn
import ultralytics.utils
from ultralytics import YOLO
import os
class DetectionResult:
    """\n    检测结果类\n    存储检测算法的输出结果\n    """
    def __init__(self):
        self.image = None
        self.result_image = None
        self.detections = []
        self.confidence = 0.0
        self.processing_time = 0.0
    def add_detection(self, x, y, width, height, confidence, label='Caries'):
        """\n        添加一个检测结果\n        参数:\n            x, y: 检测框左上角坐标\n            width, height: 检测框宽度和高度\n            confidence: 检测置信度\n            label: 标签名称\n        """
        detection = {'x': x, 'y': y, 'width': width, 'height': height, 'confidence': confidence, 'label': label}
        self.detections.append(detection)
class YOLOv11DetectionAlgorithm(QObject):
    """\n    YOLOv11检测算法实现类\n    使用YOLOv11模型进行龋齿检测\n    不再继承AlgorithmInterface，直接实现所有必要功能\n    """
    Yolo_error_occurred = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self._initialized = False
        self._model = None
        self._model_path = None
        self._parameters = {'confidence_threshold': 0.5, 'iou_threshold': 0.45, 'img_size': 640, 'device': 'cpu'}
        self._class_names = {0: 'Caries'}
    def initialize(self, model_path=None):
        """\n        初始化算法\n        参数:\n            model_path: YOLOv11模型文件路径\n        返回:\n            bool: 初始化是否成功\n        """
        try:
            if model_path:
                return self.load_model(model_path)
            else:
                return False
        except Exception as e:
            print(f'初始化算法失败: {str(e)}')
            return False
    def detect(self, image):
        """\n        执行目标检测\n        参数:\n            image: 输入图像 (numpy.ndarray)\n        返回:\n            DetectionResult: 检测结果对象\n        """
        if not self._initialized:
            print('算法未初始化')
            return None
        else:
            if self._model is None:
                print('模型未加载')
                return None
            else:
                try:
                    print('检测开始')
                    print('进度: 10%')
                    start_time = time.time()
                    print('进度: 30%')
                    processed_image = self.preprocess_image(image)
                    print('进度: 50%')
                    results = self._model(processed_image, conf=self._parameters['confidence_threshold'], iou=self._parameters['iou_threshold'], imgsz=self._parameters['img_size'], device=self._parameters['device'])
                    names_dict = self._model.names
                    print('进度: 70%')
                    result = DetectionResult()
                    result.image = image
                    result.result_image = processed_image.copy()
                    for r in results:
                        result.result_image = draw_detections(result.result_image, r, names_dict)
                    result.processing_time = (time.time() - start_time) * 1000
                    print('进度: 90%')
                    print('检测完成')
                    print('进度: 100%')
                    return result
                except Exception as e:
                    print(f'检测过程出错: {str(e)}')
                    self.Yolo_error_occurred.emit(str(e))
    def set_parameter(self, name, value):
        """\n        设置算法参数\n        参数:\n            name: 参数名称\n            value: 参数值\n        返回:\n            bool: 设置是否成功\n        """
        if name in self._parameters:
            if name == 'confidence_threshold' or name == 'iou_threshold':
                if not 0.0 <= value <= 1.0:
                        return False
            else:
                if name == 'img_size':
                    if not isinstance(value, int) or value <= 0:
                        return False
                else:
                    if name == 'device':
                        if value not in ['cpu', 'cuda']:
                            return False
            self._parameters[name] = value
            return True
        else:
            return False
    def get_parameter(self, name):
        """\n        获取算法参数\n        参数:\n            name: 参数名称\n        返回:\n            参数值或None\n        """
        return self._parameters.get(name)
    def get_available_parameters(self):
        """\n        获取所有可用参数\n        返回:\n            dict: 参数名称及其默认值\n        """
        return self._parameters.copy()
    def save_model(self, path):
        """\n        保存模型\n        参数:\n            path: 保存路径\n        返回:\n            bool: 保存是否成功\n        """
        try:
            if self._model:
                self._model.save(path)
                return True
            else:
                return False
        except Exception as e:
            print(f'保存模型失败: {str(e)}')
            return False
    def load_model(self, path):
        """\n        加载模型\n        参数:\n            path: 模型文件路径\n        返回:\n            bool: 加载是否成功\n        """
        try:
            self._model = YOLO(path)
            self._model_path = path
            self._initialized = True
            if hasattr(self._model, 'names'):
                self._class_names = self._model.names
            return True
        except Exception as e:
            print(f'加载模型失败: {str(e)}')
            return False
    def preprocess_image(self, image):
        """\n        预处理图像\n        参数:\n            image: 原始图像\n        返回:\n            numpy.ndarray: 预处理后的图像\n        """
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                return image.copy()
    def postprocess_result(self, raw_result, original_image, names_dict):
        """\n        后处理原始结果\n        参数:\n            raw_result: 算法的原始输出\n            original_image: 原始输入图像\n        返回:\n            DetectionResult: 处理后的结果对象\n        """
        result = DetectionResult()
        result.image = original_image
        result.result_image = original_image.copy()
        if len(raw_result) > 0:
            detections = raw_result[0].boxes
            for box in detections:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                label = self._class_names.get(class_id, f'Class {class_id}')
                result.add_detection(x1, y1, width, height, confidence, label)
                cv2.rectangle(result.result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                text = f'{label}: {confidence:.2f}'
                cv2.putText(result.result_image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if len(detections) > 0:
                result.confidence = float(np.mean([box.conf[0].item() for box in detections]))
                return result
            else:
                result.confidence = 0.0
        return result
    def is_initialized(self):
        """\n        检查算法是否已初始化\n        返回:\n            bool: 是否已初始化\n        """
        return self._initialized
    def get_model_path(self):
        """\n        获取当前使用的模型路径\n        返回:\n            str: 模型路径或None\n        """
        return self._model_path
    def cancel(self):
        """\n        取消正在执行的检测\n        """
        return None
def get_default_algorithm():
    """\n    获取默认的检测算法实例\n    返回:\n        YOLOv11DetectionAlgorithm: 检测算法实例\n    """
    return YOLOv11DetectionAlgorithm()
def draw_detections(frame, result, names_dict):
    """\n    在单帧上画出 YOLO11 检测结果：\n    - ht (id=0)：绿色框\n    - ct (id=1)：红色框\n    """
    if not hasattr(result, 'boxes') or result.boxes is None:
        return frame
    else:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else (-1)
            cls_name = names_dict.get(cls_id, str(cls_id))
            name_lower = str(cls_name).lower()
            if cls_id == 1 or name_lower == 'ct':
                color = (0, 0, 255)
            else:
                if cls_id == 0 or name_lower == 'ht':
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{cls_name} {conf:.2f}'
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, (-1))
            cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return frame