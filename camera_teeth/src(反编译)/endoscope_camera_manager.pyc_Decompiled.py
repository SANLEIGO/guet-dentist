# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'src\\endoscope_camera_manager.py'
# Bytecode version: 3.10.b1 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

from PyQt5.QtCore import QObject, pyqtSignal, QThread
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
class CameraDetectionWorker(QThread):
    """\n    相机检测工作线程\n    在后台线程中检测可用的相机设备\n    """
    detection_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    def run(self):
        # irreducible cflow, using cdg fallback
        # ***<module>.CameraDetectionWorker.run: Failure: Compilation Error
        print('[调试] CameraDetectionWorker线程开始执行')
        available_cameras = []
        max_camera_check = 5
        print(f'[调试] 将检查相机ID范围: 0-{max_camera_check - 1}')
        for camera_id in range(max_camera_check):
                print(f'[调试] 尝试打开相机ID: {camera_id}')
                    if cv2.os.name == 'nt':
                        cap = cv2.VideoCapture(camera_id)
                        if not cap.isOpened():
                            cap.release()
                            cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
                    else:
                        cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
                    if cap.isOpened():
                            camera_info = {'id': camera_id, 'name': f'Camera {camera_id}', 'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 'fps': cap.get(cv2.CAP_PROP_FPS), 'is_endoscope': False}
                            try:
                                ret, frame = cap.read()
                                if ret and frame is not None and (frame.size > 0) and (len(frame.shape) >= 3) and (frame.shape[2] >= 3):
                                                    b, g, r = cv2.split(frame)
                                                    r_mean = np.mean(r)
                                                    g_mean = np.mean(g)
                                                    b_mean = np.mean(b)
                                                    if r_mean > g_mean * 1.2 and r_mean > b_mean * 1.2:
                                                            camera_info['is_endoscope'] = True
                                                            camera_info['name'] = f'Endoscope Camera {camera_id}'
                            except Exception:
                                pass
                            available_cameras.append(camera_info)
                                cap.release()
                            except Exception:
                                continue
                available_cameras.sort(key=lambda x: (not x['is_endoscope'], x['id']))
                self.detection_complete.emit(available_cameras)
                    except Exception as e:
                        self.error_occurred.emit(f'相机检测错误: {str(e)}')
class EndoscopeCameraManager(QObject):
    """\n    内窥镜相机管理器\n    负责检测和初始化电脑上的内窥镜相机\n    """
    cameras_detected = pyqtSignal(list)
    camera_connected = pyqtSignal(dict)
    camera_disconnected = pyqtSignal()
    camera_error = pyqtSignal(str)
    no_camera_available = pyqtSignal()
    frame_available = pyqtSignal(np.ndarray)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._available_cameras = []
        self._current_camera = None
        self._camera_thread = None
        self._is_connected = False
        self._camera_params = {'exposure': 0, 'gain': 1.0, 'brightness': 2, 'contrast': 7}
        self._detection_worker = None
        self._last_connected_id = None
        print('内窥镜相机管理器初始化完成')
    def detect_cameras(self):
        """\n        开始检测可用的相机设备\n        在后台线程中执行以避免UI卡顿\n        """
        print('开始检测相机...')
        if self._detection_worker and self._detection_worker.isRunning():
                self._detection_worker.quit()
                self._detection_worker.wait()
        self._detection_worker = CameraDetectionWorker()
        self._detection_worker.detection_complete.connect(self._on_detection_complete)
        self._detection_worker.error_occurred.connect(self.camera_error)
        print('相机检测线程已启动')
        self._detection_worker.start()
    def _on_detection_complete(self, cameras: List[Dict]):
        """\n        相机检测完成的回调\n        参数:\n            cameras: 可用相机列表\n        """
        print(f'[调试] 收到相机检测完成信号，检测到 {len(cameras)} 个相机')
        print(f'[调试] 相机列表内容: {cameras}')
        self._available_cameras = cameras
        print('[调试] 发送cameras_detected信号给交互控制器')
        self.cameras_detected.emit(cameras)
        print('[调试] 开始执行相机选择逻辑')
        if cameras:
            print('[调试] 检测到可用相机，进入相机选择分支')
            endoscope_cameras = [cam for cam in cameras if cam['is_endoscope']]
            print(f'[调试] 内窥镜相机数量: {len(endoscope_cameras)}')
            if endoscope_cameras:
                print(f"[调试] 选择第一个内窥镜相机，ID: {endoscope_cameras[0]['id']}")
                self.connect_camera(endoscope_cameras[0]['id'])
            else:
                print(f"[调试] 选择第一个普通相机，ID: {cameras[0]['id']}")
                self.connect_camera(cameras[0]['id'])
        else:
            print('[调试] 未检测到可用相机，进入无相机处理分支')
            self.no_camera_available.emit()
            self.camera_error.emit('未检测到可用相机，请检查相机连接')
        print('[调试] 相机选择逻辑执行完成')
    def get_available_cameras(self) -> List[Dict]:
        """\n        获取已检测到的可用相机列表\n        """
        return self._available_cameras.copy()
    def connect_camera(self, camera_id: int) -> bool:
        """\n        连接指定ID的相机\n        参数:\n            camera_id: 相机ID\n        返回:\n            bool: 连接是否成功\n        """
        print(f'[调试] 尝试连接相机ID: {camera_id}')
        try:
            if camera_id < 0:
                self.camera_error.emit('无效的相机ID')
                return False
            else:
                if self._is_connected:
                    self.disconnect_camera()
                cap = None
                if cv2.os.name == 'nt':
                    backends = [cv2.CAP_ANY, cv2.CAP_MSMF]
                    for backend in backends:
                        cap = cv2.VideoCapture(camera_id, backend)
                        if cap.isOpened():
                            break
                        else:
                            if cap:
                                cap.release()
                            cap = None
                else:
                    cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
                if not cap or not cap.isOpened():
                    raise Exception(f'无法打开相机ID {camera_id}')
                else:
                    self._apply_camera_settings(cap)
                    camera_info = None
                    for cam in self._available_cameras:
                        if cam['id'] == camera_id:
                            camera_info = cam.copy()
                            break
                    if not camera_info:
                        camera_info = {'id': camera_id, 'name': f'Camera {camera_id}', 'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 'fps': cap.get(cv2.CAP_PROP_FPS), 'is_endoscope': False}
                    self._current_camera = cap
                    self._is_connected = True
                    self._last_connected_id = camera_id
                    self.camera_connected.emit(camera_info)
                    return True
        except Exception as e:
            self.camera_error.emit(f'相机连接错误: {str(e)}')
            return False
    def disconnect_camera(self, emit_signal: bool=True):
        """\n        断开当前连接的相机\n        \n        参数:\n            emit_signal: 是否发送断开连接信号，默认为True\n        """
        if self._is_connected and self._current_camera:
                self._current_camera.release()
                self._current_camera = None
                self._is_connected = False
                if emit_signal:
                    try:
                        self.camera_disconnected.emit()
                    except RuntimeError:
                        return
    def start_stream(self):
        """\n        开始相机流\n        注意：实际的图像采集循环应该在单独的线程中运行\n        这里只提供接口，具体实现可以在调用者处完成\n        """
        if not self._is_connected:
            self.camera_error.emit('未连接相机')
            return False
        else:
            return True
    def stop_stream(self):
        """\n        停止相机流\n        """
        return None
    def capture_frame(self) -> Optional[np.ndarray]:
        """\n        捕获当前帧\n        返回:\n            np.ndarray: OpenCV格式的图像，失败返回None\n        """
        if not self._is_connected or not self._current_camera:
            return None
        else:
            try:
                ret, frame = self._current_camera.read()
                if ret:
                    return frame
            except Exception as e:
                self.camera_error.emit(f'捕获图像错误: {str(e)}')
                return None
    def set_camera_parameter(self, param_name: str, value):
        """\n        设置相机参数\n        参数:\n            param_name: 参数名 (exposure, gain, brightness, contrast)\n            value: 参数值\n        """
        if param_name in self._camera_params:
            self._camera_params[param_name] = value
            if self._is_connected and self._current_camera:
                    self._apply_camera_settings(self._current_camera)
    def get_camera_parameter(self, param_name: str):
        """\n        获取相机参数\n        """
        return self._camera_params.get(param_name)
    def _apply_camera_settings(self, cap):
        # irreducible cflow, using cdg fallback
        """\n        应用相机设置\n        """
        # ***<module>.EndoscopeCameraManager._apply_camera_settings: Failure: Compilation Error
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if self._camera_params['exposure']!= 0 else 0.75)
        if self._camera_params['exposure']!= 0:
            cap.set(cv2.CAP_PROP_EXPOSURE, self._camera_params['exposure'])
        cap.set(cv2.CAP_PROP_BRIGHTNESS, self._camera_params['brightness'])
        cap.set(cv2.CAP_PROP_CONTRAST, self._camera_params['contrast'])
            cap.set(cv2.CAP_PROP_GAIN, self._camera_params['gain'])
                return
                    except Exception as e:
                        self.camera_error.emit(f'设置相机参数错误: {str(e)}')
    def is_camera_connected(self) -> bool:
        """\n        检查相机是否已连接\n        """
        return self._is_connected
    def get_endoscope_cameras(self) -> List[Dict]:
        """\n        获取检测到的内窥镜相机列表\n        """
        return [cam for cam in self._available_cameras if cam['is_endoscope']]
    def __del__(self):
        """\n        析构函数，确保释放相机资源\n        """
        self.disconnect_camera(emit_signal=False)
        if hasattr(self, '_detection_worker') and self._detection_worker:
                try:
                    self._detection_worker.quit()
                    self._detection_worker.wait()
                except RuntimeError:
                    return