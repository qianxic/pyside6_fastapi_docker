import os
import json
import base64
import requests
import traceback
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QMessageBox, QInputDialog, QFileDialog
from PySide6.QtCore import Qt, QObject, Signal, QThread, QMetaObject, Q_ARG, QPoint
import threading
import time
import logging

# 导入ZoomableLabel类
from zhuyaogongneng_docker.display import ZoomableLabel
# 创建线程间通信的信号桥
class ThreadCommunicator(QObject):
    """用于线程间通信的对象"""
    display_result_signal = Signal(object)

# 全局信号桥实例
thread_communicator = ThreadCommunicator()

# 导入检测客户端
from zhuyaogongneng_docker.function.detection_client import detect_changes, check_connection

class ExecuteChangeDetectionTask:
    def __init__(self, navigation_functions):
        """初始化变化检测任务
        
        Args:
            navigation_functions: 导航功能类实例
        """
        self.navigation_functions = navigation_functions
        
        # 结果图像
        self.result_image_path = None
        self.result_image = None
        
        # 保存用户选择的任务文件夹（可选）
        self.task_directory = None
        
        # 添加处理模式属性
        self.processing_mode = "image"  # 默认为图像模式
        
        # 使用Qt.QueuedConnection确保信号在主线程中处理
        thread_communicator.display_result_signal.connect(self._display_result, Qt.QueuedConnection)
    
    def set_processing_mode(self, mode):
        """设置处理模式
        
        Args:
            mode: 处理模式。
                 注意：此模块现在只支持图像模式处理
        """
        # 兼容性保留此方法，但不做实际模式切换
        self.processing_mode = "image"  # 始终使用图像模式
        self.log_message(f"使用图像处理模式", True)
    
    def log_message(self, message, level_or_show_in_ui=True):
        """记录消息到日志
        
        Args:
            message: 消息内容
            level_or_show_in_ui: 日志级别字符串("START"、"COMPLETE"、"ERROR")或布尔值
        """
        # 确定显示在UI中的标志和日志级别
        show_in_ui = True
        level = None
        
        # 处理参数
        if isinstance(level_or_show_in_ui, str):
            # 如果是字符串，视为日志级别
            level = level_or_show_in_ui
        else:
            # 如果是布尔值，确定是否显示在UI中
            show_in_ui = level_or_show_in_ui
            # 根据消息内容确定默认级别
            level = "ERROR" if "错误" in message or "失败" in message else "COMPLETE"
        
        # 如果需要在UI中显示，且navigation_functions有log_message方法
        if show_in_ui and hasattr(self.navigation_functions, 'log_message'):
            self.navigation_functions.log_message(message, level)
    
    def on_begin_clicked(self):
        # 检查前后时相影像是否已导入
        if not hasattr(self.navigation_functions, 'file_path') or not self.navigation_functions.file_path:
            self.log_message("错误: 未导入前时相影像", True)
            return
        
        if not hasattr(self.navigation_functions, 'file_path_after') or not self.navigation_functions.file_path_after:
            self.log_message("错误: 未导入后时相影像", True)
            return
            
        # 清除之前的结果图像，确保新结果能正确显示
        self._clear_previous_result()
            
        # 让用户选择保存文件夹
        default_dir = os.path.dirname(self.navigation_functions.file_path_after)
        output_folder = QFileDialog.getExistingDirectory(None, "选择保存文件夹", default_dir)
        
        if not output_folder:
            return
            
        # 转换所有路径为绝对路径
        before_path = os.path.abspath(self.navigation_functions.file_path).replace("\\", "/")
        after_path = os.path.abspath(self.navigation_functions.file_path_after).replace("\\", "/")
        output_path = os.path.abspath(output_folder).replace("\\", "/") 
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 记录当前处理模式
        self.log_message(f"使用{self.processing_mode}模式进行变化检测", "START")
        
        # 创建子线程执行变化检测
        detection_thread = threading.Thread(
            target=self._execute_detection,
            args=(before_path, after_path, output_path, self.processing_mode)
        )
        detection_thread.daemon = True
        detection_thread.start()
    
    def _execute_detection(self, before_path, after_path, output_path, mode="image"):
        """在子线程中执行变化检测
        
        Args:
            before_path: 前时相影像路径
            after_path: 后时相影像路径
            output_path: 输出路径
        """
        try:
            # 确保所有路径使用正斜杠并转为绝对路径
            before_path = os.path.abspath(before_path).replace("\\", "/")
            after_path = os.path.abspath(after_path).replace("\\", "/")
            output_path = os.path.abspath(output_path).replace("\\", "/")
            
            # 确保输出目录存在
            os.makedirs(output_path, exist_ok=True)
            
            self.log_message("开始执行图像变化检测，请稍后...", "START")

            try:
                task_result = detect_changes(
                    before_path=before_path,
                    after_path=after_path,
                    output_path=output_path,
                    mode="single_image" # 明确模式为 single_image
                )
            except Exception as e:
                self.log_message(f"错误: 调用变化检测接口失败: {str(e)}", "ERROR")
                thread_communicator.display_result_signal.emit(None)
                return
        
            # 处理检测结果
            logging.info(f"### [Client Single Image] Received task_result: {task_result}")
            
            # 从结果中提取重命名后的显示图像路径
            display_image_path = ""
            final_status = task_result.get("status")
            session_id = task_result.get("session_id")  # 获取session_id用于验证

            if final_status == "completed":
                display_image_path = task_result.get("display_image_path")
                logging.info(f"### [Client Single Image] Task COMPLETED. Display path: {display_image_path}, Session ID: {session_id}")
                
                # 验证返回的路径确实属于当前任务
                if session_id and display_image_path and session_id not in os.path.basename(display_image_path):
                    logging.error(f"### [Client Single Image] 返回的显示路径不属于当前任务! 路径: {display_image_path}, 当前任务ID: {session_id}")
                    thread_communicator.display_result_signal.emit(None)
                    return
            else:
                 logging.warning(f"### [Client Single Image] Task status is NOT completed: {final_status}")
            
            # 检查路径是否存在
            if not display_image_path or not os.path.exists(display_image_path):
                self.log_message(f"错误: 未找到要显示的最终结果图像。状态: {final_status}, 路径: '{display_image_path}'", "ERROR")
                thread_communicator.display_result_signal.emit(None) # 发送 None 表示失败
                return

            # 读取最终的结果图像
            self.log_message(f"读取结果图像: {display_image_path}", "INFO")
            result_img = cv2.imread(display_image_path)
            if result_img is None:
                 self.log_message(f"错误: 无法使用OpenCV读取结果图像 '{display_image_path}'", "ERROR")
                 thread_communicator.display_result_signal.emit(None) # 发送 None 表示失败
                 return
            
            # 发送结果到主线程显示
            thread_communicator.display_result_signal.emit(result_img)
            self.log_message("图像变化检测完成！", "COMPLETE")

        except Exception as e:
            logging.error(f"### [Client Single Image] Error in _execute_detection: {e}", exc_info=True)
            self.log_message(f"变化检测过程中出错: {str(e)}", "ERROR")
            thread_communicator.display_result_signal.emit(None) # 发送 None 表示失败
        

        
    def _display_result(self, result_img):
        """显示检测结果
        
        Args:
            result_img: 检测结果图像（NumPy数组）
            
        Returns:
            bool: 是否成功显示
        """
        try:
            # 检查输入
            if result_img is None:
                self.log_message("未生成结果图像，无法显示", True)
                return False
            
            # 保存结果图像
            self.result_image = result_img  # 保存原始图像数组
            
            # 使用set_result_image方法显示结果
            if hasattr(self.navigation_functions, 'set_result_image'):
                success = self.navigation_functions.set_result_image(result_img, 'memory_image')
                if success:
                    return True
                else:
                    return False
            
            # 检查必要的组件是否存在
            if not hasattr(self.navigation_functions, 'label_result'):
                self.log_message("NavigationFunctions缺少label_result属性", True)
                return False
            
            # 获取结果标签
            label_result = self.navigation_functions.label_result
            
            # 将numpy数组转换为QPixmap
            height, width = result_img.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(result_img.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            
            if pixmap.isNull():
                self.log_message("无法创建结果图像的Pixmap", True)
                return False
            
            # 保存当前结果图像路径
            if hasattr(self.navigation_functions, 'result_image_path'):
                self.navigation_functions.result_image_path = 'memory_image'  # 标记为内存中的图像
                
            # 将结果图像也传递给NavigationFunctions实例
            if hasattr(self.navigation_functions, 'result_image'):
                self.navigation_functions.result_image = result_img  # 保存原始图像数组
                
            # 使用NavigationFunctions的update_result_display方法
            if hasattr(self.navigation_functions, 'update_result_display'):
                success = self.navigation_functions.update_result_display()
                if success:
                    return True
                else:
                    self.log_message("结果图像显示失败", True)
                    return False
            else:
                self.log_message("NavigationFunctions缺少update_result_display方法", True)
                return False
                
        except Exception as e:
            self.log_message(f"显示结果图像时出错: {str(e)}", True)
            return False



    def update_navigation_functions(self, result_img):
        """更新导航函数对象的图像显示
        
        Args:
            result_img: 变化检测结果图像
        """
        try:
            if result_img is None:
                return False
            
            # 保存结果
            self.result_image = result_img  # 保存原始图像数组
            
            # 更新导航功能类的结果图像
            if hasattr(self.navigation_functions, 'set_result_image'):
                # 使用新添加的set_result_image方法，更好地处理缩放
                result = self.navigation_functions.set_result_image(result_img, 'memory_image')
                return result
            else:
                # 兼容旧版本，直接设置变量
                if hasattr(self.navigation_functions, 'result_image_path'):
                    self.navigation_functions.result_image_path = 'memory_image'  # 标记为内存中的图像
                
                if hasattr(self.navigation_functions, 'result_image'):
                    self.navigation_functions.result_image = result_img  # 保存原始图像数组
                    
                # 显示图像（调用显示方法）
                success = False
                if hasattr(self.navigation_functions, 'update_result_display'):
                    success = self.navigation_functions.update_result_display()
                
                return success
        except Exception as e:
            self.log_message(f"更新导航结果区域出错: {str(e)}", True)
            return False

    def _clear_previous_result(self):
        """清除之前的结果图像，确保新结果能正确显示"""
        try:
            # 清除当前任务实例中的结果图像
            if hasattr(self, 'result_image'):
                self.result_image = None
            if hasattr(self, 'result_image_path'):
                self.result_image_path = None
            
            # 清除NavigationFunctions中的结果图像
            if hasattr(self.navigation_functions, 'result_image'):
                self.navigation_functions.result_image = None
            if hasattr(self.navigation_functions, 'result_image_path'):
                self.navigation_functions.result_image_path = None
            
            # 清除结果显示标签的内容
            if hasattr(self.navigation_functions, 'label_result'):
                if hasattr(self.navigation_functions.label_result, 'clear'):
                    self.navigation_functions.label_result.clear()
                    # 设置提示文本
                    if hasattr(self.navigation_functions.label_result, 'setText'):
                        self.navigation_functions.label_result.setText("正在处理...")
            
            # 强制更新UI
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            self.log_message("已清除之前的结果图像", "INFO")
            
        except Exception as e:
            self.log_message(f"清除之前结果图像时出错: {str(e)}", "WARNING")


