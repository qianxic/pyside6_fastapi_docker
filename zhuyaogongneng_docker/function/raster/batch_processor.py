'''
四大块，ui、变化检测执行、渔网分割执行、工作线程
'''

'''
渔网分割功能的日志 - self.grid_log_list
批量变化检测功能的日志 - self.log_list
'''
import os
import sys
import time
import glob
import threading
import queue
import numpy as np
import cv2
import base64
import requests
import json
import atexit  # 添加atexit模块导入
from pathlib import Path
from datetime import datetime
import concurrent.futures
import multiprocessing
from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication, QProgressDialog,QPushButton
from PySide6.QtCore import Qt, Signal, QObject, QThread, QMetaObject, QTimer, QCoreApplication, Q_ARG, Slot
from PySide6.QtGui import QFont
from osgeo import gdal
import random  # 导入随机模块用于处理策略
from .detection import RasterChangeDetection, GDAL_AVAILABLE
import logging  # Added import for logging
from zhuyaogongneng_docker.function.detection_client import detect_changes, check_connection
ShapefileGenerator = None
api_manager = None

# 尝试导入GDAL库
if not GDAL_AVAILABLE:
    # GDAL已在detection模块中尝试导入
    print("警告: 从detection模块导入的GDAL不可用，栅格批处理功能可能受限")

# 全局线程池配置
CPU_COUNT = multiprocessing.cpu_count()
DEFAULT_THREAD_POOL_SIZE = max(1, CPU_COUNT - 1)  # 设置线程数为CPU核心数-1
THREAD_POOL = None
THREAD_POOL_LOCK = threading.RLock()

def get_thread_pool(max_workers=None):
    """获取或创建全局线程池"""
    global THREAD_POOL
    if max_workers is None:
        max_workers = DEFAULT_THREAD_POOL_SIZE
        
    with THREAD_POOL_LOCK:
        if THREAD_POOL is None:
            THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        return THREAD_POOL

class WorkerSignals(QObject):
    """定义工作线程的信号"""
    progress = Signal(int, str)  # 进度信号，参数为(进度百分比，处理文件名)
    task_completed = Signal(dict)  # 单个任务完成信号，参数为结果字典
    finished = Signal()  # 完成信号
    error = Signal(str)  # 错误信号，参数为错误信息

class GridSignals(QObject):
    """定义渔网分割进度信号"""
    update_progress = Signal(int, int)  # 更新进度信号，参数为(当前处理数量，总数量)
    finished = Signal()  # 处理完成信号

class RasterBatchProcessor:
    """栅格影像批处理类，用于批量处理多组栅格影像数据"""
    
    def __init__(self, navigation_functions):
        """初始化栅格影像批处理类"""
        self.navigation_functions = navigation_functions
        self.detector = RasterChangeDetection(navigation_functions)
        
        # 批处理任务队列和结果列表 (Results kept for potential future use or different logic)
        self.results = []
        self.is_processing = False
        
        # 处理线程和线程池 (Kept for structure, pool might not be directly used by RasterBatchProcessor itself anymore)
        self.processing_thread = None
        self.thread_pool = None
        self.max_workers = DEFAULT_THREAD_POOL_SIZE
        
        # 进度信号 (Old signals for local processing removed)
        # self.signals = WorkerSignals() # Keep the signals object if needed for API/Grid workers? Re-evaluate necessity.
        # --- Connections to removed methods REMOVED --- 
        # self.signals.progress.connect(self._update_progress)
        # self.signals.finished.connect(self._on_processing_finished)
        # self.signals.error.connect(self._on_processing_error)
        
        # 渔网分割信号 (Grid signals remain)
        self.grid_signals = GridCropSignals()
        self.grid_signals.update_progress.connect(self._update_grid_progress_in_main_thread)
        # Note: grid_signals.finished and grid_signals.error are connected in _start_grid_cropping
        
        # 同步锁
        self.lock = threading.RLock()
        
        # 检查GDAL是否可用
        if not GDAL_AVAILABLE:
            self.navigation_functions.log_message("警告: GDAL库未安装，栅格批处理功能将受限")
            
    
    def check_gdal_available(self):
        """检查GDAL是否可用"""
        return self.detector.check_gdal_available()
    
    def select_input_directory(self):
        """选择包含输入栅格影像的目录"""
        input_dir = QFileDialog.getExistingDirectory(
            self.navigation_functions.main_window,
            "选择输入栅格影像目录",
            str(Path.home())
        )
        
        if not input_dir:
            return None
            
        return input_dir
    
    def select_output_directory(self):
        """选择结果输出目录"""
        output_dir = QFileDialog.getExistingDirectory(
            self.navigation_functions.main_window,
            "选择结果输出目录",
            str(Path.home())
        )
        
        if not output_dir:
            return None
        self.navigation_functions.log_message(f"已选择输出目录: {output_dir}")
        return output_dir
    
    def set_max_workers(self, max_workers=None):
        """设置最大工作线程数"""
        if max_workers is None:
            max_workers = DEFAULT_THREAD_POOL_SIZE
        self.max_workers = max_workers
        self.navigation_functions.log_message(f"设置最大工作线程数: {max_workers}")
    
    def start_batch_processing(self):
        """开始批处理任务"""
        # 获取按钮引用
        start_btn = None
        if hasattr(self, 'dialog') and self.dialog is not None:
            from PySide6.QtWidgets import QPushButton
            start_btn = self.dialog.findChild(QPushButton, "开始批处理")
            if start_btn:
                start_btn.setEnabled(False)
                start_btn.setText("处理中...")
        
        try:
            # 准备绝对路径
            abs_before_dir = os.path.abspath(self.before_dir).replace("\\", "/")
            abs_after_dir = os.path.abspath(self.after_dir).replace("\\", "/")
            abs_output_dir = os.path.abspath(self.output_dir).replace("\\", "/")
            
            # 确保输出目录存在
            os.makedirs(abs_output_dir, exist_ok=True)
            
            # 检查API连接
            if not check_connection():
                self._add_log("警告: 无法连接到API服务器")
                QMessageBox.critical(self.dialog, "错误", "无法连接到API服务器，请检查服务是否启动")
                if start_btn:
                    start_btn.setEnabled(True)
                    start_btn.setText("开始执行")
                return

            # self._add_log(f"服务正常，开始准备批处理") # REMOVED
                
            # 构建API请求数据
            data = {
                "mode": "batch_raster",  # 设置模式为批量栅格处理
                "before_path": abs_before_dir,
                "after_path": abs_after_dir,
                "output_path": abs_output_dir
            }
            
            # 创建批处理处理类
            class BatchSignals(QObject):
                finished = Signal(dict)  # 处理完成信号，传递结果
                error = Signal(str)      # 错误信号
                log = Signal(str)        # 日志信号
                
            class BatchWorker(QObject):
                def __init__(self, data, endpoint_url):
                    super().__init__()
                    self.data = data
                    self.endpoint_url = endpoint_url # This is now potentially unused
                    self.signals = BatchSignals()
                    self.is_running = False
                
                def run(self):
                    """执行批处理任务"""
                    if self.is_running:
                        return
                        
                    self.is_running = True
                    try:
                        # 提取数据
                        before_path = self.data.get("before_path", "")
                        after_path = self.data.get("after_path", "")
                        output_path = self.data.get("output_path", "")
                        mode = self.data.get("mode", "batch_raster")
                        
                        # Call detect_changes ONCE and get the final result
                        # self.signals.log.emit(f"调用变化检测接口(栅格批处理)并等待结果...") # REMOVED
                        final_task_result = detect_changes(
                            before_path=before_path,
                            after_path=after_path,
                            output_path=output_path,
                            mode=mode
                        )
                        # self.signals.log.emit(f"变化检测接口返回结果: status={final_task_result.get('status')}") # REMOVED
                        
                        # 获取最终的 task_id
                        final_task_id = final_task_result.get("task_id", "未知TaskID")
                        
                        # 返回最终结果
                        self.signals.finished.emit({
                            "task_id": final_task_id, # Use final task_id
                            "result": final_task_result # Emit the FINAL result dict
                        })
                        
                    except Exception as e:
                        import traceback
                        error_info = traceback.format_exc()
                        self.signals.error.emit(f"调用批处理API时出错: {str(e)}\n{error_info}")
                    finally:
                        self.is_running = False
            
            
            # 创建并启动工作线程
            self.batch_thread = QThread()
            # Pass data directly, endpoint_url is not strictly needed by worker now
            self.batch_worker = BatchWorker(data, None) 
            self.batch_worker.moveToThread(self.batch_thread)
            
            # 连接信号
            self.batch_thread.started.connect(self.batch_worker.run)
            self.batch_worker.signals.finished.connect(self._on_batch_api_result)
            self.batch_worker.signals.error.connect(self._on_batch_api_error)
            self.batch_worker.signals.log.connect(self._add_log)
            self.batch_worker.signals.finished.connect(self.batch_thread.quit)
            
            # 启动线程
            self.batch_thread.start()
            
        except Exception as e:
            import traceback
            error_msg = f"准备批处理任务时出错: {str(e)}"
            self._add_log(f"### {error_msg}")
            self._add_log(traceback.format_exc())
            QMessageBox.critical(self.dialog, "错误", error_msg)
            
            # 重新启用开始按钮
            if start_btn:
                start_btn.setEnabled(True)
                start_btn.setText("开始执行")
    
    def _on_processing_finished(self):
        """处理完成时触发 - 确保在UI线程中执行"""
        try:
            # 确保在UI线程中更新控件
            if QThread.currentThread() != QCoreApplication.instance().thread():
                # 不在主线程中，使用QMetaObject在主线程中调用
                QMetaObject.invokeMethod(
                    self,
                    "_on_processing_finished_in_main_thread",
                    Qt.QueuedConnection
                )
            else:
                # 已经在主线程中
                self._on_processing_finished_in_main_thread()
        except Exception as e:
            print(f"处理完成回调失败: {str(e)}")
    
    def _on_processing_error(self, error_msg):
        """处理错误时触发
        
        Args:
            error_msg: 错误信息
        """
        try:
            # 启用开始按钮
            if hasattr(self, 'dialog') and self.dialog is not None:
                from PySide6.QtWidgets import QPushButton
                start_btn = self.dialog.findChild(QPushButton, "开始批处理")
                if start_btn:
                    start_btn.setEnabled(True)
                    
            # 显示错误消息
            from PySide6.QtCore import QTimer
            def show_error():
                QMessageBox.critical(
                    self.dialog, 
                    "处理错误", 
                    f"批处理过程中发生错误：\n{error_msg}"
                )
                
            QTimer.singleShot(100, show_error)
            
        except Exception as e:
            print(f"处理错误回调失败: {str(e)}")
    

    def show_dialog(self):
        """显示栅格批处理对话框"""
        try:
            # 检查GDAL是否可用
            if not self.check_gdal_available():
                self.navigation_functions.log_message("栅格批处理功能需要GDAL库，但未能找到")
                QMessageBox.critical(
                    self.navigation_functions.main_window,
                    "依赖缺失",
                    "栅格批处理功能需要GDAL库，请安装GDAL后再试。\n"
                    "可以使用: pip install GDAL 进行安装。"
                )
                return False
                
            # 导入所需模块
            from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
                                         QLabel, QPushButton, QListWidget, QProgressBar, 
                                         QFileDialog, QTabWidget, QWidget, QSplitter, QMessageBox,
                                         QLineEdit)
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QFont
            
            # 尝试导入主题管理器
            try:
                from zhuyaogongneng_docker.theme_manager import ThemeManager
            except ImportError:
                # 备选导入路径
                try:
                    from theme_manager import ThemeManager
                except ImportError:
                    # 如果找不到主题管理器，使用简单的样式
                    class ThemeManager:
                        @staticmethod
                        def get_dialog_style(is_dark):
                            return ""
                        @staticmethod
                        def get_dialog_button_style(is_dark):
                            return ""
                        @staticmethod
                        def get_transparent_container_style():
                            return "background-color: transparent;"
                        @staticmethod
                        def get_colors(is_dark):
                            return {"text": "#333333", "background_secondary": "#F5F5F5", 
                                   "border": "#CCCCCC", "button_primary_bg": "#0078D7",
                                   "info_icon": "#2196F3", "warning_icon": "#FFD700"}
                        @staticmethod
                        def get_list_widget_style(is_dark):
                            return ""
                        @staticmethod
                        def get_log_text_style(is_dark):
                            return ""
            
            # 获取主题
            is_dark_theme = hasattr(self.navigation_functions, 'is_dark_theme') and self.navigation_functions.is_dark_theme
            
            # 创建对话框
            dialog = QDialog(self.navigation_functions.main_window)
            dialog.setWindowTitle("批量处理")
            dialog.resize(800, 550)
            
            # 设置对话框样式
            dialog.setStyleSheet(ThemeManager.get_dialog_style(is_dark_theme))
            
            # 创建主布局
            main_layout = QVBoxLayout(dialog)
            main_layout.setContentsMargins(15, 15, 15, 15)
            main_layout.setSpacing(10)
            
            # 创建标题
            title_label = QLabel("批量化栅格变化检测")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
            title_label.setStyleSheet(f"color: {ThemeManager.get_colors(is_dark_theme)['text']};")
            main_layout.addWidget(title_label)
            
            # 创建选项卡部件
            tab_widget = QTabWidget()
            
            # 创建两个选项卡页面
            grid_crop_tab = QWidget()  # 渔网分割选项卡
            setup_tab = QWidget()      # 数据设置选项卡
            
            # 设置对象名称
            grid_crop_tab.setObjectName("grid_crop_tab")
            setup_tab.setObjectName("setup_tab")
            
            # 保存引用
            self.grid_crop_tab = grid_crop_tab
            self.setup_tab = setup_tab
            
            # 添加选项卡
            tab_widget.addTab(grid_crop_tab, "渔网分割")
            tab_widget.addTab(setup_tab, "开始检测")
            
            # 初始化渔网分割选项卡
            grid_layout = QVBoxLayout(grid_crop_tab)
            grid_layout.setContentsMargins(15, 15, 15, 15)
            grid_layout.setSpacing(15)
            
            # 添加渔网分割设置区域
            grid_group_layout = QVBoxLayout()
            
            
            # 添加导入导出文件夹选择区域
            folder_layout = QVBoxLayout()
            
            # 创建影像输入目录选择
            input_layout = QHBoxLayout()
            input_label = QLabel("影像输入目录:")
            self.grid_input_dir_label = QLabel("未选择")
            self.grid_input_dir_button = QPushButton("浏览...")
            
            # 设置按钮样式
            self.grid_input_dir_button.setStyleSheet(ThemeManager.get_dialog_button_style(is_dark_theme))
            self.grid_input_dir_button.setFixedSize(80, 32)
            self.grid_input_dir_button.setFont(QFont("Microsoft YaHei UI", 9))
            
            input_layout.addWidget(input_label)
            input_layout.addWidget(self.grid_input_dir_label, 1)
            input_layout.addWidget(self.grid_input_dir_button)
            
            # 创建输出目录选择
            output_layout = QHBoxLayout()
            output_label = QLabel("输出目录:")
            self.grid_output_dir_label = QLabel("未选择")
            self.grid_output_dir_button = QPushButton("浏览...")
            
            # 设置按钮样式
            self.grid_output_dir_button.setStyleSheet(ThemeManager.get_dialog_button_style(is_dark_theme))
            self.grid_output_dir_button.setFixedSize(80, 32)
            self.grid_output_dir_button.setFont(QFont("Microsoft YaHei UI", 9))
            
            output_layout.addWidget(output_label)
            output_layout.addWidget(self.grid_output_dir_label, 1)
            output_layout.addWidget(self.grid_output_dir_button)
            
            # 添加到文件夹布局
            folder_layout.addLayout(input_layout)
            folder_layout.addLayout(output_layout)
            
            # 添加到网格设置布局
            grid_group_layout.addLayout(folder_layout)
            
            # 添加渔网分割大小输入
            grid_size_layout = QHBoxLayout()
            grid_size_label = QLabel("裁剪网格大小:")
            self.grid_size_input = QLineEdit("2,2")
            self.grid_size_input.setPlaceholderText("格式: N,N (例如: 2,2)")
            self.grid_size_input.setMaximumWidth(120)
            self.grid_size_input.setToolTip("请输入裁剪网格大小，格式为N,N，如2,2表示裁剪为2×2的网格。注意：逗号需要在英文状态下输入")
            
            # 设置输入框样式
            colors = ThemeManager.get_colors(is_dark_theme)
            self.grid_size_input.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {colors['background_secondary']};
                    color: {colors['text']};
                    border: 1px solid {colors['border']};
                    border-radius: 4px;
                    padding: 5px;
                }}
                QLineEdit:focus {{
                    border: 1px solid {colors['button_primary_bg']};
                }}
            """)
            
            grid_size_layout.addWidget(grid_size_label)
            grid_size_layout.addWidget(self.grid_size_input)
            grid_size_layout.addStretch()
            
            grid_group_layout.addLayout(grid_size_layout)
            
            # 添加逗号输入提示
            comma_tip_label = QLabel("注意: 输入网格大小时，请确保逗号在英文状态下输入")
            comma_tip_label.setStyleSheet(f"color: {ThemeManager.get_colors(is_dark_theme)['info_icon']};")
            grid_group_layout.addWidget(comma_tip_label)
            
            # 添加到主布局
            grid_layout.addLayout(grid_group_layout)
            
            # 创建日志区域
            log_label = QLabel("处理进度")
            log_label.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
            self.grid_log_list = QListWidget()
            self.grid_log_list.setStyleSheet(ThemeManager.get_list_widget_style(is_dark_theme))
            self.grid_log_list.setMinimumHeight(250)
            grid_layout.addWidget(log_label)
            grid_layout.addWidget(self.grid_log_list)
            
            # 添加开始执行按钮
            # 创建透明容器用于按钮
            grid_button_container = QWidget()
            grid_button_container.setStyleSheet(ThemeManager.get_transparent_container_style())
            grid_button_layout = QHBoxLayout(grid_button_container)
            grid_button_layout.setContentsMargins(0, 10, 0, 0)
            
            # 创建开始执行按钮
            self.grid_start_button = QPushButton("开始执行")
            
            # 使用对话框按钮样式
            dialog_button_style = ThemeManager.get_dialog_button_style(is_dark_theme)
            self.grid_start_button.setStyleSheet(dialog_button_style)
            
            # 设置按钮尺寸
            self.grid_start_button.setFixedSize(120, 36)
            
            # 设置按钮字体
            self.grid_start_button.setFont(QFont("Microsoft YaHei UI", 9, QFont.Bold))
            
            # 连接信号
            self.grid_start_button.clicked.connect(self._start_grid_cropping)
            
            # 连接目录选择按钮信号
            self.grid_input_dir_button.clicked.connect(self._select_grid_input_dir)
            self.grid_output_dir_button.clicked.connect(self._select_grid_output_dir)
            
            # 添加按钮到布局，居中显示
            grid_button_layout.addStretch()
            grid_button_layout.addWidget(self.grid_start_button)
            grid_button_layout.addStretch()
            
            # 添加按钮容器到布局
            grid_layout.addWidget(grid_button_container)
            
            # 初始化数据设置选项卡
            setup_layout = QVBoxLayout(setup_tab)
            setup_layout.setContentsMargins(15, 15, 15, 15)
            setup_layout.setSpacing(15)
            
            # 创建前时相影像目录选择
            before_layout = QHBoxLayout()
            before_label = QLabel("前时相栅格目录:")
            self.before_dir_edit = QLabel("未选择")
            self.before_dir_edit.setStyleSheet(f"color: {ThemeManager.get_colors(is_dark_theme)['text']};")
            
            before_btn = QPushButton("浏览...")
            before_btn.setStyleSheet(ThemeManager.get_dialog_button_style(is_dark_theme))
            before_btn.setFixedSize(80, 32)
            before_btn.setFont(QFont("Microsoft YaHei UI", 9))
            before_btn.clicked.connect(self._select_before_dir)
            
            before_layout.addWidget(before_label)
            before_layout.addWidget(self.before_dir_edit, 1)
            before_layout.addWidget(before_btn)
            
            # 创建后时相影像目录选择
            after_layout = QHBoxLayout()
            after_label = QLabel("后时相栅格目录:")
            self.after_dir_edit = QLabel("未选择")
            self.after_dir_edit.setStyleSheet(f"color: {ThemeManager.get_colors(is_dark_theme)['text']};")
            
            after_btn = QPushButton("浏览...")
            after_btn.setStyleSheet(ThemeManager.get_dialog_button_style(is_dark_theme))
            after_btn.setFixedSize(80, 32)
            after_btn.setFont(QFont("Microsoft YaHei UI", 9))
            after_btn.clicked.connect(self._select_after_dir)
            
            after_layout.addWidget(after_label)
            after_layout.addWidget(self.after_dir_edit, 1)
            after_layout.addWidget(after_btn)
            
            # 创建输出目录选择
            output_layout = QHBoxLayout()
            output_label = QLabel("结果输出目录:")
            self.output_dir_edit = QLabel("未选择")
            self.output_dir_edit.setStyleSheet(f"color: {ThemeManager.get_colors(is_dark_theme)['text']};")
            
            output_btn = QPushButton("浏览...")
            output_btn.setStyleSheet(ThemeManager.get_dialog_button_style(is_dark_theme))
            output_btn.setFixedSize(80, 32)
            output_btn.setFont(QFont("Microsoft YaHei UI", 9))
            output_btn.clicked.connect(self._select_output_dir)
            
            output_layout.addWidget(output_label)
            output_layout.addWidget(self.output_dir_edit, 1)
            output_layout.addWidget(output_btn)
            
            # 添加到设置布局
            setup_layout.addLayout(before_layout)
            setup_layout.addLayout(after_layout)
            setup_layout.addLayout(output_layout)
            
            # 添加提示信息
            tip_label = QLabel("注意: 确保前时相和后时相栅格目录中的文件数量和名称一一对应")
            tip_label.setStyleSheet(f"color: {ThemeManager.get_colors(is_dark_theme)['info_icon']};")
            setup_layout.addWidget(tip_label)
            
            # 添加日志标签
            log_label = QLabel("处理日志")
            log_label.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
            setup_layout.addWidget(log_label)
            self.log_list = QListWidget()
            self.log_list.setStyleSheet(ThemeManager.get_list_widget_style(is_dark_theme))
            self.log_list.setMinimumHeight(250)
            setup_layout.addWidget(self.log_list)
            
            # 创建文件列表控件(隐藏的，但用于存储文件列表)
            self.before_list = QListWidget()
            self.before_list.setVisible(False)
            setup_layout.addWidget(self.before_list)
            
            self.after_list = QListWidget()
            self.after_list.setVisible(False)
            setup_layout.addWidget(self.after_list)
            
            # 创建按钮布局
            button_container = QWidget()
            button_container.setStyleSheet(ThemeManager.get_transparent_container_style())
            button_layout = QHBoxLayout(button_container)
            button_layout.setContentsMargins(0, 10, 0, 0)
            
            # 添加开始按钮
            start_button = QPushButton("开始执行")
            start_button.setObjectName("开始批处理")
            start_button.setStyleSheet(ThemeManager.get_dialog_button_style(is_dark_theme))
            start_button.setFixedSize(120, 36)
            start_button.setFont(QFont("Microsoft YaHei UI", 9, QFont.Bold))
            start_button.clicked.connect(self._start_batch_processing)
            
            
            # 添加按钮到布局，居中显示
            button_layout.addStretch()
            button_layout.addWidget(start_button)
            button_layout.addStretch()
            
            setup_layout.addWidget(button_container)
            
            # 添加选项卡到主布局
            main_layout.addWidget(tab_widget)
            
            # 应用样式 
            colors = ThemeManager.get_colors(is_dark_theme)
            dialog.setStyleSheet(dialog.styleSheet() + f"""
                QTabWidget::pane {{
                    border: 1px solid {colors['border']};
                    background-color: {colors['background']};
                }}
                QTabWidget::tab-bar {{
                    left: 5px;
                }}
                QTabBar::tab {{
                    background-color: {colors['background_secondary']};
                    color: {colors['text']};
                    padding: 8px 12px;
                    margin-right: 2px;
                    border: 1px solid {colors['border']};
                    border-bottom: none;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }}
                QTabBar::tab:selected {{
                    background-color: {colors['button_primary_bg']};
                    color: {colors['button_primary_text']};
                }}
                QTabBar::tab:hover:!selected {{
                    background-color: {colors['button_secondary_hover']};
                }}
            """)
            
            # 保存对话框引用
            self.dialog = dialog
            
            # 显示对话框
            dialog.exec()
            
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"显示栅格批处理对话框失败: {str(e)}"
            self.navigation_functions.log_message(error_msg)
            self.navigation_functions.log_message(traceback.format_exc())
            
            # 显示错误对话框
            QMessageBox.critical(
                self.navigation_functions.main_window,
                "错误",
                f"显示栅格批处理对话框失败:\n{str(e)}"
            )
            
            return False
    
    def _select_before_dir(self):
        """选择前时相目录"""
        directory = QFileDialog.getExistingDirectory(
            self.dialog,
            "选择前时相栅格影像目录",
            str(Path.home())
        )
        
        if directory:
            # 设置目录路径
            self.before_dir = directory
            self.before_dir_edit.setText(directory)
            # 扫描目录
            self._scan_directory(directory, is_before=True)
    
    def _select_after_dir(self):
        """选择后时相目录"""
        directory = QFileDialog.getExistingDirectory(
            self.dialog,
            "选择后时相栅格影像目录",
            str(Path.home())
        )
        
        if directory:
            # 设置目录路径
            self.after_dir = directory
            self.after_dir_edit.setText(directory)
            # 扫描目录
            self._scan_directory(directory, is_before=False)
            
    def _select_output_dir(self):
        """选择输出目录"""
        directory = QFileDialog.getExistingDirectory(
            self.dialog,
            "选择结果输出目录",
            str(Path.home())
        )
        
        if directory:
            self.output_dir = directory
            self.output_dir_edit.setText(directory)
            self._add_log(f"已选择输出目录: {directory}")

    def _scan_directory(self, directory, is_before=True):
        """扫描目录并添加文件到列表"""
        if not directory or not os.path.exists(directory):
            return
        
        try:
            # 获取支持的栅格文件扩展名
            extensions = ['.tif', '.tiff', '.img']
            # 使用集合来自动去重
            unique_files = set()
            
            # 查找所有扩展名的文件
            for ext in extensions:
                # 查找小写扩展名
                pattern_lower = os.path.join(directory, f"*{ext.lower()}")
                for file_path in glob.glob(pattern_lower):
                    # 添加绝对路径到集合
                    unique_files.add(os.path.abspath(file_path))
                
                # 如果需要在不区分大小写的系统上避免重复添加大写扩展名，
                # 可以在这里移除对大写扩展名的搜索，或者依赖set去重
                # 例如，在Windows上，glob("*.tif") 和 glob("*.TIF") 可能返回相同文件
                # # 查找大写扩展名 (可选，如果需要明确处理大小写不同的情况)
                # pattern_upper = os.path.join(directory, f"*{ext.upper()}")
                # for file_path in glob.glob(pattern_upper):
                #     unique_files.add(os.path.abspath(file_path))
            
            # 将集合转换为列表并排序
            files = sorted(list(unique_files))
            
            # 清空列表
            target_list = self.before_list if is_before else self.after_list
            target_list.clear()
            
            # 添加文件到列表
            for file in files:
                target_list.addItem(file) # target_list 现在存储的是绝对路径
            
            # 更新日志和保存扫描结果（使用去重后的文件列表）
            # 保存扫描结果
            if is_before:
                self.before_images = files
                # Assuming self.preview_list is the correct widget for this log
                # If not, replace self.preview_list.addItem with self._add_log
                # self._add_log(f"前时相目录扫描完成，找到 {len(files)} 个图像文件")
                # Check if preview_list exists and is intended for this log message
                if hasattr(self, 'preview_list') and self.preview_list is not None:
                    self.preview_list.addItem(f"前时相目录扫描完成，找到 {len(files)} 个图像文件")
                else:
                    # Fallback to regular log if preview_list is not available/suitable
                    self._add_log(f"前时相目录扫描完成，找到 {len(files)} 个图像文件")
            else:
                self.after_images = files
                # Same check for after_images
                if hasattr(self, 'preview_list') and self.preview_list is not None:
                    self.preview_list.addItem(f"后时相目录扫描完成，找到 {len(files)} 个图像文件")
                else:
                    self._add_log(f"后时相目录扫描完成，找到 {len(files)} 个图像文件")
                    
        except Exception as e:
            self._add_log(f"扫描目录时出错: {str(e)}")
    
    def _add_log(self, message):
        """添加日志消息到日志列表"""
        try:
            # 确保在UI线程中调用
            if QThread.currentThread() != QCoreApplication.instance().thread():
                QMetaObject.invokeMethod(
                    self,
                    "_add_log_in_main_thread",
                    Qt.QueuedConnection,
                    Q_ARG(str, message)
                )
            else:
                self._add_log_in_main_thread(message)
        except Exception as e:
            print(f"添加日志失败: {str(e)}")
    
    @Slot(str)
    def _add_log_in_main_thread(self, message):
        """在主线程中添加日志"""
        try:
            # # 添加时间戳 # REMOVED
            # timestamp = datetime.now().strftime("%H:%M:%S") # REMOVED
            # log_entry = f"[{timestamp}] {message}" # REMOVED
            log_entry = message # Use message directly
            
            # 添加到日志列表
            self.log_list.addItem(log_entry)
            
            # 滚动到底部
            self.log_list.scrollToBottom()
        except Exception as e:
            print(f"在主线程中添加日志失败: {str(e)}")
    
    def _start_batch_processing(self):
        """开始批量栅格处理"""
        # 检查目录选择
        if not hasattr(self, 'before_dir') or not self.before_dir:
            QMessageBox.warning(self.dialog, "警告", "请先选择前时相栅格目录")
            return
            
        if not hasattr(self, 'after_dir') or not self.after_dir:
            QMessageBox.warning(self.dialog, "警告", "请先选择后时相栅格目录")
            return
            
        if not hasattr(self, 'output_dir') or not self.output_dir:
            QMessageBox.warning(self.dialog, "警告", "请先选择输出目录")
            return
        
        # 检查文件数
        if self.before_list.count() == 0 or self.after_list.count() == 0:
            QMessageBox.warning(self.dialog, "警告", "前时相或后时相目录中未发现栅格文件")
            return
            
        # 禁用开始按钮，防止重复点击
        start_btn = self.dialog.findChild(QPushButton, "开始批处理")
        if start_btn:
            start_btn.setEnabled(False)
            start_btn.setText("处理中...")
            
        # 清空日志
        self.log_list.clear()
        self._add_log("开始批量栅格变化检测处理，这可能需要一点时间...")
        
        try:
            # 准备绝对路径
            abs_before_dir = os.path.abspath(self.before_dir).replace("\\", "/")
            abs_after_dir = os.path.abspath(self.after_dir).replace("\\", "/")
            abs_output_dir = os.path.abspath(self.output_dir).replace("\\", "/")
            
            # 确保输出目录存在
            os.makedirs(abs_output_dir, exist_ok=True)
            
            # 检查API连接
            if not check_connection():
                self._add_log("警告: 无法连接到API服务器")
                QMessageBox.critical(self.dialog, "错误", "无法连接到API服务器，请检查服务是否启动")
                if start_btn:
                    start_btn.setEnabled(True)
                    start_btn.setText("开始执行")
                return

            # self._add_log(f"服务正常，开始准备批处理") # REMOVED
                
            # 构建API请求数据
            data = {
                "mode": "batch_raster",  # 设置模式为批量栅格处理
                "before_path": abs_before_dir,
                "after_path": abs_after_dir,
                "output_path": abs_output_dir
            }
            
            # 创建批处理处理类
            class BatchSignals(QObject):
                finished = Signal(dict)  # 处理完成信号，传递结果
                error = Signal(str)      # 错误信号
                log = Signal(str)        # 日志信号
                
            class BatchWorker(QObject):
                def __init__(self, data, endpoint_url):
                    super().__init__()
                    self.data = data
                    self.endpoint_url = endpoint_url # This is now potentially unused
                    self.signals = BatchSignals()
                    self.is_running = False
                
                def run(self):
                    """执行批处理任务"""
                    if self.is_running:
                        return
                        
                    self.is_running = True
                    try:
                        # 提取数据
                        before_path = self.data.get("before_path", "")
                        after_path = self.data.get("after_path", "")
                        output_path = self.data.get("output_path", "")
                        mode = self.data.get("mode", "batch_raster")
                        
                        # Call detect_changes ONCE and get the final result
                        # self.signals.log.emit(f"调用变化检测接口(栅格批处理)并等待结果...") # REMOVED
                        final_task_result = detect_changes(
                            before_path=before_path,
                            after_path=after_path,
                            output_path=output_path,
                            mode=mode
                        )
                        # self.signals.log.emit(f"变化检测接口返回结果: status={final_task_result.get('status')}") # REMOVED
                        
                        # 获取最终的 task_id
                        final_task_id = final_task_result.get("task_id", "未知TaskID")
                        
                        # 返回最终结果
                        self.signals.finished.emit({
                            "task_id": final_task_id, # Use final task_id
                            "result": final_task_result # Emit the FINAL result dict
                        })
                        
                    except Exception as e:
                        import traceback
                        error_info = traceback.format_exc()
                        self.signals.error.emit(f"调用批处理API时出错: {str(e)}\n{error_info}")
                    finally:
                        self.is_running = False
            

            # 创建并启动工作线程
            self.batch_thread = QThread()
            # Pass data directly, endpoint_url is not strictly needed by worker now
            self.batch_worker = BatchWorker(data, None) 
            self.batch_worker.moveToThread(self.batch_thread)
            
            # 连接信号
            self.batch_thread.started.connect(self.batch_worker.run)
            self.batch_worker.signals.finished.connect(self._on_batch_api_result)
            self.batch_worker.signals.error.connect(self._on_batch_api_error)
            self.batch_worker.signals.log.connect(self._add_log)
            self.batch_worker.signals.finished.connect(self.batch_thread.quit)
            
            # 启动线程
            self.batch_thread.start()
            
        except Exception as e:
            import traceback
            error_msg = f"准备批处理任务时出错: {str(e)}"
            self._add_log(f"### {error_msg}")
            self._add_log(traceback.format_exc())
            QMessageBox.critical(self.dialog, "错误", error_msg)
            
            # 重新启用开始按钮
            if start_btn:
                start_btn.setEnabled(True)
                start_btn.setText("开始执行")
    


    def _on_batch_api_result(self, result_data):
        """处理批处理结果"""
        try:
            task_id = result_data.get("task_id")
            task_result = result_data.get("result")
            
            # 检查任务是否成功
            if task_result and task_result.get("status") == "completed": # Check for completed specifically
                # ADDED LOG inside completed block
                display_path = task_result.get("display_image_path") # Assuming raster also provides this
                self._add_log(f"✅ 批量栅格处理任务已完成!")
                
                # 获取处理时间信息 (保持原有逻辑，增加None检查)
                inner_result = task_result.get("result") # Get the inner dictionary
                if isinstance(inner_result, dict) and "processing_time" in inner_result:
                    processing_time = inner_result.get("processing_time") # Access inner_result
                    if isinstance(processing_time, dict):
                        total_time = processing_time.get("total", 0)
                        # Optional: Add log for processing time
                        self.preview_list.addItem(f"✅ 批量处理任务已完成!") # Adjusted text
                QMessageBox.information(self.dialog, "完成", f"批量处理任务成功完成！")
            else:
                # ADDED LOG for failure/other status
                error_detail = task_result.get("message") if task_result else "无结果"
                error_msg = f"批处理任务 {task_id} 失败或状态异常: {task_result.get('status', '未知状态')}, 详情: {error_detail}"
                self._add_log(f"### {error_msg}")
                QMessageBox.warning(self.dialog, "失败", error_msg)
        
        except Exception as e:
            import traceback
            error_msg = f"处理批处理结果时出错: {str(e)}"
            self._add_log(f"### {error_msg}")
            self._add_log(traceback.format_exc())
        
        finally:
            # 无论成功或失败，重新启用开始按钮
            start_btn = self.dialog.findChild(QPushButton, "开始批处理")
            if start_btn:
                start_btn.setEnabled(True)
                start_btn.setText("开始执行")
    
    def _on_batch_api_error(self, error_msg):
        """处理API错误"""
        self._add_log(f"### {error_msg}")
        QMessageBox.critical(self.dialog, "API错误", error_msg)
        
        # 重新启用开始按钮
        start_btn = self.dialog.findChild(QPushButton, "开始批处理")
        if start_btn:
            start_btn.setEnabled(True)
            start_btn.setText("开始执行")
    

    def _select_grid_input_dir(self):
        """选择渔网分割输入目录"""
        try:
            dir_path = QFileDialog.getExistingDirectory(
                self.dialog,
                "选择栅格影像输入目录",
                str(Path.home())
            )
            if dir_path:
                self.grid_input_dir_label.setText(dir_path)
                self._scan_grid_input_directory(dir_path)
        except Exception as e:
            self._add_grid_log(f"选择网格输入目录失败: {str(e)}")

    def _select_grid_output_dir(self):
        """选择渔网分割输出目录"""
        try:
            dir_path = QFileDialog.getExistingDirectory(
                self.dialog,
                "选择输出目录",
                str(Path.home())
            )
            if dir_path:
                self.grid_output_dir_label.setText(dir_path)
                self._add_grid_log(f"已选择渔网分割输出目录: {dir_path}")
        except Exception as e:
            self._add_grid_log(f"选择网格输出目录失败: {str(e)}")

    def _scan_grid_input_directory(self, directory):
        """扫描网格输入目录中的栅格文件"""
        try:
            # 清空之前的日志信息
            if hasattr(self, 'grid_log_list') and self.grid_log_list is not None:
                self.grid_log_list.clear()
                
            # 设置支持的栅格文件扩展名（只使用小写）
            extensions = ['.tif', '.tiff', '.img', '.dem', '.hgt']
            
            # 使用集合存储文件路径，确保唯一性
            unique_files = set()
            
            # 扫描目录中的文件，只搜索小写后缀
            for ext in extensions:
                pattern_files = glob.glob(os.path.join(directory, f"*{ext}"))
                for file in pattern_files:
                    # 使用绝对路径确保唯一性
                    abs_path = os.path.abspath(file)
                    unique_files.add(abs_path)
            
            # 转换为有序列表并按文件名排序
            files = sorted(list(unique_files))
            
            # 记录日志
            self._add_grid_log(f"已扫描网格输入目录，找到 {len(files)} 个栅格文件")
            
            # 保存文件列表到对象
            self.grid_images = files
            
            # 显示每个文件的详细信息
            if files:
                self._add_grid_log("正在读取栅格影像信息...")
                
                # 限制最多显示的文件数量，改为只显示1个文件
                max_files_to_show = min(1, len(files))
                
                for i, file_path in enumerate(files[:max_files_to_show]):
                    try:
                        # 使用GDAL读取栅格信息
                        dataset = gdal.Open(file_path)
                        
                        if dataset:
                            # 获取基本信息
                            filename = os.path.basename(file_path)
                            width = dataset.RasterXSize
                            height = dataset.RasterYSize
                            bands = dataset.RasterCount
                            
                            # 获取地理信息
                            geotransform = dataset.GetGeoTransform()
                            projection = dataset.GetProjection()
                            has_geo = geotransform is not None and projection
                            
                            # 计算文件大小
                            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                            
                            # 添加到日志
                            file_info = f"文件 {i+1}: {filename}"
                            self._add_grid_log(f"{file_info}")
                            self._add_grid_log(f"  - 尺寸: {width}×{height} 像素, {bands} 个波段")
                            self._add_grid_log(f"  - 文件大小: {file_size:.2f} MB")
                            
                            # 打印详细的地理参考信息
                            if has_geo:
                                self._add_grid_log(f"  - 地理参考: 有")
                                self._add_grid_log(f"    |- 地理变换参数: [{geotransform[0]:.6f}, {geotransform[1]:.6f}, {geotransform[2]:.6f}, {geotransform[3]:.6f}, {geotransform[4]:.6f}, {geotransform[5]:.6f}]")
                                
                                # 简化投影信息显示
                                if "PROJCS" in projection:
                                    proj_name = projection.split('"')[1] if '"' in projection else "自定义投影"
                                    self._add_grid_log(f"    |- 投影类型: {proj_name}")
                                elif "GEOGCS" in projection:
                                    geo_name = projection.split('"')[1] if '"' in projection else "地理坐标系"
                                    self._add_grid_log(f"    |- 坐标系: {geo_name}")
                                
                                # 检查和显示空间参考系统代码(EPSG)
                                srs = dataset.GetSpatialRef()
                                if srs:
                                    epsg = srs.GetAuthorityCode(None)
                                    if epsg:
                                        self._add_grid_log(f"    |- 空间参考标识: EPSG:{epsg}")
                            else:
                                self._add_grid_log(f"  - 地理参考: 无")
                            
                            # 关闭数据集
                            dataset = None
                        else:
                            self._add_grid_log(f"文件 {i+1}: {os.path.basename(file_path)} - 无法读取")
                    except Exception as e:
                        self._add_grid_log(f"读取文件 {os.path.basename(file_path)} 信息失败: {str(e)}")
                
                if len(files) > max_files_to_show:
                    self._add_grid_log(f"共 {len(files)} 个文件，还有 {len(files) - max_files_to_show} 个文件未显示 ...")
            
        except Exception as e:
            self._add_grid_log(f"扫描网格输入目录失败: {str(e)}")
            import traceback
            self._add_grid_log(traceback.format_exc())

    def _start_grid_cropping(self):
        """开始渔网分割处理"""
        try:
            # 检查是否有正在运行的任务
            if hasattr(self, 'grid_thread') and self.grid_thread is not None and self.grid_thread.isRunning():
                QMessageBox.warning(self.dialog, "警告", "已有渔网分割任务正在运行！")
                return
                
            # 获取目录路径
            input_dir = self.grid_input_dir_label.text()
            output_dir = self.grid_output_dir_label.text()
            
            # 检查目录是否已选择
            if input_dir == "未选择" or output_dir == "未选择":
                QMessageBox.warning(self.dialog, "参数错误", "请先选择输入目录和输出目录")
                return
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取网格大小
            try:
                grid_size = self.grid_size_input.text().strip()
                rows, cols = map(int, grid_size.split(","))
                if rows <= 0 or cols <= 0:
                    QMessageBox.warning(self.dialog, "参数错误", "网格行数和列数必须大于0")
                    return
            except ValueError:
                QMessageBox.warning(self.dialog, "参数错误", "网格大小格式错误，请使用 N,N 格式 (如: 2,2)")
                return
                
            # 检查图像列表
            if not hasattr(self, 'grid_images') or len(self.grid_images) == 0:
                QMessageBox.warning(self.dialog, "参数错误", "未找到可处理的图像文件")
                return
                            
            # 记录总影像数，用于计算进度
            total_images = len(self.grid_images)
            
            # 添加进度跟踪变量到类实例
            self.grid_processed_count = 0
            self.grid_total_count = total_images
            self.grid_last_percent = 0
            
            # 获取系统内存信息，以便优化处理
            memory_info = self._get_system_memory_info()
            
            # 禁用开始按钮，防止重复点击
            self.grid_start_button.setEnabled(False)
            self.grid_start_button.setText("处理中...")
            
            # 在创建新的线程和工作器前，清理旧的资源
            self._cleanup_grid_resources()
                
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 添加初始日志
            self._add_grid_log("=" * 40)
            self._add_grid_log(f"开始渔网分割处理，共 {total_images} 个文件...")
            
            # 创建线程进行处理
            self.grid_worker = GridCropWorker(self.grid_images, output_dir, rows, cols, self.grid_signals)
            self.grid_thread = QThread()
            self.grid_worker.moveToThread(self.grid_thread)
            
            # 连接信号
            self.grid_thread.started.connect(self.grid_worker.run)
            self.grid_worker.signals.finished.connect(self._on_grid_processing_finished)
            self.grid_worker.signals.error.connect(self._on_grid_crop_error)
            
            # 启动线程
            self.grid_thread.start()
        except Exception as e:
            import traceback
            error_msg = f"启动渔网分割失败: {str(e)}"
            self._add_grid_log(error_msg)
            self._add_grid_log(traceback.format_exc())
            
            # 重新启用开始按钮
            self.grid_start_button.setEnabled(True)
            self.grid_start_button.setText("开始执行")
    
    def _cleanup_grid_resources(self):
        """清理渔网分割相关的资源，包括线程、工作对象和信号连接"""
        try:
            # 断开所有信号连接
            if hasattr(self, 'grid_worker') and self.grid_worker is not None:
                try:
                    # 显式断开与 grid_worker.signals 的连接
                    if hasattr(self.grid_worker, 'signals'):
                        try:
                            self.grid_worker.signals.finished.disconnect()
                        except (TypeError, RuntimeError):
                            # 信号可能未连接或已断开
                            pass
                        try:
                            self.grid_worker.signals.error.disconnect()
                        except (TypeError, RuntimeError):
                            # 信号可能未连接或已断开
                            pass
                        try:
                            self.grid_worker.signals.update_progress.disconnect()
                        except (TypeError, RuntimeError):
                            # 信号可能未连接或已断开
                            pass
                except Exception as e:
                    # 记录但不抛出异常，确保清理继续
                    print(f"断开 grid_worker 信号连接时出错: {str(e)}")
            
            # 断开与 grid_signals 的连接
            if hasattr(self, 'grid_signals'):
                try:
                    self.grid_signals.update_progress.disconnect()
                except (TypeError, RuntimeError):
                    # 信号可能未连接或已断开
                    pass
                try:
                    self.grid_signals.finished.disconnect()
                except (TypeError, RuntimeError):
                    # 信号可能未连接或已断开
                    pass
                try:
                    self.grid_signals.error.disconnect()
                except (TypeError, RuntimeError):
                    # 信号可能未连接或已断开
                    pass
                    
            # 停止工作器
            if hasattr(self, 'grid_worker') and self.grid_worker is not None:
                if hasattr(self.grid_worker, 'stop'):
                    self.grid_worker.stop()
                self.grid_worker.deleteLater()
                self.grid_worker = None
            
            # 停止线程
            if hasattr(self, 'grid_thread') and self.grid_thread is not None:
                if self.grid_thread.isRunning():
                    self.grid_thread.quit()
                    if not self.grid_thread.wait(500):  # 等待最多0.5秒
                        self.grid_thread.terminate()
                self.grid_thread.deleteLater()
                self.grid_thread = None
            
            # 移除信号对象的引用
            # self.grid_signals = None  # 不要移除，因为我们在初始化时创建了它，它在类的生命周期内应该保持不变
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"清理渔网分割资源时出错: {str(e)}")

    def _on_grid_processing_finished(self):
        """渔网分割处理完成时触发 - 确保在UI线程中执行"""
        try:
            # 确保在UI线程中更新控件
            if QThread.currentThread() != QCoreApplication.instance().thread():
                # 不在主线程中，使用QMetaObject在主线程中调用
                QMetaObject.invokeMethod(
                    self,
                    "_on_grid_processing_finished_in_main_thread",
                    Qt.QueuedConnection
                )
            else:
                # 已经在主线程中
                self._on_grid_processing_finished_in_main_thread()
        except Exception as e:
            print(f"渔网分割完成回调失败: {str(e)}")

    def _on_grid_processing_finished_in_main_thread(self):
        """渔网分割处理完成后的回调函数 (在主线程中执行)"""
        try:
            self._add_grid_log("=" * 40)
            self._add_grid_log("渔网分割处理完成！")
            self._add_grid_log("=" * 40)

            # 首先断开所有信号连接，避免信号累积
            if hasattr(self, 'grid_worker') and self.grid_worker is not None:
                try:
                    # 显式断开信号连接
                    if hasattr(self.grid_worker, 'signals'):
                        try:
                            self.grid_worker.signals.finished.disconnect()
                        except (TypeError, RuntimeError):
                            # 信号可能未连接或已断开
                            pass
                        try:
                            self.grid_worker.signals.error.disconnect()
                        except (TypeError, RuntimeError):
                            # 信号可能未连接或已断开
                            pass
                        try:
                            self.grid_worker.signals.update_progress.disconnect()
                        except (TypeError, RuntimeError):
                            # 信号可能未连接或已断开
                            pass
                except Exception as e:
                    self._add_grid_log(f"断开信号连接时出错: {str(e)}")

            if hasattr(self, 'grid_start_button') and self.grid_start_button:
                self.grid_start_button.setEnabled(True)
                self.grid_start_button.setText("开始执行")

            # 清理渔网分割相关的线程和工作对象
            if hasattr(self, 'grid_worker') and self.grid_worker is not None:
                self.grid_worker.deleteLater()  # 使用deleteLater确保Qt安全删除
                self.grid_worker = None
            if hasattr(self, 'grid_thread') and self.grid_thread is not None:
                if self.grid_thread.isRunning():
                    self.grid_thread.quit()
                    self.grid_thread.wait(500) # 短暂等待线程退出
                self.grid_thread.deleteLater()
                self.grid_thread = None
            
            QMessageBox.information(self.dialog, "渔网分割完成", "渔网分割处理完成！")

        except Exception as e:
            error_msg = f"处理渔网分割完成回调时出错: {str(e)}"
            self._add_grid_log(error_msg)
            import traceback
            self._add_grid_log(traceback.format_exc())
            QMessageBox.critical(self.dialog, "回调错误", error_msg)

    def _get_system_memory_info(self):
        """获取系统内存信息"""
        try:
            # 使用psutil获取系统内存信息
            import psutil
            vm = psutil.virtual_memory()
            
            # 转换为GB
            total_gb = vm.total / (1024**3)
            available_gb = vm.available / (1024**3)
            
            return {
                'total_gb': total_gb,
                'available_gb': available_gb,
                'percent': vm.percent
            }
        except Exception as e:
            print(f"获取系统内存信息失败: {str(e)}")
            # 返回默认值
            return {
                'total_gb': 8.0,
                'available_gb': 4.0,
                'percent': 50.0
            }

    def _add_grid_log(self, message):
        """添加日志到渔网分割日志列表，不添加到批量检测日志"""
        try:
            # 确保在UI线程中更新控件
            if QThread.currentThread() != QCoreApplication.instance().thread():
                # 不在主线程中，使用QMetaObject在主线程中调用
                QMetaObject.invokeMethod(
                    self,
                    "_add_grid_log_in_main_thread",
                    Qt.QueuedConnection,
                    Q_ARG(str, message)
                )
            else:
                # 已经在主线程中
                self._add_grid_log_in_main_thread(message)
        except Exception as e:
            print(f"添加渔网分割日志失败: {str(e)}")

    @Slot(str)
    def _add_grid_log_in_main_thread(self, message):
        """在主线程中添加日志到渔网分割日志列表的实现"""
        try:
            if hasattr(self, 'grid_log_list') and self.grid_log_list is not None:
                # # 创建带时间戳的日志项目 # REMOVED
                # log_item = f"[{datetime.now().strftime('%H:%M:%S')}] {message}" # REMOVED
                log_item = message # Use message directly
                self.grid_log_list.addItem(log_item)
                
                # 确保立即更新UI
                QApplication.processEvents()
                
                # 滚动到底部
                self.grid_log_list.scrollToBottom()
                
                # 管理日志列表大小，避免过多项目
                MAX_LOG_ITEMS = 1000
                current_count = self.grid_log_list.count()
                if current_count > MAX_LOG_ITEMS:
                    # 保留最新的日志，删除旧的
                    for i in range(current_count - MAX_LOG_ITEMS):
                        self.grid_log_list.takeItem(0)
        
            # 注意：不要添加到主窗口日志或批量检测日志
        except Exception as e:
            print(f"在主线程中添加渔网分割日志失败: {str(e)}")

    @Slot(int, int)
    def _update_grid_progress_in_main_thread(self, current, total):
        """在主线程中更新渔网分割进度，仅记录日志"""
        try:
            # 验证进度值的有效性
            if current < 0 or total <= 0 or current > total:
                return
                
            # 计算百分比
            percent = int(current * 100 / total)
            
            # 每10%的进度或首个和最后一个任务时添加日志
            if percent % 10 == 0 or current == 1 or current == total:
                self._add_grid_log(f"渔网分割进度: {percent}% ({current}/{total})")
        except Exception as e:
            print(f"更新渔网分割进度失败: {str(e)}")

    @Slot(str)
    def _on_grid_crop_error(self, error_msg: str):
        """处理渔网分割中单个网格处理的错误信息"""
        self._add_grid_log(f"错误: {error_msg}")

# 添加渔网分割工作线程相关类
class GridCropSignals(QObject):
    """定义渔网分割进度信号"""
    update_progress = Signal(int, int)  # 更新进度信号，参数为(当前处理数量，总数量)
    finished = Signal()  # 处理完成信号
    error = Signal(str) # 添加错误信号

class GridCropWorker(QObject):
    """渔网分割工作线程 (简化版)"""
    
    def __init__(self, grid_images, output_dir, rows, cols, signals):
        """
        初始化渔网分割工作线程 (简化版)
        Args:
            grid_images: 要裁剪的图像列表 (已经是绝对路径字符串)
            output_dir: 输出目录 (字符串)
            rows: 网格行数
            cols: 网格列数
            signals: GridCropSignals 实例
        """
        super().__init__()
        self.grid_images = [os.path.abspath(str(img)) for img in grid_images] # Ensure absolute paths
        self.output_dir = os.path.abspath(str(output_dir))
        self.rows = rows
        self.cols = cols
        self.signals = signals # Use the passed signals object
        self.is_running = False
        self._lock = threading.Lock()
        self.thread_pool = None

    def stop(self):
        """标记停止处理过程"""
        with self._lock:
            self.is_running = False
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False, cancel_futures=True)
    
    def run(self):
        """执行渔网分割处理 (简化版)"""
        with self._lock:
            if self.is_running:
                return # Already running
        self.is_running = True
        
        success_count = 0
        failed_count = 0
        total_files = len(self.grid_images)
        start_time = time.time()

        if total_files == 0:
            self.signals.finished.emit()
            with self._lock:
                self.is_running = False
            return

        # 简化工作线程数
        num_workers = max(1, (os.cpu_count() or 1) - 1)

        try:
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)

            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
            futures = []

            # 提交所有任务
            for img_path in self.grid_images:
                with self._lock:
                    if not self.is_running:
                        break # Stop requested
                    future = self.thread_pool.submit(self._process_single_image, img_path, self.output_dir, self.rows, self.cols)
                    futures.append(future)
                    
            # 处理结果
            for future in concurrent.futures.as_completed(futures):
                with self._lock:
                    if not self.is_running:
                        break # Stop requested during result processing

                try:
                    success, msg = future.result() # msg contains details or error
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        self.signals.error.emit(msg) # Emit error for failed tasks
                except Exception as e:
                    failed_count += 1
                    self.signals.error.emit(f"任务执行异常: {e}")
                finally:
                    # Update progress regardless of success/failure
                    current_count = success_count + failed_count
                    # Use the connected signal directly
                    self.signals.update_progress.emit(current_count, total_files)

        except Exception as e:
            self.signals.error.emit(f"渔网分割主过程出错: {e}")
        finally:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True) # Wait for tasks if not stopped
                self.thread_pool = None

            total_time = time.time() - start_time
            time_str = f"{total_time/60:.1f}分钟" if total_time > 60 else f"{total_time:.1f}秒"
            logging.info(f"GridCropWorker (Raster) finished. Time: {time_str}, Success: {success_count}, Failed: {failed_count}")

            self.signals.finished.emit()
            with self._lock:
                self.is_running = False
        
    def _process_single_image(self, image_file, output_dir, rows, cols):
        """处理单个图像的渔网分割 (简化版, 无缓存)
        
        Args:
            image_file: 图像文件路径 (str)
            output_dir: 输出目录 (str)
            rows: 网格行数
            cols: 网格列数
            
        Returns:
            tuple: (处理是否成功, 处理详细信息或错误信息)
        """
        dataset = None
        output_dataset = None
        temp_datasets = []
        file_lock = threading.Lock() # Local lock for file writing within this task
        
        try:
            image_file = str(image_file)
            output_dir = str(output_dir)
            
            file_name = os.path.basename(image_file)
            base_name, ext = os.path.splitext(file_name)
            if not ext or ext.lower() not in ['.tif', '.tiff', '.img']:
                 ext = '.tif' # Default to tif if unknown/missing
            
            image_output_dir = output_dir # Use the main output dir
            
            try:
                dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
                if dataset is None:
                    return False, f"无法打开栅格文件: {file_name}"
            except Exception as e:
                return False, f"打开栅格文件 {file_name} 失败: {str(e)}"
                
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            bands = dataset.RasterCount
            datatype = dataset.GetRasterBand(1).DataType
            projection = dataset.GetProjection()
            geo_transform = dataset.GetGeoTransform()
            has_geo = geo_transform is not None and projection
            
            grid_width = width // cols
            grid_height = height // rows
            
            successful_grids = 0
            total_grids = rows * cols
            
            driver = gdal.GetDriverByName("GTiff")
            
            for row in range(rows):
                for col in range(cols):
                    x_offset = col * grid_width
                    y_offset = row * grid_height
                    x_size = width - x_offset if col == cols - 1 else grid_width
                    y_size = height - y_offset if row == rows - 1 else grid_height

                    if x_size <= 0 or y_size <= 0:
                        logging.warning(f"Skipping grid {base_name}_r{row+1}c{col+1} ({file_name}) due to invalid dimensions: W={x_size}, H={y_size}")
                        continue # Skip this grid if dimensions are invalid

                    grid_name = f"{base_name}_r{row+1}c{col+1}{ext}"
                    output_file = os.path.join(image_output_dir, grid_name)
                
                    with file_lock:
                        output_dataset = None # Ensure it's reset each loop iteration
                        try:
                            output_dataset = driver.Create(output_file, x_size, y_size, bands, datatype)
                            if output_dataset is None:
                                continue # Skip this grid if creation fails

                            temp_datasets.append(output_dataset) # Track for cleanup

                            if has_geo:
                                new_geo_transform = list(geo_transform)
                                new_geo_transform[0] = geo_transform[0] + x_offset * geo_transform[1]
                                new_geo_transform[3] = geo_transform[3] + y_offset * geo_transform[5]
                                output_dataset.SetGeoTransform(tuple(new_geo_transform))
                                output_dataset.SetProjection(projection)

                            for band_idx in range(1, bands + 1):
                                output_band = output_dataset.GetRasterBand(band_idx)
                                input_band = dataset.GetRasterBand(band_idx)
                                nodata_value = input_band.GetNoDataValue()
                                if nodata_value is not None:
                                    output_band.SetNoDataValue(nodata_value)

                                # Read directly, no caching/preloading
                                band_data = input_band.ReadAsArray(x_offset, y_offset, x_size, y_size)
                                output_band.WriteArray(band_data)
                                band_data = None # Hint for GC (Corrected indentation)

                            output_dataset.FlushCache()
                            # output_dataset = None # Explicitly release reference NOW handled in finally
                            successful_grids += 1

                        except Exception as grid_e:
                            # Log error for this specific grid but continue
                            # self.signals.error.emit(f"处理网格 r{row+1}c{col+1} ({file_name}) 失败: {grid_e}") # Too noisy potentially
                            pass # Silently continue for now (Corrected indentation)
                        finally:
                            if output_dataset is not None: # Ensure cleanup if error occurred mid-write or on success
                                try:
                                    # output_dataset.FlushCache() # Already flushed if successful
                                    output_dataset = None # Release reference
                                except Exception: # Ignore potential errors during cleanup
                                    pass # (Corrected indentation and structure)

            if successful_grids == total_grids:
                return True, f"成功处理 {file_name}: {successful_grids} 个网格"
            elif successful_grids > 0:
                return False, f"{file_name}: 部分成功 ({successful_grids}/{total_grids})"
            else:
                return False, f"{file_name}: 所有网格处理失败"
                
        except Exception as e:
            return False, f"处理 {os.path.basename(image_file)} 整体失败: {e}"
        finally:
            # Ensure all resources are released
            for ds in temp_datasets:
                if ds is not None:
                    try:
                        ds.FlushCache()
                        ds = None
                    except: pass
            if dataset is not None:
                try:
                    dataset = None
                except: pass
            # No explicit GC needed here, let Python handle it
                
# The end of the simplified GridCropWorker class definition

