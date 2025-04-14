import os
import sys
import time
import glob
import re
import threading
import queue
import atexit
import logging
import traceback
from datetime import datetime
from pathlib import Path
import concurrent.futures
import multiprocessing
import random
import psutil
import numpy as np
import cv2
from PIL import Image
from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication, QProgressDialog, QProgressBar, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal, QObject, QThread, QMetaObject, QTimer, QCoreApplication, Q_ARG, Slot
import gc

# 导入API路径连接器
try:
    from change3d_api_docker.path_connector import path_connector
except ImportError:
    try:
        from change3d_api_docker.path_connector import path_connector
    except ImportError:
        path_connector = None

# 尝试导入Qt相关模块
from PySide6.QtCore import (QObject, Signal, Qt, QThread, QTimer,
                           QMetaObject, QCoreApplication, Q_ARG, Slot)
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QListWidget, QProgressBar,
                              QFileDialog, QTabWidget, QWidget, QSplitter,
                              QMessageBox, QLineEdit, QApplication, QMenu,
                              QToolButton)
from PySide6.QtGui import QFont, QAction, QCursor

# 导入ThemeManager
try:
    from zhuyaogongneng_docker.theme_manager import ThemeManager
except ImportError:
    try:
        from theme_manager import ThemeManager
    except ImportError:
        pass


logging.getLogger('root').setLevel(logging.ERROR)  # 将root日志级别设为ERROR，抑制WARNING消息
api_manager = None
CPU_COUNT = multiprocessing.cpu_count()
DEFAULT_THREAD_POOL_SIZE = max(1, CPU_COUNT - 1)  # 设置线程数为CPU核心数-1
GLOBAL_THREAD_POOL = None
THREAD_POOL_LOCK = threading.RLock()
def get_thread_pool():
    """创建全局线程池"""
    global GLOBAL_THREAD_POOL
    with THREAD_POOL_LOCK:
        if GLOBAL_THREAD_POOL is None or GLOBAL_THREAD_POOL._shutdown:
            # 获取系统内存信息
            mem_info = psutil.virtual_memory()
            total_gb = mem_info.total / (1024**3)  # 总内存(GB)
            
            # 根据系统资源动态调整线程池大小
            pool_size = max(1, CPU_COUNT - 1)  # 设为CPU核心数-1
                
            GLOBAL_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)
            logging.info(f"创建线程池: {pool_size}个工作线程 (CPU: {CPU_COUNT}核, 内存: {total_gb:.1f}GB)")
        return GLOBAL_THREAD_POOL
    

    '''
    四大块，ui、变化检测执行、渔网分割执行、工作线程
    '''



#ui开始
class BatchProcessingDialog(QDialog):
    """批量化影像变化检测对话框"""
    
    def __init__(self, navigation_functions, parent=None):
        """初始化批处理对话框
        
        Args:
            navigation_functions: 导航函数对象，包含主程序相关功能接口
            parent: 父窗口
        """
        super().__init__(parent)
        self.navigation_functions = navigation_functions
        
        # 获取深色主题状态
        self.is_dark_theme = navigation_functions.is_dark_theme
        
        # 初始化参数
        self.before_images = []  # 前时相影像列表
        self.after_images = []   # 后时相图像列表
        self.grid_images = []    # 渔网分割影像列表
        
        self.before_image_dir = "" # 前时相目录
        self.after_image_dir = ""  # 后时相目录
        self.output_dir = ""       # 输出目录
        self.grid_image_dir = ""   # 渔网分割目录
        self.grid_output_dir = ""  # 渔网分割输出目录
        
        # 初始化网格大小
        self.grid_rows = 2
        self.grid_cols = 2
        
        # 标志变量，用于防止重复读取
        self.is_scanning_grid = False
        
        # 初始化线程和工作对象
        self.batch_thread = None
        self.batch_worker = None
        self.grid_thread = None
        self.grid_worker = None
        
        # 进度条引用
        self.progress_bar = None
        
        # 标记应用程序是否正在关闭
        self.is_shutting_down = False
        
        # 移除API可用性检查 - 将在点击执行按钮时检查
        self.api_available = path_connector is not None
        # self.api_connected = False # 延迟检查
        
        # 注册窗口关闭时的资源清理
        self.destroyed.connect(self.cleanup_resources)
        
        # 配置日志
        self.setup_logging()
        
        # 初始化UI（会应用当前的主题）
        self.init_ui()
        
        # 连接主题变化的信号
        if hasattr(navigation_functions, 'theme_changed_signal'):
            navigation_functions.theme_changed_signal.connect(self.on_theme_changed)
    
    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口标题和大小
        self.setWindowTitle("批量处理")
        self.resize(800, 550)  # 减小窗口大小以适应内容
        
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # 创建标题
        title_label = QLabel("批量化影像变化检测")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
        
        # 创建选项卡部件
        self.tab_widget = QTabWidget()
        
        # 创建两个选项卡页面，删除结果查看选项卡
        self.grid_crop_tab = QWidget()  # 渔网分割选项卡
        self.setup_tab = QWidget()      # 数据设置选项卡
        
        # 添加选项卡
        self.tab_widget.addTab(self.grid_crop_tab, "渔网分割")
        self.tab_widget.addTab(self.setup_tab, "开始检测")

        # 应用主题样式到整个对话框
        self.apply_theme()
        
        # 添加标题到主布局
        title_label.setStyleSheet(f"color: {ThemeManager.get_colors(self.is_dark_theme)['text']};")
        main_layout.addWidget(title_label)
        
        # 初始化各选项卡内容
        self.init_grid_crop_tab()  # 初始化渔网分割选项卡
        self.init_setup_tab()      # 初始化数据设置选项卡
        
        # 添加选项卡到主布局
        main_layout.addWidget(self.tab_widget)
        
    def on_theme_changed(self):
        """主题变化时的响应函数"""
        # 更新主题状态
        self.is_dark_theme = self.navigation_functions.is_dark_theme
        # 应用新主题
        self.update_theme()
    
    def setup_logging(self):
        """配置日志系统"""
        # 已禁用日志输出
        pass
    
    def cleanup_resources(self):
        """清理所有资源"""
        self.is_shutting_down = True
        self.stop_all_threads()
        
        # 清理进度条
        if hasattr(self, 'progress_bar') and self.progress_bar is not None:
            try:
                self.progress_bar.setParent(None)
                self.progress_bar.deleteLater()
                self.progress_bar = None
            except:
                pass
                
        # 删除grid_progress_bar的清理
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        try:
            # 停止所有线程
            self.stop_all_threads()
            # 标记正在关闭
            self.is_shutting_down = True
            
            # 清理UI资源
            if hasattr(self, 'progress_bar') and self.progress_bar is not None:
                try:
                    self.progress_bar.setParent(None)
                    self.progress_bar.deleteLater()
                    self.progress_bar = None
                except:
                    pass
                    
            # 记录关闭事件
            logging.info("批处理窗口已关闭")
        except Exception as e:
            logging.error(f"关闭窗口时出错: {str(e)}")
        finally:
            # 确保窗口能够正常关闭
            event.accept()
    
    def stop_all_threads(self):
        """停止所有运行中的线程和工作器"""
        try:
            # 停止批处理线程
            if hasattr(self, 'batch_thread') and self.batch_thread and self.batch_thread.isRunning():
                if hasattr(self, 'batch_worker') and self.batch_worker:
                    self.batch_worker.is_running = False
                
                self.batch_thread.quit()
                if not self.batch_thread.wait(2000):  # 等待最多2秒
                    self.batch_thread.terminate()
            
            # 停止渔网分割线程
            if hasattr(self, 'grid_thread') and self.grid_thread and self.grid_thread.isRunning():
                if hasattr(self, 'grid_worker') and self.grid_worker:
                    self.grid_worker.stop()  # 使用专门的停止方法
                
                self.grid_thread.quit()
                if not self.grid_thread.wait(2000):  # 等待最多2秒
                    self.grid_thread.terminate()
            
            # 清理引用
            self.batch_worker = None
            self.batch_thread = None
            self.grid_worker = None
            self.grid_thread = None
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"停止线程时出错: {str(e)}")
    
    def init_grid_crop_tab(self):
        """初始化渔网分割选项卡"""
        # 创建布局
        layout = QVBoxLayout(self.grid_crop_tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # 添加渔网分割大小设置区域
        grid_group_layout = QVBoxLayout()
        

        # 添加导入导出文件夹选择区域
        folder_layout = QVBoxLayout()
        
        # 创建影像输入目录选择
        input_layout = QHBoxLayout()
        input_label = QLabel("影像输入目录:")
        self.after_dir_label_crop = QLabel("未选择")
        self.after_dir_button_crop = QPushButton("浏览...")
        
        # 设置按钮样式
        self.after_dir_button_crop.setStyleSheet(ThemeManager.get_dialog_button_style(self.is_dark_theme))
        self.after_dir_button_crop.setFixedSize(80, 32)
        self.after_dir_button_crop.setFont(QFont("Microsoft YaHei UI", 9))
        
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.after_dir_label_crop, 1)
        input_layout.addWidget(self.after_dir_button_crop)
        
        # 创建输出目录选择
        output_layout = QHBoxLayout()
        output_label = QLabel("结果输出目录:")
        self.output_dir_label_crop = QLabel("未选择")
        self.output_dir_button_crop = QPushButton("浏览...")
        
        # 设置按钮样式
        self.output_dir_button_crop.setStyleSheet(ThemeManager.get_dialog_button_style(self.is_dark_theme))
        self.output_dir_button_crop.setFixedSize(80, 32)
        self.output_dir_button_crop.setFont(QFont("Microsoft YaHei UI", 9))
        
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_dir_label_crop, 1)
        output_layout.addWidget(self.output_dir_button_crop)
        
        # 添加到文件夹布局
        folder_layout.addLayout(input_layout)
        folder_layout.addLayout(output_layout)
        
        # 添加到网格设置布局
        grid_group_layout.addLayout(folder_layout)
        
        # 添加渔网分割大小输入
        grid_layout = QHBoxLayout()
        grid_label = QLabel("裁剪网格大小:")
        self.grid_size_input = QLineEdit("2,2")
        self.grid_size_input.setPlaceholderText("格式: N,N (例如: 2,2)")
        self.grid_size_input.setMaximumWidth(120)
        self.grid_size_input.setToolTip("请输入裁剪网格大小，格式为N,N，如2,2表示裁剪为2×2的网格。注意：逗号需要在英文状态下输入")
        
        # 设置输入框样式
        colors = ThemeManager.get_colors(self.is_dark_theme)
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
        
        grid_layout.addWidget(grid_label)
        grid_layout.addWidget(self.grid_size_input)
        grid_layout.addStretch()
        
        grid_group_layout.addLayout(grid_layout)
        
        # 添加逗号输入提示
        self.comma_tip_label = QLabel("注意: 输入网格大小时，请确保逗号在英文状态下输入")
        self.comma_tip_label.setStyleSheet(f"color: {ThemeManager.get_colors(self.is_dark_theme)['info_icon']};")
        grid_group_layout.addWidget(self.comma_tip_label)

        
        # 添加到主布局
        layout.addLayout(grid_group_layout)
        
        # 创建日志区域（合并进度和日志）
        log_label = QLabel("处理日志")
        log_label.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        layout.addWidget(log_label)
        self.grid_log_list = QListWidget()
        self.grid_log_list.setStyleSheet(ThemeManager.get_log_text_style(self.is_dark_theme))
        
        # 添加到主布局
        layout.addWidget(log_label)
        layout.addWidget(self.grid_log_list)
        
        # 添加开始执行按钮
        # 创建透明容器用于按钮
        button_container = QWidget()
        button_container.setStyleSheet(ThemeManager.get_transparent_container_style())
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        # 创建开始执行按钮
        self.grid_start_button = QPushButton("开始执行")
        
        # 使用对话框按钮样式
        dialog_button_style = ThemeManager.get_dialog_button_style(self.is_dark_theme)
        self.grid_start_button.setStyleSheet(dialog_button_style)
        
        # 设置按钮尺寸
        self.grid_start_button.setFixedSize(120, 36)
        
        # 设置按钮字体，与数据设置标签页一致
        self.grid_start_button.setFont(QFont("Microsoft YaHei UI", 9, QFont.Bold))
        
        # 连接信号
        self.grid_start_button.clicked.connect(self.start_grid_cropping)
        
        # 连接目录选择按钮信号
        self.after_dir_button_crop.clicked.connect(self.select_after_dir_crop)
        self.output_dir_button_crop.clicked.connect(self.select_output_dir_crop)
        
        # 添加按钮到布局，居中显示
        button_layout.addStretch()
        button_layout.addWidget(self.grid_start_button)
        button_layout.addStretch()
        
        # 添加按钮容器到布局
        layout.addWidget(button_container)
        
    def init_setup_tab(self):
        """初始化数据设置选项卡"""
        # 创建布局
        layout = QVBoxLayout(self.setup_tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # 保存布局引用以便后续使用
        self.setup_tab_layout = layout
        
        # 创建前时相影像目录选择
        before_layout = QHBoxLayout()
        before_label = QLabel("前时相影像目录:")
        self.before_dir_label = QLabel("未选择")
        self.before_dir_button = QPushButton("浏览...")
        
        # 设置按钮样式 - 使用对话框按钮样式
        self.before_dir_button.setStyleSheet(ThemeManager.get_dialog_button_style(self.is_dark_theme))
        self.before_dir_button.setFixedSize(80, 32)
        self.before_dir_button.setFont(QFont("Microsoft YaHei UI", 9))
        
        before_layout.addWidget(before_label)
        before_layout.addWidget(self.before_dir_label, 1)
        before_layout.addWidget(self.before_dir_button)
        
        # 创建后时相影像目录选择
        after_layout = QHBoxLayout()
        after_label = QLabel("后时相影像目录:")
        self.after_dir_label = QLabel("未选择")
        self.after_dir_button = QPushButton("浏览...")
        
        # 设置按钮样式
        self.after_dir_button.setStyleSheet(ThemeManager.get_dialog_button_style(self.is_dark_theme))
        self.after_dir_button.setFixedSize(80, 32)
        self.after_dir_button.setFont(QFont("Microsoft YaHei UI", 9))
        
        after_layout.addWidget(after_label)
        after_layout.addWidget(self.after_dir_label, 1)
        after_layout.addWidget(self.after_dir_button)
        
        # 创建输出目录选择
        output_layout = QHBoxLayout()
        output_label = QLabel("结果输出目录:")
        self.output_dir_label = QLabel("未选择")
        self.output_dir_button = QPushButton("浏览...")
        
        # 设置按钮样式
        self.output_dir_button.setStyleSheet(ThemeManager.get_dialog_button_style(self.is_dark_theme))
        self.output_dir_button.setFixedSize(80, 32)
        self.output_dir_button.setFont(QFont("Microsoft YaHei UI", 9))
        
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_dir_label, 1)
        output_layout.addWidget(self.output_dir_button)
        
        # 添加到主布局
        layout.addLayout(before_layout)
        layout.addLayout(after_layout)
        layout.addLayout(output_layout)
        
        # 添加提示信息
        tip_label = QLabel("注意: 确保前时相和后时相影像目录中的文件数量和名称一一对应")
        tip_label.setStyleSheet(f"color: {ThemeManager.get_colors(self.is_dark_theme)['info_icon']};")
        layout.addWidget(tip_label)
        
        # 添加日志标签
        log_label = QLabel("处理日志")
        log_label.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        layout.addWidget(log_label)
        
        # 添加预览列表，放大高度
        self.preview_list = QListWidget()
        self.preview_list.setStyleSheet(ThemeManager.get_list_widget_style(self.is_dark_theme))
        self.preview_list.setMinimumHeight(250)  # 增加最小高度
        self.preview_list.setMaximumHeight(350)  # 增加最大高度
        layout.addWidget(self.preview_list)
        
        # 添加开始执行按钮
        # 创建透明容器用于按钮
        button_container = QWidget()
        button_container.setStyleSheet(ThemeManager.get_transparent_container_style())
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        # 创建开始执行按钮
        self.start_button = QPushButton("开始执行")
        
        # 使用对话框按钮样式
        dialog_button_style = ThemeManager.get_dialog_button_style(self.is_dark_theme)
        self.start_button.setStyleSheet(dialog_button_style)
        
        # 设置按钮尺寸
        self.start_button.setFixedSize(120, 36)
        
        # 设置按钮字体
        self.start_button.setFont(QFont("Microsoft YaHei UI", 9, QFont.Bold))
        
        # 连接信号
        self.start_button.clicked.connect(self.start_batch_processing)
        
        # 添加按钮到布局，居中显示
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addStretch()
        
        # 添加按钮容器到布局
        layout.addWidget(button_container)
        
        # 连接按钮信号
        self.before_dir_button.clicked.connect(self.select_before_dir)
        self.after_dir_button.clicked.connect(self.select_after_dir)
        self.output_dir_button.clicked.connect(self.select_output_dir)
        
    def apply_theme(self):
        """应用当前主题到对话框"""
        colors = ThemeManager.get_colors(self.is_dark_theme)
        
        # 设置窗口背景和边框
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {colors['background']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
            }}
            QLabel {{
                color: {colors['text']};
            }}
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
            QComboBox {{
                background-color: {colors['background_secondary']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                padding: 5px;
                min-height: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors['background_secondary']};
                color: {colors['text']};
                selection-background-color: {colors['button_primary_bg']};
                selection-color: white;
                border: 1px solid {colors['border']};
            }}
            QListWidget {{
                background-color: {colors['background_secondary']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 5px;
            }}
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
        
    def update_theme(self):
        """更新组件的主题样式"""
        # 获取当前主题状态
        self.is_dark_theme = self.navigation_functions.is_dark_theme
        
        # 应用主题样式
        self.apply_theme()
        
        # 更新各组件样式
        colors = ThemeManager.get_colors(self.is_dark_theme)
        
        # 更新日志组件样式
        log_style = ThemeManager.get_log_text_style(self.is_dark_theme)
        self.preview_list.setStyleSheet(log_style)
        self.grid_log_list.setStyleSheet(log_style)
        
        # 更新按钮样式
        dialog_button_style = ThemeManager.get_dialog_button_style(self.is_dark_theme)
        
        # 更新数据设置标签页按钮样式
        self.start_button.setStyleSheet(dialog_button_style)
        self.before_dir_button.setStyleSheet(dialog_button_style)
        self.after_dir_button.setStyleSheet(dialog_button_style)
        self.output_dir_button.setStyleSheet(dialog_button_style)
        
        # 更新格网裁剪标签页按钮样式
        self.grid_start_button.setStyleSheet(dialog_button_style)
        self.after_dir_button_crop.setStyleSheet(dialog_button_style)
        self.output_dir_button_crop.setStyleSheet(dialog_button_style)
        
        # 更新网格大小输入框样式
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
        
        # 更新标签文字颜色
        for label in self.findChildren(QLabel):
            label.setStyleSheet(f"color: {colors['text']};")
            
        # 更新tab样式
        self.tab_widget.setStyleSheet(f"""
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
        
        # 更新警告标签颜色
        if hasattr(self, 'comma_tip_label'):
            self.comma_tip_label.setStyleSheet(f"color: {colors['info_icon']};")

        # 确保提示标签也使用info_icon颜色
        for child in self.findChildren(QLabel):
            if child.text().startswith("注意:"):
                child.setStyleSheet(f"color: {colors['info_icon']};")



#ui结束
#变化检测开始

    def select_before_dir(self):
        """选择前时相影像目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择前时相影像目录")
        if directory:
            # 如果选择了新目录，清除日志和之前的前时相数据
            if self.before_image_dir != directory:
                self.before_images = []
                # 清空日志，但保留后时相数据（如果有）
                self.preview_list.clear()
            
            # 保存目录路径
            self.before_image_dir = directory
            self.before_dir_label.setText(directory)
            
            # 扫描目录获取图像列表
            self.scan_directory(directory, is_before=True)
            
    def select_after_dir(self):
        """选择后时相目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择后时相目录")
        if directory:
            # 如果选择了新目录，清除之前的后时相数据
            if self.after_image_dir != directory:
                self.after_images = []
                # 不清空日志，保留前时相信息
            
            # 保存目录路径
            self.after_image_dir = directory
            self.after_dir_label.setText(directory)
            
            # 扫描目录获取图像列表
            self.scan_directory(directory, is_before=False)
            
    def select_output_dir(self):
        """选择输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择结果输出目录")
        if directory:
            # 保存目录路径
            self.output_dir = directory
            self.output_dir_label.setText(directory)
            
            # 添加选择输出目录的消息
            self.preview_list.addItem(f"输出目录: {directory}")
            
    def scan_directory(self, directory, is_before=True):
        """扫描目录中的图像文件"""
        try:
            # 使用集合去重，避免重复扫描同一文件
            image_files = set()
            
            # 统一扩展名大小写
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            
            # 使用os.walk扫描目录中的所有文件
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    file_lower = filename.lower()
                    if any(file_lower.endswith(ext) for ext in image_extensions):
                        abs_path = os.path.abspath(os.path.join(root, filename))
                        image_files.add(abs_path)
                         
            # 转换为有序列表
            files = sorted(list(image_files))
            
            # 保存扫描结果
            if is_before:
                self.before_images = files
                self.preview_list.addItem(f"前时相目录扫描完成，找到 {len(files)} 个图像文件")
            else:
                self.after_images = files
                self.preview_list.addItem(f"后时相目录扫描完成，找到 {len(files)} 个图像文件")
                
            
            # 检查前后影像数量是否匹配
            if hasattr(self, 'before_images') and hasattr(self, 'after_images'):
                if self.before_images and self.after_images:
                    if len(self.before_images) != len(self.after_images):
                        msg = f"警告: 前后时相影像数量不匹配 (前: {len(self.before_images)}, 后: {len(self.after_images)})"
                        self.preview_list.addItem(msg)
                        QMessageBox.warning(self, "警告", msg)
                    else:
                        pass
            
            return files
                
        except Exception as e:
            error_msg = f"扫描目录失败: {str(e)}"
            self.preview_list.addItem(error_msg)
            return []

    def add_log(self, message):
        """添加日志到日志列表（线程安全）"""
        # 已禁用日志输出
        pass

    def _add_log_impl(self, message):
        """实际添加日志的实现"""
        # 已禁用日志输出
        pass
            


    def _update_api_batch_paths(self):
        """更新API批处理路径，当三个目录都已设置时调用"""
        try:
            if path_connector is not None and path_connector.check_connection():
                # 确保三个目录都已设置
                if not (hasattr(self, 'before_image_dir') and self.before_image_dir and 
                        hasattr(self, 'after_image_dir') and self.after_image_dir and 
                        hasattr(self, 'output_dir') and self.output_dir):
                    return
                
                # 使用绝对路径确保API能正确识别
                abs_before_dir = os.path.abspath(self.before_image_dir)
                abs_after_dir = os.path.abspath(self.after_image_dir)
                abs_output_dir = os.path.abspath(self.output_dir)
                
                # 确保输出目录存在
                if not os.path.exists(abs_output_dir):
                    os.makedirs(abs_output_dir, exist_ok=True)
                
                # 直接调用API进行批处理路径处理
                path_connector.process_batch_image_paths(
                    abs_before_dir,
                    abs_after_dir,
                    abs_output_dir
                )
                return True
                
        except Exception:
            # 不输出API相关错误信息
            pass
        return False

    def start_batch_processing(self):
        """开始批处理，调用batch_image端点处理整个目录"""
        # 禁用开始按钮
        if hasattr(self, 'start_button'):
            self.start_button.setEnabled(False)
            self.start_button.setText("处理中...")
        
        # 清空日志并添加开始消息
        self.preview_list.clear()
        self.preview_list.addItem("开始批量图像变化检测处理，这可能需要一点时间...") # ADDED START MESSAGE

        # 检查API可用性
        if path_connector is None:
            error_msg = "警告: API路径连接器未初始化"
            self.preview_list.addItem(error_msg)
            QMessageBox.critical(self, "错误", "API连接器未初始化，无法执行批处理")
            if hasattr(self, 'start_button'):
                self.start_button.setEnabled(True)
                self.start_button.setText("开始执行")
            return
            
        if not path_connector.check_connection():
            # 使用与栅格一致的措辞
            error_msg = "警告: 无法连接到API服务器"
            self.preview_list.addItem(error_msg)
            QMessageBox.critical(self, "错误", "无法连接到API服务器，请检查服务是否启动")
            if hasattr(self, 'start_button'):
                self.start_button.setEnabled(True)
                self.start_button.setText("开始执行")
            return
        
        try:
            # 构建请求数据
            # ... (路径处理保持不变)
            abs_before_dir = os.path.abspath(self.before_image_dir).replace("\\", "/")
            abs_after_dir = os.path.abspath(self.after_image_dir).replace("\\", "/")
            abs_output_dir = os.path.abspath(self.output_dir).replace("\\", "/")

            # 确保输出目录存在
            os.makedirs(abs_output_dir, exist_ok=True)
            
            data = {
                "mode": "batch_image", # 确认使用batch_image模式
                "before_path": abs_before_dir,  # Use host path directly
                "after_path": abs_after_dir,    # Use host path directly
                "output_path": abs_output_dir   # Use host path directly
            }

            # 向API发送批处理请求
            endpoint_url = f"{path_connector.api_url}/detect/batch_image" # This might be unused now
            
            # 创建工作线程
            self.batch_thread = QThread()
            self.batch_worker = BatchImageProcessingWorker(data, endpoint_url) # Pass corrected data
            self.batch_worker.moveToThread(self.batch_thread)
            
            # 连接信号
            self.batch_thread.started.connect(self.batch_worker.run)
            self.batch_worker.signals.finished.connect(self.on_batch_api_result)
            self.batch_worker.signals.error.connect(self.on_batch_api_error)
            # 使用与栅格一致的 lambda 连接日志信号
            self.batch_worker.signals.log.connect(lambda msg: self.preview_list.addItem(msg)) 
            self.batch_worker.signals.finished.connect(self.batch_thread.quit)
            
            # 启动线程
            self.batch_thread.start()
            
        except Exception as e:
            import traceback
            error_info = traceback.format_exc()
            error_msg = f"准备批处理任务时出错: {str(e)}"
            self.preview_list.addItem(f"### {error_msg}") # Add ### prefix like raster
            self.preview_list.addItem(error_info) # Keep traceback for detail
            QMessageBox.critical(self, "错误", error_msg)
            
            # 重新启用开始按钮
            if hasattr(self, 'start_button'):
                self.start_button.setEnabled(True)
                self.start_button.setText("开始执行")
    
    def on_batch_api_result(self, result_data):
        """处理批处理结果"""
        try:
            task_id = result_data.get("task_id")
            task_result = result_data.get("result")
            
            # 检查任务是否成功 (使用与栅格一致的状态检查和日志)
            if task_result and task_result.get("status") == "completed":
                # ADDED LOG inside completed block
                display_path = task_result.get("display_image_path")
                self.preview_list.addItem(f"✅ 批量处理任务已完成!") # Adjusted text
                QMessageBox.information(self, "完成", f"批量处理任务成功完成！")
                # TODO: Add logic here to actually display the image using display_path if needed
            else:
                # ADDED LOG for failure/other status
                error_detail = task_result.get("message") if task_result else "无结果"
                error_msg = f"批处理任务 {task_id} 失败或状态异常: {task_result.get('status', '未知状态')}, 详情: {error_detail}"
                self.preview_list.addItem(f"### {error_msg}")
                QMessageBox.warning(self, "失败", error_msg)
        

        except Exception as e:
            error_msg = f"处理批处理结果时出错: {str(e)}"
            # Optional: Add traceback if needed
            # import traceback
            # self.preview_list.addItem(traceback.format_exc())
        
        finally:
            # 无论成功或失败，重新启用开始按钮
            if hasattr(self, 'start_button') and self.start_button is not None:
                self.start_button.setEnabled(True)
                self.start_button.setText("开始执行")
                
            # 清理线程资源
            self.clean_batch_resources()
    


    def on_batch_api_error(self, error_msg):
        """处理批处理错误"""
        # 使用与栅格一致的错误消息格式
        self.preview_list.addItem(f"### {error_msg}")
        QMessageBox.critical(self, "API错误", error_msg) # Keep title as API错误 or change to 错误?
        
        # 重新启用开始按钮
        if hasattr(self, 'start_button') and self.start_button is not None:
            self.start_button.setEnabled(True)
            self.start_button.setText("开始执行")
            
        # 清理线程资源
        self.clean_batch_resources()




    def clean_batch_resources(self):
        """清理批量处理相关资源"""
        try:
            # 当线程结束时，释放工作器资源
            if hasattr(self, 'batch_worker') and self.batch_worker is not None:
                if hasattr(self.batch_worker, 'signals'):
                    try:
                        # 断开所有信号连接
                        if hasattr(self.batch_worker.signals, 'finished'):
                            self.batch_worker.signals.finished.disconnect()
                        if hasattr(self.batch_worker.signals, 'error'):
                            self.batch_worker.signals.error.disconnect()
                        if hasattr(self.batch_worker.signals, 'log'):
                            self.batch_worker.signals.log.disconnect()
                    except Exception:
                        pass
                self.batch_worker.deleteLater()
                self.batch_worker = None
                
            # 清理线程
            if hasattr(self, 'batch_thread') and self.batch_thread is not None:
                if self.batch_thread.isRunning():
                    self.batch_thread.quit()
                    if not self.batch_thread.wait(1000):  # 等待1秒
                        self.batch_thread.terminate()
                self.batch_thread = None
                
            self.add_log("批量处理资源已清理")
        except Exception as e:
            self.add_log(f"清理资源时出错: {str(e)}")



            
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        try:
            # 确保所有线程都被终止
            self.stop_all_threads()
            
            # 显式删除UI相关资源
            self._cleanup_ui_resources()
        except Exception as e:
            # 已禁用日志输出
            pass

    def _cleanup_ui_resources(self):
        """清理UI资源"""
        try:
            # 清理进度条资源
            for progress_bar_name in ['progress_bar', 'batch_progress_bar']:
                if hasattr(self, progress_bar_name) and getattr(self, progress_bar_name) is not None:
                    try:
                        progress_bar = getattr(self, progress_bar_name)
                        progress_bar.setParent(None)
                        progress_bar.deleteLater()
                        setattr(self, progress_bar_name, None)
                    except:
                        pass
                        
            # 确保工作器已被清理
            for worker_name in ['grid_worker', 'batch_worker']:
                if hasattr(self, worker_name) and getattr(self, worker_name) is not None:
                    try:
                        worker = getattr(self, worker_name)
                        worker.deleteLater()
                        setattr(self, worker_name, None)
                    except:
                        pass
            
            # 确保所有信号连接已断开
            for signals_name in ['grid_signals', 'batch_signals']:
                if hasattr(self, signals_name):
                    try:
                        signals = getattr(self, signals_name)
                        signals.disconnect()
                    except:
                        pass
        except Exception as e:
            # 已禁用日志输出
            pass






#变化检测结束
    def select_after_dir_crop(self):
        """选择格网裁剪影像目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择影像输入目录")
        if directory:
            # 如果选择了新目录，清除旧数据
            if self.grid_image_dir != directory:
                self.grid_images = []
                
            self.grid_image_dir = directory
            self.after_dir_label_crop.setText(directory)
            
            
            # 清空日志并扫描目录，确保UI更新
            self.grid_log_list.clear()
            QApplication.processEvents()
            
            # 扫描目录中的图像
            self.scan_grid_directory(directory)
            
    def select_output_dir_crop(self):
        """选择格网裁剪输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择结果输出目录")
        if directory:
            self.grid_output_dir = directory
            self.output_dir_label_crop.setText(directory)
            # 添加选择输出目录的消息
            self.add_grid_log(f"已选择输出目录: {directory}")
            
            # 如果对应的保存复选框存在，确保它被勾选
            if hasattr(self, 'save_grid_cb') and self.save_grid_cb is not None:
                self.save_grid_cb.setChecked(True)
            
    def scan_grid_directory(self, directory):
        """扫描格网裁剪目录中的图像文件"""
        try:
            # 清空之前的日志信息
            self.grid_log_list.clear()
                    
            # 设置支持的图像文件扩展名
            extensions = ['.jpg', '.jpeg', '.png']
            
            # 使用集合去重，避免重复扫描同一文件
            image_files = set()
            
            # 使用os.walk扫描目录中的所有文件
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    file_lower = filename.lower()
                    file_upper = filename.upper()
                    if any(file_lower.endswith(ext) for ext in extensions) or any(file_upper.endswith(ext.upper()) for ext in extensions):
                        abs_path = os.path.abspath(os.path.join(root, filename))
                        image_files.add(abs_path)
            
            # 将集合转换为有序列表
            files = sorted(list(image_files))
            
            # 记录日志
            self.add_grid_log(f"已扫描网格输入目录，找到 {len(files)} 个图像文件")
            
            # 保存文件列表到对象
            self.grid_images = files
            
            # 显示文件详情（最多显示1个，避免UI卡顿）
            max_display = min(1, len(files))
            for i in range(max_display):
                file_path = files[i]
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
                self.add_grid_log(f"文件 {i+1}: {file_name} ({file_size:.2f} MB)")
                
            if len(files) > max_display:
                self.add_grid_log(f"... 还有 {len(files) - max_display} 个文件未显示")
                
            return files
        except Exception as e:
            self.add_grid_log(f"扫描网格输入目录失败: {str(e)}")
            import traceback
            self.add_grid_log(traceback.format_exc())
            return []

    def start_grid_cropping(self):
        """开始渔网分割处理"""
        try:
            # 检查是否有正在运行的任务
            if hasattr(self, 'grid_thread') and self.grid_thread is not None and self.grid_thread.isRunning():
                QMessageBox.warning(self, "警告", "已有渔网分割任务正在运行！")
                return
                
            # 获取输入目录
            input_dir = self.after_dir_label_crop.text().strip()
            if input_dir == "未选择" or not os.path.exists(input_dir):
                QMessageBox.warning(self, "错误", "请选择有效的输入目录！")
                return
                
            # 获取输出目录
            output_dir = self.output_dir_label_crop.text().strip()
            if output_dir == "未选择":
                QMessageBox.warning(self, "错误", "请选择输出目录！")
                return
                
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取网格大小
            try:
                # 解析网格大小，格式为"行,列"
                grid_size_text = self.grid_size_input.text().strip()
                rows, cols = map(int, grid_size_text.split(","))
                
                # 检查有效性
                if rows <= 0 or cols <= 0:
                    raise ValueError("网格大小必须为正整数")
                    
                # 保存到对象中
                self.grid_rows = rows
                self.grid_cols = cols
                
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无效的网格大小: {str(e)}\n请使用格式: 行数,列数（例如: 2,2）")
                return
                
            # 检查文件列表是否存在，如果已扫描过则不需要再次扫描
            if not hasattr(self, 'grid_images') or not self.grid_images:
                self.grid_images = self.scan_grid_directory(input_dir)
                
            # 如果文件列表为空，提示用户
            if not self.grid_images:
                QMessageBox.warning(self, "错误", "未找到有效的图像文件！")
                return
                        
            # 检查文件路径是否重复
            unique_files = set()
            for file_path in self.grid_images:
                unique_files.add(os.path.abspath(str(file_path)))
                
            if len(unique_files) < len(self.grid_images):
                self.add_grid_log(f"警告: 发现 {len(self.grid_images) - len(unique_files)} 个重复文件路径，已自动去重")
                self.grid_images = sorted(list(unique_files))
                
            # 禁用开始按钮，防止重复点击
            self.grid_start_button.setEnabled(False)
            self.grid_start_button.setText("处理中...")
            
            # 创建信号和工作线程
            # 添加GridCropSignals类的定义
            class GridCropSignals(QObject):
                update_progress = Signal(int, int)  # 当前进度和总进度
                finished = Signal()  # 处理完成信号
                error = Signal(str)  # Add an error signal
                
            self.grid_signals = GridCropSignals()
            self.grid_worker = GridCropWorker(self, self.grid_images, output_dir, rows, cols, self.grid_signals)
            self.grid_thread = QThread()
            self.grid_worker.moveToThread(self.grid_thread)
            
            # 连接信号
            self.grid_signals.update_progress.connect(self._update_grid_progress_in_main_thread)
            self.grid_signals.finished.connect(self._on_grid_processing_finished_in_main_thread)
            self.grid_signals.error.connect(self._on_grid_processing_error_in_main_thread)
            self.grid_thread.started.connect(self.grid_worker.run)
            self.grid_thread.start()

            
        except Exception as e:
            self.add_grid_log(f"启动渔网分割失败: {str(e)}")
            import traceback
            self.add_grid_log(traceback.format_exc())
            
            # 重新启用开始按钮
            if hasattr(self, 'grid_start_button'):
                self.grid_start_button.setEnabled(True)
                self.grid_start_button.setText("开始执行")

    def clean_grid_resources(self):
        """清理渔网分割相关资源"""
        try:
            # 停止渔网分割线程
            if hasattr(self, 'grid_thread') and self.grid_thread and self.grid_thread.isRunning():
                # 先停止工作器
                if hasattr(self, 'grid_worker') and self.grid_worker:
                    self.grid_worker.stop()
                
                # 尝试等待线程完成
                self.grid_thread.quit()
                if not self.grid_thread.wait(3000):  # 等待最多3秒
                    self.grid_thread.terminate()
                    # 已禁用日志输出
            
            # 清理引用
            self.grid_worker = None
            self.grid_thread = None
            self.grid_signals = None
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 已禁用日志输出
        except Exception as e:
            # 已禁用日志输出
            pass

    def update_grid_progress(self, current, total):
        pass

    def on_grid_processing_finished(self):
        pass

    def add_grid_log(self, message):
        """添加日志到渔网分割日志列表，避免控制台打印"""
        try:
            # 避免使用 QMetaObject.invokeMethod 方式
            if QThread.currentThread() != QCoreApplication.instance().thread():
                # 使用信号-槽机制传递消息
                if not hasattr(self, '_log_bridge'):
                    # 创建持久的信号桥接器
                    class LogBridge(QObject):
                        log_signal = Signal(str)
                    self._log_bridge = LogBridge()
                    self._log_bridge.log_signal.connect(self._add_grid_log_impl, Qt.QueuedConnection)
                
                self._log_bridge.log_signal.emit(message)
            else:
                # 直接调用
                self._add_grid_log_impl(message)
        except Exception:
            # 不输出异常信息
            pass

    def _add_grid_log_impl(self, message):
        """在主线程中实现日志添加"""
        try:
            # 创建带时间戳的日志项
            log_item = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
            
            # 添加到日志列表
            if hasattr(self, 'grid_log_list') and self.grid_log_list is not None:
                self.grid_log_list.addItem(log_item)
                self.grid_log_list.scrollToBottom()
        except Exception:
            # 不输出异常信息
            pass


    def _safe_ui_update(self, update_func):
        """安全地更新UI，避免QBasicTimer相关的跨线程错误
        
        Args:
            update_func: 执行UI更新的函数
        """
        try:
            # 确保UI更新发生在主线程
            if QThread.currentThread() == QCoreApplication.instance().thread():
                # 已经在主线程，直接更新
                update_func()
            else:
                # 不在主线程，使用Qt的信号机制将操作放到主线程执行
                # 创建一个临时QObject用于信号传递
                class TempSignalBridge(QObject):
                    signal = Signal(object)
                
                signal_bridge = TempSignalBridge()
                signal_bridge.signal.connect(lambda f: f(), Qt.QueuedConnection)
                signal_bridge.signal.emit(update_func)
                
                # 给Qt事件循环一些时间来处理排队的操作
                QCoreApplication.processEvents()
        except Exception as e:
            # 已禁用日志输出
            pass


    def on_grid_error(self, error_msg):
        """处理渔网分割错误"""
        try:
            self.add_grid_log(f"错误: {error_msg}")
            QMessageBox.critical(self, "处理错误", error_msg)
        except Exception as e:
            logging.error(f"处理渔网分割错误时出错: {str(e)}")

    @Slot(int, int)
    def _update_grid_progress_in_main_thread(self, current, total):
        """在主线程中更新渔网分割进度"""
        try:
            # 验证进度值的有效性
            if current < 0 or total <= 0 or current > total:
                return
                
            # 计算百分比
            percent = int(current * 100 / total)
            
            # 每10%的进度或首个和最后一个任务时添加日志
            if percent % 10 == 0 or current == 1 or current == total:
                self.add_grid_log(f"渔网分割进度: {percent}% ({current}/{total})")
                
        except Exception as e:
            # 已禁用日志输出
            pass

    @Slot()
    def _on_grid_processing_finished_in_main_thread(self):
        """渔网分割处理完成后的回调函数"""
        try:
            # 添加完成日志
            self.add_grid_log("="*40)
            self.add_grid_log("渔网分割处理完成！")
            self.add_grid_log("="*40)
            
            # 重新启用开始按钮
            self.grid_start_button.setEnabled(True)
            self.grid_start_button.setText("开始执行")
            
            # 清理资源 - 确保线程安全释放
            self._cleanup_grid_resources()
            
            # 显示完成消息
            QMessageBox.information(self, "渔网分割完成", "渔网分割处理完成！")
        
        except Exception as e:
            self.add_grid_log(f"处理完成回调中出错: {str(e)}")
            import traceback
            self.add_grid_log(traceback.format_exc())
            

    def _cleanup_grid_resources(self):
        """安全清理渔网分割相关资源"""
        try:
            # 停止工作器
            if hasattr(self, 'grid_worker') and self.grid_worker:
                self.grid_worker.stop()
            
            # 停止线程
            if hasattr(self, 'grid_thread') and self.grid_thread and self.grid_thread.isRunning():
                self.grid_thread.quit()
                if not self.grid_thread.wait(2000):  # 等待最多2秒
                    self.grid_thread.terminate()
            
            # 清理引用
            self.grid_worker = None
            self.grid_thread = None
            
            # 断开信号连接
            if hasattr(self, 'grid_signals'):
                try:
                    self.grid_signals.disconnect()
                except:
                    pass
                self.grid_signals = None
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            pass

    def _on_grid_processing_error_in_main_thread(self, error_msg):
        """处理渔网分割错误"""
        try:
            self.add_grid_log(f"错误: {error_msg}")
            QMessageBox.critical(self, "处理错误", error_msg)
        except Exception as e:
            logging.error(f"处理渔网分割错误时出错: {str(e)}")



#主要界面及工作线程类
class BatchProcessing:
    """批量处理模块入口类"""
    
    def __init__(self, navigation_functions):
        """初始化批量处理模块
        
        Args:
            navigation_functions: 导航函数接口对象
        """
        self.navigation_functions = navigation_functions
        self.batch_dialog = None
        
        # 检查API可用性
        self.api_available = path_connector is not None
        if self.api_available:
            self.navigation_functions.log_message("API路径处理服务已加载")
        
    def show_batch_processing_dialog(self):
        """显示批量处理对话框"""
        try:
            if self.batch_dialog is None:
                self.batch_dialog = BatchProcessingDialog(self.navigation_functions)
            else:
                # 强制清理日志列表
                if hasattr(self.batch_dialog, "grid_log_list") and self.batch_dialog.grid_log_list is not None:
                    self.batch_dialog.grid_log_list.clear()
                if hasattr(self.batch_dialog, "preview_list") and self.batch_dialog.preview_list is not None:
                    self.batch_dialog.preview_list.clear()
            
            # 确保对话框主题设置是最新的
            if hasattr(self.batch_dialog, "update_theme"):
                # 在显示之前强制更新主题设置
                self.batch_dialog.is_dark_theme = self.navigation_functions.is_dark_theme
                self.batch_dialog.update_theme()
                
            self.batch_dialog.show()
            self.batch_dialog.activateWindow()
            
        except Exception as e:
            logging.error(f"显示批量处理对话框时出错: {str(e)}")
            logging.error(traceback.format_exc())

class ProcessSignals(QObject):
    update_progress = Signal(int, int)  # 当前进度和总进度
    finished = Signal()  # 处理完成信号
    
class BatchProcessSignals(QObject):
    finished = Signal()  # 处理完成信号
    


class GridCropWorker(QObject):
    """渔网分割处理的工作线程类 (简化版)"""

    def __init__(self, dialog, grid_images, output_dir, rows, cols, signals):
        """初始化渔网分割工作线程"""
        super().__init__()
        self.dialog = dialog # Still needed for context potentially, or remove if not used
        self.grid_images = [Path(os.path.abspath(str(img))) for img in grid_images] # Ensure absolute paths
        self.output_dir = Path(output_dir)
        self.rows = rows
        self.cols = cols
        self.signals = signals
        self.is_running = False
        self._lock = threading.Lock() # Simple lock for is_running flag
        self.thread_pool = None # Initialize pool as None
        # Removed cache, process_pool, complex locks, thread_local, processed_files, state_lock

    def stop(self):
        """标记停止处理过程"""
        with self._lock:
            self.is_running = False
        # Attempt to shutdown the pool if it exists
        if self.thread_pool:
            # Non-blocking shutdown, tasks might not finish
            self.thread_pool.shutdown(wait=False, cancel_futures=True)

    def run(self):
        """运行渔网分割处理 (简化版)"""
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

        # Simplified worker count
        num_workers = max(1, (os.cpu_count() or 1) -1) # Default to CPU count - 1

        try:
            # Use ThreadPoolExecutor directly
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
            futures = []

            # Submit all tasks
            for img_path in self.grid_images:
                with self._lock:
                    if not self.is_running:
                        break # Stop requested
                # Submit task to process one image
                future = self.thread_pool.submit(self._process_grid_image, str(img_path), str(self.output_dir))
                futures.append(future)

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                with self._lock:
                    if not self.is_running:
                         # If stop was requested after task completion, don't process further
                         # Cancel remaining futures if possible (ThreadPoolExecutor limitation)
                        # for f in futures: f.cancel() # Note: cancel doesn't reliably stop running threads
                        break

                try:
                    success, msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        # Emit error signal for failed tasks
                        self.signals.error.emit(msg)
                except Exception as e:
                    failed_count += 1
                    # Emit error signal for exceptions during task execution
                    self.signals.error.emit(f"任务执行出错: {e}")
                finally:
                    # Update progress regardless of success/failure
                    current_count = success_count + failed_count
                    self.signals.update_progress.emit(current_count, total_files)

        except Exception as e:
            self.signals.error.emit(f"渔网分割处理过程中发生严重错误: {e}")
        finally:
            # Ensure pool is shut down
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True) # Wait for running tasks to complete if not stopped
                self.thread_pool = None

            total_time = time.time() - start_time
            time_str = f"{total_time/60:.1f}分钟" if total_time > 60 else f"{total_time:.1f}秒"
            logging.info(f"GridCropWorker finished. Total time: {time_str}, Success: {success_count}, Failed: {failed_count}") # Use logging

            # Emit finished signal
            self.signals.finished.emit()
            with self._lock:
                self.is_running = False

    def _process_grid_image(self, image_file, output_path):
        """处理单个图像为网格 (无缓存)"""
        try:
            # 确保输出目录存在 (线程安全)
            # os.makedirs(output_path, exist_ok=True) # Do this once before starting workers

            # 读取图像
            image = cv2.imread(image_file)
            if image is None:
                return False, f"无法读取图像: {os.path.basename(image_file)}"

            file_name = os.path.basename(image_file)
            base_name, ext = os.path.splitext(file_name)
            if not ext or ext.lower() not in ['.jpg', '.jpeg', '.png']:
                ext = '.png'

            height, width = image.shape[:2]
            grid_height = height // self.rows
            grid_width = width // self.cols
            successful_grids = 0
            file_lock = threading.Lock() # Lock for writing files from this task

            for row in range(self.rows):
                for col in range(self.cols):
                    y = row * grid_height
                    x = col * grid_width
                    h = height - y if row == self.rows - 1 else grid_height
                    w = width - x if col == self.cols - 1 else grid_width

                    grid_image = image[y:y+h, x:x+w]
                    grid_filename = f"{base_name}_r{row+1}c{col+1}{ext}"
                    grid_path = os.path.join(output_path, grid_filename)

                    with file_lock: # Protect file writing just in case
                        success = cv2.imwrite(grid_path, grid_image)
                        if success:
                            successful_grids += 1

            if successful_grids == self.rows * self.cols:
                return True, f"成功处理 {file_name}"
            elif successful_grids > 0:
                 # Log partial success as warning/error? For now, treat as failure for simplicity
                 return False, f"{file_name}: 部分成功 ({successful_grids}/{self.rows*self.cols})"
            else:
                return False, f"{file_name}: 所有网格处理失败"

        except Exception as e:
            return False, f"处理 {os.path.basename(image_file)} 出错: {e}"

    # --- Removed _process_grid_image_external static method ---

# 注册退出时线程池清理函数
def cleanup_thread_pool():
    """程序退出时清理线程池资源"""
    global GLOBAL_THREAD_POOL
    if GLOBAL_THREAD_POOL is not None:
        try:
            GLOBAL_THREAD_POOL.shutdown(wait=False)
            GLOBAL_THREAD_POOL = None
        except Exception as e:
            logging.error(f"关闭线程池时出错: {str(e)}")

# 注册退出处理函数
atexit.register(cleanup_thread_pool)


class BatchImageProcessingWorker(QObject):
    """批量图像变化检测工作线程类"""
    
    def __init__(self, data, endpoint_url):
        """初始化批处理工作线程"""
        super().__init__()
        self.data = data
        self.endpoint_url = endpoint_url # May not be needed if using path_connector.detect_changes
        self.signals = BatchImageProcessingSignals()
        self.is_running = False
    
    def run(self):
        """执行批处理任务"""
        if self.is_running:
            return
            
        self.is_running = True
        try:
            # 导入路径连接器 (keep local import just in case)
            try:
                from change3d_api_docker.path_connector import path_connector
            except ImportError:
                try:
                    from change3d_api_docker.path_connector import path_connector
                except ImportError:
                    path_connector = None
                    
            if path_connector is None or not path_connector.check_connection():
                self.signals.error.emit("无法连接到API服务器，请检查连接后重试")
                self.is_running = False
                return

            # 提取数据
            before_path = self.data.get("before_path", "")
            after_path = self.data.get("after_path", "")
            output_path = self.data.get("output_path", "")
            mode = self.data.get("mode", "batch_image")


            self.signals.log.emit(f"等待最终结果...")
            final_task_result = path_connector.detect_changes(
                before_path=before_path,
                after_path=after_path,
                output_path=output_path,
                mode=mode
            )

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
            # 使用与栅格一致的错误消息格式
            self.signals.error.emit(f"调用批处理API时出错: {str(e)}\n{error_info}")
        finally:
            self.is_running = False

class BatchImageProcessingSignals(QObject):
    """批处理信号类"""
    finished = Signal(dict)  # 处理完成信号，传递结果
    error = Signal(str)      # 错误信号
    log = Signal(str)        # 日志信号
