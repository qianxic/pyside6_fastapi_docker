from PySide6.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QPushButton, QFrame, QGroupBox, QTextEdit, QGridLayout, 
    QMessageBox, QMenuBar, QMenu, QStatusBar, QFileDialog, QGraphicsView,
    QGraphicsScene, QCheckBox, QRadioButton, QComboBox, QDialog, QSplitter,
    QLineEdit, QStackedWidget, QSizePolicy, QTextBrowser, QInputDialog
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QMetaObject, Q_ARG, Slot, QPoint
from PySide6.QtGui import QIcon, QPixmap, QImage, QFont, QAction, QCursor, QColor, QPainter, QLinearGradient, QBrush
import os
import sys
import time
import logging
import platform
import threading
import numpy as np
import cv2
import traceback
from datetime import datetime
from pathlib import Path
from functools import partial
from PIL import Image
from .display import NavigationFunctions, ZoomableLabel
from .theme_manager import ThemeManager
from .function.import_before_image import ImportBeforeImage
from .function.import_after_image import ImportAfterImage
from .function.change_cd import ExecuteChangeDetectionTask
from .function.clear_task import ClearTask
from .function.fishnet_fenge import GridCropping
from .function.image_display import ImageDisplay
from display import NavigationFunctions, ZoomableLabel
from theme_manager import ThemeManager
from .function import ImportBeforeImage,ImportAfterImage,ExecuteChangeDetectionTask,ClearTask,GridCropping,ImageDisplay


'''
主要功能：应用程序的主界面和核心逻辑
实现了RemoteSensingApp主窗口类：整个应用的核心框架
实现了HomePage首页界面类：显示欢迎信息和进入主界面按钮
创建并管理各种UI组件：图像显示区域、按钮导航栏、日志区域等
初始化、组织和连接各个功能模块
实现主题切换、帮助显示等应用级功能
处理图像导入、处理和结果导出等操作

'''

class HomePage(QWidget):
    def __init__(self, parent=None, is_dark_theme=False):
        """初始化首页界面"""
        super().__init__(parent)
        self.is_dark_theme = is_dark_theme
        # --- Add flags to track background task completion --- 
        self.docker_ready = False
        self.cleanup_done = False
        self.docker_success = False # Track success specifically
        self.cleanup_success = False # Track success specifically
        # --- 
        self.init_ui()
        
        # 设置与主页面一致的大小策略，确保尺寸一致性
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 信号会在应用程序完全初始化后连接
    
    def get_image_mode_button(self):
        """获取图像版按钮"""
        return self.btn_image_mode
    
    def get_raster_mode_button(self):
        """获取影像版按钮"""
        return self.btn_raster_mode
    
    def set_buttons_enabled(self, enabled=True):
        """设置按钮是否可用"""
        if hasattr(self, 'btn_image_mode') and hasattr(self, 'btn_raster_mode'):
             self.btn_image_mode.setEnabled(enabled)
             self.btn_raster_mode.setEnabled(enabled)
        
    def show_loading_message(self, message):
        """显示加载消息"""
        if hasattr(self, 'loading_label'):
            self.loading_label.setText(message)
            self.loading_label.show()
        
    def hide_loading_message(self):
        """隐藏加载消息"""
        if hasattr(self, 'loading_label'):
            self.loading_label.hide()
    
    @Slot(str)
    def update_loading_message(self, message):
        if hasattr(self, 'loading_label'):
            # Display the latest message from background tasks
            # Add timestamp for clarity?
            # timestamp = datetime.now().strftime("%H:%M:%S")
            # self.loading_label.setText(f"[{timestamp}] {message}")
            self.loading_label.setText(message)
            print(f"UI Status Update: {message}") # Console log for debugging
            # Ensure the UI updates
            QApplication.processEvents()

    @Slot(bool, str)
    def handle_task_completion(self, success, task_name):
        print(f"Task Completed: {task_name}, Success: {success}") # Console log
        task_finished_successfully = False

        if "清理" in task_name:
            self.cleanup_done = True
            self.cleanup_success = success # Store success state
            if not success:
                 self.update_loading_message("[错误] 清理共享目录失败。请检查权限或文件占用情况。")
        elif "Docker" in task_name:
            self.docker_ready = True
            self.docker_success = success # Store success state
            if not success:
                 self.update_loading_message("[错误] Docker容器或API服务启动失败。请检查Docker是否运行及网络连接。")

        # Check if BOTH tasks are finished
        if self.docker_ready and self.cleanup_done:
            # Check if BOTH tasks were successful
            if self.docker_success and self.cleanup_success:
                 print("后台任务成功完成，启用按钮。")
                 self.set_buttons_enabled(True)
                 self.hide_loading_message() # Hide loading message on full success
            else:
                 print("后台任务完成，但至少有一个失败，按钮保持禁用。")
                 self.set_buttons_enabled(False)
                 # Keep the last error message visible in loading_label
                 # Optionally, consolidate error messages here
                 error_msg = "初始化失败: "
                 if not self.cleanup_success: error_msg += "清理目录失败。"
                 if not self.docker_success: error_msg += "Docker/API启动失败。"
                 self.update_loading_message(error_msg)
    
    def init_ui(self):
        """初始化首页UI"""
        # 首先清除现有布局和小部件
        if self.layout():
            # 递归删除所有子组件
            def deleteItems(layout):
                if layout is not None:
                    while layout.count():
                        item = layout.takeAt(0)
                        widget = item.widget()
                        if widget is not None:
                            widget.deleteLater()
                        else:
                            deleteItems(item.layout())
            deleteItems(self.layout())
            # 删除旧布局
            QWidget().setLayout(self.layout())
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 获取当前主题颜色
        colors = ThemeManager.get_colors(self.is_dark_theme)

        # 设置窗口背景和文本颜色
        self.setStyleSheet(f"background-color: {colors['background']}; color: {colors['text']};")
        
        # 系统标题
        self.title_label = QLabel("遥感影像变化检测系统 V1.1") # Store reference
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Microsoft YaHei UI", 24, QFont.Bold))
        self.title_label.setStyleSheet(f"color: {colors['text']};") # Use theme color
        
        # 系统描述
        self.description = QLabel("本系统提供遥感影像的变化检测功能，支持图像标准化、网格分割、变化检测等多种功能。") # Store reference
        self.description.setAlignment(Qt.AlignCenter)
        self.description.setFont(QFont("Microsoft YaHei UI", 12))
        self.description.setWordWrap(True)
        self.description.setStyleSheet(f"color: {colors['text']};") # Use theme color
        
        # 两个模式选择按钮
        self.btn_image_mode = QPushButton("图像版")
        self.btn_image_mode.setObjectName("imageVersionButton")
        self.btn_image_mode.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        self.btn_image_mode.setMinimumSize(120, 40)
        
        self.btn_raster_mode = QPushButton("影像版")
        self.btn_raster_mode.setObjectName("rasterVersionButton")
        self.btn_raster_mode.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        self.btn_raster_mode.setMinimumSize(120, 40)
        
        # 使用主题管理器获取按钮样式
        self.btn_image_mode.setStyleSheet(ThemeManager.get_primary_button_style(self.is_dark_theme))
        self.btn_raster_mode.setStyleSheet(ThemeManager.get_primary_button_style(self.is_dark_theme))
        
        # 添加组件到主布局
        main_layout.addStretch(1)
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.description)
        main_layout.addStretch(1)
        
        # 模式选择说明
        self.mode_description = QLabel("请选择您要使用的模式:") # Store reference
        self.mode_description.setAlignment(Qt.AlignCenter)
        self.mode_description.setFont(QFont("Microsoft YaHei UI", 11))
        self.mode_description.setStyleSheet(f"color: {colors['text']};") # Use theme color
        main_layout.addWidget(self.mode_description)
        
        # 模式选择提示
        self.image_mode_tip = QLabel("图像版：用于普通图像变化检测，不生成矢量边界") # Store reference
        self.image_mode_tip.setAlignment(Qt.AlignCenter)
        self.image_mode_tip.setFont(QFont("Microsoft YaHei UI", 9))
        self.image_mode_tip.setStyleSheet(f"color: {colors['secondary_text']};") # Use secondary theme color
        
        self.raster_mode_tip = QLabel("影像版：用于栅格影像变化检测，可生成矢量边界(.shp)文件") # Store reference
        self.raster_mode_tip.setAlignment(Qt.AlignCenter)
        self.raster_mode_tip.setFont(QFont("Microsoft YaHei UI", 9))
        self.raster_mode_tip.setStyleSheet(f"color: {colors['secondary_text']};") # Use secondary theme color
        
        main_layout.addWidget(self.image_mode_tip)
        main_layout.addWidget(self.raster_mode_tip)
        main_layout.addSpacing(10)
        
        # 按钮并排布局
        btn_container = QHBoxLayout()
        btn_container.addStretch(1)
        btn_container.addWidget(self.btn_image_mode)
        btn_container.addSpacing(40)  # 两个按钮之间增加间距
        btn_container.addWidget(self.btn_raster_mode)
        btn_container.addStretch(1)
        main_layout.addLayout(btn_container)
        
        # 创建加载提示标签 - 放在左下角
        self.loading_label = QLabel("正在初始化系统组件...") # Initial message
        self.loading_label.setAlignment(Qt.AlignCenter) # Center align maybe better
        self.loading_label.setFont(QFont("Microsoft YaHei UI", 10))
        self.loading_label.setStyleSheet(f"color: {colors['secondary_text']};") # Use secondary theme color
        self.loading_label.setWordWrap(True)
        self.loading_label.show() # Make sure it's visible initially
        
        # 添加伸缩空间，确保loading_label在底部
        main_layout.addStretch(1)
        # 添加加载标签到布局底部
        main_layout.addWidget(self.loading_label)
        
        # No need to re-apply styles here as they were set using theme colors above
        
    def update_theme(self, is_dark_theme):
        """更新主题"""
        if self.is_dark_theme != is_dark_theme:
            self.is_dark_theme = is_dark_theme
            # 获取新主题颜色
            colors = ThemeManager.get_colors(self.is_dark_theme)

            # 更新基础样式
            self.setStyleSheet(f"background-color: {colors['background']}; color: {colors['text']};")

            # 更新所有标签的样式
            if hasattr(self, 'title_label'):
                 self.title_label.setStyleSheet(f"color: {colors['text']};")
            if hasattr(self, 'description'):
                 self.description.setStyleSheet(f"color: {colors['text']};")
            if hasattr(self, 'mode_description'):
                 self.mode_description.setStyleSheet(f"color: {colors['text']};")
            if hasattr(self, 'image_mode_tip'):
                 self.image_mode_tip.setStyleSheet(f"color: {colors['secondary_text']};")
            if hasattr(self, 'raster_mode_tip'):
                 self.raster_mode_tip.setStyleSheet(f"color: {colors['secondary_text']};")
            if hasattr(self, 'loading_label'):
                 self.loading_label.setStyleSheet(f"color: {colors['secondary_text']};")

            # 更新按钮样式
            if hasattr(self, 'btn_image_mode'):
                 self.btn_image_mode.setStyleSheet(ThemeManager.get_primary_button_style(self.is_dark_theme))
            if hasattr(self, 'btn_raster_mode'):
                 self.btn_raster_mode.setStyleSheet(ThemeManager.get_primary_button_style(self.is_dark_theme))

class RemoteSensingApp(QMainWindow):
    def __init__(self):
        """初始化GUI界面"""
        super().__init__()
        
        # 配置日志
        
        # 设置窗口属性
        self.setWindowTitle("遥感影像变化检测系统 V1.0")
        self.setGeometry(50, 50, 800, 500)
        
        # 注释掉这一行来恢复最大化窗口功能
        # self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        
        # 主题变量 (默认设为浅色主题)
        self.is_dark_theme = False
        
        # 应用当前主题
        self.apply_theme()
        
        # 初始化状态栏
        self.statusBar().showMessage("正在加载应用组件...")
        
        # 创建中央控件和主栈式布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建栈式窗口管理器
        self.stacked_widget = QStackedWidget(self.central_widget)
        
        # 创建首页和主界面容器（先不初始化主界面内容，提高启动速度）
        self.home_page = HomePage(self.stacked_widget, self.is_dark_theme)
        self.main_page = QWidget(self.stacked_widget)
        
        # 确保两个页面使用相同的大小策略
        self.home_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 将页面添加到栈式窗口
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.main_page)
        
        # 设置中央布局
        central_layout = QVBoxLayout(self.central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(self.stacked_widget)
        
        # 默认显示首页
        self.stacked_widget.setCurrentIndex(0)
        
        # 暂时禁用首页按钮，等所有模块加载完成后再启用
        self.home_page.set_buttons_enabled(False)
        
        # 初始化标志
        self.main_page_initialized = False
        self.modules_initialized = False
        
        # 创建空的navigation_functions属性，等到实际需要时再初始化
        self.navigation_functions = None
        
        # 连接首页按钮信号
        self.connect_home_page_signals()
        
        # 预加载服务器模型
        self.preload_server_model()
        
        # 使用单个定时器启动同步初始化过程
        QTimer.singleShot(100, self.initialize_all_components)
        
    def preload_server_model(self):
        """从服务器预加载模型检查点权重"""
        preload_thread = threading.Thread(target=self._load_model_checkpoint, daemon=True)
        preload_thread.start()
    
    def _load_model_checkpoint(self):
        """在后台线程中执行模型检查点加载"""
        try:
            # 导入必要的模块
            from change3d_api.path_connector import path_connector
            
            # 创建测试图像进行预热
            import os
            import cv2
            import numpy as np
            import tempfile
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp()
            
            # 创建测试图像 - 使用256x256尺寸而不是64x64，以匹配模型预期的处理块大小
            test_img1 = np.ones((256, 256, 3), dtype=np.uint8) * 200
            test_img2 = np.ones((256, 256, 3), dtype=np.uint8) * 100
            test_img2[50:150, 50:150, :] = 250  # 更大的变化区域
            
            # 保存测试图像
            test_img1_path = os.path.join(temp_dir, "test_before.png")
            test_img2_path = os.path.join(temp_dir, "test_after.png")
            output_path = os.path.join(temp_dir, "test_result.png")
            
            cv2.imwrite(test_img1_path, test_img1)
            cv2.imwrite(test_img2_path, test_img2)
            
            # 直接调用API进行变化检测，触发模型加载
            result = path_connector.detect_single_image(
                before_path=test_img1_path,
                after_path=test_img2_path,
                output_path=output_path
            )
            
            # 等待任务完成
            if result.get("task_id"):
                path_connector.wait_for_task_completion(result.get("task_id"))
            
            # 清理临时文件
            try:
                os.remove(test_img1_path)
                os.remove(test_img2_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rmdir(temp_dir)
            except Exception:
                pass
                
        except Exception:
            pass
    
    def initialize_all_components(self):
        """初始化所有组件，包括主界面和栅格模块"""
        try:
            # 设置进度状态
            self.statusBar().showMessage("正在初始化主界面...", 2000)
            self.home_page.show_loading_message("正在初始化主界面...")
            
            # 初始化主界面 UI
            if not self.main_page_initialized:
                self.init_main_page()
                self.main_page_initialized = True
            
            # 初始化基础功能模块
            if not hasattr(self, 'modules_initialized') or not self.modules_initialized:
                self.statusBar().showMessage("正在初始化功能模块...", 2000)
                self.home_page.show_loading_message("正在初始化功能模块...")
                self.init_function_modules()
            
            # 初始化栅格模块
            self.statusBar().showMessage("正在加载栅格影像处理模块...", 2000)
            self.home_page.show_loading_message("正在加载栅格影像处理模块...")
            self.initialize_raster_modules()
            
            # 所有模块加载完成后，启用首页按钮
            self.statusBar().showMessage("系统初始化完成", 2000)
            # 删除初始化完成的提示信息显示
            self.home_page.hide_loading_message()
            self.home_page.set_buttons_enabled(True)
            
        except Exception as e:
           pass
    
    def initialize_raster_modules(self):
        """初始化栅格模块"""
        try:
            # 首先检查GDAL是否可用
            try:
                # 尝试导入GDAL
                try:
                    import gdal
                    gdal_available = True
                    print("成功导入GDAL")
                except ImportError:
                    try:
                        from osgeo import gdal
                        gdal_available = True
                    except ImportError:
                        gdal_available = False
                        return
                
                # GDAL可用，继续加载栅格模块
                self.gdal_available = gdal_available
                
                # 加载栅格模块
                if self.navigation_functions is not None:
                    from zhuyaogongneng_docker.function.raster import RasterChangeDetection
                    from zhuyaogongneng_docker.function.raster import RasterBatchProcessor
                    from zhuyaogongneng_docker.function.raster.import_module import RasterImporter
                    from zhuyaogongneng_docker.function.raster.grid import RasterGridCropping
                    
                    # 初始化各种栅格处理类
                    self.raster_cd = RasterChangeDetection(self.navigation_functions)
                    self.raster_batch_processor = RasterBatchProcessor(self.navigation_functions)
                    self.raster_importer = RasterImporter(self.navigation_functions)
                    self.raster_grid_cropping = RasterGridCropping(self.navigation_functions)
                    
                    # 设置ShapefileGenerator和RasterExporter为None，以避免后续使用时报错
                    self.shp_generator = None
                    self.raster_exporter = None
                    
                    # 标记栅格模块已加载
                    self.raster_modules_loaded = True
                    
                    # 加载次要模块
                    self._load_secondary_raster_modules()
                else:
                    print("导航功能尚未初始化，无法加载栅格模块")
                    self.statusBar().showMessage("导航功能初始化失败，部分功能不可用", 3000)
            
            except Exception as e:
                print(f"加载GDAL和栅格模块时出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
                self.statusBar().showMessage("加载栅格模块时出错，部分功能不可用", 3000)
        
        except Exception as e:
            print(f"初始化栅格模块时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.statusBar().showMessage("初始化过程中出错，部分功能可能不可用", 3000)
    
    def connect_home_page_signals(self):
        """连接首页按钮事件"""
        # 确保首页UI存在
        if not hasattr(self, 'home_page') or self.home_page is None:
            print("错误: 首页UI还未创建")
            return False
        
        # 切换到图像模式前先清理界面
        def switch_to_image_mode():
            # 如果清理任务已初始化，先清理界面
            if hasattr(self, 'clear_task') and self.clear_task is not None:
                self.clear_task.clear_interface()
            # 切换到图像模式
            self.switch_to_main_page("image")
            
        # 切换到栅格模式前先清理界面
        def switch_to_raster_mode():
            # 如果清理任务已初始化，先清理界面
            if hasattr(self, 'clear_task') and self.clear_task is not None:
                self.clear_task.clear_interface()
            # 切换到栅格模式
            self.switch_to_main_page("raster")
        
        # 连接图像版和影像版按钮信号到清理界面的函数
        self.home_page.get_image_mode_button().clicked.connect(switch_to_image_mode)
        self.home_page.get_raster_mode_button().clicked.connect(switch_to_raster_mode)
        
        # 不显示状态栏消息
        # self.statusBar().showMessage("正在加载系统组件，请稍候...", 10000)
        
        return True

    def init_main_page(self):
        """初始化主界面页面"""
        
        # 创建布局
        main_layout = QVBoxLayout(self.main_page)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # 创建顶部按钮导航栏
        self.create_button_group(main_layout)
        
        # 创建主分割器，垂直分割上下两部分
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setChildrenCollapsible(False) # 防止区域被完全折叠
        main_layout.addWidget(main_splitter)
        
        # 创建上部窗口容器
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        
        # 创建下部窗口容器 - 使用相同的设计策略
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)
        
        # 添加到主分割器
        main_splitter.addWidget(top_widget)
        main_splitter.addWidget(bottom_widget)
        
        # 创建标签和组
        self.create_before_image_group(top_layout)
        self.create_after_image_group(top_layout)
        
        # 创建日志区域和输出结果区域（左右并排）- 使用与上半部分相同的策略
        log_group = self.create_log_group(None)
        output_group = self.create_output_group(None)
        
        # 直接添加到底部布局，不使用分割器，但保持1:3的比例（使用权重1和3）
        bottom_layout.addWidget(log_group, 1)  # 设置比例权重为1
        bottom_layout.addWidget(output_group, 3)  # 设置比例权重为3，保持1:3比例
        
        # 设置固定的分割比例
        main_splitter.setSizes([3, 2])  # 按3:2比例分割
        
        # 保存分割器引用，以便在窗口调整大小时维持比例
        self.main_splitter = main_splitter
        
        # 初始化功能模块 (会创建NavigationFunctions实例)
        success = self.init_function_modules()
        
        # 连接按钮点击事件
        if success:
            self.connect_buttons()
        else:
            print("由于功能模块初始化失败，按钮事件连接可能不完整")

        # 初始化后立即应用按钮样式
        self.apply_button_styles()
        
    
    def switch_to_main_page(self, mode="image"):
        """切换到主界面，支持异步加载"""
        print(f"开始切换到主界面... 模式: {mode}")
        # 保存当前模式
        self.current_mode = mode
        
        # 设置鼠标为等待状态，提示用户正在加载
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # 检查主界面是否已初始化
            if not self.main_page_initialized:
                self.statusBar().showMessage("正在初始化主界面...", 2000)
                self.init_main_page()
                self.main_page_initialized = True
                # 根据初始化结果显示适当的消息
                if hasattr(self, 'modules_initialized') and self.modules_initialized:
                    self.statusBar().showMessage("功能模块加载完成", 2000)
                else:
                    self.statusBar().showMessage("部分功能模块可能不可用", 3000)
            
            # 确保日志使用正确的颜色
            if hasattr(self, 'text_log'):
                # 刷新日志文本颜色
                self.refresh_log_text_color()
            
            # 如果从一个模式切换到另一个模式，先清理界面
            if hasattr(self, 'previous_mode') and self.previous_mode != mode and self.previous_mode is not None:
                # 调用清理任务来清除界面
                if hasattr(self, 'clear_task') and self.clear_task is not None:
                    print(f"正在清理界面，从 {self.previous_mode} 模式切换到 {mode} 模式")
                    self.clear_task.clear_interface()
                    # 确保界面更新
                    QApplication.processEvents()
            
            # 记录当前模式，用于下次切换比较
            self.previous_mode = mode
            
            # 根据模式更新UI和功能
            self.update_ui_for_mode(mode)
            
            # 如果是栅格模式，确保栅格模块已加载 - 如果是预加载模式，这里可能已经完成
            if mode == "raster" and (not hasattr(self, 'raster_modules_loaded') or not self.raster_modules_loaded):
                # 尝试使用栅格模块，如果未加载会触发加载
                self.load_raster_specific_modules()
            
            # 更新窗口标题
            title_suffix = "图像版" if mode == "image" else "影像版"
            self.setWindowTitle(f"遥感影像变化检测系统 V1.0 - {title_suffix}")
                
            # 切换到主界面页
            self.stacked_widget.setCurrentIndex(1)
            
            # 记录主界面已加载 - 减少日志输出
            if self.navigation_functions is not None:
                # 避免重复日志
                #self.navigation_functions.log_message(f"已切换至主界面 ({title_suffix})")
                pass
                
            # 简短提示消息
            if not self.statusBar().currentMessage():
                self.statusBar().showMessage(f"主界面已加载，当前模式: {title_suffix}", 2000)
            
            # 强制重新连接按钮，确保正确的模式特定功能被连接
            self.connect_buttons()
            
                  
            print("主界面切换完成")
 
        except Exception as e:
            import traceback
            print(f"切换到主界面时出错: {str(e)}")
            print(traceback.format_exc())
        finally:
            # 恢复鼠标状态
            QApplication.restoreOverrideCursor()

    def load_mode_specific_modules(self, mode):
        """加载特定模式的模块和功能 (兼容保留，预加载模式下不会被调用)"""
        try:
            print(f"加载{mode}模式特定模块...")
            
            if mode == "image":
                # 图像模式 - 确保只加载图像模式需要的模块
                if hasattr(self, 'navigation_functions'):
                    # 减少日志输出
                    # self.navigation_functions.log_message("加载图像版功能模块")
                    pass
                    
                # 确保原有功能模块可用
                if not hasattr(self, 'import_before_image') or self.import_before_image is None:
                    self.init_function_modules()
                
                # 清理可能存在的栅格模块引用，避免干扰 (不再需要清理，预加载模式下保留所有模块)
                # for attr in ['raster_cd', 'raster_importer', 'raster_batch_processor', 
                #            'raster_exporter', 'shp_generator']:
                #     if hasattr(self, attr):
                #         delattr(self, attr)
                
            elif mode == "raster":
                # 影像模式 - 需要支持生成SHP文件的额外功能
                if hasattr(self, 'navigation_functions'):
                    # 减少日志输出
                    # self.navigation_functions.log_message("加载影像版功能模块")
                    pass
                
                # 确保原有功能模块可用
                if not hasattr(self, 'import_before_image') or self.import_before_image is None:
                    self.init_function_modules()
                
                # 加载栅格影像特有的处理模块
                # 在预加载模式下，栅格模块已在启动时加载，这里只检查是否加载成功
                if not hasattr(self, 'raster_modules_loaded') or not self.raster_modules_loaded:
                    success = self.load_raster_specific_modules()
                    if not success:
                        print("栅格处理模块加载失败，某些功能可能不可用")
                        if hasattr(self, 'navigation_functions'):
                            self.navigation_functions.log_message("栅格处理模块加载失败，某些功能可能不可用")
                            self.navigation_functions.log_message("请安装GDAL库 (pip install GDAL) 并确保依赖项配置正确")
                            # 显示状态栏信息
                            self.statusBar().showMessage("栅格处理功能不可用，请安装GDAL库", 5000)
            
            # 更新UI元素以匹配当前模式
            self.update_ui_for_mode(mode)
            
        except Exception as e:
            print(f"加载模式特定模块时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
    def load_raster_specific_modules(self):
        """加载栅格影像处理所需的特定模块"""
        # 如果已经加载过影像版模块，直接返回
        if hasattr(self, 'raster_modules_loaded') and self.raster_modules_loaded:
            return True
        
        # 如果已经预加载且GDAL可用，使用complete_raster_modules_init完成初始化
        if hasattr(self, 'raster_modules_preloaded') and self.raster_modules_preloaded and \
           hasattr(self, 'gdal_available') and self.gdal_available:
            return self.complete_raster_modules_init()
            
        try:
            # 先检查GDAL是否可用
            try:
                import gdal
                gdal_available = True
            except ImportError:
                try:
                    from osgeo import gdal
                    gdal_available = True
                except ImportError:
                    gdal_available = False
                    self.navigation_functions.log_message("警告: GDAL库未安装，栅格影像处理功能将不可用")
                    return False
            
            if not gdal_available:
                self.navigation_functions.log_message("无法加载栅格处理模块: GDAL库缺失")
                return False
            
            # 使用单独的线程延迟加载模块
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # 保存GDAL可用状态
            self.gdal_available = gdal_available
            
            # 完成初始化的剩余部分
            return self.complete_raster_modules_init()
            
        except Exception as e:
            import traceback
            self.navigation_functions.log_message(f"加载栅格模块失败: {str(e)}", "ERROR")
            self.navigation_functions.log_message(traceback.format_exc(), "ERROR")
            
            # 恢复鼠标
            QApplication.restoreOverrideCursor()
            return False
    
    def _load_secondary_raster_modules(self):
        """延迟加载次要栅格处理模块"""
        try:
            # 使用模块列表来批量导入
            secondary_modules = [
                "zhuyaogongneng.function.raster.utils",
                "zhuyaogongneng.function.raster.visualization"
            ]
            
            for module_name in secondary_modules:
                try:
                    __import__(module_name)
                except ImportError:
                    pass
        except Exception:
            pass

    def update_ui_for_mode(self, mode):
        """根据当前模式更新UI元素"""
        # 获取当前模式的显示文本
        mode_text = "图像版" if mode == "image" else "影像版"
        
        # 如果已经初始化了主页面相关的UI组件，更新它们
        if hasattr(self, 'btn_begin') and self.btn_begin is not None:
            if mode == "image":
                self.btn_begin.setText("开始解译")
            else:
                self.btn_begin.setText("开始检测")
                
        # 更新组标题以反映当前模式
        if hasattr(self, 'group_output') and self.group_output is not None:
            title = "解译结果" if mode == "image" else "变化检测结果"
            self.group_output.setTitle(title)
        
        # 记录模式切换
        if hasattr(self, 'navigation_functions') and self.navigation_functions is not None:
            self.navigation_functions.log_message(f"已切换到{mode_text}模式")

    def create_before_image_group(self, parent_layout):
        """创建前时相影像组"""
        self.group_before = QGroupBox("前时相影像")
        layout_before = QVBoxLayout(self.group_before)
        layout_before.setContentsMargins(8, 16, 8, 8)  # 增加内边距
        
        # 创建可缩放标签
        self.label_before = ZoomableLabel()
        self.label_before.setAlignment(Qt.AlignCenter)
        self.label_before.setText("前时相影像")
        
        # 使用主题管理器设置样式
        self.label_before.setStyleSheet(ThemeManager.get_image_label_style(self.is_dark_theme))
        
        # 修改最小尺寸，使其更合理且能自动适应窗口
        self.label_before.setMinimumSize(300, 300)
        
        # 使用更稳定的尺寸策略，避免频繁调整
        self.label_before.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.group_before.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout_before.addWidget(self.label_before)
        if parent_layout:
            parent_layout.addWidget(self.group_before, 1)  # 设置比例权重为1，确保平分区域
    
    def create_after_image_group(self, parent_layout):
        """创建后时相影像组"""
        self.group_after = QGroupBox("后时相影像")
        layout_after = QVBoxLayout(self.group_after)
        layout_after.setContentsMargins(8, 16, 8, 8)  # 增加内边距
        
        # 创建可缩放标签
        self.label_after = ZoomableLabel()
        self.label_after.setAlignment(Qt.AlignCenter)
        self.label_after.setText("后时相影像")
        
        # 使用主题管理器设置样式
        self.label_after.setStyleSheet(ThemeManager.get_image_label_style(self.is_dark_theme))
        
        # 修改最小尺寸，使其更合理且能自动适应窗口
        self.label_after.setMinimumSize(300, 300)
        
        # 使用更稳定的尺寸策略，避免频繁调整
        self.label_after.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.group_after.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout_after.addWidget(self.label_after)
        if parent_layout:
            parent_layout.addWidget(self.group_after, 1)  # 设置比例权重为1，确保平分区域
    
    def refresh_log_text_color(self):
        """刷新日志文本框中所有文本的颜色，使其与当前主题一致"""
        if hasattr(self, 'text_log'):
            # 获取当前文本并清空
            current_text = self.text_log.toPlainText()
            self.text_log.clear()
            
            # 根据当前主题确定文本颜色
            if self.is_dark_theme:
                self.text_log.setTextColor(Qt.GlobalColor.white)
            else:
                self.text_log.setTextColor(Qt.GlobalColor.black)
            
            # 直接添加文本，不使用HTML格式
            self.text_log.setPlainText(current_text)
    
    def create_log_group(self, parent_layout):
        """创建日志组"""
        self.group_log = QGroupBox("系统日志")
        layout_log = QVBoxLayout(self.group_log)
        layout_log.setContentsMargins(8, 16, 8, 8)  # 增加内边距
        
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setFont(QFont("Microsoft YaHei UI", 9))  # 设置合适的字体和大小
        
        # 设置文本颜色为白色或黑色，根据主题
        if self.is_dark_theme:
            self.text_log.setTextColor(Qt.GlobalColor.white)
        else:
            self.text_log.setTextColor(Qt.GlobalColor.black)
            
        # 设置文本框样式
        self.text_log.setStyleSheet(ThemeManager.get_log_text_style(self.is_dark_theme))
        
        # 设置尺寸策略，确保可以自由缩放但保持合理的最小尺寸
        self.text_log.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.group_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 设置最小尺寸以确保可见性
        self.text_log.setMinimumSize(200, 300)
        # 为日志组设置最小尺寸
        self.group_log.setMinimumSize(250, 350)
        
        layout_log.addWidget(self.text_log)
        
        if parent_layout:
            parent_layout.addWidget(self.group_log, 1)  # 设置比例权重为1，与解译结果1:3比例
        
        return self.group_log
    
    def create_output_group(self, parent_layout):
        """创建输出结果组"""
        self.group_output = QGroupBox("解译结果")
        layout_output = QVBoxLayout(self.group_output)
        layout_output.setContentsMargins(8, 16, 8, 8)  # 增加内边距
        
        # 创建可缩放标签
        self.label_result = ZoomableLabel()
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setText("未生成结果")
        
        # 使用主题管理器设置样式
        self.label_result.setStyleSheet(ThemeManager.get_image_label_style(self.is_dark_theme))
        
        # 修改最小尺寸，使其更合理且能自动适应窗口
        self.label_result.setMinimumSize(300, 300)
        
        # 使用更稳定的尺寸策略，避免频繁调整
        self.label_result.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.group_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 设置解译结果组的最小尺寸
        self.group_output.setMinimumSize(450, 350)
        
        # 添加标签到布局
        layout_output.addWidget(self.label_result)
        
        # 添加到父布局
        if parent_layout:
            parent_layout.addWidget(self.group_output, 3)  # 设置比例权重为3，与日志区域1:3比例
        
        return self.group_output
    
    def show_help(self):
        """显示帮助信息对话框"""
        # 创建帮助对话框
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("系统帮助")
        help_dialog.setWindowIcon(QIcon(":/icons/help.png"))
        help_dialog.setMinimumSize(700, 500)
        
        # 设置对话框布局
        layout = QVBoxLayout()
        help_dialog.setLayout(layout)
        
        # 创建HTML帮助内容
        help_text = QTextBrowser()
        help_text.setOpenExternalLinks(True)
        
        # 根据当前模式显示不同的帮助内容
        current_mode = getattr(self, 'current_mode', 'image')
        
        if current_mode == "image":
            help_html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Microsoft YaHei UI', sans-serif; margin: 15px; }}
                    h2 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                    ul {{ list-style-type: square; }}
                    li {{ margin: 10px 0; }}
                    .note {{ background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 10px; margin: 15px 0; }}
                    .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 15px 0; }}
                </style>
            </head>
            <body>
                <h2>遥感影像变化检测系统使用指南</h2>
                <p>本系统用于处理和分析遥感影像，检测两个时相之间的变化。</p>
                
                <h3>基本操作流程：</h3>
                <ol>
                    <li><b>渔网分割</b>: 对大尺寸影像进行裁剪，便于分析处理。</li>
                    <li><b>导入前时相影像</b>: 导入较早的遥感影像。</li>
                    <li><b>导入后时相影像</b>: 导入较新的遥感影像。</li>
                    <li><b>开始解译</b>: 运行变化检测算法分析两个时相影像的差异。</li>
                    <li><b>批量处理</b>: 批量处理多个遥感影像对。</li>
                </ol>

                <h3>功能按钮说明：</h3>
                <ul>
                    <li><b>首页</b>: 返回系统首页。</li>
                    <li><b>渔网分割</b>: 将大尺寸影像分割成若干小块，便于处理和分析。</li>
                    <li><b>导入前时相影像</b>: 导入较早时期的影像。</li>
                    <li><b>导入后时相影像</b>: 导入较新时期的影像。</li>
                    <li><b>开始解译</b>: 运行变化检测算法，分析两个时相之间的变化。</li>
                    <li><b>批量处理</b>: 批量处理多对影像。</li>
                    <li><b>切换主题</b>: 在浅色和深色主题之间切换。</li>
                    <li><b>清空界面</b>: 清除当前加载的所有影像和分析结果。</li>
                    <li><b>帮助</b>: 显示本帮助信息。</li>
                </ul>
                
                <p>版本：1.0.0 | 开发者：qianxiR</p>
            </body>
            </html>
            """
        else:
            # 栅格模式帮助内容
            help_html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Microsoft YaHei UI', sans-serif; margin: 15px; }}
                    h2 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                    ul {{ list-style-type: square; }}
                    li {{ margin: 10px 0; }}
                    .note {{ background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 10px; margin: 15px 0; }}
                    .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 15px 0; }}
                </style>
            </head>
            <body>
                <h2>栅格影像处理模式使用指南</h2>
                <p>栅格模式专为处理GeoTIFF等栅格格式影像设计，支持保留地理坐标信息。</p>
                
                <h3>基本操作流程：</h3>
                <ol>
                    <li><b>渔网分割</b>: 对大尺寸栅格影像进行裁剪。</li>
                    <li><b>导入前时相影像</b>: 导入较早的栅格影像。</li>
                    <li><b>导入后时相影像</b>: 导入较新的栅格影像。</li>
                    <li><b>开始解译</b>: 运行变化检测算法分析两个时相栅格影像的差异。</li>
                    <li><b>批量处理</b>: 批量处理多个栅格影像对。</li>
                </ol>

                <h3>功能按钮说明：</h3>
                <ul>
                    <li><b>首页</b>: 返回系统首页。</li>
                    <li><b>渔网分割</b>: 将大尺寸栅格影像分割成若干小块，保留坐标信息。</li>
                    <li><b>导入前时相影像</b>: 导入较早时期的栅格影像。</li>
                    <li><b>导入后时相影像</b>: 导入较新时期的栅格影像。</li>
                    <li><b>开始解译</b>: 运行栅格变化检测算法，分析两个时相之间的变化。</li>
                    <li><b>批量处理</b>: 批量处理多对栅格影像。</li>
                    <li><b>切换主题</b>: 在浅色和深色主题之间切换。</li>
                    <li><b>清空界面</b>: 清除当前加载的所有栅格影像和分析结果。</li>
                    <li><b>帮助</b>: 显示本帮助信息。</li>
                </ul>
                
                <p>版本：1.0.0 | 开发者：qianxiR</p>
            </body>
            </html>
            """
            
        help_text.setHtml(help_html)
        layout.addWidget(help_text)
        
        # 创建关闭按钮
        close_button = QPushButton("关闭")
        close_button.setFixedSize(120, 32)
        close_button.setStyleSheet(ThemeManager.get_dialog_button_style(self.is_dark_theme))
        close_button.clicked.connect(help_dialog.close)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        # 显示对话框
        help_dialog.exec_()

    def resizeEvent(self, event):
        """处理窗口调整大小事件，确保所有子界面按比例调整"""
        # 调用父类的resizeEvent
        super().resizeEvent(event)
        
        # 获取当前窗口大小
        window_size = self.size()
        
        # 如果是主界面，确保所有元素按比例调整
        current_index = getattr(self, 'stacked_widget', None)
        if current_index and current_index.currentIndex() == 1:  # 主界面索引为1
            # 调整主界面的布局
            self._adjust_main_page_layout(window_size)
        
        # 通知所有观察者窗口尺寸变化
        if hasattr(self, 'resize_observers'):
            for observer in self.resize_observers:
                try:
                    if callable(observer):
                        observer(window_size)
                except Exception as e:
                    print(f"通知尺寸变化观察者出错: {str(e)}")
    
    def add_resize_observer(self, observer):
        """添加窗口尺寸变化的观察者
        
        Args:
            observer: 一个接收QSize参数的可调用对象
        """
        if not hasattr(self, 'resize_observers'):
            self.resize_observers = []
            
        if callable(observer) and observer not in self.resize_observers:
            self.resize_observers.append(observer)
    
    def remove_resize_observer(self, observer):
        """移除窗口尺寸变化的观察者
        
        Args:
            observer: 要移除的观察者
        """
        if hasattr(self, 'resize_observers') and observer in self.resize_observers:
            self.resize_observers.remove(observer)
    
    def _adjust_main_page_layout(self, window_size):
        """调整主界面布局大小
        
        Args:
            window_size: 当前窗口大小
        """
        # 如果主分割器已初始化，调整分割比例
        if hasattr(self, 'main_splitter'):
            # 主分割器保持3:2的比例
            main_sizes = self.main_splitter.sizes()
            if sum(main_sizes) > 0:  # 防止除零错误
                current_ratio = main_sizes[0] / sum(main_sizes)
                target_ratio = 0.6  # 3/(3+2) = 0.6，即3:2的比例
                
                if abs(current_ratio - target_ratio) > 0.03:  # 比例偏离超过3%时调整
                    total_size = sum(main_sizes)
                    self.main_splitter.setSizes([int(total_size * target_ratio), int(total_size * (1 - target_ratio))])
            
        # 确保所有组件的最小尺寸
        self._ensure_minimum_sizes()
            
        # 调整图像显示区域的大小和比例
        self._adjust_image_display_areas()
    
    def _adjust_image_display_areas(self):
        """调整图像显示区域的大小和比例"""
        # 确保所有图像显示区域的最小尺寸保持一致
        min_size = 300
        
        # 调整前后时相和结果图像显示区域
        for label in [self.label_before, self.label_after, self.label_result]:
            if hasattr(label, 'setMinimumSize'):
                # 设置最小尺寸
                label.setMinimumSize(min_size, min_size)
                
                # 重新计算图像显示比例（如果有原始图像）
                if hasattr(label, 'original_pixmap') and label.original_pixmap is not None:
                    # 如果ZoomableLabel有_calculate_initial_scale方法，调用它重新计算比例
                    if hasattr(label, '_calculate_initial_scale'):
                        label._calculate_initial_scale()
                    
                    # 如果ZoomableLabel有update_display方法，调用它更新显示
                    if hasattr(label, 'update_display'):
                        label.update_display()
        
        # 确保所有组件的最小尺寸
        self._ensure_minimum_sizes()
        
        # 确保处理完后更新所有UI元素
        QApplication.processEvents()
    
    def _ensure_minimum_sizes(self):
        """确保所有UI组件的最小尺寸设置得到尊重"""
        # 确保日志组和解译结果组的最小尺寸设置得到尊重
        if hasattr(self, 'group_log'):
            self.group_log.setMinimumSize(250, 350)
            
        if hasattr(self, 'group_output'):
            self.group_output.setMinimumSize(450, 350)
            
        # 确保日志文本区域的最小尺寸
        if hasattr(self, 'text_log'):
            self.text_log.setMinimumSize(200, 300)
            
    def init_function_modules(self):
        """初始化功能模块"""
        # 初始化导航功能模块并确保所有子模块获取正确的主题信息
        self.navigation_functions = NavigationFunctions(self.label_before, self.label_after, self.label_result, self.text_log)
        # 设置NavigationFunctions的main_window引用，方便子模块访问
        self.navigation_functions.main_window = self
        # 设置当前主题信息，确保子模块使用正确的主题
        self.navigation_functions.is_dark_theme = self.is_dark_theme
        
        # 确保日志文本颜色正确
        if self.is_dark_theme:
            self.text_log.setTextColor(Qt.GlobalColor.white)
        else:
            self.text_log.setTextColor(Qt.GlobalColor.black)
            
        # 初始化各功能模块
        self.grid_cropping = GridCropping(self.navigation_functions)
        self.import_before_image = ImportBeforeImage(self.navigation_functions)
        self.import_after_image = ImportAfterImage(self.navigation_functions)
        self.execute_change_detection = ExecuteChangeDetectionTask(
            self.navigation_functions
        )
        # 添加: 设置初始处理模式
        if hasattr(self, 'current_mode') and hasattr(self.execute_change_detection, 'set_processing_mode'):
            self.execute_change_detection.set_processing_mode(self.current_mode)
        self.clear_task = ClearTask(
            self.navigation_functions,
            self.text_log
        )
        self.image_display = ImageDisplay(self.navigation_functions)
        
        # 初始化批量处理模块
        try:
            from zhuyaogongneng_docker.function.batch_processing import BatchProcessing
            self.batch_processing = BatchProcessing(self.navigation_functions)
        except ImportError:
            from function.batch_processing import BatchProcessing
            self.batch_processing = BatchProcessing(self.navigation_functions)
        
        # 为ZoomableLabel添加窗口尺寸变化的观察者
        self._register_zoomable_label_observers()
        
        # 记录初始化结果
        self.modules_initialized = True
        
        self.navigation_functions.log_message("所有功能模块初始化成功")
        
        return True
        
    def _register_zoomable_label_observers(self):
        """为ZoomableLabel注册窗口尺寸变化的观察者"""
        # 为三个图像显示区域添加窗口尺寸变化的观察者
        for label in [self.label_before, self.label_after, self.label_result]:
            if hasattr(label, '_calculate_initial_scale') and hasattr(label, 'update_display'):
                # 创建闭包函数，确保每个观察者都有正确的label引用
                def create_observer(label_obj):
                    def observer(size):
                        # 只有当标签有原始图像时才重新计算缩放比例
                        if hasattr(label_obj, 'original_pixmap') and label_obj.original_pixmap is not None:
                            label_obj._calculate_initial_scale()
                            label_obj.update_display()
                    return observer
                
                # 添加观察者
                self.add_resize_observer(create_observer(label))

    def apply_theme(self):
        """应用当前主题样式"""
        # 应用主题样式表
        self.setStyleSheet(ThemeManager.get_theme_style(self.is_dark_theme))
            
        # 确保首页主题与应用主题一致
        if hasattr(self, 'home_page'):
            if self.home_page.is_dark_theme != self.is_dark_theme:
                self.home_page.update_theme(self.is_dark_theme)

    def show_batch_processing(self):
        """显示批量处理对话框"""
        # 判断是否是影像版模式
        is_raster_mode = hasattr(self, 'current_mode') and self.current_mode == "raster"
        
        if is_raster_mode:
            self.show_raster_batch_processing()
        else:
            # 懒加载模式 - 只在第一次使用时初始化
            if not hasattr(self, 'batch_processing') or self.batch_processing is None:
                # 设置等待光标
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    from zhuyaogongneng_docker.function.batch_processing import BatchProcessing
                    self.batch_processing = BatchProcessing(self.navigation_functions)
                finally:
                    # 恢复光标
                    QApplication.restoreOverrideCursor()
            
            self.batch_processing.show_batch_processing_dialog()

    def apply_button_styles(self):
        """应用按钮样式"""
        # 首页按钮使用主要按钮样式
        self.btn_home.setStyleSheet(ThemeManager.get_primary_button_style(self.is_dark_theme))
        
        # 解译和导出按钮使用主要按钮样式
        self.btn_begin.setStyleSheet(ThemeManager.get_primary_button_style(self.is_dark_theme))
        self.btn_batch.setStyleSheet(ThemeManager.get_primary_button_style(self.is_dark_theme))
        
        # 导入按钮使用次要按钮样式
        self.btn_import.setStyleSheet(ThemeManager.get_secondary_button_style(self.is_dark_theme))
        self.btn_import_after.setStyleSheet(ThemeManager.get_secondary_button_style(self.is_dark_theme))
        self.btn_crop.setStyleSheet(ThemeManager.get_secondary_button_style(self.is_dark_theme))
        
        # 功能性按钮使用工具按钮样式
        self.btn_theme.setStyleSheet(ThemeManager.get_utility_button_style(self.is_dark_theme))
        self.btn_help.setStyleSheet(ThemeManager.get_utility_button_style(self.is_dark_theme))
        self.btn_clear.setStyleSheet(ThemeManager.get_utility_button_style(self.is_dark_theme))

    def switch_to_home_page(self):
        """切换到首页"""
        # 在切换到首页前，先清理界面
        if hasattr(self, 'clear_task') and self.clear_task is not None:
            print("准备切换到首页，先清理界面")
            self.clear_task.clear_interface()
            # 确保界面更新
            QApplication.processEvents()
            
        # 确保首页使用当前主题
        if self.home_page.is_dark_theme != self.is_dark_theme:
            self.home_page.update_theme(self.is_dark_theme)
        
        # 确保日志使用正确的颜色
        if hasattr(self, 'text_log'):
            # 刷新日志文本颜色
            self.refresh_log_text_color()
        
            
        # 切换到首页
        self.stacked_widget.setCurrentIndex(0)
        
        # 记录首页已加载
        if self.navigation_functions is not None:
            self.navigation_functions.log_message("已切换至首页")

    def connect_buttons(self):
        """连接按钮点击事件"""
        # 先断开所有可能存在的连接，避免连接混乱
        buttons_to_disconnect = [
            self.btn_import, self.btn_import_after, self.btn_begin,
            self.btn_batch, self.btn_crop,
            self.btn_home, self.btn_clear, self.btn_help,
            self.btn_theme
        ]
        
        for btn in buttons_to_disconnect:
            try:
                # 先尝试将按钮信号从所有槽断开
                btn.clicked.disconnect()
            except TypeError:
                # 如果没有连接或出现TypeError，忽略
                pass
            except Exception:
                # 忽略其他断开连接失败的情况
                pass
        
        # 获取当前模式
        current_mode = getattr(self, 'current_mode', 'image')  # 默认为图像模式
        
        # 先连接通用按钮
        self.btn_home.clicked.connect(self.switch_to_home_page)
        self.btn_clear.clicked.connect(self.clear_task.clear_interface)
        self.btn_help.clicked.connect(self.show_help)
        self.btn_theme.clicked.connect(self.toggle_theme)
        
        # 根据当前模式连接不同的功能
        if current_mode == "image":
            # 图像模式 - 使用原有图像处理模块，确保不使用栅格模块
            # 裁剪和导入功能
            self.btn_import.clicked.connect(self.import_before_image.on_import_clicked)
            self.btn_import_after.clicked.connect(self.import_after_image.import_after_image)
            
            # 检测和批处理功能
            self.btn_begin.clicked.connect(self.execute_change_detection.on_begin_clicked)
            self.btn_batch.clicked.connect(self.show_batch_processing)
            
            # 渔网裁剪功能
            self.btn_crop.clicked.connect(self.grid_cropping.crop_image)
            
            # 记录日志
            self.navigation_functions.log_message("已切换到图像模式")
                
        elif current_mode == "raster":
            # 栅格模式 - 使用栅格影像处理模块
            # 裁剪通用功能
            self.btn_crop.clicked.connect(self.start_raster_grid_cropping)
            
            # 导入功能
            if hasattr(self, 'raster_importer') and self.raster_importer is not None:
                self.btn_import.clicked.connect(self.raster_importer.import_before_image)
                self.btn_import_after.clicked.connect(self.raster_importer.import_after_image)
            else:
                self.btn_import.clicked.connect(lambda: self.show_raster_module_missing("导入前时相栅格影像"))
                self.btn_import_after.clicked.connect(lambda: self.show_raster_module_missing("导入后时相栅格影像"))
            
            # 变化检测功能 
            if hasattr(self, 'raster_cd') and self.raster_cd is not None:
                self.btn_begin.clicked.connect(self.raster_cd.on_begin_clicked)
            else:
                self.btn_begin.clicked.connect(lambda: self.show_raster_module_missing("变化检测"))
            
            # 批处理功能
            if hasattr(self, 'raster_batch_processor') and self.raster_batch_processor is not None:
                self.btn_batch.clicked.connect(self.show_raster_batch_processing)
            else:
                self.btn_batch.clicked.connect(lambda: self.show_raster_module_missing("批量处理"))
            
            # 记录日志
            self.navigation_functions.log_message("已切换到栅格模式")

    def show_raster_module_missing(self, feature_name="栅格影像处理模块"):
        """显示栅格模块缺失的错误信息"""
        QMessageBox.critical(self, "模块缺失",
                           f"{feature_name}不可用。\n\n"
                           "请安装GDAL库（pip install GDAL）并确保依赖项配置正确。")
    
    def show_raster_batch_processing(self):
        """显示栅格影像批量处理对话框"""
        # 检查是否有专门的栅格批处理模块
        if hasattr(self, 'raster_batch_processor'):
            self.raster_batch_processor.show_dialog()
        else:
            # 如果没有，显示提示
            QMessageBox.information(self, "功能不可用",
                                   "栅格影像批量处理功能尚未实现。\n\n"
                                   "您可以使用单张处理功能依次处理影像。")

    def create_button_group(self, parent_layout):
        """创建按钮组"""
        button_layout = QHBoxLayout()
        
        # 创建首页按钮
        self.btn_home = QPushButton("首页")
        self.btn_home.setIcon(QIcon(":/icons/home.png"))
        button_layout.addWidget(self.btn_home)
        
        # 创建渔网裁剪按钮
        self.btn_crop = QPushButton("渔网分割")
        self.btn_crop.setIcon(QIcon(":/icons/crop.png"))
        
        # 创建导入前后时相影像按钮
        self.btn_import = QPushButton("导入前时相影像")
        self.btn_import.setIcon(QIcon(":/icons/import.png"))
        
        self.btn_import_after = QPushButton("导入后时相影像")
        self.btn_import_after.setIcon(QIcon(":/icons/import_after.png"))
        
        # 创建开始解译按钮
        self.btn_begin = QPushButton("开始解译")
        self.btn_begin.setIcon(QIcon(":/icons/play.png"))
        
        # 创建批量处理按钮
        self.btn_batch = QPushButton("批量处理")
        self.btn_batch.setIcon(QIcon(":/icons/batch.png"))
        
        # 创建功能按钮
        self.btn_theme = QPushButton("切换主题")
        self.btn_theme.setIcon(QIcon(":/icons/theme.png"))
        
        self.btn_clear = QPushButton("清空界面")
        self.btn_clear.setIcon(QIcon(":/icons/clear.png"))
        
        self.btn_help = QPushButton("帮助")
        self.btn_help.setIcon(QIcon(":/icons/help.png"))
        
        # 添加所有按钮到布局（除首页按钮外，已在前面添加）
        for btn in [self.btn_crop, self.btn_import, self.btn_import_after, self.btn_begin, self.btn_batch, self.btn_theme, self.btn_clear, self.btn_help]:
            button_layout.addWidget(btn)
            # 设置固定高度并增加间距
            btn.setFixedHeight(32)
            btn.setFont(QFont("Microsoft YaHei UI", 9))
        
        # 设置首页按钮高度和字体
        self.btn_home.setFixedHeight(32)
        self.btn_home.setFont(QFont("Microsoft YaHei UI", 9, QFont.Bold))
            
        # 设置按钮间距
        button_layout.setSpacing(8)
        
        parent_layout.addLayout(button_layout)
        
        # 在按钮下方添加分隔线
        self.nav_separator = QWidget()
        self.nav_separator.setFixedHeight(1)
        self.nav_separator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.nav_separator.setStyleSheet(ThemeManager.get_separator_style(self.is_dark_theme))
        parent_layout.addWidget(self.nav_separator)

    def toggle_theme(self):
        """切换深浅主题并更新首页主题"""
        # 切换主题
        self.is_dark_theme = not self.is_dark_theme
        
        # 记录当前主题
        theme_name = "深色" if self.is_dark_theme else "浅色"
        
        # 应用主题样式
        self.setStyleSheet(ThemeManager.get_app_stylesheet(self.is_dark_theme))
        
        # 更新首页主题
        if hasattr(self, 'home_page'):
            self.home_page.update_theme(self.is_dark_theme)
        
        # 重新连接首页信号
        if hasattr(self, 'connect_home_page_signals'):
            self.connect_home_page_signals()
        
        # 更新顶部容器样式
        if hasattr(self, 'top_container'):
            self.top_container.setStyleSheet(f"background-color: {ThemeManager.get_colors(self.is_dark_theme)['header_bg']};")
        
        # 更新标题标签样式
        if hasattr(self, 'title_label'):
            self.title_label.setStyleSheet(f"color: {ThemeManager.get_colors(self.is_dark_theme)['header_text']};")
        
        # 更新图像显示区域样式
        image_labels = [self.label_before, self.label_after, self.label_result]
        for label in image_labels:
            if hasattr(label, 'setStyleSheet'):
                label.setStyleSheet(ThemeManager.get_image_label_style(self.is_dark_theme))
        
        # 更新日志文本区域样式
        if hasattr(self, 'text_log'):
            # 设置样式
            self.text_log.setStyleSheet(ThemeManager.get_log_text_style(self.is_dark_theme))
            # 刷新所有日志文本颜色
            self.refresh_log_text_color()
        
        # 更新分隔线样式
        if hasattr(self, 'nav_separator'):
            self.nav_separator.setStyleSheet(ThemeManager.get_separator_style(self.is_dark_theme))
        
        # 更新所有按钮的尺寸和字体
        all_buttons = [
            self.btn_home, self.btn_crop, self.btn_import, 
            self.btn_import_after, self.btn_begin, self.btn_batch,
            self.btn_theme, self.btn_clear, self.btn_help
        ]
        
        for btn in all_buttons:
            btn.setFixedHeight(32)
            btn.setFont(QFont("Microsoft YaHei UI", 9))
        
        # 主页按钮字体加粗
        self.btn_home.setFont(QFont("Microsoft YaHei UI", 9, QFont.Bold))
        
        # 使用统一的方法应用按钮样式
        self.apply_button_styles()
        
        # 更新导航功能中的主题标志
        if hasattr(self, 'navigation_functions'):
            self.navigation_functions.is_dark_theme = self.is_dark_theme
            self.navigation_functions.log_message(f"已切换至{theme_name}主题")

    def start_raster_grid_cropping(self):
        """启动栅格网格裁剪功能"""
        try:
            # 检查是否已加载栅格模块
            if not hasattr(self, 'raster_grid_cropping') or self.raster_grid_cropping is None:
                # 导入并初始化栅格网格裁剪类
                try:
                    from zhuyaogongneng_docker.function.raster.grid import RasterGridCropping
                    self.raster_grid_cropping = RasterGridCropping(self.navigation_functions)
                    self.navigation_functions.log_message("已加载栅格网格裁剪模块")
                except ImportError as e:
                    self.navigation_functions.log_message(f"无法导入栅格网格裁剪模块: {str(e)}", "ERROR")
                    self.show_raster_module_missing("栅格渔网裁剪")
                    return

            # 使用等待光标
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            try:
                # 选择输入文件
                input_file = self.raster_grid_cropping.select_input_file()
                if not input_file:
                    QApplication.restoreOverrideCursor()
                    return
                    
                # 选择输出目录
                output_dir = self.raster_grid_cropping.select_output_directory()
                if not output_dir:
                    QApplication.restoreOverrideCursor()
                    return
                
                # 恢复光标，以便用户输入
                QApplication.restoreOverrideCursor()
                    
                # 提示用户输入网格大小
                rows, ok = QInputDialog.getInt(self, "设置网格行数", "请输入要将影像分割的行数:", 2, 1, 100, 1)
                if not ok:
                    return
                    
                cols, ok = QInputDialog.getInt(self, "设置网格列数", "请输入要将影像分割的列数:", 2, 1, 100, 1)
                if not ok:
                    return
                
                # 设置等待光标
                QApplication.setOverrideCursor(Qt.WaitCursor)
                    
                # 执行栅格网格裁剪
                self.navigation_functions.log_message(f"开始栅格网格裁剪: {input_file}，网格大小: {rows}x{cols}")
                
                # 使用异常捕获块，确保资源释放
                try:
                    success = self.raster_grid_cropping.start_grid_cropping(input_file, output_dir, rows, cols)
                    
                    if success:
                        self.navigation_functions.log_message("栅格网格裁剪完成", "COMPLETE")
                        # 显示完成消息
                        QMessageBox.information(self, "裁剪完成", f"栅格网格裁剪已完成!\n子图像已保存到: {output_dir}")
                    else:
                        self.navigation_functions.log_message("栅格网格裁剪失败", "ERROR")
                        QMessageBox.warning(self, "裁剪失败", "栅格网格裁剪过程中出现错误，请查看日志了解详情。")
                except Exception as e:
                    import traceback
                    self.navigation_functions.log_message(f"栅格网格裁剪过程中出错: {str(e)}", "ERROR")
                    self.navigation_functions.log_message(traceback.format_exc(), "ERROR")
                    QMessageBox.critical(self, "裁剪错误", f"栅格网格裁剪失败: {str(e)}")
                finally:
                    # 手动触发垃圾回收，回收内存
                    import gc
                    gc.collect()
                    
                    # 确保恢复光标
                    QApplication.restoreOverrideCursor()
                    
                    # 处理Qt事件队列，确保UI响应
                    QApplication.processEvents()
            except Exception as e:
                # 恢复光标
                QApplication.restoreOverrideCursor()
                
                # 显示错误
                self.navigation_functions.log_message(f"准备栅格裁剪过程中出错: {str(e)}", "ERROR")
                QMessageBox.critical(self, "操作错误", f"准备栅格裁剪过程中出错: {str(e)}")
                
        except Exception as e:
            import traceback
            self.navigation_functions.log_message(f"栅格网格裁剪初始化过程中出错: {str(e)}", "ERROR")
            self.navigation_functions.log_message(traceback.format_exc(), "ERROR")
            
            # 确保恢复光标
            try:
                QApplication.restoreOverrideCursor()
            except:
                pass

    def complete_raster_modules_init(self):
        """完成栅格模块的初始化"""
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # 从函数导入需要的栅格处理类
            from zhuyaogongneng_docker.function.raster import RasterChangeDetection
            from zhuyaogongneng_docker.function.raster import RasterBatchProcessor
            from zhuyaogongneng_docker.function.raster.import_module import RasterImporter
            from zhuyaogongneng_docker.function.raster.grid import RasterGridCropping
            
            # 初始化各种栅格处理类
            self.raster_cd = RasterChangeDetection(self.navigation_functions)
            self.raster_batch_processor = RasterBatchProcessor(self.navigation_functions)
            self.raster_importer = RasterImporter(self.navigation_functions)
            self.raster_grid_cropping = RasterGridCropping(self.navigation_functions)
            
            # 设置ShapefileGenerator和RasterExporter为None，以避免后续使用时报错
            self.shp_generator = None
            self.raster_exporter = None
            
            # 标记栅格模块已加载
            self.raster_modules_loaded = True
            
            # 加载次要模块
            self._load_secondary_raster_modules()
            
            self.navigation_functions.log_message("栅格处理模块加载完成")
            QApplication.restoreOverrideCursor()
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"完成栅格模块初始化失败: {str(e)}"
            if hasattr(self, 'navigation_functions'):
                self.navigation_functions.log_message(error_msg, "ERROR")
                self.navigation_functions.log_message(traceback.format_exc(), "ERROR")
            else:
                print(error_msg)
                print(traceback.format_exc())
                
            QApplication.restoreOverrideCursor()
            return False

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 创建主窗口并显示
    window = RemoteSensingApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
