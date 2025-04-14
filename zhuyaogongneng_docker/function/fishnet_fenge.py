import os
import sys
import cv2
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QDialog, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QWidget
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication
from pathlib import Path
import shutil
import traceback

# 添加项目根目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 使用绝对导入
from display import ZoomableLabel

# 引入主题管理器
try:
    from theme_manager import ThemeManager
except ImportError:


        from .theme_utils import ThemeManager


class GridCropping:
    def __init__(self, navigation_functions=None):
        """初始化渔网裁剪处理类
        
        Args:
            navigation_functions: NavigationFunctions实例，用于日志记录和功能访问（批处理模式下可能为None）
        """
        self.navigation_functions = navigation_functions
    
    def log_message(self, message, level="INFO"):
        """记录日志消息
        
        Args:
            message: 日志消息
            level: 日志级别（INFO, ERROR等）
        """
        # 检查是否可以访问主日志
        if hasattr(self, 'navigation_functions') and self.navigation_functions is not None:
            # 使用带级别的日志记录（如果navigation_functions支持）
            if hasattr(self.navigation_functions, 'log_message'):
                try:
                    # 尝试使用可能支持日志级别的新接口
                    self.navigation_functions.log_message(message, level)
                except TypeError:
                    # 回退到旧接口
                    self.navigation_functions.log_message(message)
    
    def select_image_file(self, title="选择图像文件"):
        """根据当前模式选择合适的图像文件
        
        Args:
            title: 文件选择对话框标题
            
        Returns:
            str: 选择的文件路径，未选择返回空字符串
        """
        # 获取当前模式（从主窗口）
        current_mode = "image"  # 默认为图像模式
        if self.navigation_functions:
            if hasattr(self.navigation_functions, 'main_window') and hasattr(self.navigation_functions.main_window, 'current_mode'):
                current_mode = self.navigation_functions.main_window.current_mode
                
        # 根据模式设置文件过滤器
        if current_mode == "raster":
            # 栅格模式 - 使用栅格影像格式过滤器
            file_filter = "TIFF文件 (*.tif *.tiff);;GeoTIFF文件 (*.tif *.tiff);;栅格影像文件 (*.tif *.tiff *.img *.dem *.hgt);;所有文件 (*.*)"
            if self.navigation_functions:
                self.log_message("当前为栅格模式，仅显示栅格影像格式")
        else:
            # 图像模式 - 使用普通图像格式过滤器
            file_filter = "图像文件 (*.png *.jpg *.jpeg);;所有文件 (*.*)"
            if self.navigation_functions:
                self.log_message("当前为图像模式，显示普通图像格式")
                
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            None, 
            title, 
            "", 
            file_filter)
                
        return file_path
    
    def crop_image(self, is_before=True):
        """通过文件选择对话框选择图像进行网格裁剪
        
        Args:
            is_before: 是否为前时相图像
        
        Returns:
            bool: 是否成功执行
        """
        try:
            # 导入Qt库
            from PySide6.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QInputDialog, QApplication
            from PySide6.QtGui import QImage, QPixmap

            # 使用等待光标
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            try:
                # 选择输入文件
                file_path = self.select_image_file("选择要进行网格裁剪的图像")
                if not file_path:
                    QApplication.restoreOverrideCursor()
                    self.log_message("未选择文件")
                    return False
                
                # 选择输出目录
                save_dir = QFileDialog.getExistingDirectory(None, "选择保存目录")
                if not save_dir:
                    QApplication.restoreOverrideCursor()
                    self.log_message("未选择保存目录")
                    return False
                
                # 恢复光标，以便用户输入
                QApplication.restoreOverrideCursor()
                
                # 提示用户输入网格行数和列数
                rows, ok = QInputDialog.getInt(None, "设置网格行数", "请输入要将影像分割的行数:", 2, 1, 100, 1)
                if not ok:
                    self.log_message("未指定网格行数")
                    return False
                
                cols, ok = QInputDialog.getInt(None, "设置网格列数", "请输入要将影像分割的列数:", 2, 1, 100, 1)
                if not ok:
                    self.log_message("未指定网格列数")
                    return False
                
                # 设置等待光标
                QApplication.setOverrideCursor(Qt.WaitCursor)
                
                # 记录文件和目录信息
                self.log_message(f"开始图像网格裁剪: {file_path}，网格大小: {rows}x{cols}")
                self.log_message(f"选择图像文件: {file_path}")
                self.log_message(f"选择保存目录: {save_dir}")
                
                # 处理图像
                self.log_message("处理图像...")
                last_files = self._crop_image_grid(file_path, rows, cols, save_dir, is_before)
                
                # 恢复光标
                QApplication.restoreOverrideCursor()
                
                # 创建网格示例图并显示完成对话框
                if last_files:
                    # 生成网格预览图（内存中的图像数据）
                    grid_preview_img = self._generate_grid_preview(file_path, rows, cols, save_dir)
                    self.log_message("网格裁剪完成！", "COMPLETE")
                    
                    # 显示完成对话框，提供预览选项
                    if grid_preview_img is not None:
                        # 检查是否使用深色主题
                        is_dark_theme = hasattr(self.navigation_functions, 'is_dark_theme') and self.navigation_functions.is_dark_theme
                        
                        # 创建对话框
                        self.log_message("创建裁剪完成对话框", "INFO")
                        dialog = QDialog()
                        dialog.setWindowTitle("裁剪完成")
                        dialog.setFixedSize(350, 180)
                        
                        # 使用ThemeManager设置对话框样式
                        dialog.setStyleSheet(ThemeManager.get_dialog_style(is_dark_theme))
                        
                        # 创建布局
                        layout = QVBoxLayout(dialog)
                        layout.setSpacing(15)
                        layout.setContentsMargins(25, 25, 25, 25)
                        
                        # 创建提示标签
                        label = QLabel("网格裁剪完成，裁剪结果保存至目标文件夹。\n你要查看哪个影像？")
                        label.setAlignment(Qt.AlignCenter)
                        
                        # 使用ThemeManager设置标签样式
                        label.setStyleSheet(ThemeManager.get_dialog_label_style(is_dark_theme))
                        
                        layout.addWidget(label)
                        
                        # 创建按钮容器，设置透明背景
                        button_container = QWidget()
                        button_container.setStyleSheet(ThemeManager.get_transparent_container_style())
                        button_layout = QHBoxLayout(button_container)
                        button_layout.setContentsMargins(0, 10, 0, 0)
                        button_layout.setSpacing(15)
                        
                        # 使用ThemeManager获取对话框按钮样式
                        button_style = ThemeManager.get_dialog_button_style(is_dark_theme)
                        
                        # 创建按钮
                        btn_preview = QPushButton("裁剪示意图")
                        btn_cropped = QPushButton("裁剪块")
                        btn_cancel = QPushButton("不查看")
                        
                        # 设置按钮样式
                        btn_preview.setStyleSheet(button_style)
                        btn_cropped.setStyleSheet(button_style)
                        btn_cancel.setStyleSheet(button_style)
                        
                        # 添加按钮到布局
                        button_layout.addStretch()
                        button_layout.addWidget(btn_preview)
                        button_layout.addWidget(btn_cropped)
                        button_layout.addWidget(btn_cancel)
                        button_layout.addStretch()
                        
                        # 添加按钮容器到布局
                        layout.addWidget(button_container)
                        
                        # 转换网格预览图，准备显示
                        height, width, channel = grid_preview_img.shape
                        bytes_per_line = 3 * width
                        # 将OpenCV BGR格式转换为RGB格式
                        preview_rgb = cv2.cvtColor(grid_preview_img, cv2.COLOR_BGR2RGB)
                        q_img = QImage(preview_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        
                        # 连接按钮信号
                        def on_preview_clicked():
                            # 创建图像预览对话框
                            preview_dialog = QDialog()
                            preview_dialog.setWindowTitle("网格裁剪示意图")
                            preview_layout = QVBoxLayout(preview_dialog)
                            
                            # 创建可缩放标签显示预览图
                            from display import ZoomableLabel
                            preview_label = ZoomableLabel()
                            preview_label.setMinimumSize(width, height)
                            pixmap = QPixmap.fromImage(q_img)
                            preview_label.original_pixmap = pixmap
                            preview_label.scale_factor = 1.0
                            preview_label.update_display()
                            
                            # 添加到布局
                            preview_layout.addWidget(preview_label)
                            
                            # 添加关闭按钮
                            close_btn = QPushButton("关闭")
                            close_btn.clicked.connect(preview_dialog.close)
                            close_btn.setStyleSheet(button_style)
                            
                            preview_layout.addWidget(close_btn)
                            preview_dialog.setMinimumSize(800, 600)
                            preview_dialog.exec_()
                        
                        btn_preview.clicked.connect(on_preview_clicked)
                        btn_cropped.clicked.connect(lambda: self._show_cropped_images_browser(last_files, dialog))
                        btn_cancel.clicked.connect(dialog.reject)
                        
                        dialog.exec_()
                    else:
                        self.log_message("无法生成网格预览图", "ERROR")

                return True
                
            except Exception as e:
                QApplication.restoreOverrideCursor()  # 确保恢复光标
                raise e
                
        except Exception as e:
            import traceback
            self.log_message(f"图像网格裁剪出错: {str(e)}")
            self.log_message(traceback.format_exc())
            return False
    
    def _show_cropped_images_browser(self, file_list, parent_dialog=None):
        """显示裁剪块的浏览器，允许用户逐张预览并加载裁剪后的图像块
        
        Args:
            save_dir: 保存目录
            file_list: 裁剪后的文件列表
            is_before: 是否为前时相
            parent_dialog: 父对话框，如果有的话
        """
        # 如果没有文件，直接返回
        if not file_list:
            return
            
        # 导入Qt库
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget, QHBoxLayout
        from PySide6.QtCore import Qt, QSize
        from PySide6.QtGui import QPixmap, QImage
        
        # 检查是否使用深色主题
        is_dark_theme = hasattr(self.navigation_functions, 'is_dark_theme') and self.navigation_functions.is_dark_theme
        
        # 创建浏览对话框
        browser = QDialog()
        browser.setWindowTitle("裁剪图像浏览器")
        browser.setMinimumSize(600, 550)
        
        # 使用ThemeManager设置浏览器样式
        browser.setStyleSheet(ThemeManager.get_dialog_style(is_dark_theme))
        
        # 创建主布局
        main_layout = QVBoxLayout(browser)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # 创建信息标签
        info_label = QLabel()
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignCenter)
        
        # 使用ThemeManager设置标签样式
        info_label.setStyleSheet(ThemeManager.get_dialog_label_style(is_dark_theme))
        
        main_layout.addWidget(info_label)
        
        # 创建滚动区域以容纳图像
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: transparent; border: none;")
        
        # 创建图像容器和图像标签
        image_container = QWidget()
        image_container.setStyleSheet(ThemeManager.get_container_style(is_dark_theme))
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(10, 10, 10, 10)
        image_layout.setSpacing(5)
        
        # 创建图像标签
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(400, 300)
        image_label.setStyleSheet("background-color: transparent;")
        
        # 添加到图像容器
        image_layout.addWidget(image_label)
        
        # 设置滚动区域的内容为图像容器
        scroll_area.setWidget(image_container)
        
        # 添加滚动区域到主布局
        main_layout.addWidget(scroll_area)
        
        # 创建导航按钮容器
        nav_container = QWidget()
        nav_container.setStyleSheet(ThemeManager.get_container_style(is_dark_theme))
        nav_layout = QHBoxLayout(nav_container)
        nav_layout.setContentsMargins(10, 10, 10, 10)
        nav_layout.setSpacing(10)
        
        # 使用ThemeManager获取按钮样式
        button_style = ThemeManager.get_dialog_button_style(is_dark_theme)
        
        # 创建导航按钮
        btn_prev = QPushButton("上一张")
        btn_prev.setStyleSheet(button_style)
        btn_prev.setFixedWidth(100)
        
        btn_next = QPushButton("下一张")
        btn_next.setStyleSheet(button_style)
        btn_next.setFixedWidth(100)
        
        # 当前图像索引
        current_index = [0]  # 使用列表以便在闭包中修改
        total_images = len(file_list)
        
        # 更新图像显示函数
        def update_image_display():
            if not file_list or current_index[0] >= len(file_list):
                return
                
            current_path = file_list[current_index[0]]
            file_name = os.path.basename(current_path)
            
            # 更新信息文本
            info_label.setText(f"图像 {current_index[0]+1}/{total_images}: {file_name}")
            
            # 加载并显示图像
            try:
                # 使用PIL读取图像以避免中文路径问题
                img_pil = Image.open(current_path)
                img = np.array(img_pil)
                
                if img is not None:
                    # 如果是RGB格式，需要转换颜色空间
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img_rgb = img  # PIL读取的已经是RGB
                    else:
                        # 灰度图转RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    
                    # 转换为Qt图像
                    qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], 
                                  img_rgb.shape[1] * 3, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    
                    # 确定最大显示尺寸，考虑滚动区域尺寸
                    container_width = scroll_area.width() - 30
                    container_height = scroll_area.height() - 30
                    
                    # 调整图像大小以适应显示区域，保持宽高比
                    scaled_pixmap = pixmap.scaled(
                        container_width, 
                        container_height,
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    
                    # 设置图像到标签
                    image_label.setPixmap(scaled_pixmap)
                    
                    # 记录日志
                    self.log_message(f"显示图像: {file_name}, 尺寸: {img.shape[1]}x{img.shape[0]}")
                else:
                    image_label.setText("无法加载图像")
                    self.log_message(f"无法加载图像: {current_path}")
            except Exception as e:
                image_label.setText(f"加载错误: {str(e)}")
                self.log_message(f"加载图像出错: {str(e)}")
                import traceback
                self.log_message(traceback.format_exc())
        
        # 设置按钮点击事件
        def on_prev_clicked():
            current_index[0] = (current_index[0] - 1) % total_images
            update_image_display()
            
        def on_next_clicked():
            current_index[0] = (current_index[0] + 1) % total_images
            update_image_display()
            
        # 连接按钮事件
        btn_prev.clicked.connect(on_prev_clicked)
        btn_next.clicked.connect(on_next_clicked)
        
        # 添加按钮到导航布局
        nav_layout.addStretch()
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        nav_layout.addStretch()
        
        # 添加导航容器到主布局
        main_layout.addWidget(nav_container)
        
        # 创建操作按钮容器
        action_container = QWidget()
        action_container.setStyleSheet(ThemeManager.get_container_style(is_dark_theme))
        action_layout = QHBoxLayout(action_container)
        action_layout.setContentsMargins(10, 10, 10, 10)
        action_layout.setSpacing(15)
        
        # 创建操作按钮
        btn_load_before = QPushButton("加载为前时相")
        btn_load_before.setStyleSheet(button_style)
        
        btn_load_after = QPushButton("加载为后时相")
        btn_load_after.setStyleSheet(button_style)
        
        btn_close = QPushButton("关闭")
        btn_close.setStyleSheet(button_style)
        
        # 设置加载按钮点击事件
        def load_current_as_before():
            if current_index[0] < len(file_list):
                self._load_as_time(file_list[current_index[0]], True, browser)
                
        def load_current_as_after():
            if current_index[0] < len(file_list):
                self._load_as_time(file_list[current_index[0]], False, browser)
        
        # 关闭按钮只关闭当前浏览器窗口，而不影响父对话框
        def close_browser():
            browser.accept()  # 使用accept而不是reject，这样不会影响父对话框
        
        # 连接按钮事件
        btn_load_before.clicked.connect(load_current_as_before)
        btn_load_after.clicked.connect(load_current_as_after)
        btn_close.clicked.connect(close_browser)  # 使用新的关闭函数
        
        # 添加按钮到操作布局
        action_layout.addStretch()
        action_layout.addWidget(btn_load_before)
        action_layout.addWidget(btn_load_after)
        action_layout.addWidget(btn_close)
        action_layout.addStretch()
        
        # 添加操作容器到主布局
        main_layout.addWidget(action_container)
        
        # 初始化显示第一张图像
        update_image_display()
        
        # 窗口大小变化时自动调整图像大小
        def on_resize(event):
            update_image_display()
            QDialog.resizeEvent(browser, event)
        
        # 覆盖浏览器的resize事件
        browser.resizeEvent = on_resize
        
        # 显示对话框 - 使用exec而不是show确保模态行为
        browser.exec()
    
    def _generate_grid_preview(self, image_path, rows, cols, save_dir):
        """生成网格预览图
        
        Args:
            image_path: 原始图像路径
            rows: 网格行数
            cols: 网格列数
            save_dir: 保存目录
            
        Returns:
            numpy.ndarray: 生成的预览图像数据
        """
        try:
            # 获取文件名（不带路径和扩展名）
            file_name = os.path.basename(image_path)
            base_name, ext = os.path.splitext(file_name)
            
            # 读取原始图像
            img = cv2.imread(image_path)
            if img is None:
                self.log_message(f"无法读取图像: {image_path}")
                return None
            
            # 获取图像尺寸
            height, width = img.shape[:2]
            
            # 根据图像尺寸计算网格大小
            grid_width = width // cols
            grid_height = height // rows
            
            # 创建副本以绘制网格
            grid_img = img.copy()
            
            # 定义网格线颜色和厚度
            line_color = (0, 255, 0)  # 绿色
            line_thickness = max(1, min(width, height) // 200)  # 根据图像大小调整线条粗细
            
            # 绘制水平线
            for i in range(1, rows):
                y = i * grid_height
                cv2.line(grid_img, (0, y), (width, y), line_color, line_thickness)
            
            # 绘制垂直线
            for i in range(1, cols):
                x = i * grid_width
                cv2.line(grid_img, (x, 0), (x, height), line_color, line_thickness)
            
            # 为每个网格添加序号
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(width, height) / 1000)  # 根据图像大小调整字体大小
            
            for row in range(rows):
                for col in range(cols):
                    # 计算网格左上角坐标（添加少量偏移以避免直接贴边）
                    x = col * grid_width + 10
                    y = row * grid_height + 30  # 垂直方向多偏移一些，以便文字完全显示
                    
                    # 计算文本
                    text = f"r{row+1}c{col+1}"
                    text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
                    
                    # 绘制文本背景框
                    bg_width = text_size[0] + 10
                    bg_height = text_size[1] + 10
                    bg_x1 = max(0, x - 5)
                    bg_y1 = max(0, y - text_size[1] - 5)
                    bg_x2 = min(width, bg_x1 + bg_width)
                    bg_y2 = min(height, bg_y1 + bg_height)
                    
                    # 用半透明矩形作为背景
                    overlay = grid_img.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                    
                    # 应用透明度
                    alpha = 0.7
                    cv2.addWeighted(overlay, alpha, grid_img, 1 - alpha, 0, grid_img)
                    
                    # 在背景上绘制文本
                    cv2.putText(grid_img, text, (x, y), font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
            
            # 调整图像尺寸以便于显示
            standardized_width, standardized_height = self._normalize_grid_preview_size(grid_img)
            if standardized_width != width or standardized_height != height:
                grid_img = cv2.resize(grid_img, (standardized_width, standardized_height), 
                                     interpolation=cv2.INTER_AREA)
            
            return grid_img
            
        except Exception as e:
            import traceback
            self.log_message(f"生成网格预览图出错: {str(e)}")
            self.log_message(traceback.format_exc())
            return None
            
    def _normalize_grid_preview_size(self, img):
        """计算标准化的预览图尺寸
        
        Args:
            img: 图像数据
        
        Returns:
            tuple: (标准化宽度, 标准化高度)
        """
        try:
            # 获取图像尺寸
            height, width = img.shape[:2]
            
            # 标准化尺寸为最大边长不超过800像素
            max_size = 800
            
            # 如果图像尺寸小于标准尺寸，直接返回原始尺寸
            if width <= max_size and height <= max_size:
                return width, height
            
            # 根据原始宽高比例计算标准化尺寸
            aspect_ratio = width / height
            
            if width > height:
                standardized_width = max_size
                standardized_height = int(max_size / aspect_ratio)
            else:
                standardized_height = max_size
                standardized_width = int(max_size * aspect_ratio)
            
            return standardized_width, standardized_height
            
        except Exception as e:
            self.log_message(f"计算标准化网格预览尺寸时出错: {str(e)}")
            return 800, 600  # 默认值
    
    def _crop_image_grid(self, file_path, rows, cols, save_dir, is_before=True):
        """将图像裁剪为网格
        
        Args:
            file_path: 图像文件路径
            rows: 网格行数
            cols: 网格列数
            save_dir: 保存目录路径
            is_before: 是否为前时相图像
            
        Returns:
            list: 生成的子图像文件列表
        """
        try:
            # 获取文件名（不带路径和扩展名）
            file_name = os.path.basename(file_path)
            base_name, ext = os.path.splitext(file_name)
            
            # 读取图像
            img = cv2.imread(file_path)
            if img is None:
                self.log_message(f"无法读取图像: {file_path}")
                return []
                
            # 获取图像尺寸
            height, width = img.shape[:2]
            
            # 根据图像尺寸计算网格大小
            grid_width = width // cols
            grid_height = height // rows
            
            # 创建保存子图像的列表
            cropped_files = []
            
            # 裁剪每个网格
            for row in range(rows):
                for col in range(cols):
                    # 计算当前网格的边界
                    x = col * grid_width
                    y = row * grid_height
                    
                    # 确保最后一行/列包含所有剩余像素
                    if col == cols - 1:
                        w = width - x
                    else:
                        w = grid_width
                        
                    if row == rows - 1:
                        h = height - y
                    else:
                        h = grid_height
                        
                        # 裁剪当前网格
                    crop = img[y:y+h, x:x+w]
                    
                    # 创建保存路径
                    crop_filename = f"{base_name}_r{row+1}c{col+1}{ext}"
                    crop_path = os.path.join(save_dir, crop_filename)
                    
                    # 保存裁剪后的图像
                    cv2.imwrite(crop_path, crop)
                    
                    # 添加到文件列表
                    cropped_files.append(crop_path)
            
            self.log_message(f"成功裁剪 {len(cropped_files)} 个子图像")
            return cropped_files
            
        except Exception as e:
            import traceback
            self.log_message(f"裁剪图像网格出错: {str(e)}")
            self.log_message(traceback.format_exc())
            return []

    def _load_as_time(self, image_path, is_before, dialog=None):
        """加载图像为特定时相
        
        Args:
            image_path: 图像路径
            is_before: 是否为前时相
            dialog: 对话框实例，如果要关闭
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                self.log_message(f"错误: 文件不存在 {image_path}")
                return False
                
            # 加载图像并设置保持原始尺寸
            if is_before:
                # 设置文件路径
                self.navigation_functions.file_path = image_path
                
                # 确保标签已经是ZoomableLabel
                if not isinstance(self.navigation_functions.label_before, ZoomableLabel):
                    self.log_message("前时相标签不是ZoomableLabel类型，尝试转换")
                    self.navigation_functions.label_before = self.navigation_functions.replace_with_zoomable_label(self.navigation_functions.label_before)
                
                # 直接加载图像到标签
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    self.navigation_functions.label_before.original_pixmap = pixmap
                    self.navigation_functions.label_before.scale_factor = 1.0
                    self.navigation_functions.label_before.offset = QPoint(0, 0)
                    self.navigation_functions.label_before.update_display()
                    self.log_message(f"已将图像 {os.path.basename(image_path)} 加载为前时相")
                else:
                    self.log_message(f"无法加载图像: {image_path}")
                    return False
            else:
                # 设置文件路径
                self.navigation_functions.file_path_after = image_path
                
                # 确保标签已经是ZoomableLabel
                if not isinstance(self.navigation_functions.label_after, ZoomableLabel):
                    self.log_message("后时相标签不是ZoomableLabel类型，尝试转换")
                    self.navigation_functions.label_after = self.navigation_functions.replace_with_zoomable_label(self.navigation_functions.label_after)
                
                # 直接加载图像到标签
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    self.navigation_functions.label_after.original_pixmap = pixmap
                    self.navigation_functions.label_after.scale_factor = 1.0
                    self.navigation_functions.label_after.offset = QPoint(0, 0)
                    self.navigation_functions.label_after.update_display()
                    self.log_message(f"已将图像 {os.path.basename(image_path)} 加载为后时相")
                else:
                    self.log_message(f"无法加载图像: {image_path}")
                    return False
            
            # 如果提供了对话框实例，关闭它
            if dialog:
                dialog.accept()
                
            return True
        except Exception as e:
            self.log_message(f"加载图像时出错: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            return False

    def _show_grid_size_dialog(self):
        """显示网格大小输入对话框
        
        Returns:
            tuple: (grid_size_text, ok)，其中grid_size_text是输入的网格大小，ok是是否点击了确定按钮
        """
        try:
            from PySide6.QtWidgets import QInputDialog
            
            # 获取是否为深色主题
            is_dark_theme = hasattr(self.navigation_functions, 'is_dark_theme') and self.navigation_functions.is_dark_theme
            
            # 检查是否有主题管理器
            if 'ThemeManager' in globals():
                dialog_style = ThemeManager.get_dialog_style(is_dark_theme)
            else:
                dialog_style = "QInputDialog { background-color: #202124; color: #f7f7f8; }" if is_dark_theme else "QInputDialog { background-color: #ffffff; color: #333333; }"
            
            # 创建输入对话框并直接设置样式
            dialog = QInputDialog()
            dialog.setStyleSheet(dialog_style)
            
            # 显示输入对话框
            grid_size_text, ok = dialog.getText(
                None, 
                "输入网格大小", 
                "请输入网格大小 (例如: 3 表示3x3=9个网格):", 
                text="3"
            )
            
            return grid_size_text, ok
            
        except Exception as e:
            self.log_message(f"显示网格大小对话框时出错: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            return "3", False  # 返回默认值和False表示未成功获取用户输入
