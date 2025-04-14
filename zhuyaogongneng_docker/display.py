# daohanglan.py
import cv2
import logging
from datetime import datetime
from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox, QInputDialog, QApplication, QTextEdit, QScrollBar, QDialog, QVBoxLayout, QPushButton, QGridLayout
from PySide6.QtGui import QPixmap, QImage, QPainter, Qt, QWheelEvent, QMouseEvent, QResizeEvent
from PySide6.QtCore import QEvent, Qt, QPoint
import os
import tempfile
from PIL import Image
from pathlib import Path

'''
1. display.py
主要功能：负责图像显示和交互功能
实现了ZoomableLabel类：支持图像缩放、平移和区域选择
实现了NavigationFunctions类：管理图像显示和日志记录
提供图像交互功能：鼠标滚轮缩放、拖动图像、双击重置视图
处理图像尺寸调整和显示逻辑
实现日志记录系统，可同时输出到UI界面和文件

'''

class ZoomableLabel(QLabel):#定义图像为缩放的标签类
    """可缩放的标签类，支持鼠标滚轮缩放图像和拖动"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.setAlignment(Qt.AlignCenter)
        # 启用鼠标追踪，以便能够接收鼠标事件
        self.setMouseTracking(True)
        # 设置焦点策略，使标签可以接收键盘焦点
        self.setFocusPolicy(Qt.StrongFocus)
        # 安装事件过滤器，以便能够处理鼠标滚轮事件
        self.installEventFilter(self)
        
        # 拖动相关变量
        self.dragging = False
        self.drag_start_position = QPoint()
        self.offset = QPoint(0, 0)  # 图像偏移量
        
        # 当前显示的图像尺寸
        self.current_pixmap_size = None
        
        # 区域选择相关变量
        self.selecting = False  # 是否正在选择区域
        self.selection_start = QPoint()  # 选择区域的起始点
        self.selection_end = QPoint()  # 选择区域的结束点
        self.selection_active = False  # 是否有活动的选择区域
        
        # 选择模式
        self.selection_mode = False  # 是否处于选择模式
        
        # 是否保持图像原始大小的标志
        self.preserve_original_size = False
        
        # 接受鼠标双击事件
        self.setAcceptDrops(True)

    def set_pixmap(self, pixmap):
        """设置原始图像并显示"""
        self.original_pixmap = pixmap
        # 强制重置缩放和偏移，确保重新计算
        self.scale_factor = 0.0  # 设置一个非有效值，强制计算新值
        self.offset = QPoint(0, 0)  # 重置偏移量
        self.selection_active = False  # 重置选择状态
        
        # 自动计算合适的缩放比例以适应窗口大小
        if pixmap and not pixmap.isNull():
            # 计算初始缩放比例
            self._calculate_initial_scale()
            
            # 确保图像居中显示
            self.current_pixmap_size = (pixmap.width() * self.scale_factor, 
                                       pixmap.height() * self.scale_factor)
        
        # 更新显示
        self.update_display()

    def _calculate_initial_scale(self):
        """计算初始缩放比例，使图像刚好填满显示区域的高度，同时保持图像宽高比例"""
        if not self.original_pixmap:
            self.scale_factor = 1.0  # 确保有默认值
            return
            
        # 获取标签尺寸
        label_width = self.width()
        label_height = self.height()
        
        # 获取图像尺寸
        pixmap_width = self.original_pixmap.width()
        pixmap_height = self.original_pixmap.height()
        
        # 如果标签尺寸为零，无法计算缩放比例，使用默认值
        if label_width <= 10 or label_height <= 10:
            self.scale_factor = 1.0
            return
            
        # 如果图像尺寸为零，无法计算缩放比例
        if pixmap_width <= 0 or pixmap_height <= 0:
            self.scale_factor = 1.0
            return
            
        # 首先计算一个适合高度的缩放比例
        # 为了留出一点边距，使用95%的高度
        height_scale = (label_height * 0.95) / pixmap_height
        
        # 确保图像的宽度也不会超出标签宽度
        if pixmap_width * height_scale > label_width * 0.95:
            # 如果按高度缩放后宽度超出，则使用宽度缩放
            width_scale = (label_width * 0.95) / pixmap_width
            # 选择较小的缩放比例，确保图像完全可见
            self.scale_factor = width_scale
        else:
            # 否则使用高度缩放
            self.scale_factor = height_scale
        
        # 确保缩放比例在合理范围内
        if self.scale_factor <= 0.05:
            self.scale_factor = 0.05

    def reset_view(self):
        """重置视图到初始状态，使用与初始加载相同的缩放策略"""
        if self.original_pixmap:
            # 不直接设为1.0，而是重新计算适当的缩放比例
            # self.scale_factor = 1.0
            self.offset = QPoint(0, 0)  # 重置偏移量
            self.selection_active = False  # 重置选择状态
            self.selecting = False
            self.selection_mode = False
            
            # 重新计算合适的缩放比例
            self._calculate_initial_scale()
            
            # 更新显示
            self.update_display()
            self.setCursor(Qt.ArrowCursor)  # 恢复鼠标光标
            return True
        return False

    def enter_selection_mode(self):
        """进入区域选择模式"""
        if self.original_pixmap:
            self.selection_mode = True
            self.selection_active = False
            self.setCursor(Qt.CrossCursor)  # 设置十字光标
            self.update()  # 更新显示

    def exit_selection_mode(self):
        """退出区域选择模式"""
        self.selection_mode = False
        self.selecting = False
        self.setCursor(Qt.ArrowCursor)  # 恢复默认光标
        self.update()  # 更新显示

    def get_selected_area(self):
        """获取选择区域在原始图像上的坐标
        
        返回：
            tuple: (x, y, width, height) 或 None（如果没有选择区域）
        """
        if not self.selection_active or not self.original_pixmap:
            return None
            
        # 将选择区域的坐标从显示坐标转换为原始图像坐标
        
        # 获取标签和图像的尺寸信息
        label_width = self.width()
        label_height = self.height()
        
        # 如果当前有缩放
        if self.scale_factor != 1.0:
            # 确定图像在标签中的位置
            scaled_width = int(self.current_pixmap_size[0])
            scaled_height = int(self.current_pixmap_size[1])
            
            # 计算图像在标签中的偏移量（居中显示）
            x_offset = (label_width - scaled_width) // 2 + self.offset.x()
            y_offset = (label_height - scaled_height) // 2 + self.offset.y()
            
            # 计算选择区域在缩放图像上的坐标
            sel_x1 = min(self.selection_start.x(), self.selection_end.x()) - x_offset
            sel_y1 = min(self.selection_start.y(), self.selection_end.y()) - y_offset
            sel_width = abs(self.selection_end.x() - self.selection_start.x())
            sel_height = abs(self.selection_end.y() - self.selection_start.y())
            
            # 确保坐标在图像范围内
            sel_x1 = max(0, sel_x1)
            sel_y1 = max(0, sel_y1)
            
            if sel_x1 >= scaled_width or sel_y1 >= scaled_height:
                return None
                
            # 限制宽度和高度不超出图像边界
            sel_width = min(sel_width, scaled_width - sel_x1)
            sel_height = min(sel_height, scaled_height - sel_y1)
            
            # 转换回原始图像的坐标
            orig_x = int(sel_x1 / self.scale_factor)
            orig_y = int(sel_y1 / self.scale_factor)
            orig_width = int(sel_width / self.scale_factor)
            orig_height = int(sel_height / self.scale_factor)
            
            return (orig_x, orig_y, orig_width, orig_height)
        else:
            # 获取原始图像的尺寸
            pixmap_width = self.original_pixmap.width()
            pixmap_height = self.original_pixmap.height()
            
            # 计算图像在标签中的缩放比例
            scale_x = pixmap_width / self.current_pixmap_size[0]
            scale_y = pixmap_height / self.current_pixmap_size[1]
            
            # 计算图像在标签中的偏移量（居中显示）
            x_offset = (label_width - self.current_pixmap_size[0]) // 2
            y_offset = (label_height - self.current_pixmap_size[1]) // 2
            
            # 计算选择区域在显示图像上的坐标
            sel_x1 = min(self.selection_start.x(), self.selection_end.x()) - x_offset
            sel_y1 = min(self.selection_start.y(), self.selection_end.y()) - y_offset
            sel_width = abs(self.selection_end.x() - self.selection_start.x())
            sel_height = abs(self.selection_end.y() - self.selection_start.y())
            
            # 确保坐标在图像范围内
            sel_x1 = max(0, sel_x1)
            sel_y1 = max(0, sel_y1)
            
            if sel_x1 >= self.current_pixmap_size[0] or sel_y1 >= self.current_pixmap_size[1]:
                return None
                
            # 限制宽度和高度不超出图像边界
            sel_width = min(sel_width, self.current_pixmap_size[0] - sel_x1)
            sel_height = min(sel_height, self.current_pixmap_size[1] - sel_y1)
            
            # 转换为原始图像的坐标
            orig_x = int(sel_x1 * scale_x)
            orig_y = int(sel_y1 * scale_y)
            orig_width = int(sel_width * scale_x)
            orig_height = int(sel_height * scale_y)
            
            return (orig_x, orig_y, orig_width, orig_height)

    def update_display(self):
        """更新显示，根据当前缩放因子和偏移量重新绘制图像"""
        if not self.original_pixmap:
            # 如果没有原始图像，清空显示
            super().clear()
            return
            
        # 获取标签的尺寸
        label_width = self.width()
        label_height = self.height()
        
        # 如果标签尺寸过小，暂不更新
        if label_width <= 10 or label_height <= 10:
            return
            
        # 创建新的pixmap用于绘制，使用标签当前大小
        display_pixmap = QPixmap(label_width, label_height)
        display_pixmap.fill(Qt.transparent)  # 使背景透明
        
        # 创建绘图器
        painter = QPainter(display_pixmap)
        
        # 计算缩放后的图像尺寸
        pixmap_width = self.original_pixmap.width() * self.scale_factor
        pixmap_height = self.original_pixmap.height() * self.scale_factor
        
        # 保存当前显示的图像尺寸，用于其他计算
        self.current_pixmap_size = (pixmap_width, pixmap_height)
        
        # 计算图像在标签中的位置（严格居中显示）
        x = (label_width - pixmap_width) / 2 + self.offset.x()
        y = (label_height - pixmap_height) / 2 + self.offset.y()
        
        # 绘制原始图像
        painter.drawPixmap(int(x), int(y), int(pixmap_width), int(pixmap_height), self.original_pixmap)
        
        # 如果有选择区域且选择是活跃的，绘制选择矩形
        if self.selection_active:
            # 设置矩形颜色和样式
            painter.setPen(Qt.red)  # 设置红色边框
            
            # 计算选择矩形的坐标
            sel_x = min(self.selection_start.x(), self.selection_end.x())
            sel_y = min(self.selection_start.y(), self.selection_end.y())
            sel_width = abs(self.selection_end.x() - self.selection_start.x())
            sel_height = abs(self.selection_end.y() - self.selection_start.y())
            
            # 绘制矩形
            painter.drawRect(sel_x, sel_y, sel_width, sel_height)
        
        # 结束绘图
        painter.end()
        
        # 在设置pixmap之前保存当前大小
        current_size = self.size()
        
        # 设置标签的pixmap
        super().setPixmap(display_pixmap)
        
        # 确保大小不变
        if current_size.width() > 10 and current_size.height() > 10:
            self.resize(current_size)

    def can_drag(self):
        """判断当前是否可以拖动图像"""
        if not self.original_pixmap or not self.current_pixmap_size:
            return False
            
        # 如果图像实际显示尺寸大于标签尺寸，或缩放比例大于1.0，则可以拖动
        label_width = self.width()
        label_height = self.height()
        pixmap_width, pixmap_height = self.current_pixmap_size
        
        # 无论放大还是缩小，只要图像显示尺寸超过标签尺寸或缩放比例不是1.0，就允许拖动
        return pixmap_width > label_width or pixmap_height > label_height or self.scale_factor != 1.0

    def eventFilter(self, obj, event):
        """事件过滤器，处理鼠标滚轮事件和鼠标拖动事件"""
        if obj == self and self.original_pixmap:
            # 处理鼠标双击事件 - 恢复原始视图
            if event.type() == QEvent.MouseButtonDblClick:
                mouse_event = QMouseEvent(event)
                if mouse_event.button() == Qt.LeftButton:
                    # 重置视图到原始状态
                    self.reset_view()
                    return True
            
            # 处理鼠标滚轮事件
            elif event.type() == QEvent.Wheel:
                wheel_event = QWheelEvent(event)
                # 获取滚轮角度增量
                delta = wheel_event.angleDelta().y()
                
                # 根据滚轮方向调整缩放因子
                if delta > 0:  # 向上滚动，放大
                    self.scale_factor *= 1.2  # 加大缩放步长
                else:  # 向下滚动，缩小
                    self.scale_factor /= 1.2  # 加大缩放步长
                
                # 限制缩放范围
                self.scale_factor = max(0.1, min(self.scale_factor, 10.0))  # 增加最大缩放倍数
                
                # 更新显示
                self.update_display()
                
                # 根据当前状态更新鼠标光标
                if self.selection_mode:
                    self.setCursor(Qt.CrossCursor)
                elif self.can_drag():
                    self.setCursor(Qt.OpenHandCursor)  # 无论放大还是缩小，只要可拖动就显示手形光标
                else:
                    self.setCursor(Qt.ArrowCursor)
                    
                return True
            
            # 处理鼠标按下事件
            elif event.type() == QEvent.MouseButtonPress:
                mouse_event = QMouseEvent(event)
                if mouse_event.button() == Qt.LeftButton:
                    if self.selection_mode:
                        # 在选择模式下开始绘制选择框
                        self.selecting = True
                        self.selection_active = True
                        self.selection_start = mouse_event.position().toPoint()
                        self.selection_end = self.selection_start  # 初始化为相同点
                        self.update()  # 更新显示
                        return True
                    elif self.can_drag():
                        # 在非选择模式下，如果可以拖动图像，开始拖动
                        self.dragging = True
                        self.drag_start_position = mouse_event.position().toPoint()
                        self.setCursor(Qt.ClosedHandCursor)  # 更改鼠标光标为抓取状态
                        return True
            
            # 处理鼠标移动事件
            elif event.type() == QEvent.MouseMove:
                mouse_event = QMouseEvent(event)
                
                if self.selecting:
                    # 更新选择区域的结束点
                    self.selection_end = mouse_event.position().toPoint()
                    self.update()  # 更新显示
                    return True
                elif self.dragging:
                    # 计算鼠标移动的距离
                    delta = mouse_event.position().toPoint() - self.drag_start_position
                    # 更新偏移量
                    self.offset += delta
                    # 更新拖动起始位置
                    self.drag_start_position = mouse_event.position().toPoint()
                    # 更新显示
                    self.update_display()
                    return True
                # 如果鼠标悬停在图像上且可以拖动，显示手形光标（无论是放大还是缩小）
                elif not self.dragging and self.can_drag() and not self.selection_mode:
                    self.setCursor(Qt.OpenHandCursor)
                    return True
            
            # 处理鼠标释放事件
            elif event.type() == QEvent.MouseButtonRelease:
                mouse_event = QMouseEvent(event)
                if mouse_event.button() == Qt.LeftButton:
                    if self.selecting:
                        # 完成选择区域的绘制
                        self.selecting = False
                        self.selection_end = mouse_event.position().toPoint()
                        
                        # 检查选择区域是否有效（不是一个点）
                        if self.selection_start.x() == self.selection_end.x() and self.selection_start.y() == self.selection_end.y():
                            self.selection_active = False
                        
                        self.update()  # 更新显示
                        return True
                    elif self.dragging:
                        self.dragging = False
                        # 鼠标释放后根据当前状态设置光标
                        if self.can_drag() and not self.selection_mode:
                            self.setCursor(Qt.OpenHandCursor)  # 只要图像可拖动就显示手形光标
                        elif self.selection_mode:
                            self.setCursor(Qt.CrossCursor)
                        else:
                            self.setCursor(Qt.ArrowCursor)
                        return True
            
            # 处理鼠标离开事件
            elif event.type() == QEvent.Leave:
                if not self.dragging and not self.selection_mode:
                    self.setCursor(Qt.ArrowCursor)
                return False
        
        return super().eventFilter(obj, event)

    def resizeEvent(self, event: QResizeEvent):
        """处理缩放标签的调整大小事件"""
        super().resizeEvent(event)
        
        # 仅在有图像时才更新显示
        if self.original_pixmap:
            # 获取原尺寸和新尺寸
            old_size = event.oldSize()
            new_size = event.size()
            
            # 检查尺寸变化是否明显 - 避免小幅度调整引起刷新
            size_changed = (abs(old_size.width() - new_size.width()) > 5 or 
                           abs(old_size.height() - new_size.height()) > 5)
            
            # 调整显示比例以适应新大小
            if size_changed:
                # 如果尺寸变化明显，重新计算最佳缩放比例
                old_scale = self.scale_factor
                self._calculate_initial_scale()
                
                # 如果是首次显示，还需要设置resize_count标志
                if not hasattr(self, 'resize_count'):
                    self.resize_count = 1
                else:
                    self.resize_count += 1
            
            # 无论如何都更新显示，确保图像正确显示在新尺寸的容器中
            self.update_display()

    def force_resize_recalculate(self):
        """强制重新计算缩放比例并更新显示"""
        if self.original_pixmap:
            old_scale = self.scale_factor
            self._calculate_initial_scale()
            self.update_display()

    def clear(self):
        """完全清除图像内容"""
        # 清除原始图像
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.current_pixmap_size = None
        self.selection_active = False
        self.selecting = False
        self.selection_mode = False
        
        # 调用父类的clear方法
        super().clear()
        
        # 重置鼠标光标
        self.setCursor(Qt.ArrowCursor)

class NavigationFunctions:
    """导航功能类 - 管理图像显示和日志记录"""
    
    def __init__(self, label_before, label_after, label_result, text_log):
        """初始化导航功能
        
        Args:
            label_before: 前时相标签
            label_after: 后时相标签
            label_result: 结果显示标签
            text_log: 日志文本框
        """
        # 确保标签是ZoomableLabel类型
        self.label_before = self.replace_with_zoomable_label(label_before)
        self.label_after = self.replace_with_zoomable_label(label_after)
        self.label_result = self.replace_with_zoomable_label(label_result)
        
        # 为标签设置名称，方便调试
        if hasattr(self.label_before, 'setObjectName'):
            self.label_before.setObjectName("label_before")
        if hasattr(self.label_after, 'setObjectName'):
            self.label_after.setObjectName("label_after")
        if hasattr(self.label_result, 'setObjectName'):
            self.label_result.setObjectName("label_result")
        
        # 设置日志框引用
        self.text_log = text_log
        
        # 初始化文件路径变量
        self.file_path = None  # 前时相图像路径
        self.file_path_after = None  # 后时相图像路径
        self.result_image_path = None  # 结果图像路径
        
        # 变化检测结果
        self.result_image = None
        self.cached_result_image = None
        
        # 存储原始图像尺寸
        self.before_image_original_size = None
        self.after_image_original_size = None
        self.result_image_original_size = None  # 添加结果图像的原始尺寸属性
        
        # 设置主题
        self.is_dark_theme = False  # 默认为浅色主题
        
        # 添加渔网掩膜标志
        self.has_fishnet_mask = False
        
        # 添加图像标准化标志
        self.is_image_standardized = False
        
        # 添加历史记录列表
        self.history_records = []
        
        # 注释掉setup_logging调用，避免重复初始化日志系统
        # self.setup_logging()
        
        # 保存标签的缩放状态
        self._label_states = {}
        
        # 初始化日志
        self.log_message("系统初始化完成")

    def replace_with_zoomable_label(self, label):
        """将标准标签替换为可缩放标签（如果还不是）"""
        if not isinstance(label, ZoomableLabel):
            # 如果标签不是ZoomableLabel类型，需要进行转换
            parent = label.parent()
            name = label.objectName()
            geometry = label.geometry()
            
            # 创建新的可缩放标签替换原标签
            new_label = ZoomableLabel(parent=parent)
            new_label.setObjectName(name)
            new_label.setGeometry(geometry)
            new_label.setAlignment(label.alignment())
            new_label.setMinimumSize(label.minimumSize())
            
            # 如果原标签有布局，尝试将新标签添加到相同位置
            if parent and parent.layout():
                try:
                    # 尝试找到原标签在布局中的位置
                    for i in range(parent.layout().count()):
                        if parent.layout().itemAt(i).widget() == label:
                            # 移除原标签
                            parent.layout().removeWidget(label)
                            # 添加新标签到相同位置
                            parent.layout().insertWidget(i, new_label)
                            break
                except Exception as e:
                    self.log_message(f"替换标签布局时出错: {str(e)}")
            
            # 把原标签的属性传递给新标签
            if label.pixmap():
                new_label.setPixmap(label.pixmap())
            
            # 隐藏原标签
            label.hide()
            # 显示新标签
            new_label.show()
            
            # 返回新创建的标签
            return new_label
        
        # 如果已经是ZoomableLabel，直接返回
        return label

    def log_message(self, message, level="INFO"):
        """
        记录消息到日志文件和界面
        
        Args:
            message: 消息内容
            level: 日志级别，可以是"INFO"(常规信息),"START"(功能开始),"COMPLETE"(功能完成),"ERROR"(错误信息)
                   只有START, COMPLETE和ERROR级别会显示在UI中
        """
        try:
            # 记录到日志文件 - 所有消息都记录到文件
            logging.info(message)
            
            # 决定是否显示在UI上
            show_in_ui = level in ["START", "COMPLETE", "ERROR"]
            
            # 记录到UI - 只有指定级别的消息才会显示
            if self.text_log and show_in_ui:
                # 根据级别调整消息格式
                if level == "START":
                    formatted_text = f"▶ 开始: {message}"
                elif level == "COMPLETE":
                    formatted_text = f"✓ 完成: {message}"
                elif level == "ERROR":
                    formatted_text = f"❌ 错误: {message}"
                else:
                    formatted_text = message
                
                # 记录消息前确保文本使用正确的颜色
                if hasattr(self, 'is_dark_theme'):
                    try:
                        # 根据主题获取固定的文本颜色
                        text_color = "white" if self.is_dark_theme else "black"

                        # 使用HTML格式应用文本颜色
                        formatted_message = f"<span style='color:{text_color};'>{formatted_text}</span>"
                        # 添加格式化后的消息
                        self.text_log.append(formatted_message)
                    except Exception as e:
                        # 如果设置颜色失败，不影响主流程，使用默认文本添加
                        print(f"设置文本颜色失败: {str(e)}")
                        self.text_log.append(formatted_text)
                else:
                    # 如果没有主题属性，使用默认文本添加
                    self.text_log.append(formatted_text)
                
                # 自动滚动到底部
                self.text_log.verticalScrollBar().setValue(self.text_log.verticalScrollBar().maximum())
        except Exception as e:
            # 如果记录消息失败，至少尝试在UI中显示
            if self.text_log:
                self.text_log.append(f"记录消息失败: {str(e)}")
                self.text_log.append(f"原始消息: {message}")
                # 自动滚动到底部
                self.text_log.verticalScrollBar().setValue(self.text_log.verticalScrollBar().maximum())
    
    def update_image_display(self, is_before=None):
        """更新图像显示
        
        Args:
            is_before: 指定更新哪个图像。True表示前时相，False表示后时相，None表示两者都更新
        """
        try:
            # 确保处理前应用所有挂起的UI更新，使标签尺寸准确
            QApplication.processEvents()
            
            # 更新前时相图像(如果is_before为True或None)
            if (is_before is None or is_before) and self.file_path:
                try:
                    pixmap = QPixmap(self.file_path)
                    if not pixmap.isNull():
                        # 保存标签原始状态（如果有）
                        self._save_label_state(self.label_before)
                        # 保存原始尺寸
                        self.before_image_original_size = (pixmap.width(), pixmap.height())
                        # 正常设置pixmap，确保居中显示
                        self.label_before.set_pixmap(pixmap)
                        # 强制重新计算缩放比例
                        # self.label_before.force_resize_recalculate()
                    else:
                        self.log_message(f"无法加载前时相图像: {self.file_path}", "ERROR")
                except Exception as e:
                    self.log_message(f"显示前时相图像时出错: {str(e)}", "ERROR")
            
            # 更新后时相图像(如果is_before为False或None)
            if (is_before is None or is_before is False) and self.file_path_after:
                try:
                    pixmap = QPixmap(self.file_path_after)
                    if not pixmap.isNull():
                        # 保存标签原始状态（如果有）
                        self._save_label_state(self.label_after)
                        # 保存原始尺寸
                        self.after_image_original_size = (pixmap.width(), pixmap.height())
                        # 正常设置pixmap，确保居中显示
                        self.label_after.set_pixmap(pixmap)
                        # 强制重新计算缩放比例
                        # self.label_after.force_resize_recalculate()
                    else:
                        self.log_message(f"无法加载后时相图像: {self.file_path_after}", "ERROR")
                except Exception as e:
                    self.log_message(f"显示后时相图像时出错: {str(e)}", "ERROR")
                
            # 更新解译结果图像(如果is_before为None，表示更新所有图像)
            if is_before is None and hasattr(self, 'result_image') and self.result_image is not None:
                # 调用专门的解译结果图像更新方法
                self.update_result_display()
            
            # 最后确保所有UI更新都被应用    
            QApplication.processEvents()
                
        except Exception as e:
            self.log_message(f"更新图像显示时出错: {str(e)}", "ERROR")

    # 添加显示图像信息的方法
    def show_image_info(self):
        """显示图像的详细信息"""
        self.log_message("正在获取图像详细信息", "START")
        

    
        
        # 显示解译结果图像信息
        if hasattr(self, 'result_image') and self.result_image is not None:
            
            if hasattr(self, 'result_image_path') and self.result_image_path:
                self.log_message(f"结果路径: {self.result_image_path}", "INFO")
            else:
                self.log_message(f"结果路径: 内存图像", "INFO")
                
            if hasattr(self, 'result_image_original_size') and self.result_image_original_size:
                width, height = self.result_image_original_size
                self.log_message(f"图像尺寸: {width} x {height} 像素", "INFO")
            elif self.result_image is not None:
                height, width = self.result_image.shape[:2]
                self.log_message(f"图像尺寸: {width} x {height} 像素", "INFO")
        else:
            self.log_message("解译结果图像未生成", "INFO")
        
        self.log_message("图像信息获取完成", "COMPLETE")

    def add_to_history(self, operation, details=""):
        """添加操作到历史记录
        
        Args:
            operation: 操作名称
            details: 操作详情
        """
        try:
            # 获取当前时间
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 记录日志 - 历史记录一般是重要操作，设为COMPLETE级别
            self.log_message(f"{operation}: {details}", "COMPLETE")
            
            # 保存到历史记录列表
            history_item = {
                "timestamp": timestamp,
                "operation": operation,
                "details": details
            }
            self.history_records.append(history_item)
            
        except Exception as e:
            self.log_message(f"添加历史记录时出错: {str(e)}", "ERROR")
            
    def clear_history(self):
        """清除所有历史记录"""
        self.history_records = []

    def _save_label_state(self, label):
        """保存标签的缩放状态"""
        if hasattr(label, 'objectName') and label.objectName():
            key = label.objectName()
            if hasattr(label, 'scale_factor') and hasattr(label, 'offset'):
                self._label_states[key] = {
                    'scale_factor': label.scale_factor,
                    'offset': QPoint(label.offset)
                }
    
    def _restore_label_state(self, label):
        """恢复标签的缩放状态"""
        if hasattr(label, 'objectName') and label.objectName():
            key = label.objectName()
            if key in self._label_states:
                # 恢复之前保存的状态
                state = self._label_states[key]
                label.scale_factor = state['scale_factor']
                label.offset = QPoint(state['offset'])
                # 更新显示
                label.update_display()

    def update_result_display(self):
        """专门更新解译结果图像显示，使用与前后时相图像相同的处理策略"""
        try:
            # 确保处理前应用所有挂起的UI更新，使标签尺寸准确
            QApplication.processEvents()
            
            if hasattr(self, 'result_image') and self.result_image is not None:
                # 将numpy数组转换为QPixmap
                height, width = self.result_image.shape[:2]
                bytes_per_line = 3 * width
                q_image = QImage(self.result_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                if not pixmap.isNull():
                    # 保存标签原始状态（如果有）
                    self._save_label_state(self.label_result)
                    
                    # 保存原始尺寸
                    self.result_image_original_size = (pixmap.width(), pixmap.height())
                    
                    # 正常设置pixmap，确保居中显示
                    self.label_result.set_pixmap(pixmap)
                    
                    # 添加到历史记录
                    
                    return True
                else:
                    self.log_message("无法创建解译结果图像的Pixmap", "ERROR")
                    return False
            else:
                self.log_message("解译结果图像不存在或为空", "INFO")
                return False
        except Exception as e:
            self.log_message(f"显示解译结果图像时出错: {str(e)}", "ERROR")
            import traceback
            self.log_message(traceback.format_exc(), "ERROR")
            return False

    def set_result_image(self, image_data, image_path=None):
        """设置解译结果图像并显示
        
        Args:
            image_data: numpy数组形式的图像数据
            image_path: 可选的图像路径，如果不提供则标记为内存图像
            
        Returns:
            bool: 是否成功设置和显示图像
        """
        try:
            # 保存图像数据
            self.result_image = image_data
            
            # 保存图像路径
            if image_path:
                self.result_image_path = image_path
            else:
                self.result_image_path = 'memory_image'  # 标记为内存中的图像
                
            # 显示图像
            result = self.update_result_display()
            
            if result:
                return True
            else:
                return False
        except Exception as e:
            self.log_message(f"设置解译结果图像时出错: {str(e)}", "ERROR")
            import traceback
            self.log_message(traceback.format_exc(), "ERROR")
            return False
    