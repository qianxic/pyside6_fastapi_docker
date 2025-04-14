import os
import numpy as np
import logging
from pathlib import Path
from PIL import Image
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import cv2

class ImageDisplay:
    def __init__(self, navigation_functions):
        """
        初始化图像显示模块
        
        Args:
            navigation_functions: NavigationFunctions实例，用于日志记录和图像显示
        """
        self.navigation_functions = navigation_functions
        
    def display_image(self, file_path, is_before=True):
        """
        显示图像
        
        Args:
            file_path: 图像文件路径
            is_before: 是否为前时相图像
        """
        try:
            # 使用常规方法加载图像
            pixmap = QPixmap(file_path)
            
            if not pixmap.isNull():
                # 获取图像尺寸，存储原始尺寸信息
                width = pixmap.width()
                height = pixmap.height()
                
                # 保存原始尺寸信息（用于后续可能的操作）
                if is_before:
                    self.navigation_functions.before_image_original_size = (width, height)
                else:
                    self.navigation_functions.after_image_original_size = (width, height)
                
                # 根据是前时相还是后时相选择不同的标签
                label = self.navigation_functions.label_before if is_before else self.navigation_functions.label_after
                
                # 设置图像到可缩放标签
                if hasattr(label, 'set_pixmap'):
                    label.set_pixmap(pixmap)
                    self.navigation_functions.log_message(f"{'前' if is_before else '后'}时相影像加载成功 (放大图像后，双击可恢复原始视图)")
                else:
                    # 如果标签不是可缩放标签，则直接设置
                    label.setPixmap(pixmap.scaled(
                        label.width(), 
                        label.height(), 
                        aspectRatioMode=Qt.KeepAspectRatio
                    ))
                    self.navigation_functions.log_message(f"{'前' if is_before else '后'}时相影像加载成功")
            else:
                self.navigation_functions.log_message("无法加载图像：图像为空")
                    
        except Exception as e:
            self.navigation_functions.log_message(f"加载图像时出错: {str(e)}")
            import traceback
            self.navigation_functions.log_message(traceback.format_exc())
