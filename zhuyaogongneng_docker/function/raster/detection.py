"""
栅格影像变化检测模块
用于处理栅格影像数据的变化检测，支持多种检测算法
"""

# 抑制TIFF相关警告
import os
os.environ['QT_LOGGING_RULES'] = "qt.imageformats.tiff=false"

import sys
import traceback
import time
import json
import numpy as np
import cv2
from pathlib import Path
import threading
import logging
from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from osgeo import gdal
GDAL_AVAILABLE = True

# 尝试导入路径连接器
from zhuyaogongneng_docker.function.detection_client import detect_changes, check_connection


# 尝试导入GDAL以支持栅格影像处理

class RasterChangeDetection:
    """栅格影像变化检测类"""
    
    def __init__(self, navigation_functions):
        """初始化栅格影像变化检测类"""
        self.navigation_functions = navigation_functions
        self.result_image = None
        self.before_image_path = None
        self.after_image_path = None
        self.threshold = 30
        
        # 检查API可用性
        self.api_available = check_connection()
        
        # 检查GDAL是否可用
        if not GDAL_AVAILABLE:
            self.navigation_functions.log_message("警告: GDAL库未安装，栅格影像变化检测功能将受限", "WARNING")
    
    # 增加统一格式的日志方法
    def log_message(self, message, level="COMPLETE"):
        """记录消息到日志，与图像版本保持一致的格式
        
        Args:
            message: 消息内容
            level: 日志级别，可以是"INFO"、"COMPLETE"、"ERROR"等
        """
        if hasattr(self.navigation_functions, 'log_message'):
            self.navigation_functions.log_message(message, level)
            
    def check_gdal_available(self):
        """检查GDAL是否可用"""
        return GDAL_AVAILABLE

    def on_begin_clicked(self):
        """处理开始检测按钮点击事件"""
        # 检查前后时相影像是否已导入
        if not hasattr(self.navigation_functions, 'file_path') or not self.navigation_functions.file_path:
            self.log_message("错误: 未导入前时相影像", "ERROR")
            return False
        
        if not hasattr(self.navigation_functions, 'file_path_after') or not self.navigation_functions.file_path_after:
            self.log_message("错误: 未导入后时相影像", "ERROR")
            return False
            
        # 让用户选择保存文件夹
        default_dir = os.path.dirname(self.navigation_functions.file_path_after)
        output_folder = QFileDialog.getExistingDirectory(None, "选择保存文件夹", default_dir)
        
        if not output_folder:
            # 用户取消了操作
            return False
            
        # 转换所有路径为绝对路径
        before_path = os.path.abspath(self.navigation_functions.file_path).replace("\\", "/")
        after_path = os.path.abspath(self.navigation_functions.file_path_after).replace("\\", "/")
        output_path = os.path.abspath(output_folder).replace("\\", "/") 
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 记录当前处理模式 - 与图像版本保持一致的格式
        self.log_message("使用raster模式进行变化检测", "START")
        
        # 创建子线程执行变化检测
        detection_thread = threading.Thread(
            target=self._execute_detection,
            args=(before_path, after_path, output_path)
        )
        detection_thread.daemon = True
        detection_thread.start()
        
        return True
    
    def _execute_detection(self, before_path, after_path, output_path):
        """在子线程中执行变化检测
        
        Args:
            before_path: 前时相影像路径
            after_path: 后时相影像路径
            output_path: 输出路径
        """
        try:
            # 确保所有路径使用正斜杠并为绝对路径
            before_path = os.path.abspath(before_path).replace("\\", "/")
            after_path = os.path.abspath(after_path).replace("\\", "/")
            output_path = os.path.abspath(output_path).replace("\\", "/")
            
            # 确保输出目录存在
            os.makedirs(output_path, exist_ok=True)
            
            self.log_message("开始执行栅格变化检测，请稍后...", "START")
            
            try:
                # 调用检测接口执行变化检测
                task_result = detect_changes(
                    before_path=before_path,
                    after_path=after_path,
                    output_path=output_path,
                    mode="single_raster"
                )
            except Exception as e:
                self.log_message(f"错误: 调用变化检测接口失败: {str(e)}", "ERROR")
                self.on_detection_finished((False, None))
                return
            
            # -----------------------------------------------------

            # --- 直接处理 detect_changes 返回的最终结果 --- 

            # 从最终结果中获取重命名后的显示图像路径
            display_image_path = ""
            final_status = task_result.get("status")
            vector_files = [] # Initialize vector files list

            if final_status == "completed":
                display_image_path = task_result.get("display_image_path")
                
                # 获取矢量文件列表（路径现在指向最终用户目录下的文件）
                final_output_dir = task_result.get("output_path")
                if final_output_dir:
                    vectors_subdir = os.path.join(final_output_dir, "vectors")
                    changes_shp = os.path.join(vectors_subdir, "changes.shp")
                    changes_geojson = os.path.join(vectors_subdir, "changes.geojson")
                    if os.path.exists(changes_shp): vector_files.append(changes_shp)
                    if os.path.exists(changes_geojson): vector_files.append(changes_geojson)
                    # Add auxiliary files if shapefile exists
                    if os.path.exists(changes_shp):
                        for ext in [".dbf", ".shx", ".prj", ".cpg", ".qpj"]:
                            aux_file = os.path.join(vectors_subdir, f"changes{ext}")
                            if os.path.exists(aux_file):
                                vector_files.append(aux_file)
                
                if vector_files:
                    self.log_message("生成的矢量文件 (在最终输出目录中):", "INFO")
                    for vf in vector_files:
                        self.log_message(f"  - {vf}", "INFO")
                        
            else:
                logging.warning(f"### [Client Single Raster] Task status is NOT completed: {final_status}") # ADDED LOG
            
            # 检查路径是否存在
            if not display_image_path or not os.path.exists(display_image_path):
                self.log_message(f"错误: 未找到要显示的最终结果图像。状态: {final_status}, 路径: '{display_image_path}'", "ERROR")
                self.on_detection_finished((False, None))
                return
                
            # 读取最终的掩码图像 (栅格通常是 TIF)
            self.log_message(f"正在读取检测结果图像: {display_image_path}", "INFO")
            # Use GDAL to read the result raster to preserve geo-info if needed later
            mask_dataset = gdal.Open(display_image_path) if GDAL_AVAILABLE else None
            if mask_dataset:
                 mask_img = mask_dataset.ReadAsArray()
                 # Handle multi-band masks if necessary, assuming single band for now
                 if len(mask_img.shape) > 2:
                     mask_img = mask_img[0] # Take the first band 
                 mask_dataset = None # Close dataset
            else:
                # Fallback to OpenCV if GDAL fails or is unavailable
                mask_img = cv2.imread(display_image_path, cv2.IMREAD_GRAYSCALE) 
            
            if mask_img is None:
                 self.log_message(f"错误: 无法读取结果栅格图像 '{display_image_path}'", "ERROR")
                 self.on_detection_finished((False, None))
                 return
            
            # 处理结果
            self.log_message("影像变化检测完成，结果已输出到指定位置", "COMPLETE")
            self.on_detection_finished((True, mask_img))
            
        except Exception as e:
            logging.error(f"### [Client Single Raster] Error in _execute_detection: {e}", exc_info=True) # ADDED LOG
            self.log_message(f"影像变化检测过程中出错: {str(e)}", "ERROR")
            self.on_detection_finished((False, None))

    



    def use_api_detection(self):
        """使用API进行变化检测 - ，重定向到on_begin_clicked方法"""
        self.on_begin_clicked()
    
    
    def on_detection_finished(self, result):
        """变化检测完成回调"""
        success, result_image = result
        
        if success and result_image is not None:
            
            # 保存结果
            self.result_image = result_image
            
            # 显示结果
            if hasattr(self.navigation_functions, 'label_result'):
                self.display_result(self.result_image, self.navigation_functions.label_result)
        else:
            self.log_message("解译失败: 未能生成有效的结果图像", "ERROR")
    

    
    def has_detection_result(self):
        """检查是否有变化检测结果"""
        return self.result_image is not None
    
    def display_result(self, image, label):
        """
        在标签上显示结果图像
        
        Args:
            image: 要显示的图像数组
            label: 目标显示标签
        """
        try:
            # 确保标签存在且支持图像显示
            if not hasattr(label, 'setPixmap') and not hasattr(label, 'set_pixmap'):
                self.log_message("错误: 标签对象不支持图像显示", "ERROR")
                return
                
            # 转换图像格式 - 无需检查图像是否为空，直接处理
            if image is not None and len(image.shape) >= 2:
                if len(image.shape) == 3:
                    height, width, channel = image.shape
                    bytes_per_line = 3 * width
                    # BGR转RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # 创建QImage
                    q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                else:
                    height, width = image.shape
                    # 单通道图像转换为RGB以确保兼容性
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    height, width, channel = image_rgb.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # 创建QPixmap
                pixmap = QPixmap.fromImage(q_image)
                
                # 检查是否是ZoomableLabel类型
                if hasattr(label, 'set_pixmap'):
                    # 使用ZoomableLabel的set_pixmap方法
                    label.set_pixmap(pixmap)
                else:
                    # 常规QLabel
                    label.setPixmap(pixmap)
                    label.setScaledContents(True)
                
            else:
                self.log_message("警告: 图像数据无效，无法显示", "WARNING")
                
        except Exception as e:
            self.log_message(f"显示结果图像出错: {str(e)}", "ERROR")
    
    def set_detection_parameters(self, threshold=30):
        """设置变化检测参数"""
        self.threshold = threshold
        self.navigation_functions.log_message(f"已设置变化检测参数: 阈值={threshold}", "INFO")
    
    def read_image_from_path(self, image_path):
        """从路径读取影像
        
        Args:
            image_path: 影像文件路径
            
        Returns:
            读取的影像数组
        """
        try:
            if GDAL_AVAILABLE:
                # 使用GDAL读取地理参考影像
                dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
                if dataset:
                    # 获取影像尺寸和波段数
                    width = dataset.RasterXSize
                    height = dataset.RasterYSize
                    bands = dataset.RasterCount
                    
                    # 读取影像数据
                    if bands >= 3:
                        # 多波段，读取RGB
                        red = dataset.GetRasterBand(1).ReadAsArray()
                        green = dataset.GetRasterBand(2).ReadAsArray()
                        blue = dataset.GetRasterBand(3).ReadAsArray()
                        
                        # 创建RGB图像
                        image = np.dstack((blue, green, red))
                    else:
                        # 单波段
                        image = dataset.GetRasterBand(1).ReadAsArray()
                    
                    # 关闭数据集
                    dataset = None
                    
                    return image
            
            # 如果GDAL不可用或GDAL读取失败，尝试使用OpenCV
            image = cv2.imread(image_path)
            return image
            
        except Exception as e:
            self.navigation_functions.log_message(f"读取影像出错: {str(e)}", "ERROR")
            return None
    