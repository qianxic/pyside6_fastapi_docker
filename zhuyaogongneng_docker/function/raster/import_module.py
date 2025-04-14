"""
栅格影像导入模块
用于导入前后时相栅格影像，保留地理参考信息
支持多种栅格格式，包括GeoTIFF、TIFF等
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import tempfile
from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

# 尝试导入GDAL库
try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    pass

# 尝试导入路径连接器
try:
    from change3d_api_docker.path_connector import path_connector
except ImportError:
    pass

class RasterImporter:
    """栅格影像导入类，用于导入和处理栅格影像数据"""
    
    def __init__(self, navigation_functions):
        """初始化栅格影像导入类"""
        self.navigation_functions = navigation_functions
        self.temp_dir = tempfile.mkdtemp()  # 创建临时目录
        
        # 记录前后时相影像信息
        self.before_image_path = None
        self.after_image_path = None
        self.before_projection = None
        self.after_projection = None
        self.before_geotransform = None
        self.after_geotransform = None
        
        # 检查GDAL是否可用
        if not GDAL_AVAILABLE:
            self.navigation_functions.log_message("警告: GDAL库未安装，栅格影像导入功能将受限")
            
        # 检查API可用性
        self.api_available = path_connector is not None
        if self.api_available:
            self.navigation_functions.log_message("API路径处理服务已加载")
    
    def check_gdal_available(self):
        """检查GDAL是否可用，如果不可用则提示用户"""
        if not GDAL_AVAILABLE:
            self.navigation_functions.log_message("错误: 栅格影像导入需要GDAL库，但未能找到")
            QMessageBox.critical(
                self.navigation_functions.main_window, 
                "依赖缺失", 
                "栅格影像导入需要GDAL库，请安装GDAL后再试。\n"
                "可以使用: pip install GDAL 进行安装。"
            )
            return False
        return True
    
    def select_raster_file(self, title="选择栅格影像文件", initial_dir=None):
        """
        选择栅格影像文件
        
        参数:
            title: 对话框标题
            initial_dir: 初始目录
            
        返回:
            选择的文件路径，如果取消则返回空字符串
        """
        # 检查GDAL是否可用
        if not self.check_gdal_available():
            return ""
            
        # 打开文件选择对话框
        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        
        if initial_dir:
            file_dialog.setDirectory(initial_dir)
            
        file_path, _ = file_dialog.getOpenFileName(
            self.navigation_functions.main_window,
            title,
            "",
            "栅格影像文件 (*.tif *.tiff *.img *.dem *.hgt);;GeoTIFF文件 (*.tif *.tiff);;所有文件 (*.*)",
            options=options
        )
        
        return file_path
    
    def import_before_image(self):
        """导入前时相栅格影像"""
        # 记录开始信息
        
        # 使用通用的文件选择函数
        file_path = self.select_raster_file("选择前时相栅格影像")
        
        if not file_path:
            self.navigation_functions.log_message("未选择文件", "INFO")
            return False
            
        # 记录已选择的文件
        self.navigation_functions.log_message(f"已选择前时相影像: {file_path}", "INFO")
            
        # 加载栅格影像
        result = self.load_raster_image(file_path, is_before=True)
        
        if result is not None:
            # 记录文件路径
            self.before_image_path = file_path
            # 保存到navigation_functions用于API路径处理
            if hasattr(self.navigation_functions, 'raster_file_path'):
                self.navigation_functions.raster_file_path = file_path
                
            # 保存文件路径到navigation_functions中
            self.navigation_functions.file_path = file_path
                
            # 显示影像 - 使用多种方法尝试显示
            self.display_before_image(result)
            
            # 确保影像被正确存储
            self.navigation_functions.before_image = result
            
            # 记录导入成功
            self.navigation_functions.log_message("前时相影像导入完成", "COMPLETE")
            
            # 通知API路径已更新（如果API可用且后时相已导入）
            if self.api_available and self.after_image_path:
                try:
                    # 尝试使用API处理路径
                    result = path_connector.process_single_raster_paths(
                        self.before_image_path,
                        self.after_image_path
                    )
                    self.navigation_functions.log_message("已通知API路径更新", "INFO")
                    self.navigation_functions.log_message(f"API临时目录: {result.get('temp_dir')}", "INFO")
                    self.navigation_functions.log_message(f"API输出路径: {result.get('output_path')}", "INFO")
                except Exception as e:
                    self.navigation_functions.log_message(f"API路径更新通知失败: {str(e)}", "INFO")
                    
            return True
        else:
            # 记录导入失败
            self.navigation_functions.log_message("前时相影像导入失败", "ERROR")
            return False
    
    def import_after_image(self):
        """导入后时相栅格影像"""
        # 记录开始信息
        
        # 使用通用的文件选择函数，如果已有前时相影像，从其所在目录开始浏览
        initial_dir = os.path.dirname(self.before_image_path) if self.before_image_path else None
        file_path = self.select_raster_file("选择后时相栅格影像", initial_dir)
        
        if not file_path:
            self.navigation_functions.log_message("未选择文件", "INFO")
            return False
            
        # 记录已选择的文件
        self.navigation_functions.log_message(f"已选择后时相影像: {file_path}", "INFO")
            
        # 加载栅格影像
        result = self.load_raster_image(file_path, is_before=False)
        
        if result is not None:
            # 记录文件路径
            self.after_image_path = file_path
            # 保存到navigation_functions用于API路径处理
            if hasattr(self.navigation_functions, 'raster_file_path_after'):
                self.navigation_functions.raster_file_path_after = file_path
                
            # 保存文件路径到navigation_functions中
            self.navigation_functions.file_path_after = file_path
                
            # 显示影像 - 使用多种方法尝试显示
            self.display_after_image(result)
            
            # 确保影像被正确存储
            self.navigation_functions.after_image = result
            
            # 记录导入成功
            self.navigation_functions.log_message("后时相影像导入完成", "COMPLETE")
            
            # 检查前后时相投影是否一致
            self.check_projections()
            
            # 通知API路径已更新（如果API可用且前时相已导入）
            if self.api_available and self.before_image_path:
                try:
                    # 尝试使用API处理路径
                    result = path_connector.process_single_raster_paths(
                        self.before_image_path,
                        self.after_image_path
                    )
                    self.navigation_functions.log_message("已通知API路径更新", "INFO")
                    self.navigation_functions.log_message(f"API临时目录: {result.get('temp_dir')}", "INFO")
                    self.navigation_functions.log_message(f"API输出路径: {result.get('output_path')}", "INFO")
                except Exception as e:
                    self.navigation_functions.log_message(f"API路径更新通知失败: {str(e)}", "INFO")
            
            return True
        else:
            # 记录导入失败
            self.navigation_functions.log_message("后时相影像导入失败", "ERROR")
            return False
    
    def load_raster_image(self, image_path, is_before=True):
        """
        加载栅格影像，并保存投影信息
        
        参数:
            image_path: 栅格影像路径
            is_before: 是否为前时相影像
            
        返回:
            加载后的OpenCV格式影像，失败返回None
        """
        if not self.check_gdal_available():
            return None
            
        try:
            # 使用GDAL打开栅格文件
            dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
            if dataset is None:
                self.navigation_functions.log_message(f"无法打开栅格文件: {image_path}", "ERROR")
                return None
                
            # 获取投影信息
            projection = dataset.GetProjection()
            geotransform = dataset.GetGeoTransform()
            
            # 获取影像尺寸和波段数
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            band_count = dataset.RasterCount
            
            self.navigation_functions.log_message(f"栅格尺寸: {width}×{height}, 波段数: {band_count}", "INFO")
            
            # 读取波段数据
            try:
                if band_count >= 3:  # RGB影像
                    # 读取3个波段数据
                    red_band = dataset.GetRasterBand(1).ReadAsArray()
                    green_band = dataset.GetRasterBand(2).ReadAsArray()
                    blue_band = dataset.GetRasterBand(3).ReadAsArray()
                    
                    # 创建OpenCV BGR图像
                    img = np.dstack((blue_band, green_band, red_band))
                    
                else:  # 单波段影像
                    # 读取单波段数据
                    band = dataset.GetRasterBand(1).ReadAsArray()
                    
                    # 转为三通道灰度图
                    img = np.dstack((band, band, band))
            except Exception as e:
                # 尝试使用替代方法
                self.navigation_functions.log_message("尝试使用替代方法读取栅格数据", "INFO")
                
                # 创建临时文件
                temp_file = os.path.join(self.temp_dir, f"temp_{os.path.basename(image_path)}")
                
                # 使用GDAL转换为临时TIFF文件
                driver = gdal.GetDriverByName('GTiff')
                temp_ds = driver.CreateCopy(temp_file, dataset)
                temp_ds = None  # 关闭数据集
                
                # 使用OpenCV读取临时文件
                img = cv2.imread(temp_file)
                
                # 如果仍然失败，抛出异常
                if img is None:
                    raise Exception("无法使用替代方法读取栅格数据")
            
            # 保存投影信息
            if is_before:
                self.before_projection = projection
                self.before_geotransform = geotransform
                # 同时保存到检测模块
                if hasattr(self.navigation_functions, 'raster_cd'):
                    self.navigation_functions.raster_cd.before_image_path = image_path
                    self.navigation_functions.raster_cd.projection_info = projection
                    self.navigation_functions.raster_cd.geotransform = geotransform
            else:
                self.after_projection = projection
                self.after_geotransform = geotransform
                # 同时保存到检测模块
                if hasattr(self.navigation_functions, 'raster_cd'):
                    self.navigation_functions.raster_cd.after_image_path = image_path
            
            # 关闭数据集
            dataset = None
            
            # 保存影像到导航函数
            if is_before:
                self.navigation_functions.before_image = img
            else:
                self.navigation_functions.after_image = img
            
            # 返回图像
            return img
            
        except Exception as e:
            self.navigation_functions.log_message(f"加载栅格影像失败: {str(e)}", "ERROR")
            
            # 尝试使用OpenCV直接读取
            try:
                self.navigation_functions.log_message("尝试使用OpenCV直接读取影像", "INFO")
                img = cv2.imread(image_path)
                if img is not None:
                    # 保存影像到导航函数
                    if is_before:
                        self.navigation_functions.before_image = img
                    else:
                        self.navigation_functions.after_image = img
                    
                    return img
            except:
                pass
                
            return None
    
    def check_projections(self):
        """检查前后时相投影是否一致，如果不一致则提示用户"""
        if self.before_projection and self.after_projection:
            if self.before_projection != self.after_projection:
                self.navigation_functions.log_message("警告: 前后时相栅格影像投影不一致，可能影响变化检测结果")
                QMessageBox.warning(
                    self.navigation_functions.main_window,
                    "投影不一致",
                    "前后时相栅格影像投影不一致，可能影响变化检测结果。\n"
                    "建议先将影像投影转换为相同的坐标系统。"
                )
            else:
                self.navigation_functions.log_message("前后时相栅格影像投影一致")
        
        if self.before_geotransform and self.after_geotransform:
            # 检查分辨率是否匹配
            before_resolution = (self.before_geotransform[1], self.before_geotransform[5])
            after_resolution = (self.after_geotransform[1], self.after_geotransform[5])
            
            if before_resolution != after_resolution:
                self.navigation_functions.log_message(f"警告: 前后时相栅格影像分辨率不一致: "
                                                    f"前时相 {before_resolution}, 后时相 {after_resolution}")
                QMessageBox.warning(
                    self.navigation_functions.main_window,
                    "分辨率不一致",
                    "前后时相栅格影像分辨率不一致，可能影响变化检测结果。\n"
                    "建议先将影像重采样为相同的分辨率。"
                )
            else:
                self.navigation_functions.log_message("前后时相栅格影像分辨率一致")
    
    def get_raster_info(self, raster_path):
        """
        获取栅格影像信息
        
        参数:
            raster_path: 栅格影像路径
            
        返回:
            包含栅格信息的字典，失败返回None
        """
        if not self.check_gdal_available():
            return None
            
        try:
            # 使用GDAL打开栅格文件
            dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
            if dataset is None:
                self.navigation_functions.log_message(f"无法打开栅格文件: {raster_path}")
                return None
                
            # 获取基本信息
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            band_count = dataset.RasterCount
            projection = dataset.GetProjection()
            geotransform = dataset.GetGeoTransform()
            
            # 获取栅格波段信息
            bands_info = []
            for i in range(1, band_count + 1):
                band = dataset.GetRasterBand(i)
                band_info = {
                    'index': i,
                    'type': gdal.GetDataTypeName(band.DataType),
                    'min': band.GetMinimum() or '',
                    'max': band.GetMaximum() or '',
                    'nodata': band.GetNoDataValue() or ''
                }
                bands_info.append(band_info)
            
            # 获取投影信息
            projection_info = "未定义"
            if projection:
                srs = osr.SpatialReference()
                srs.ImportFromWkt(projection)
                projection_info = srs.ExportToProj4()
            
            # 获取分辨率
            if geotransform:
                x_resolution = geotransform[1]
                y_resolution = abs(geotransform[5])
            else:
                x_resolution = y_resolution = 0
            
            # 关闭数据集
            dataset = None
            
            # 返回信息字典
            return {
                'path': raster_path,
                'width': width,
                'height': height,
                'band_count': band_count,
                'projection': projection_info,
                'x_resolution': x_resolution,
                'y_resolution': y_resolution,
                'bands': bands_info,
                'geotransform': geotransform
            }
            
        except Exception as e:
            import traceback
            error_msg = f"获取栅格信息失败: {str(e)}"
            self.navigation_functions.log_message(error_msg)
            self.navigation_functions.log_message(traceback.format_exc())
            return None
    
    def display_raster_info(self, raster_path):
        """
        显示栅格影像信息对话框
        
        参数:
            raster_path: 栅格影像路径
            
        返回:
            是否成功显示信息
        """
        # 获取栅格信息
        info = self.get_raster_info(raster_path)
        if not info:
            return False
        
        # 构建信息文本
        info_text = f"文件路径: {info['path']}\n"
        info_text += f"影像尺寸: {info['width']}×{info['height']} 像素\n"
        info_text += f"波段数量: {info['band_count']}\n"
        info_text += f"投影信息: {info['projection']}\n"
        info_text += f"空间分辨率: {info['x_resolution']} (X), {info['y_resolution']} (Y)\n\n"
        
        info_text += "波段信息:\n"
        for band in info['bands']:
            info_text += f"  波段 {band['index']}: 类型 {band['type']}"
            if band['min'] and band['max']:
                info_text += f", 范围 {band['min']} - {band['max']}"
            if band['nodata']:
                info_text += f", 无数据值 {band['nodata']}"
            info_text += "\n"
        
        # 显示信息对话框
        QMessageBox.information(
            self.navigation_functions.main_window,
            "栅格影像信息",
            info_text
        )
        
        return True
    
    def display_before_image(self, image):
        """显示前时相栅格影像
        
        参数:
            image: 图像数组（OpenCV格式）
        """
        try:
            # 使用最可靠的方法显示图像
            if hasattr(self.navigation_functions, 'update_image_display'):
                self.navigation_functions.update_image_display(is_before=True)
                return
                
            # 如果update_image_display不可用，使用_display_image_to_label方法
            if hasattr(self.navigation_functions, 'label_before'):
                self._display_image_to_label(image, self.navigation_functions.label_before)
                return
                
            # 如果都不可用，记录警告
            self.navigation_functions.log_message("警告: 无法显示前时相影像，未找到有效的显示方法", "WARNING")
            
        except Exception as e:
            self.navigation_functions.log_message(f"显示前时相影像时出错: {str(e)}", "ERROR")
    
    def display_after_image(self, image):
        """显示后时相栅格影像
        
        参数:
            image: 图像数组（OpenCV格式）
        """
        try:
            # 使用最可靠的方法显示图像
            if hasattr(self.navigation_functions, 'update_image_display'):
                self.navigation_functions.update_image_display(is_before=False)
                return
                
            # 如果update_image_display不可用，使用_display_image_to_label方法
            if hasattr(self.navigation_functions, 'label_after'):
                self._display_image_to_label(image, self.navigation_functions.label_after)
                return
                
            # 如果都不可用，记录警告
            self.navigation_functions.log_message("警告: 无法显示后时相影像，未找到有效的显示方法", "WARNING")
            
        except Exception as e:
            self.navigation_functions.log_message(f"显示后时相影像时出错: {str(e)}", "ERROR")
    
    def _display_image_to_label(self, image, label):
        """将图像显示到标签
        
        参数:
            image: OpenCV格式图像数组
            label: 目标标签
        """
        try:
            # 检查image是否为None
            if image is None:
                self.navigation_functions.log_message("错误: 图像数据为空，无法显示", "ERROR")
                return
                
            # 检查图像维度
            if len(image.shape) < 2:
                self.navigation_functions.log_message("错误: 图像数据格式不正确，无法显示", "ERROR")
                return
                
            # 转换为RGB
            if len(image.shape) == 3:
                # OpenCV图像是BGR格式，需要转换为RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # 单通道图像转换为RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # 创建QImage
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 创建QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            
            # 确保标签实例存在
            if label is None:
                self.navigation_functions.log_message("错误: 标签对象为空，无法显示图像", "ERROR")
                return
                
            # 如果标签有set_pixmap方法（ZoomableLabel），则使用它
            if hasattr(label, 'set_pixmap'):
                label.set_pixmap(pixmap)
            else:
                # 否则直接设置pixmap
                label.setPixmap(pixmap)
                label.setScaledContents(True)
            
            # 手动更新标签显示
            label.update()
            
            # 确保图像被导入到主应用程序
            if hasattr(self.navigation_functions, 'main_window') and self.navigation_functions.main_window is not None:
                if hasattr(self.navigation_functions.main_window, 'repaint'):
                    self.navigation_functions.main_window.repaint()
            
        except Exception as e:
            self.navigation_functions.log_message(f"显示图像时出错: {str(e)}", "ERROR")
    
    def execute_detection(self):
        """执行影像检测"""
        self.navigation_functions.log_message("开始执行影像检测", "INFO")
        
        # 验证必要条件
        if not self.task_name or self.task_name.strip() == "":
            self.navigation_functions.log_message("任务名称不能为空", "ERROR")
            return

        if not self.before_image_path or not os.path.exists(self.before_image_path):
            self.navigation_functions.log_message("变化前影像路径不存在", "ERROR")
            return

        if not self.after_image_path or not os.path.exists(self.after_image_path):
            self.navigation_functions.log_message("变化后影像路径不存在", "ERROR")
            return

        # 获取文件名
        b_file_name = os.path.basename(self.before_image_path)
        a_file_name = os.path.basename(self.after_image_path)
        self.navigation_functions.log_message(f"变化前影像: {b_file_name}", "INFO")
        self.navigation_functions.log_message(f"变化后影像: {a_file_name}", "INFO")

        try:
            # 读取影像
            self.navigation_functions.log_message("正在读取影像...", "INFO")
            before_image = cv2.imread(self.before_image_path, cv2.IMREAD_UNCHANGED)
            after_image = cv2.imread(self.after_image_path, cv2.IMREAD_UNCHANGED)
            
            if before_image is None or after_image is None:
                self.navigation_functions.log_message("无法读取影像文件", "ERROR")
                return

            # 显示影像 - 使用最简单的方法
            if hasattr(self.navigation_functions, 'update_image_display'):
                self.navigation_functions.update_image_display(self.before_image_path, is_before=True)
                self.navigation_functions.update_image_display(self.after_image_path, is_before=False)
            
            # 保存影像数据
            self.before_image_data = before_image
            self.after_image_data = after_image
            
            # 更新状态
            self.status = "图像已加载"
            self.navigation_functions.log_message("影像加载完成", "COMPLETE")
            
            # 通知API路径已更新
            if hasattr(self.navigation_functions, 'notify_api_path_updated'):
                self.navigation_functions.notify_api_path_updated('before_image_path', self.before_image_path)
                self.navigation_functions.notify_api_path_updated('after_image_path', self.after_image_path)
                self.navigation_functions.log_message("已通知API路径更新", "INFO")
            
            return True
            
        except Exception as e:
            self.navigation_functions.log_message(f"执行影像检测时出错: {str(e)}", "ERROR")
            return False 