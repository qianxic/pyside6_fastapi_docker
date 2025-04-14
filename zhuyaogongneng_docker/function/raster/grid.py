"""
栅格网格裁剪模块
用于将栅格影像按网格划分并裁剪，保留地理参考信息
支持多种栅格格式，包括GeoTIFF、TIFF等
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
import tempfile
from datetime import datetime
from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication, QProgressDialog
from PySide6.QtCore import Qt, Signal, QObject

# 尝试导入栅格处理模块
try:
    from zhuyaogongneng_docker.function.raster.import_module import RasterImporter
    HAS_RASTER_IMPORTER = True
except ImportError:
    HAS_RASTER_IMPORTER = False

# 尝试导入GDAL库
try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    try:
        import gdal
        GDAL_AVAILABLE = True
    except ImportError:
        GDAL_AVAILABLE = False

class RasterGridCropping:
    """栅格影像网格裁剪类，将栅格影像按网格裁剪，保留地理参考信息"""
    
    def __init__(self, navigation_functions):
        """初始化栅格网格裁剪类"""
        self.navigation_functions = navigation_functions
        self.temp_dir = tempfile.mkdtemp()  # 创建临时目录
        
        # 创建RasterImporter实例用于文件选择
        try:
            self.raster_importer = RasterImporter(navigation_functions)
        except:
            self.raster_importer = None
            
        # 检查GDAL是否可用
        if not GDAL_AVAILABLE:
            self.navigation_functions.log_message("警告: GDAL库未安装，栅格网格裁剪功能将受限")
    
    def check_gdal_available(self):
        """检查GDAL是否可用，如果不可用则提示用户"""
        if not GDAL_AVAILABLE:
            self.navigation_functions.log_message("错误: 栅格网格裁剪需要GDAL库，但未能找到")
            QMessageBox.critical(
                self.navigation_functions.main_window, 
                "依赖缺失", 
                "栅格网格裁剪需要GDAL库，请安装GDAL后再试。\n"
                "可以使用: pip install GDAL 进行安装。"
            )
            return False
        return True
    
    def select_input_file(self):
        """选择输入栅格文件"""
        # 如果可以使用RasterImporter的选择文件方法
        if hasattr(self, 'raster_importer') and self.raster_importer is not None:
            file_path = self.raster_importer.select_raster_file("选择栅格影像文件")
        else:
            # 备用方法：使用通用栅格文件过滤器
            file_path, _ = QFileDialog.getOpenFileName(
                self.navigation_functions.main_window,
                "选择栅格影像文件",
                str(Path.home()),
                "栅格影像文件 (*.tif *.tiff *.img *.dem *.hgt);;GeoTIFF文件 (*.tif *.tiff);;ERDAS IMG文件 (*.img);;所有文件 (*.*)"
            )
            
            # 检查文件扩展名
            if file_path:
                ext = os.path.splitext(file_path)[1].lower()
                valid_extensions = ['.tif', '.tiff', '.img', '.dem', '.hgt']
                if ext not in valid_extensions:
                    self.navigation_functions.log_message(f"警告: 所选文件不是支持的栅格格式: {file_path}")
                    QMessageBox.warning(
                        self.navigation_functions.main_window,
                        "格式不支持",
                        f"请选择支持的栅格影像格式文件: {', '.join(valid_extensions)}"
                    )
                    return None
        
        if not file_path:
            return None
            
        self.navigation_functions.log_message(f"已选择栅格文件: {file_path}")
        return file_path
    
    def select_output_directory(self):
        """选择输出目录"""
        output_dir = QFileDialog.getExistingDirectory(
            self.navigation_functions.main_window,
            "选择网格裁剪输出目录",
            str(Path.home())
        )
        
        if not output_dir:
            return None
            
        self.navigation_functions.log_message(f"已选择输出目录: {output_dir}")
        return output_dir
    
    def start_grid_cropping(self, input_file, output_dir, rows, cols):
        """
        开始网格裁剪任务
        
        参数:
            input_file: 输入栅格文件路径
            output_dir: 输出目录
            rows: 网格行数
            cols: 网格列数
            
        返回:
            是否成功启动裁剪任务
        """
        # 检查GDAL是否可用
        if not self.check_gdal_available():
            return False
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            self.navigation_functions.log_message(f"错误: 输入文件不存在: {input_file}")
            return False
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建进度对话框，使用类变量以确保不被过早释放
        self.progress_dialog = QProgressDialog("正在执行网格裁剪...", "取消", 0, 100)
        self.progress_dialog.setWindowTitle("栅格网格裁剪")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)  # 立即显示
        self.progress_dialog.show()
        QApplication.processEvents()
        
        # 使用 try-finally 确保对话框始终会关闭
        try:
            # 更新进度
            self.progress_dialog.setValue(10)
            QApplication.processEvents()
            
            # 裁剪影像
            cropped_files = self.crop_raster_grid(input_file, output_dir, rows, cols, self.progress_dialog)
            
            # 更新进度并关闭对话框
            self.progress_dialog.setValue(100)
            QApplication.processEvents()
            
            # 判断是否成功
            if cropped_files and len(cropped_files) > 0:
                self.navigation_functions.log_message(f"网格裁剪完成，共生成 {len(cropped_files)} 个子图像")
                return True
            else:
                self.navigation_functions.log_message("网格裁剪失败，未生成任何子图像")
                return False
            
        except Exception as e:
            import traceback
            error_msg = f"网格裁剪过程中发生错误: {str(e)}"
            self.navigation_functions.log_message(error_msg)
            self.navigation_functions.log_message(traceback.format_exc())
            
            # 显示错误对话框
            try:
                QMessageBox.critical(self.navigation_functions.main_window, "裁剪错误", error_msg)
            except:
                # 如果无法显示错误对话框，只记录日志
                pass
                
            return False
        finally:
            # 确保关闭进度对话框，防止悬挂资源
            try:
                self.progress_dialog.close()
                # 延迟删除对象，避免立即回收可能导致的闪退
                self.progress_dialog.deleteLater()
                self.progress_dialog = None
                # 手动触发垃圾回收
                import gc
                gc.collect()
            except:
                pass
            
            # 刷新UI事件队列
            QApplication.processEvents()
    
    def crop_raster_grid(self, raster_path, output_dir, rows, cols, progress_dialog=None):
        """
        将栅格影像裁剪为网格
        
        参数:
            raster_path: 栅格文件路径
            output_dir: 输出目录
            rows: 网格行数
            cols: 网格列数
            progress_dialog: 进度对话框，用于更新进度
            
        返回:
            生成的子图像文件路径列表
        """
        # 变量初始化
        dataset = None
        generated_files = []
        
        try:
            # 使用GDAL打开栅格文件
            dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
            if dataset is None:
                self.navigation_functions.log_message(f"无法打开栅格文件: {raster_path}")
                return []
            
            # 获取栅格信息
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            band_count = dataset.RasterCount
            geotransform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()
            data_type = dataset.GetRasterBand(1).DataType
            
            # 记录栅格信息
            self.navigation_functions.log_message(f"栅格尺寸: {width}×{height}，波段数: {band_count}")
            
            # 计算每个网格的大小
            cell_width = width // cols
            cell_height = height // rows
            
            # 记录网格信息
            self.navigation_functions.log_message(f"网格大小: {rows}×{cols}，每个网格: {cell_width}×{cell_height}")
            
            # 输出目录对象
            output_path = Path(output_dir)
            
            # 执行网格裁剪
            generated_files = self._process_grid_regions(dataset, rows, cols, width, height, band_count, 
                                                        cell_width, cell_height, geotransform, projection, 
                                                        data_type, output_path, raster_path, progress_dialog)
            
            # 手动释放dataset内存
            dataset = None
            
            # 记录完成信息
            self.navigation_functions.log_message(f"栅格网格裁剪完成，共生成 {len(generated_files)} 个子图像")
            
            return generated_files
            
        except Exception as e:
            import traceback
            error_msg = f"裁剪栅格网格失败: {str(e)}"
            self.navigation_functions.log_message(error_msg)
            self.navigation_functions.log_message(traceback.format_exc())
            
            return []
        finally:
            # 确保关闭数据集，释放资源
            if dataset is not None:
                dataset = None
                
            # 手动触发垃圾回收
            import gc
            gc.collect()
    
    def _process_grid_regions(self, dataset, rows, cols, width, height, band_count, cell_width, cell_height, 
                             geotransform, projection, data_type, output_path, raster_path, progress_dialog=None):
        """处理裁剪区域的辅助方法"""
        generated_files = []
        try:
            # 创建裁剪队列
            crop_regions = []
            for row in range(rows):
                for col in range(cols):
                    # 计算当前网格的像素坐标
                    x_start = col * cell_width
                    y_start = row * cell_height
                    x_end = min((col + 1) * cell_width, width)
                    y_end = min((row + 1) * cell_height, height)
                    
                    # 计算当前网格的地理坐标变换
                    x_offset = x_start
                    y_offset = y_start
                    
                    # 调整地理变换参数
                    new_geotransform = list(geotransform)
                    # 调整原点坐标
                    new_geotransform[0] = geotransform[0] + x_offset * geotransform[1]
                    new_geotransform[3] = geotransform[3] + y_offset * geotransform[5]
                    
                    # 生成编号索引
                    grid_index = row * cols + col
                    
                    # 添加到裁剪队列
                    crop_regions.append((x_start, y_start, x_end, y_end, new_geotransform, grid_index))
            
            # 计算总裁剪任务数
            total_crops = len(crop_regions)
            
            # 记录开始裁剪
            self.navigation_functions.log_message(f"开始裁剪 {total_crops} 个网格...")
            
            # 循环处理每个裁剪区域
            for i, (x_start, y_start, x_end, y_end, new_geotransform, grid_index) in enumerate(crop_regions):
                try:
                    # 计算当前网格的宽度和高度
                    crop_width = x_end - x_start
                    crop_height = y_end - y_start
                    
                    # 忽略过小的区域
                    if crop_width < 10 or crop_height < 10:
                        self.navigation_functions.log_message(f"网格 {grid_index} 太小，已跳过")
                        continue
                    
                    # 创建输出文件名
                    base_name = Path(raster_path).stem
                    output_filename = f"{base_name}_{grid_index:05d}.tif"
                    output_file = output_path / output_filename
                    
                    # 处理数据类型，确保使用适合的数据类型
                    # GDAL数据类型可能不是Python直接支持的，所以需要检查
                    try:
                        # 检查数据类型是否有效
                        band_datatype = data_type
                        sample_band = dataset.GetRasterBand(1)
                        if sample_band.DataType != data_type:
                            self.navigation_functions.log_message(f"警告: 数据类型不匹配，使用第一个波段的数据类型")
                            band_datatype = sample_band.DataType
                    except Exception as e:
                        self.navigation_functions.log_message(f"获取数据类型失败，使用默认类型: {str(e)}")
                        band_datatype = gdal.GDT_Float32  # 使用安全的默认类型
                    
                    # 创建输出数据集，使用更安全的创建方式
                    try:
                        # 获取驱动程序
                        driver = gdal.GetDriverByName('GTiff')
                        
                        # 创建输出数据集，添加创建选项
                        # 使用BIGTIFF=IF_SAFER确保大文件支持
                        # 使用COMPRESS=LZW进行无损压缩
                        creation_options = ['COMPRESS=LZW', 'BIGTIFF=IF_SAFER', 'TILED=YES']
                        crop_ds = driver.Create(
                            str(output_file),
                            crop_width,
                            crop_height,
                            band_count,
                            band_datatype,
                            options=creation_options
                        )
                        
                        if crop_ds is None:
                            raise Exception(f"无法创建输出数据集: {output_file}")
                        
                        # 设置地理变换和投影信息
                        crop_ds.SetGeoTransform(new_geotransform)
                        crop_ds.SetProjection(projection)
                        
                        # 复制所有元数据
                        crop_ds.SetMetadata(dataset.GetMetadata())
                        
                        # 复制每个波段的数据
                        for band_idx in range(1, band_count + 1):
                            try:
                                # 获取源波段
                                band = dataset.GetRasterBand(band_idx)
                                
                                # 复制波段级元数据
                                src_metadata = band.GetMetadata()
                                
                                # 直接读取当前裁剪区域数据，减少内存使用
                                data = band.ReadAsArray(x_start, y_start, crop_width, crop_height)
                                if data is None:
                                    raise Exception(f"无法读取波段 {band_idx} 的数据")
                                
                                # 获取目标波段
                                crop_band = crop_ds.GetRasterBand(band_idx)
                                
                                # 设置元数据
                                crop_band.SetMetadata(src_metadata)
                                
                                # 设置NoData值（如果存在）
                                nodata_value = band.GetNoDataValue()
                                if nodata_value is not None:
                                    crop_band.SetNoDataValue(nodata_value)
                                
                                # 写入数据
                                crop_band.WriteArray(data)
                                
                                # 计算统计信息
                                crop_band.ComputeStatistics(False)
                                
                                # 手动释放波段数据内存
                                data = None
                                crop_band = None
                                band = None
                            except Exception as band_error:
                                self.navigation_functions.log_message(f"处理波段 {band_idx} 失败: {str(band_error)}")
                                continue
                        
                        # 关闭子图像数据集
                        crop_ds.FlushCache()  # 刷新缓存确保写入
                        crop_ds = None
                        
                        # 添加到生成的文件列表
                        generated_files.append(str(output_file))
                    except Exception as e:
                        self.navigation_functions.log_message(f"创建输出数据集失败: {str(e)}")
                        continue
                    
                    # 更新进度
                    if progress_dialog and i % max(1, total_crops // 20) == 0:  # 最多更新20次进度
                        progress = 20 + int(80 * (i + 1) / total_crops)
                        progress_dialog.setValue(progress)
                        QApplication.processEvents()
                        
                    # 定期执行垃圾回收，避免内存占用过高
                    if i % 5 == 0:  # 更频繁地执行垃圾回收
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    self.navigation_functions.log_message(f"处理网格 {grid_index} 失败: {str(e)}")
                    # 继续处理其他网格，不中断整个过程
                    import gc
                    gc.collect()  # 处理完每个网格后进行垃圾回收
            
            return generated_files
            
        except Exception as e:
            self.navigation_functions.log_message(f"创建裁剪队列失败: {str(e)}")
            return generated_files
    