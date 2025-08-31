import os
import sys
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import time
import argparse
# import geopandas as gpd # Not used
import logging

# 将项目路径添加到系统路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.insert(0, project_dir)

# 确保change3d_docker模块可以被找到
change3d_docker_path = os.path.join(project_dir, "change3d_docker")
if os.path.exists(change3d_docker_path):
    sys.path.insert(0, change3d_docker_path)

# 解决OpenMP冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# 设置正确的模型路径 (Docker 容器内绝对路径)
default_model_path = "/app/change3d_docker/checkpoint/checkpoint.pth.tar"

# 尝试导入 GDAL
try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    # print("警告: GDAL 库未安装，某些栅格功能可能受限") # Removed warning print

class DetectionMode(str, Enum):
    """处理模式枚举"""
    single_image = "single_image"      # 普通图像单张处理
    single_raster = "single_raster"    # 栅格影像单张处理
    batch_image = "batch_image"        # 普通图像批处理
    batch_raster = "batch_raster"      # 栅格影像批处理

# 直接导入所需模块
import torch
# 导入单文件处理函数
from change3d_docker.scripts_app.large_image_BCD import process_and_save as process_and_save_single_image
from change3d_docker.scripts_app.large_raster_BCD import process_and_save as process_and_save_single_raster
# 导入批量处理函数 (使用别名)
from change3d_docker.scripts_app.batch_image_BCD import process_and_save as process_and_save_batch_image
from change3d_docker.scripts_app.batch_raster_BCD import process_and_save as process_and_save_batch_raster

class ChangeDetectionModel:
    """变化检测模型服务"""
    
    def __init__(self):
        """初始化模型服务"""
        self._model_loaded = True # Assume loaded for now
        
        # t1, t2, output目录的Docker容器内路径
        self.t1_dir = "/app/change3d_api_docker/t1"
        self.t2_dir = "/app/change3d_api_docker/t2" 
        self.output_dir = "/app/change3d_api_docker/output"
    
    def _validate_and_convert_path(self, path: str) -> str:
        """验证并转换路径，确保它是Docker容器内有效的路径 (移除打印)
        
        Args:
            path: 原始路径
            
        Returns:
            str: 转换后的路径 (或原始路径如果无法转换)
        """

        return path # 直接返回路径，让后续的 os.path.exists 处理
    
    def run_detection(self, mode: str, before_path: str, after_path: str, 
                     output_path: str) -> Dict[str, Any]:
        """执行变化检测 (移除打印)
        
        Args:
            mode: 处理模式
            before_path: 前时相路径
            after_path: 后时相路径
            output_path: 输出路径
            
        Returns:
            Dict: 处理结果
        """

        
        # 验证并转换路径 (移除打印)
        before_path = self._validate_and_convert_path(before_path)
        after_path = self._validate_and_convert_path(after_path)
        
        # 确保输出目录存在
        # ### 移除调试日志 ###
        output_dir = None
        if mode in [DetectionMode.single_image, DetectionMode.single_raster]:
            output_dir = os.path.dirname(output_path) # For single files, ensure the parent dir exists
        elif mode in [DetectionMode.batch_image, DetectionMode.batch_raster]:
             output_dir = output_path # For batch, output_path *is* the directory
        
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                # print(f"已创建输出目录: {output_dir}") 
            except Exception as e:
                 # print(f"创建输出目录失败: {str(e)}")
                return {"status": "error", "message": f"创建输出目录失败: {str(e)}"} 
        
        # 根据模式选择处理方法
        if mode == DetectionMode.single_image:
            return self._run_single_image(before_path, after_path, output_path)
        elif mode == DetectionMode.single_raster:
            return self._run_single_raster(before_path, after_path, output_path)
        elif mode == DetectionMode.batch_image:
            return self._run_batch_image(before_path, after_path, output_path) # Pass output_path as output_dir
        elif mode == DetectionMode.batch_raster:
            return self._run_batch_raster(before_path, after_path, output_path) # Pass output_path as output_dir
        else:
            return {"status": "error", "message": f"不支持的处理模式: {mode}"}
    
    def _create_args(self, before_path, after_path, output_path):
        """创建参数对象 (内部辅助函数，无打印)"""
        args = argparse.Namespace()
        args.before_path = before_path
        args.after_path = after_path
        args.output_path = output_path
        args.checkpoint = default_model_path
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        args.pretrained = None 
        args.patch_size = 256
        args.in_height = 256
        args.in_width = 256
        args.stride_ratio = 0.5
        args.batch_size = 16 # Default, might be adjusted later
        args.overlap_weights = True
        args.num_class = 1
        args.num_perception_frame = 1
        args.dataset = 'CD'
        args.block_size = 512 # Default block size for processing large images
        args.model_arch = 'siam_unet'
        args.raw_output = False
        args.binary_mask = True
        args.save_binary_mask = True # API should ensure output exists
        args.auto_memory = False # Keep manual batch size for API consistency
        args.save_result = True # Keep True for quadview generation
        args.max_images = 0 
        args.file_ext = '.png,.jpg,.jpeg,.tif,.tiff' 
        args.ignore_geo = False
        args.band_indices = '1,2,3'
        args.warp_projection = None
        args.export_shapefile = True
        args.export_geojson = True
        args.min_polygon_area = 100.0
        args.simplify_tolerance = 0.5
        args.attribute_change_type = True
        args.calculate_area = True
        args.merge_vectors = True 
        args.merged_file_name = 'merged_changes'
        args.quiet = True # Keep quiet for API calls
        args.save_visualization = True # Keep True for quadview
        
        return args
    
    def _run_single_image(self, before_path: str, after_path: str, output_path: str) -> Dict[str, Any]:
        """执行单图像变化检测 (移除打印)"""
        # ### 移除调试日志 ###
        # Ensure output_path is a file path, not a directory for single image
        if os.path.isdir(output_path):
            before_basename = os.path.basename(before_path)
            before_name, _ = os.path.splitext(before_basename)
            # Construct the expected result PNG path within the directory
            output_file_path = os.path.join(output_path, f"{before_name}_result.png")
        else:
             output_file_path = output_path # Assume it's already a file path

        args = self._create_args(before_path, after_path, output_file_path)
        args.save_binary_mask = True
        args.save_result = True 
        
        # ### 移除计时和结果打印 ###
        # start_time = time.time()
        result = process_and_save_single_image(args)
        return result 
    
    def _run_single_raster(self, before_path: str, after_path: str, output_path: str) -> Dict[str, Any]:
        """执行单栅格变化检测 (移除打印)"""
        # ### 移除调试日志 ###
        # Ensure output_path is a file path for single raster (e.g., mask.tif)
        if os.path.isdir(output_path):
            before_basename = os.path.basename(before_path)
            before_name, _ = os.path.splitext(before_basename)
            # Construct the expected mask TIF path within the directory
            output_file_path = os.path.join(output_path, f"{before_name}_mask.tif")
        else:
             output_file_path = output_path

        args = self._create_args(before_path, after_path, output_file_path)
        args.save_binary_mask = True
        args.export_shapefile = True
        args.export_geojson = True
        args.save_visualization = True
        
        # ### 移除计时和结果打印 ###
        # start_time = time.time()
        result = process_and_save_single_raster(args)
        return result

    def _run_batch_image(self, before_dir: str, after_dir: str, output_dir: str) -> Dict[str, Any]:
        """执行批量图像变化检测 (移除打印)"""
        args = self._create_args(before_dir, after_dir, output_dir) # output_dir is correct here
        args.save_binary_mask = True
        args.save_result = True
        
        result = process_and_save_batch_image(args)
        return result # Directly return the result from the script

    def _run_batch_raster(self, before_dir: str, after_dir: str, output_dir: str) -> Dict[str, Any]:
        """执行批量栅格变化检测 (移除打印)"""
        args = self._create_args(before_dir, after_dir, output_dir)
        args.save_binary_mask = True
        args.export_shapefile = True
        args.export_geojson = True
        args.save_visualization = True
        result = process_and_save_batch_raster(args)

        return result

# 创建单例对象
detection_model = ChangeDetectionModel()