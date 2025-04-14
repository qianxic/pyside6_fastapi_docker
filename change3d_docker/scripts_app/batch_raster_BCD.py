#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 批量处理带地理信息的栅格影像并导出矢量结果
# 基于inference_large_raster_BCD.py，增加了批量处理功能

# 批量栅格影像批量推理示例:
'''
python scripts_app/batch_raster_BCD.py `
  --before_path "dataes/test_raster/whole_image_2012/t1" `
  --after_path "dataes/test_raster/whole_image_2016/t2" `
  --output_path "results/raster_batch_results"
'''

import os
import sys
import argparse
import numpy as np
from skimage import io
from tqdm import tqdm
import time
import warnings
import glob
from pathlib import Path
import pandas as pd
import math
import torch
import torch.nn.functional as F
import cv2

# 修改导入方式，使用sys.path添加父目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.trainer import Trainer
from data.transforms import BCDTransforms
from osgeo import gdal, ogr, osr
# 启用GDAL异常，使错误更容易捕获和处理
gdal.UseExceptions()
gdal_version = gdal.VersionInfo()
HAS_GDAL = True
# 尝试导入GeoPandas
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, MultiLineString
HAS_GEOPANDAS = True

import rasterio
from rasterio import features
HAS_RASTERIO = True

# 添加全局模型缓存字典
MODEL_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='批量栅格影像变化检测推理')
    parser.add_argument('--before_path', type=str, required=True, 
                        help='前时相栅格影像目录')
    parser.add_argument('--after_path', type=str, required=True,
                        help='后时相栅格影像目录')
    parser.add_argument('--output_path', type=str, default='results/raster_batch_results',
                        help='输出主目录')
    
    # 以下参数设为不可见的默认值
    args = parser.parse_args()
    
    # 添加默认参数
    args.checkpoint = "checkpoint/checkpoint.pth.tar"
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.file_ext = '.tif,.tiff'
    args.in_height = 256
    args.in_width = 256
    args.num_perception_frame = 1
    args.num_class = 1
    args.save_binary_mask = True
    args.patch_size = 0
    args.max_patch_size = 1024
    args.min_patch_size = 256
    args.auto_patch_divisor = 16
    args.stride_ratio = 0.75
    args.batch_size = 2
    args.overlap_weights = False
    args.raw_output = False
    args.dataset = 'LEVIR-CD'
    args.auto_memory = True
    args.max_images = 0
    
    # 栅格处理相关参数
    args.ignore_geo = False
    args.band_indices = '1,2,3'
    args.warp_projection = None
    
    # 矢量导出相关参数
    args.export_shapefile = True
    args.export_geojson = True
    args.vector_output_dir = None
    args.min_polygon_area = 10.0
    args.simplify_tolerance = 0.5
    args.attribute_change_type = True
    args.calculate_area = True
    
    # 矢量合并相关参数
    args.merge_vectors = True
    args.merged_file_name = 'merged_changes'
    
    # 输出详细程度控制
    args.quiet = True
    args.save_visualization = True
    
    # 为了向后兼容，保留原始参数名，但使用新的参数名
    args.pre_dir = args.before_path
    args.post_dir = args.after_path
    args.output_dir = args.output_path
    
    return args

def load_model(args):
    """加载模型和权重，使用缓存机制避免重复加载"""
    # 构建缓存键 - 使用检查点路径和设备信息
    cache_key = f"{args.checkpoint}_{args.device}"
    
    # 确保模型输入尺寸有效
    if args.in_height <= 0 or args.in_width <= 0:
        args.in_height = 256
        args.in_width = 256
    
    # 检查缓存中是否已存在模型
    if cache_key in MODEL_CACHE:
        # 从缓存获取模型
        model = MODEL_CACHE[cache_key]
        
        # 检查模型参数是否与当前需求匹配
        current_perception_frame_size = (args.num_perception_frame, args.in_height, args.in_width)
        if hasattr(model.encoder, 'perception_frames'):
            try:
                model_perception_frame_size = (
                    model.encoder.perception_frames.shape[2],
                    model.encoder.perception_frames.shape[3],
                    model.encoder.perception_frames.shape[4]
                )
                
                # 如果尺寸不匹配，则重新创建感知帧
                if current_perception_frame_size != model_perception_frame_size:
                    # 创建新的感知帧参数
                    new_perception_frames = torch.randn(
                        1, 3, args.num_perception_frame, args.in_height, args.in_width, 
                        requires_grad=True,
                        device=args.device
                    )
                    # 替换模型中的感知帧
                    model.encoder.perception_frames = torch.nn.Parameter(new_perception_frames)
            except:
                pass
        
        return model
    
    # 创建模型 - 确保此时args.in_height和args.in_width是有效的
    model = Trainer(args).to(args.device)
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if 'state_dict' in checkpoint:
        # 尝试加载模型参数，但忽略感知帧参数，因为尺寸可能不匹配
        try:
            # 常规加载尝试
            model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError as e:
            # 大多数错误是由尺寸不匹配导致的
            # 获取模型状态字典和检查点状态字典
            model_state_dict = model.state_dict()
            checkpoint_state_dict = checkpoint['state_dict']
            
            # 创建新的状态字典，只包含匹配的参数
            new_state_dict = {}
            for k, v in checkpoint_state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        new_state_dict[k] = v
            
            # 加载匹配的参数
            model.load_state_dict(new_state_dict, strict=False)
    else:
        # 尝试直接加载整个模型
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            # 处理尺寸不匹配错误
            # 获取模型状态字典和检查点状态字典
            model_state_dict = model.state_dict()
            
            # 创建新的状态字典，只包含匹配的参数
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        new_state_dict[k] = v
            
            # 加载匹配的参数
            model.load_state_dict(new_state_dict, strict=False)
    
    # 设置模型为评估模式
    model.eval()
    
    # 将模型保存到缓存
    MODEL_CACHE[cache_key] = model
    
    return model

def get_transform(args):
    """获取验证/测试用的变换"""
    # 保存原始patch_size
    original_patch_size = args.patch_size
    
    # 确保变换使用固定的尺寸256x256
    temp_args = argparse.Namespace(**vars(args))  # 创建args的副本
    temp_args.in_height = 256
    temp_args.in_width = 256
    temp_args.patch_size = 256  # 确保使用256作为patch_size来获取transform
    
    # 获取变换
    _, val_transform = BCDTransforms.get_transform_pipelines(temp_args)
    
    # 恢复原始patch_size
    args.patch_size = original_patch_size
    
    return val_transform

def read_geotiff(raster_path, selected_bands=None):
    """读取GeoTIFF栅格文件，保留地理参考信息
    
    Args:
        raster_path: 栅格文件路径
        selected_bands: 要读取的波段索引列表，索引从1开始
        
    Returns:
        tuple: (图像数组, 地理变换参数, 投影信息, 原始数据集)
    """
    try:
        # 打开栅格数据集
        ds = gdal.Open(raster_path)
        if ds is None:
            raise IOError(f"无法打开栅格文件: {raster_path}")
        
        # 获取栅格信息
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands_count = ds.RasterCount
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        # 统一处理地理变换参数的精度和类型
        geo_transform = list(geo_transform)
        # 对坐标值（第1和第4个参数）使用2位小数
        geo_transform[0] = float(f"{geo_transform[0]:.2f}")  # X坐标
        geo_transform[3] = float(f"{geo_transform[3]:.2f}")  # Y坐标
        # 对分辨率值（第2、第3、第5、第6个参数）使用6位小数
        geo_transform[1] = float(f"{geo_transform[1]:.6f}")  # X分辨率
        geo_transform[2] = float(f"{geo_transform[2]:.6f}")  # 行旋转
        geo_transform[4] = float(f"{geo_transform[4]:.6f}")  # 列旋转
        geo_transform[5] = float(f"{geo_transform[5]:.6f}")  # Y分辨率
        geo_transform = tuple(geo_transform)
        
        # 如果未指定波段，默认使用RGB波段（如果存在）
        if selected_bands is None:
            if bands_count >= 3:
                selected_bands = [1, 2, 3]  # RGB波段（GDAL波段索引从1开始）
            else:
                selected_bands = [1]  # 只选第一个波段
        
        # 确保选择的波段不超过可用波段数
        selected_bands = [b for b in selected_bands if b <= bands_count]
        if not selected_bands:
            raise ValueError(f"选择的波段{selected_bands}超出了可用范围(1-{bands_count})")
        
        # 读取选定的波段
        if len(selected_bands) == 1:
            # 单波段读取
            band = ds.GetRasterBand(selected_bands[0])
            image_array = band.ReadAsArray()
            # 转为RGB（三通道相同）
            image = np.stack([image_array, image_array, image_array], axis=2)
        else:
            # 多波段读取
            bands = []
            for band_idx in selected_bands[:3]:  # 最多取前3个波段作为RGB
                band = ds.GetRasterBand(band_idx)
                bands.append(band.ReadAsArray())
            
            # 创建RGB图像数组
            while len(bands) < 3:
                # 如果不足3个波段，复制最后一个波段
                bands.append(bands[-1])
            
            image = np.stack(bands, axis=2)
        
        # 数据类型转换和归一化
        if image.dtype != np.uint8:
            # 自动缩放到0-255
            for i in range(image.shape[2]):
                channel = image[:,:,i]
                # 获取NoData值并安全处理
                try:
                    nodata_value = band.GetNoDataValue()
                    if nodata_value is not None:
                        non_nodata = (channel != nodata_value)
                    else:
                        non_nodata = np.ones_like(channel, dtype=bool)
                except Exception as e:
                    non_nodata = np.ones_like(channel, dtype=bool)
                
                if np.any(non_nodata):
                    # 使用百分位数进行拉伸，避免极值影响
                    try:
                        min_val = np.percentile(channel[non_nodata], 2)
                        max_val = np.percentile(channel[non_nodata], 98)
                        # 避免同值问题
                        if min_val == max_val:
                            min_val = np.min(channel[non_nodata])
                            max_val = np.max(channel[non_nodata])
                            if min_val == max_val:
                                min_val = 0
                                max_val = 255
                        
                        # 应用拉伸
                        normalized = np.zeros_like(channel, dtype=np.float32)
                        normalized[non_nodata] = np.clip(channel[non_nodata], min_val, max_val)
                        normalized[non_nodata] = ((normalized[non_nodata] - min_val) / (max_val - min_val) * 255)
                        image[:,:,i] = normalized.astype(np.uint8)
                    except Exception as e:
                        # 简单线性拉伸作为备选方案
                        min_val = np.min(channel)
                        max_val = np.max(channel)
                        if min_val != max_val:
                            image[:,:,i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else:
                            image[:,:,i] = np.zeros_like(channel, dtype=np.uint8)
            
            # 确保最终结果是uint8
            image = image.astype(np.uint8)
        
        # 检查图像是否有效
        if np.isnan(image).any() or np.isinf(image).any():
            image = np.nan_to_num(image, nan=0, posinf=255, neginf=0).astype(np.uint8)
        
        return image, geo_transform, projection, ds
        
    except Exception as e:
        raise

def create_sliding_windows(image_shape, patch_size, stride):
    """创建滑动窗口坐标列表，确保所有窗口大小一致"""
    h, w = image_shape[:2]
    windows = []
    
    # 只在能完整容纳窗口的区域创建窗口
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            windows.append((x, y, x + patch_size, y + patch_size))
    
    return windows

def create_weight_map(patch_size, stride, weight_type='gaussian'):
    """创建权重图，用于融合重叠区域"""
    if weight_type == 'gaussian':
        # 创建二维高斯权重
        sigma = patch_size / 4
        x = np.linspace(-patch_size/2, patch_size/2, patch_size)
        y = np.linspace(-patch_size/2, patch_size/2, patch_size)
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        # 归一化
        kernel = kernel / kernel.max()
        return kernel
    elif weight_type == 'linear':
        # 创建线性边缘衰减权重
        x = np.linspace(0, 1, patch_size//2)
        y = np.linspace(0, 1, patch_size//2)
        xx, yy = np.meshgrid(x, y)
        center = np.ones((patch_size//2, patch_size//2))
        top_left = xx * yy
        top_right = np.fliplr(xx) * yy
        bottom_left = xx * np.flipud(yy)
        bottom_right = np.fliplr(xx) * np.flipud(yy)
        
        kernel = np.zeros((patch_size, patch_size))
        kernel[:patch_size//2, :patch_size//2] = top_left
        kernel[:patch_size//2, patch_size//2:] = top_right
        kernel[patch_size//2:, :patch_size//2] = bottom_left
        kernel[patch_size//2:, patch_size//2:] = center
        return kernel
    else:
        # 均匀权重
        return np.ones((patch_size, patch_size))

def determine_optimal_patch_size(image_shape, args):
    """根据图像尺寸自动确定最优滑动窗口大小"""
    # 强制使用256大小，无论用户设置或图像尺寸
    return 256

def determine_optimal_stride(patch_size, args):
    """根据块大小确定合适的步长"""
    stride = int(patch_size * args.stride_ratio)
    # 确保步长至少为32像素
    stride = max(stride, 32)
    return stride

def estimate_memory_usage(patch_size, batch_size, device):
    """估计每批次的内存占用"""
    # 假设使用的是float32类型的张量
    bytes_per_value = 4
    
    # 每个patch的输入(前后时相)和输出(预测)的大小
    input_size_per_patch = 2 * 3 * patch_size * patch_size  # 两个RGB图像
    output_size_per_patch = patch_size * patch_size  # 单通道输出
    
    # 模型参数和中间特征的大致估计
    model_overhead = 100 * 1024 * 1024  # 100MB固定开销
    feature_maps = 20 * 3 * patch_size * patch_size  # 粗略估计特征图占用
    
    # 批次的总开销
    batch_memory = batch_size * (input_size_per_patch + output_size_per_patch + feature_maps) * bytes_per_value
    
    # 总内存占用 (模型 + 批次数据)
    total_memory = model_overhead + batch_memory
    
    return total_memory / (1024 * 1024)  # 转换为MB

def adjust_batch_size(patch_size, args):
    """根据滑动窗口大小和可用内存调整批处理大小"""
    # 对于CPU设备，根据patch_size调整批处理大小
    if str(args.device) == 'cpu':
        # 根据patch_size大小调整批处理大小，避免CPU内存不足
        if patch_size > 512:
            return min(args.batch_size, 2)  # 大patch使用更小的批次
        else:
            return min(args.batch_size, 4)  # 小patch可以用稍大的批次
    
    # 以下是GPU相关的逻辑
    if not args.auto_memory:
        return args.batch_size
    
    # 确保使用的是GPU设备
    if not torch.cuda.is_available():
        return args.batch_size
    
    try:
        # 获取GPU信息
        device_idx = int(str(args.device).split(':')[-1]) if ':' in str(args.device) else 0
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory / (1024 * 1024)  # 转换为MB
        allocated_memory = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
        
        # 计算可用内存 (留出20%的余量)
        available_memory = (total_memory - allocated_memory) * 0.8
        
        # 估算每个批次所需内存
        memory_per_batch = estimate_memory_usage(patch_size, 1, args.device)
        
        # 计算最优批处理大小
        optimal_batch_size = max(1, int(available_memory / memory_per_batch))
        
        # 将批处理大小限制在合理范围内
        optimal_batch_size = min(optimal_batch_size, 16)
        optimal_batch_size = max(optimal_batch_size, 1)
        
        if optimal_batch_size != args.batch_size:
            return optimal_batch_size
        else:
            return args.batch_size
            
    except Exception as e:
        # 出错时静默使用默认批大小
        return args.batch_size

def process_large_raster(pre_img_path, post_img_path, model, transform, args):
    """处理大型栅格影像，使用滑动窗口方法
    
    Returns:
        tuple: (结果掩码, 前时相图像, 后时相图像, 地理变换, 投影, 源数据集)
    """
    # 检查是否为地理栅格格式
    is_geotiff = any(ext in pre_img_path.lower() for ext in ['.tif', '.tiff'])
    geo_transform = None
    projection = None
    pre_ds = None
    
    # 确保模型输入尺寸不为0（防止除零错误）
    if args.in_height <= 0:
        args.in_height = 256
    if args.in_width <= 0:
        args.in_width = 256
    
    # 保存原始输入参数，以便在函数结束时恢复
    original_in_height = args.in_height
    original_in_width = args.in_width
    
    try:
        # 读取图像 - 静默处理
        try:
            if is_geotiff and not args.ignore_geo:
                # 获取用户选择的波段
                band_indices = [int(i) for i in args.band_indices.split(',')] if args.band_indices else None
                
                # 读取带地理参考的栅格
                pre_img, pre_geo_transform, pre_projection, pre_ds = read_geotiff(pre_img_path, band_indices)
                post_img, post_geo_transform, post_projection, post_ds = read_geotiff(post_img_path, band_indices)
                
                # 使用前时相的地理信息作为参考
                geo_transform = pre_geo_transform
                projection = pre_projection
                
                # 检查两个栅格的地理参考是否一致
                geo_transform_mismatch = False
                if len(pre_geo_transform) == len(post_geo_transform):
                    # 使用容差检查，允许小的浮点数精度差异
                    tolerance = 1e-5
                    for i in range(len(pre_geo_transform)):
                        if abs(pre_geo_transform[i] - post_geo_transform[i]) > tolerance:
                            geo_transform_mismatch = True
                            break
                else:
                    geo_transform_mismatch = True
                
                if geo_transform_mismatch or pre_projection != post_projection:
                    # 静默处理不一致的情况
                    geo_transform = pre_geo_transform
                    projection = pre_projection
            else:
                # 使用普通图像读取模式
                pre_img = io.imread(pre_img_path)
                if len(pre_img.shape) == 2:
                    pre_img = np.stack([pre_img, pre_img, pre_img], axis=2)
                elif pre_img.shape[2] > 3:
                    pre_img = pre_img[:, :, :3]
                    
                post_img = io.imread(post_img_path)
                if len(post_img.shape) == 2:
                    post_img = np.stack([post_img, post_img, post_img], axis=2)
                elif post_img.shape[2] > 3:
                    post_img = post_img[:, :, :3]
        except Exception as e:
            raise
        
        # 确保两个图像尺寸一致
        if pre_img.shape[:2] != post_img.shape[:2]:
            post_img = cv2.resize(post_img, (pre_img.shape[1], pre_img.shape[0]))
        
        # 获取图像尺寸 - 不打印信息
        h, w = pre_img.shape[:2]
        
        # 确保图像尺寸至少是256的倍数，在读取后立即填充
        MODEL_SIZE = 256
        pad_h_to_model = (MODEL_SIZE - h % MODEL_SIZE) % MODEL_SIZE
        pad_w_to_model = (MODEL_SIZE - w % MODEL_SIZE) % MODEL_SIZE
        
        if pad_h_to_model > 0 or pad_w_to_model > 0:
            pre_img = np.pad(pre_img, ((0, pad_h_to_model), (0, pad_w_to_model), (0, 0)), mode='reflect')
            post_img = np.pad(post_img, ((0, pad_h_to_model), (0, pad_w_to_model), (0, 0)), mode='reflect')
            h, w = pre_img.shape[:2]
        
        # 先确定最优滑动窗口大小 - 修复变量顺序问题
        patch_size = determine_optimal_patch_size((h, w), args)
        stride = determine_optimal_stride(patch_size, args)
        
        # 更新args中的patch_size，用于获取transform
        args.patch_size = patch_size
        
        # 重新获取transform以适应新的patch_size
        transform = get_transform(args)
        
        # 根据patch_size调整批处理大小
        batch_size = adjust_batch_size(patch_size, args)
        
        # 创建滑动窗口
        windows = create_sliding_windows((h, w), patch_size, stride)
        
        # 创建结果图 - 如果使用权重融合，总是使用浮点数类型
        if args.raw_output or args.overlap_weights:
            result_map = np.zeros((h, w), dtype=np.float32)
        else:
            result_map = np.zeros((h, w), dtype=np.uint8)
        
        # 创建权重图用于融合
        if args.overlap_weights:
            weight_map = create_weight_map(patch_size, stride)
            count_map = np.zeros((h, w), dtype=np.float32)
        
        # 创建批次
        num_batches = int(np.ceil(len(windows) / batch_size))
        
        # 开始处理 - 完全静默处理，移除tqdm进度条
        model.eval()
        with torch.no_grad():
            for batch_idx in range(num_batches):
                # 获取当前批次的窗口
                batch_windows = windows[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                
                # 准备批次数据
                batch_pre = []
                batch_post = []
                batch_windows_valid = []  # 记录有效的窗口
                
                for x1, y1, x2, y2 in batch_windows:
                    # 检查图像块尺寸是否一致
                    if x2-x1 == patch_size and y2-y1 == patch_size:
                        # 提取图像块
                        pre_patch = pre_img[y1:y2, x1:x2]
                        post_patch = post_img[y1:y2, x1:x2]
                        
                        # 应用变换并调整大小到模型的输入尺寸
                        image = np.concatenate([pre_patch, post_patch], axis=2)
                        mask = np.zeros((patch_size, patch_size), dtype=np.float32)
                        image_t, _ = transform(image, mask)
                        
                        # 分离前后时相
                        pre_t = image_t[0:3].unsqueeze(0)
                        post_t = image_t[3:6].unsqueeze(0)
                        
                        # 确保张量的尺寸匹配模型输入尺寸256x256
                        if pre_t.shape[2] != args.in_height or pre_t.shape[3] != args.in_width:
                            pre_t = F.interpolate(pre_t, size=(args.in_height, args.in_width), mode='bilinear', align_corners=False)
                            post_t = F.interpolate(post_t, size=(args.in_height, args.in_width), mode='bilinear', align_corners=False)
                        
                        batch_pre.append(pre_t)
                        batch_post.append(post_t)
                        batch_windows_valid.append((x1, y1, x2, y2))  # 只添加有效窗口
                
                # 合并批次
                if batch_pre:
                    batch_pre = torch.cat(batch_pre, dim=0).to(args.device)
                    batch_post = torch.cat(batch_post, dim=0).to(args.device)
                    
                    # 前向传播
                    outputs = model.update_bcd(batch_pre, batch_post)
                    
                    # 处理结果
                    if args.raw_output:
                        preds = outputs.squeeze(1).cpu().numpy()  # 保持浮点数预测
                    else:
                        preds = (outputs > 0.5).float().squeeze(1).cpu().numpy()  # 二值化
                    
                    # 将结果合并到完整图像
                    for idx, (x1, y1, x2, y2) in enumerate(batch_windows_valid):
                        if idx < len(preds):
                            if args.overlap_weights:
                                # 使用权重融合重叠区域
                                result_map[y1:y2, x1:x2] += preds[idx] * weight_map
                                count_map[y1:y2, x1:x2] += weight_map
                            else:
                                # 直接覆盖
                                if args.raw_output:
                                    result_map[y1:y2, x1:x2] = preds[idx]
                                else:
                                    result_map[y1:y2, x1:x2] = preds[idx] * 255
                    
                    # 清理GPU内存
                    del batch_pre, batch_post, outputs, preds
                    if torch.cuda.is_available() and str(args.device) != 'cpu':
                        torch.cuda.empty_cache()
        
        # 处理融合结果
        if args.overlap_weights:
            # 避免除零
            count_map = np.maximum(count_map, 1e-6)
            result_map = result_map / count_map
            
            if not args.raw_output:
                # 二值化并转换为 uint8
                result_map = (result_map > 0.5).astype(np.uint8) * 255
        
        return result_map, pre_img, post_img, geo_transform, projection, pre_ds
    
    finally:
        # 恢复原始参数值，无论函数是否成功执行
        args.in_height = original_in_height
        args.in_width = original_in_width

def mask_to_lines(mask, min_area=100, simplify=True, simplify_tolerance=0.5):

    # 高斯滤波处理掩码，使边界更平滑
    if mask.dtype != np.uint8:
        # 先规格化到0-255范围的uint8类型
        temp_mask = (mask * 255).astype(np.uint8)
        # 应用高斯滤波
        smoothed_mask = cv2.GaussianBlur(temp_mask, (5, 5), 0)
        # 对于浮点数掩码，执行二值化
        binary_mask = (smoothed_mask > 127).astype(np.uint8)
    else:
        # 应用高斯滤波
        smoothed_mask = cv2.GaussianBlur(mask, (5, 5), 0)
        binary_mask = (smoothed_mask > 0).astype(np.uint8)
    
    # 使用rasterio.features提取形状
    shapes = features.shapes(binary_mask, mask=binary_mask > 0, connectivity=8)
    
    # 转换为shapely几何对象
    polygons = []
    areas = []  # 存储对应的面积
    
    for geom, value in shapes:
        if value == 0:  # 跳过背景
            continue
            
        # 转换为shapely对象
        polygon = shape(geom)
        
        # 按面积过滤
        if polygon.area < min_area:
            continue
        
        # 简化多边形（如果需要）
        if simplify and simplify_tolerance > 0:
            polygon = polygon.simplify(simplify_tolerance)
            
        # 确保是有效的多边形
        if not polygon.is_valid:
            # 尝试修复无效多边形
            polygon = polygon.buffer(0)
            if not polygon.is_valid or polygon.is_empty:
                continue
                
        # 只保留多边形，不包括线要素
        if isinstance(polygon, (Polygon, MultiPolygon)):
            polygons.append(polygon)
            areas.append(polygon.area)
    
    return polygons, areas

def pixel_to_geo_coords(x, y, geo_transform):
    """将像素坐标转换为地理坐标
    
    Args:
        x, y: 像素坐标
        geo_transform: GDAL地理变换参数
        
    Returns:
        tuple: (地理x坐标, 地理y坐标)
    """
    geo_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2]
    geo_y = geo_transform[3] + x * geo_transform[4] + y * geo_transform[5]
    return geo_x, geo_y

def transform_line_to_geo(line, geo_transform):
    """将线的像素坐标转换为地理坐标
    
    Args:
        line: shapely线要素
        geo_transform: GDAL地理变换参数
        
    Returns:
        shapely.geometry: 转换后的线要素
    """
    if line.geom_type == 'LineString':
        # 转换坐标
        coords = [pixel_to_geo_coords(x, y, geo_transform) for x, y in line.coords]
        return LineString(coords)
    elif line.geom_type == 'MultiLineString':
        # 转换多多边形中的每个线
        geo_lines = []
        for l in line.geoms:
            geo_lines.append(transform_line_to_geo(l, geo_transform))
        return MultiLineString(geo_lines)
    else:
        raise ValueError(f"不支持的几何类型: {type(line)}")

def export_vector(polygons, areas, output_path, geo_transform=None, projection=None, attributes=None, export_format='shp'):
    """导出矢量文件
    
    Args:
        polygons: 变化区域的多边形要素
        areas: 对应的面积信息
        output_path: 输出文件路径
        geo_transform: 地理变换参数
        projection: 投影信息
        attributes: 属性信息字典
        export_format: 导出格式，'shp'或'geojson'
        
    Returns:
        bool: 是否成功导出
    """
    try:
        # 准备几何对象列表
        geometries = []
        
        # 处理多边形，转换坐标
        for polygon in polygons:
            # 只处理多边形类型
            if isinstance(polygon, (Polygon, MultiPolygon)):
                # 如果有地理变换参数，转换坐标
                if geo_transform is not None:
                    # 转换多边形坐标
                    if isinstance(polygon, Polygon):
                        # 转换外环
                        exterior_coords = []
                        for x, y in polygon.exterior.coords:
                            geo_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2]
                            geo_y = geo_transform[3] + x * geo_transform[4] + y * geo_transform[5]
                            exterior_coords.append((geo_x, geo_y))
                        
                        # 转换内环（如果有）
                        interior_coords = []
                        for interior in polygon.interiors:
                            hole_coords = []
                            for x, y in interior.coords:
                                geo_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2]
                                geo_y = geo_transform[3] + x * geo_transform[4] + y * geo_transform[5]
                                hole_coords.append((geo_x, geo_y))
                            if len(hole_coords) >= 3:  # 内环至少需要3个点
                                interior_coords.append(hole_coords)
                        
                        # 创建新的多边形
                        if len(exterior_coords) >= 3:  # 外环至少需要3个点
                            try:
                                if interior_coords:
                                    geo_polygon = Polygon(exterior_coords, interior_coords)
                                else:
                                    geo_polygon = Polygon(exterior_coords)
                                
                                if geo_polygon.is_valid:
                                    geometries.append(geo_polygon)
                            except:
                                pass  # 如果创建多边形失败，跳过
                    
                    # 处理MultiPolygon
                    elif isinstance(polygon, MultiPolygon):
                        geo_polygons = []
                        for single_polygon in polygon.geoms:
                            # 转换外环
                            exterior_coords = []
                            for x, y in single_polygon.exterior.coords:
                                geo_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2]
                                geo_y = geo_transform[3] + x * geo_transform[4] + y * geo_transform[5]
                                exterior_coords.append((geo_x, geo_y))
                            
                            # 转换内环（如果有）
                            interior_coords = []
                            for interior in single_polygon.interiors:
                                hole_coords = []
                                for x, y in interior.coords:
                                    geo_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2]
                                    geo_y = geo_transform[3] + x * geo_transform[4] + y * geo_transform[5]
                                    hole_coords.append((geo_x, geo_y))
                                if len(hole_coords) >= 3:  # 内环至少需要3个点
                                    interior_coords.append(hole_coords)
                            
                            # 创建新的多边形
                            if len(exterior_coords) >= 3:  # 外环至少需要3个点
                                try:
                                    if interior_coords:
                                        geo_polygon = Polygon(exterior_coords, interior_coords)
                                    else:
                                        geo_polygon = Polygon(exterior_coords)
                                    
                                    if geo_polygon.is_valid:
                                        geo_polygons.append(geo_polygon)
                                except:
                                    pass  # 如果创建多边形失败，跳过
                        
                        if geo_polygons:
                            geometries.append(MultiPolygon(geo_polygons))
                else:
                    # 如果没有地理变换，直接添加多边形
                    geometries.append(polygon)
        
        if not geometries:
            return False
        
        # 准备属性数据
        attr_data = {}
        
        # 添加默认属性
        attr_data["ID"] = list(range(1, len(geometries) + 1))
        
        # 添加面积属性
        if areas is not None:
            attr_data["Area"] = []
            attr_data["Perimeter"] = []
            
            for i, geom in enumerate(geometries):
                if i < len(areas):
                    attr_data["Area"].append(areas[i])
                    # 计算周长
                    try:
                        attr_data["Perimeter"].append(geom.length)
                    except:
                        attr_data["Perimeter"].append(0)
                else:
                    attr_data["Area"].append(0)
                    attr_data["Perimeter"].append(0)
        
        # 添加自定义属性
        if attributes is not None:
            for attr_name, attr_values in attributes.items():
                attr_data[attr_name] = []
                for i in range(len(geometries)):
                    if i < len(attr_values):
                        attr_data[attr_name].append(attr_values[i])
                    else:
                        attr_data[attr_name].append(None)
        
        # 创建GeoDataFrame
        gdf = gpd.GeoDataFrame(attr_data, geometry=geometries)
        
        # 设置坐标参考系统
        if projection and projection.strip(): # 检查 projection 是否有效且非空
            try:
                # 尝试从WKT创建CRS
                srs = osr.SpatialReference()
                srs.ImportFromWkt(projection)
                epsg = srs.GetAuthorityCode(None)
                if epsg:
                    gdf.crs = f"EPSG:{epsg}"
                else:
                    # 确保WKT字符串是有效的，才赋值
                    if srs.Validate() == ogr.OGRERR_NONE:
                         gdf.crs = projection
                    else:
                         gdf.crs = None # 明确设为None
            except Exception as e:
                gdf.crs = None # 明确设为None
        
        # 导出文件
        if export_format.lower() == 'geojson':
            gdf.to_file(output_path, driver='GeoJSON')
        else:
            gdf.to_file(output_path)
            
        return True
        
    except Exception as e:
        return False

def save_geotiff_result(mask, output_path, geo_transform=None, projection=None, src_ds=None):
    """将掩码保存为GeoTIFF格式"""
    try:
        # 获取数据类型
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            gdal_dtype = gdal.GDT_Float32
        else:
            gdal_dtype = gdal.GDT_Byte
        
        # 创建驱动
        driver = gdal.GetDriverByName('GTiff')
        
        # 创建数据集
        ds = driver.Create(
            output_path,
            mask.shape[1],  # 宽度
            mask.shape[0],  # 高度
            1,              # 波段数
            gdal_dtype,
            options=['COMPRESS=LZW']  # 使用LZW压缩
        )
        
        # 设置地理参考信息
        if geo_transform is not None:
            ds.SetGeoTransform(geo_transform)
        
        if projection is not None:
            ds.SetProjection(projection)
        
        # 写入数据
        ds.GetRasterBand(1).WriteArray(mask)
        
        # 设置NoData值
        ds.GetRasterBand(1).SetNoDataValue(0)
        
        # 刷新
        ds.FlushCache()
        
        # 关闭数据集
        ds = None
        
        return True
        
    except Exception as e:
        # 尝试普通方式保存
        try:
            cv2.imwrite(output_path, mask)
            return True
        except:
            return False

def visualize_results(pre_img, post_img, pred_mask, save_path, is_raw_output=False):
    """可视化结果 - 将前后时相图像、变化检测结果和二值掩码共四张图横排显示"""
    # 确保输入掩码有效
    if pred_mask is None or pred_mask.size == 0:
        return False
        
    # 确保掩码是二维数组
    if len(pred_mask.shape) > 2:
        pred_mask = pred_mask[:,:,0]  # 如果是多通道，取第一个通道
        
    # 确保mask尺寸与图像一致
    if pred_mask.shape[:2] != pre_img.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (pre_img.shape[1], pre_img.shape[0]), 
                              interpolation=cv2.INTER_NEAREST if not is_raw_output else cv2.INTER_LINEAR)
    
    # 如果图像太大，先缩小以便可视化
    max_size = 1200
    h, w = pre_img.shape[:2]
    if h > max_size or w > max_size:
        scale = min(max_size / h, max_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pre_img = cv2.resize(pre_img, (new_w, new_h))
        post_img = cv2.resize(post_img, (new_w, new_h))
        pred_mask = cv2.resize(pred_mask, (new_w, new_h), 
                              interpolation=cv2.INTER_NEAREST if not is_raw_output else cv2.INTER_LINEAR)
    
    # 调整预测掩码为RGB
    h, w = pre_img.shape[:2]
    
    # 创建原始二值掩码的可视化
    binary_mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
    if is_raw_output:
        # 对于概率图，先二值化
        binary = (pred_mask > 0.5).astype(np.uint8) * 255
        binary_mask_colored[..., 0] = 0  # B通道
        binary_mask_colored[..., 1] = binary  # G通道
        binary_mask_colored[..., 2] = 0  # R通道
    else:
        # 已经是二值化的结果
        binary_mask_colored[..., 0] = 0  # B通道
        binary_mask_colored[..., 1] = pred_mask  # G通道
        binary_mask_colored[..., 2] = 0  # R通道
    
    # 处理叠加在原图上的变化检测结果
    if is_raw_output:
        # 创建热力图
        pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        pred_colored = cv2.addWeighted(post_img, 0.7, heatmap, 0.3, 0)
    else:
        # 二值化结果
        pred_colored = np.copy(post_img)
        # 红色表示变化区域 - 修复尺寸不匹配问题
        # 创建掩码数组，避免广播问题
        mask_condition = pred_mask > 127
        pred_colored[..., 0] = np.where(mask_condition, 255, post_img[..., 0])
        pred_colored[..., 1] = np.where(mask_condition, 0, post_img[..., 1])
        pred_colored[..., 2] = np.where(mask_condition, 0, post_img[..., 2])
    
    # 创建一个空白画布 - 现在需要容纳4张图像
    gap = 5  # 图像之间的间隔
    canvas_width = w * 4 + gap * 3
    canvas_height = h
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 在画布上放置四张图像
    canvas[0:h, 0:w] = pre_img
    canvas[0:h, w+gap:w*2+gap] = post_img
    canvas[0:h, w*2+gap*2:w*3+gap*2] = pred_colored
    canvas[0:h, w*3+gap*3:w*4+gap*3] = binary_mask_colored
    
    # 保存结果
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 规范化路径，确保使用正确的路径分隔符
    normalized_save_path = os.path.normpath(save_path)
    
    try:
        # 直接使用PIL保存图像，跳过OpenCV的imwrite
        from PIL import Image
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(normalized_save_path)
        return True
    except Exception as e:
        return False

def get_image_pairs(args):
    """获取前后时相栅格影像对"""
    pre_dir = args.pre_dir
    post_dir = args.post_dir
    
    if not os.path.exists(pre_dir) or not os.path.exists(post_dir):
        raise ValueError(f"前后时相目录必须存在: {pre_dir}, {post_dir}")
    
    # 支持多种文件扩展名
    file_extensions = args.file_ext.split(',')
    
    # 获取所有前时相栅格文件
    pre_files = []
    for ext in file_extensions:
        pre_files.extend(glob.glob(os.path.join(pre_dir, f'*{ext}')))
    pre_files = sorted(pre_files)
    
    # 创建图像对
    image_pairs = []
    for pre_file in pre_files:
        # 提取文件名（不含路径和扩展名）
        filename = os.path.basename(pre_file)
        filename_without_ext, file_ext = os.path.splitext(filename)
        
        # 构造对应的后时相文件路径（使用相同的扩展名）
        post_file = os.path.join(post_dir, f"{filename_without_ext}{file_ext}")
        
        # 检查后时相文件是否存在
        if os.path.exists(post_file):
            image_pairs.append((pre_file, post_file, filename_without_ext))
    
    return image_pairs

def merge_vector_files(input_dir, output_file, file_format='shp', pattern='*'):
    """合并多个矢量文件为一个"""
    # 确保输入目录存在 (这里input_dir通常是vectors目录)
    if not os.path.exists(input_dir):
        return False

    # 在输入目录下创建merged子目录
    merged_output_dir = os.path.join(input_dir, 'merged')
    if not os.path.exists(merged_output_dir):
        os.makedirs(merged_output_dir, exist_ok=True)
    
    # 构建最终的输出文件完整路径
    if file_format.lower() == 'shp':
        output_file_name = f"{output_file}.shp"
    else:
        output_file_name = f"{output_file}.geojson"
    final_output_path = os.path.join(merged_output_dir, output_file_name)
    
    # 查找所有匹配的文件 (在原始vectors目录下查找)
    vector_files = glob.glob(os.path.join(input_dir, pattern))
    if not vector_files:
        return False
    
    # 读取并合并所有文件
    gdfs = []
    crs = None
    
    for file_path in vector_files:
        try:
            gdf = gpd.read_file(file_path)
            
            # 保存第一个文件的坐标系统
            if crs is None and gdf.crs is not None:
                crs = gdf.crs
            
            # 添加文件名作为源文件属性
            filename = os.path.basename(file_path)
            gdf['source_file'] = filename
            
            gdfs.append(gdf)
        except Exception as e:
            pass # Continue processing other files
    
    if not gdfs:
        return False
    
    # 合并所有GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    
    # 设置坐标参考系统
    if crs is not None:
        merged_gdf.crs = crs
    
    # 保存合并后的文件到新的子目录
    try:
        if file_format.lower() == 'shp':
            merged_gdf.to_file(final_output_path)
        else:
            merged_gdf.to_file(final_output_path, driver='GeoJSON')
        return True
    except Exception as e:
        return False

def process_batch_raster(args):
    """批量处理栅格影像变化检测，用于API调用
    
    Args:
        args: 参数对象，包含输入输出路径等信息
        
    Returns:
        Dict: 处理结果
    """
    # 开始总计时
    start_time_total = time.time()
    
    # 确保args中有quiet参数，API调用时应该静默
    if not hasattr(args, 'quiet'):
        args.quiet = True
    
    # 确保兼容性设置
    if not hasattr(args, 'pre_dir') and hasattr(args, 'before_path'):
        args.pre_dir = args.before_path
    if not hasattr(args, 'post_dir') and hasattr(args, 'after_path'):
        args.post_dir = args.after_path
    if not hasattr(args, 'output_dir') and hasattr(args, 'output_path'):
        args.output_dir = args.output_path
    
    # 确保输出路径存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取输出路径中的子目录
    result_dir = os.path.join(args.output_dir, 'result')
    mask_dir = os.path.join(args.output_dir, 'masks')
    vector_dir = os.path.join(args.output_dir, 'vectors')
    
    try:
        # 这里需要深度复制args，避免对原始args的修改
        if 'copy' not in globals():
            import copy
        args_copy = copy.deepcopy(args)
        
        # 调用main_impl函数执行实际的批处理逻辑
        status_code = main_impl(args_copy)
        
        # 收集处理后的向量文件
        all_vector_files = []
        if os.path.exists(vector_dir):
            for ext in ['.shp', '.geojson']:
                vector_files = glob.glob(os.path.join(vector_dir, f'*{ext}'))
                all_vector_files.extend(vector_files)
        
        # 收集合并后的向量文件
        merged_vector_files = []
        merged_dir = os.path.join(vector_dir, 'merged')
        if os.path.exists(merged_dir):
            for ext in ['.shp', '.geojson']:
                merged_files = glob.glob(os.path.join(merged_dir, f'*{ext}'))
                merged_vector_files.extend(merged_files)
        
        # 计算总处理时间
        total_elapsed = time.time() - start_time_total
        
        # 构建API友好的返回结果
        result = {
            "status": "success" if status_code == 0 else "failed",
            "message": f"批量栅格处理完成，状态码: {status_code}",
            "output_path": args.output_dir,
            "result_dir": result_dir,
            "mask_dir": mask_dir,
            "vector_dir": vector_dir,
            "vector_files": all_vector_files,
            "merged_vector_files": merged_vector_files,
            "processing_time": {
                "total": round(total_elapsed, 2)
            }
        }
        
        return result
        
    except Exception as e:
        # 如果处理过程出错，返回错误信息
        total_elapsed = time.time() - start_time_total
        return {
            "status": "failed",
            "message": f"批量栅格处理出错: {str(e)}",
            "output_path": args.output_dir,
            "processing_time": {
                "total": round(total_elapsed, 2)
            }
        }

# 导出为process_and_save函数，与batch_image_BCD保持一致性
process_and_save = process_batch_raster

# 将原始main函数的内容提取为main_impl，可以接收外部参数
def main_impl(args):
    """批量处理栅格影像变化检测的实际实现
    
    Args:
        args: 参数对象
        
    Returns:
        int: 处理状态码，0表示成功
    """
    # 开始内部计时
    start_time_impl = time.time()
    
    success_count = 0
    failed_count = 0
    total_process_time = 0
    all_individual_vector_files = [] # 收集所有单独生成的矢量文件路径

    try:
        # 获取所有栅格对
        image_pairs = get_image_pairs(args)
        
        total_images = len(image_pairs)
        if total_images == 0:
            return 1 # 返回失败状态码
            
        # 限制处理的图像数量
        if hasattr(args, 'max_images') and args.max_images > 0 and args.max_images < total_images:
            image_pairs = image_pairs[:args.max_images]
            total_images = len(image_pairs)

        # 加载模型
        model = load_model(args)
        
        # 获取变换
        transform = get_transform(args)
        
        # 创建输出子目录
        result_dir = os.path.join(args.output_dir, 'result')
        mask_dir = os.path.join(args.output_dir, 'masks')
        vector_dir = os.path.join(args.output_dir, 'vectors')
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)
        
        # 遍历处理每一对栅格
        for idx, (pre_path, post_path, filename) in enumerate(image_pairs):
            start_time_pair = time.time()
            individual_vector_paths = [] # 当前栅格对生成的矢量文件
            try:
                # 处理单个栅格对
                pred_mask, pre_img, post_img, geo_transform, projection, src_ds = process_large_raster(
                    pre_path, post_path, model, transform, args
                )
                
                # 检查结果有效性
                if pred_mask is None:
                    raise ValueError("模型处理返回了空的掩码")
                
                # 1. 保存GeoTIFF掩码 (如果需要)
                mask_path = None
                if args.save_binary_mask:
                    mask_filename = f"{filename}_mask.tif"
                    mask_path = os.path.join(mask_dir, mask_filename)
                    save_geotiff_result(pred_mask, mask_path, geo_transform, projection, src_ds)
                
                # 2. 保存可视化结果 (如果需要)
                result_path = None
                if args.save_visualization:
                    result_filename = f"{filename}_result.png" # 可视化通常保存为png
                    result_path = os.path.join(result_dir, result_filename)
                    visualize_results(pre_img, post_img, pred_mask, result_path, args.raw_output)
                
                # 3. 导出矢量文件 (如果需要)
                if projection and projection.strip(): # 仅当有有效投影信息时才执行矢量化
                    if args.export_shapefile or args.export_geojson:
                        polygons, areas = mask_to_lines(pred_mask, args.min_polygon_area, True, args.simplify_tolerance)
                        
                        if polygons:
                            # 创建属性
                            attributes = {}
                            if args.attribute_change_type:
                                attributes['ChangeType'] = ['Change'] * len(polygons)
                            if args.calculate_area:
                                attributes['Area_px'] = areas # 像素面积
                            
                            if args.export_shapefile:
                                shp_filename = f"{filename}.shp"
                                shp_path = os.path.join(vector_dir, shp_filename)
                                if export_vector(polygons, areas, shp_path, geo_transform, projection, attributes, 'shp'):
                                    individual_vector_paths.append(shp_path)
                            
                            if args.export_geojson:
                                geojson_filename = f"{filename}.geojson"
                                geojson_path = os.path.join(vector_dir, geojson_filename)
                                if export_vector(polygons, areas, geojson_path, geo_transform, projection, attributes, 'geojson'):
                                    individual_vector_paths.append(geojson_path)
                else:
                    pass
                
                # 处理成功
                success_count += 1
                all_individual_vector_files.extend(individual_vector_paths)
                pair_time = time.time() - start_time_pair
                total_process_time += pair_time
                
            except Exception as pair_e:
                failed_count += 1
            finally:
                # 清理单对栅格的内存
                if 'src_ds' in locals() and src_ds:
                    src_ds = None # 尝试释放数据集
                if 'pred_mask' in locals(): del pred_mask
                if 'pre_img' in locals(): del pre_img
                if 'post_img' in locals(): del post_img
                if torch.cuda.is_available() and str(args.device) != 'cpu':
                    torch.cuda.empty_cache()
        
        # 合并矢量文件
        merged_vector_files = []
        if args.merge_vectors and all_individual_vector_files:
            vector_output_dir = os.path.join(args.output_dir, 'vectors')
            merged_shp_path = os.path.join(vector_output_dir, 'merged', f"{args.merged_file_name}.shp")
            merged_geojson_path = os.path.join(vector_output_dir, 'merged', f"{args.merged_file_name}.geojson")
            
            if args.export_shapefile:
                if merge_vector_files(vector_output_dir, args.merged_file_name, 'shp', '*.shp'):
                    merged_vector_files.append(merged_shp_path)
            if args.export_geojson:
                 if merge_vector_files(vector_output_dir, args.merged_file_name, 'geojson', '*.geojson'):
                     merged_vector_files.append(merged_geojson_path)
        
        return 0  # 成功
        
    except Exception as e:
        return 1  # 失败

# 原始的命令行入口点
def main():
    """命令行入口点函数"""
    # 解析命令行参数
    args = get_args()
    
    # 调用实际实现
    status_code = main_impl(args)
    
    # 返回状态码
    return status_code

if __name__ == "__main__":
    sys.exit(main())
