#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 处理带地理信息的栅格影像并导出矢量结果
# 基于batch_raster_BCD.py，修改为单对大影像处理

# 栅格影像推理示例:
'''
python scripts_app/large_raster_BCD.py `
  --before_path "dataes/test_raster/whole_image_2012/test/image/2012_test.tif" `
  --after_path "dataes/test_raster/whole_image_2016/test/image/2016_test.tif" `
  --output_path "results/large_raster_results/raster_results/2012_2016_result.png"
'''

import os
import sys
import argparse
import numpy as np
from skimage import io
from tqdm import tqdm
import time
import warnings
import math 
import torch
import torch.nn.functional as F
import cv2
import logging

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

# Configure basic logging (can be customized further)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser(description='栅格影像变化检测推理')
    parser.add_argument('--before_path', type=str, required=True, 
                        help='前时相栅格影像路径')
    parser.add_argument('--after_path', type=str, required=True,
                        help='后时相栅格影像路径')
    parser.add_argument('--output_path', type=str, default='results/raster_results/output_result.png',
                        help='输出结果路径')
    
    # 以下参数设为不可见的默认值
    args = parser.parse_args()
    
    # 添加默认参数
    args.checkpoint = "checkpoint/checkpoint.pth.tar"
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.in_height = 256
    args.in_width = 256
    args.num_perception_frame = 1
    args.num_class = 1
    args.binary_mask = None
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
    
    # 栅格处理相关参数
    args.ignore_geo = False
    args.band_indices = '1,2,3'
    args.warp_projection = None
    
    # 矢量导出相关参数
    args.export_shapefile = True
    args.export_geojson = True
    args.vector_output_dir = None
    args.min_polygon_area = 100.0
    args.simplify_tolerance = 0.5
    args.attribute_change_type = True
    args.calculate_area = True
    
    # 为了向后兼容，保留原始参数名，但使用新的参数名
    args.pre_img = args.before_path
    args.post_img = args.after_path
    args.output = args.output_path
    
    return args

def load_model(args):
    """加载模型和权重"""
    # 构建缓存键
    cache_key = f"{args.checkpoint}_{args.device}"
    
    # 检查缓存中是否已存在模型
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    # 确保必要参数存在
    if not hasattr(args, 'dataset'):
        args.dataset = 'CD'
    if not hasattr(args, 'model_arch'):
        args.model_arch = 'siam_unet'
    if not hasattr(args, 'num_perception_frame'):
        args.num_perception_frame = 1
    
    # 设置固定尺寸
    args.in_height = 256
    args.in_width = 256
    
    # 直接创建模型并加载权重
    model = Trainer(args).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    
    # 加载状态字典
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # 设置模型为评估模式
    model.eval()
    
    # 缓存模型
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
    """读取GeoTIFF栅格文件，保留地理参考信息 (移除打印)"""
    ds = gdal.Open(raster_path)
    if ds is None:
        raise ValueError(f"无法打开栅格文件: {raster_path}")
    
    # 获取栅格信息
    width = ds.RasterXSize
    height = ds.RasterYSize
    bands_count = ds.RasterCount
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    
    # 统一处理地理变换参数的精度和类型
    geo_transform = list(geo_transform)
    geo_transform[0] = float(f"{geo_transform[0]:.2f}")  # X坐标
    geo_transform[3] = float(f"{geo_transform[3]:.2f}")  # Y坐标
    geo_transform[1] = float(f"{geo_transform[1]:.6f}")  # X分辨率
    geo_transform[2] = float(f"{geo_transform[2]:.6f}")  # 行旋转
    geo_transform[4] = float(f"{geo_transform[4]:.6f}")  # 列旋转
    geo_transform[5] = float(f"{geo_transform[5]:.6f}")  # Y分辨率
    geo_transform = tuple(geo_transform)
    
    # 设置波段
    selected_bands = [1, 2, 3]
    
    # 读取选定的波段
    bands = []
    for band_idx in selected_bands:
        band = ds.GetRasterBand(band_idx)
        bands.append(band.ReadAsArray())
    
    # 创建RGB图像数组
    image = np.stack(bands, axis=2)
    
    # 数据类型转换和归一化
    for i in range(image.shape[2]):
        channel = image[:,:,i]
        min_val = np.percentile(channel, 2)
        max_val = np.percentile(channel, 98)
        normalized = np.zeros_like(channel, dtype=np.float32)
        normalized = np.clip(channel, min_val, max_val)
        normalized = ((normalized - min_val) / (max_val - min_val) * 255)
        image[:,:,i] = normalized.astype(np.uint8)
    
    # 确保最终结果是uint8
    image = image.astype(np.uint8)
    
    # 处理NaN和无穷大
    image = np.nan_to_num(image, nan=0, posinf=255, neginf=0).astype(np.uint8)
    
    return image, geo_transform, projection, ds

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
    """使用滑窗处理大栅格影像，包含地理参考和进度条 (移除打印)"""
    
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
        
        # 确保返回结果与batch_raster_BCD.py中的函数一致
        # 返回顺序：pred_mask, pre_img, post_img, geo_transform, projection, src_ds
        return result_map, pre_img, post_img, geo_transform, projection, pre_ds
    
    finally:
        # 恢复原始参数值，无论函数是否成功执行
        args.in_height = original_in_height
        args.in_width = original_in_width

def mask_to_lines(mask, min_area=100, simplify=True, simplify_tolerance=0.5):
    """从二值掩码中提取矢量多边形
    
    Args:
        mask: 二值掩码图像
        min_area: 最小多边形面积（像素单位）
        simplify: 是否简化线
        simplify_tolerance: 简化容差
        
    Returns:
        list: 多边形列表和对应的面积列表
    """
    # 确保掩码是有效的二维数组
    if mask is None or mask.size == 0:
        return [], []
    
    # 首先确保mask是2D数组
    if len(mask.shape) > 2:
        # 如果是多通道图像，转换为单通道
        mask = mask[:, :, 0] if mask.shape[2] > 0 else np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    
    # 智能数据类型处理
    if mask.dtype != np.uint8:
        if mask.dtype == np.bool_ or mask.dtype == bool:
            # 如果是布尔类型，直接转换为0和1
            binary_mask = mask.astype(np.uint8) 
        elif np.issubdtype(mask.dtype, np.floating):
            # 如果是浮点类型，使用阈值二值化
            binary_mask = (mask > 0.5).astype(np.uint8)
        else:
            # 其他整数类型，确保只有0和1
            binary_mask = (mask > 0).astype(np.uint8)
    else:
        # 对于uint8类型，确保只有0和1或0和255
        if np.max(mask) > 1:
            binary_mask = (mask > 127).astype(np.uint8)
        else:
            binary_mask = mask
    
    # 增强边缘平滑处理 - 增加更大的核和更多迭代次数
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # 先腐蚀再膨胀（开运算）去除小噪点
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # 先膨胀再腐蚀（闭运算）填充小孔洞
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # 应用中值滤波平滑边缘
    binary_mask = cv2.medianBlur(binary_mask, 5)
    
    # 使用更强的高斯滤波平滑边界
    smoothed_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (9, 9), 2.0)
    binary_mask = (smoothed_mask > 0.5).astype(np.uint8)
    
    # 如果掩码全为0，则直接返回空列表
    if np.sum(binary_mask) == 0:
        return [], []
    
    try:
        # 使用rasterio.features提取形状
        shapes = features.shapes(binary_mask, mask=binary_mask > 0, connectivity=8)
        
        # 转换为shapely几何对象
        polygons = []
        areas = []  # 存储对应的面积
        
        for geom, value in shapes:
            if value == 0:  # 跳过背景
                continue
                
            try:
                # 转换为shapely对象
                polygon = shape(geom)
                
                # 只处理多边形类型
                if not isinstance(polygon, (Polygon, MultiPolygon)):
                    continue
                
                # 按面积过滤
                if polygon.area < min_area:
                    continue
                
                # 确保是有效的多边形
                if not polygon.is_valid:
                    # 尝试修复无效多边形
                    polygon = polygon.buffer(0)
                    if not polygon.is_valid or polygon.is_empty:
                        continue
                
                # 简化多边形（如果需要）- 增加简化强度
                if simplify and simplify_tolerance > 0:
                    # 使用Douglas-Peucker算法简化多边形，增大容差值
                    polygon = polygon.simplify(simplify_tolerance * 2, preserve_topology=True)
                    
                # 再次检查简化后的多边形
                if not polygon.is_valid or polygon.is_empty or polygon.area < min_area:
                    continue
                
                # 添加到结果列表
                polygons.append(polygon)
                areas.append(polygon.area)
            except Exception:
                # 如果处理单个多边形时出错，跳过此多边形
                continue
        
        return polygons, areas
    except Exception:
        # 如果整个处理过程出错，返回空结果
        return [], []

def pixel_to_geo_coords(x, y, geo_transform):
    """将像素坐标转换为地理坐标
    
    Args:
        x, y: 像素坐标
        geo_transform: GDAL地理变换参数
        
    Returns:
        tuple: (地理x坐标, 地理y坐标)
    """
    geo_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2]
    # 反转Y坐标计算方向
    geo_y = geo_transform[3] + x * geo_transform[4] - y * abs(geo_transform[5])
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
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果多边形列表为空，返回失败
        if not polygons or len(polygons) == 0:
            return False
        
        # 准备几何对象列表
        geometries = []
        
        # 处理多边形，转换坐标
        for polygon in polygons:
            # 只处理多边形类型
            if isinstance(polygon, (Polygon, MultiPolygon)):
                # 如果有地理变换参数，转换坐标
                if geo_transform is not None:
                    try:
                        # 转换多边形坐标
                        if isinstance(polygon, Polygon):
                            geo_polygon = transform_polygon_to_geo(polygon, geo_transform)
                            if geo_polygon and geo_polygon.is_valid and not geo_polygon.is_empty:
                                geometries.append(geo_polygon)
                        
                        # 处理MultiPolygon
                        elif isinstance(polygon, MultiPolygon):
                            geo_polygons = []
                            for single_polygon in polygon.geoms:
                                geo_polygon = transform_polygon_to_geo(single_polygon, geo_transform)
                                if geo_polygon and geo_polygon.is_valid and not geo_polygon.is_empty:
                                    geo_polygons.append(geo_polygon)
                            
                            if geo_polygons:
                                geometries.append(MultiPolygon(geo_polygons))
                    except Exception:
                        # 如果坐标转换失败，直接添加原始多边形
                        if polygon.is_valid and not polygon.is_empty:
                            geometries.append(polygon)
                else:
                    # 如果没有地理变换，直接添加多边形
                    if polygon.is_valid and not polygon.is_empty:
                        geometries.append(polygon)
        
        # 如果转换后的几何对象列表为空，返回失败
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
                    attr_data["Area"].append(float(areas[i]))
                    # 计算周长
                    try:
                        attr_data["Perimeter"].append(float(geom.length))
                    except:
                        attr_data["Perimeter"].append(0.0)
                else:
                    attr_data["Area"].append(0.0)
                    attr_data["Perimeter"].append(0.0)
        
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
        if projection:
            try:
                # 尝试从WKT创建CRS
                srs = osr.SpatialReference()
                srs.ImportFromWkt(projection)
                epsg = srs.GetAuthorityCode(None)
                if epsg:
                    gdf.crs = f"EPSG:{epsg}"
                else:
                    # 尝试直接设置WKT
                    gdf.crs = projection
            except Exception:
                # 如果设置投影失败，尝试使用通用投影
                try:
                    # 默认使用WGS84
                    gdf.crs = "EPSG:4326"
                except:
                    pass
        else:
            # 如果没有提供投影信息，使用默认投影
            try:
                gdf.crs = "EPSG:4326"
            except:
                pass
        
        # 导出文件
        try:
            if export_format.lower() == 'geojson':
                gdf.to_file(output_path, driver='GeoJSON')
            else:
                # 使用ESRI Shapefile驱动
                gdf.to_file(output_path, driver='ESRI Shapefile')
            return True
        except Exception:
            # 如果导出失败，尝试修复几何对象
            # 注意：此处不做复杂修复，以减少不必要的错误处理
            try:
                # 修复几何对象
                fixed_geometries = []
                for geom in geometries:
                    if not geom.is_valid:
                        fixed_geom = geom.buffer(0)
                        if fixed_geom.is_valid and not fixed_geom.is_empty:
                            fixed_geometries.append(fixed_geom)
                    else:
                        fixed_geometries.append(geom)
                
                if fixed_geometries:
                    gdf = gpd.GeoDataFrame(attr_data, geometry=fixed_geometries)
                    if projection:
                        try:
                            srs = osr.SpatialReference()
                            srs.ImportFromWkt(projection)
                            epsg = srs.GetAuthorityCode(None)
                            if epsg:
                                gdf.crs = f"EPSG:{epsg}"
                            else:
                                gdf.crs = "EPSG:4326"
                        except:
                            gdf.crs = "EPSG:4326"
                    
                    # 再次尝试导出
                    if export_format.lower() == 'geojson':
                        gdf.to_file(output_path, driver='GeoJSON')
                    else:
                        gdf.to_file(output_path, driver='ESRI Shapefile')
                    return True
                else:
                    return False
            except:
                return False
    except Exception:
        return False

def transform_polygon_to_geo(polygon, geo_transform):
    """将多边形的像素坐标转换为地理坐标
    
    Args:
        polygon: shapely多边形要素
        geo_transform: GDAL地理变换参数
        
    Returns:
        shapely.geometry.Polygon: 转换后的多边形
    """
    if not isinstance(polygon, Polygon) or not polygon.is_valid:
        return None
    
    try:
        # 转换外环
        exterior_coords = []
        for x, y in polygon.exterior.coords:
            geo_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2]
            # 反转Y坐标计算方向
            geo_y = geo_transform[3] + x * geo_transform[4] - y * abs(geo_transform[5])
            exterior_coords.append((geo_x, geo_y))
        
        # 检查外环是否有足够的点
        if len(exterior_coords) < 3:
            return None
        
        # 转换内环（如果有）
        interior_coords = []
        for interior in polygon.interiors:
            hole_coords = []
            for x, y in interior.coords:
                geo_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2]
                # 反转Y坐标计算方向
                geo_y = geo_transform[3] + x * geo_transform[4] - y * abs(geo_transform[5])
                hole_coords.append((geo_x, geo_y))
            if len(hole_coords) >= 3:  # 内环至少需要3个点
                interior_coords.append(hole_coords)
        
        # 创建新的多边形
        if interior_coords:
            geo_polygon = Polygon(exterior_coords, interior_coords)
        else:
            geo_polygon = Polygon(exterior_coords)
        
        # 验证多边形有效性
        if not geo_polygon.is_valid:
            geo_polygon = geo_polygon.buffer(0)
            if not geo_polygon.is_valid or geo_polygon.is_empty:
                return None
        
        return geo_polygon
    except Exception:
        return None

def save_geotiff_result(mask, output_path, geo_transform=None, projection=None, src_ds=None):
    """保存带有地理参考信息的GeoTIFF结果 (移除打印)"""
    try:
        # 对掩码进行边缘平滑处理，减少齿状边缘
        if mask.dtype == np.uint8:
            # 对二值掩码应用平滑处理
            # 使用中值滤波平滑边缘
            smooth_mask = cv2.medianBlur(mask, 5)
            # 使用形态学操作进一步平滑边缘
            kernel = np.ones((3, 3), np.uint8)
            smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            # 使用高斯模糊后重新二值化
            smooth_mask = cv2.GaussianBlur(smooth_mask.astype(np.float32), (5, 5), 1.0)
            mask = (smooth_mask > 127).astype(np.uint8) * 255
            
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
    """生成并保存可视化结果（包括四联图） (移除打印)"""
    # 确保输入掩码有效
    if pred_mask is None or pred_mask.size == 0:
        print("警告: 预测掩码为空，无法可视化")
        return False
        
    # 确保掩码是二维数组
    if len(pred_mask.shape) > 2:
        pred_mask = pred_mask[:,:,0]  # 如果是多通道，取第一个通道
        
    # 确保mask尺寸与图像一致
    if pred_mask.shape[:2] != pre_img.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (pre_img.shape[1], pre_img.shape[0]), 
                              interpolation=cv2.INTER_NEAREST if not is_raw_output else cv2.INTER_LINEAR)
    
    # 边缘平滑处理 - 对于可视化效果
    if not is_raw_output:
        # 使用中值滤波平滑边缘
        smooth_mask = cv2.medianBlur(pred_mask, 5)
        # 使用形态学操作进一步平滑边缘
        kernel = np.ones((3, 3), np.uint8)
        smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # 使用高斯模糊
        if smooth_mask.dtype != np.float32:
            smooth_mask = smooth_mask.astype(np.float32)
        smooth_mask = cv2.GaussianBlur(smooth_mask, (5, 5), 1.0)
        pred_mask = (smooth_mask > 127).astype(np.uint8) * 255
    
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
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (0, 0, 0)  # 黑色
    
    # 计算标题位置
    title1 = "Pre-event Image"
    title2 = "Post-event Image"
    title3 = "Change Detection"
    title4 = "Binary Mask"
    
    title1_size = cv2.getTextSize(title1, font, font_scale, font_thickness)[0]
    title2_size = cv2.getTextSize(title2, font, font_scale, font_thickness)[0]
    title3_size = cv2.getTextSize(title3, font, font_scale, font_thickness)[0]
    title4_size = cv2.getTextSize(title4, font, font_scale, font_thickness)[0]
    
    title1_x = int(w / 2 - title1_size[0] / 2)
    title2_x = int(w + gap + w / 2 - title2_size[0] / 2)
    title3_x = int(w * 2 + gap * 2 + w / 2 - title3_size[0] / 2)
    title4_x = int(w * 3 + gap * 3 + w / 2 - title4_size[0] / 2)
    
    # 在画布顶部添加标题
    canvas = cv2.copyMakeBorder(canvas, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(canvas, title1, (title1_x, 20), font, font_scale, text_color, font_thickness)
    cv2.putText(canvas, title2, (title2_x, 20), font, font_scale, text_color, font_thickness)
    cv2.putText(canvas, title3, (title3_x, 20), font, font_scale, text_color, font_thickness)
    cv2.putText(canvas, title4, (title4_x, 20), font, font_scale, text_color, font_thickness)
    
    # 保存结果
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 确保文件扩展名正确
    _, file_ext = os.path.splitext(save_path)
    if not file_ext or file_ext.lower() not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        # 如果没有扩展名或扩展名不正确，默认添加.png
        save_path = save_path + '.png' if not file_ext else save_path.rsplit('.', 1)[0] + '.tif'
    
    try:
        # 使用PIL来保存图像，PIL可以更好地处理中文路径
        from PIL import Image
        # 保存为RGB格式
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(save_path)
        return True
    except Exception as e:
        pass

def process_and_save(args):
    """主处理流程，加载数据、模型，执行推理并保存结果 (移除打印)"""
    
    start_total_time = time.time()
    results = {
        "status": "processing",
        "message": "开始处理...",
        "output_path": None,
        "quad_view_path": None,
        "vector_files": [],
        "processing_time": {}
    }
    
    # 1. 加载模型
    model_load_start = time.time()
    try:
        model = load_model(args)
    except Exception as e:
        results["status"] = "failed"
        results["message"] = f"加载模型失败: {str(e)}"
        return results
    model_load_time = time.time() - model_load_start
    results["processing_time"]["model_load"] = model_load_time
    
    # 2. 获取数据变换
    transform = get_transform(args)

    # 3. 执行大图像处理
    process_start_time = time.time()
    pred_mask, pre_img, post_img, geo_transform, projection, src_ds = process_large_raster(
        args.before_path, args.after_path, model, transform, args
    )
    process_time = time.time() - process_start_time
    results["processing_time"]["process"] = process_time
    
    # 4. 后处理和保存
    post_start_time = time.time()
    try:
        # 确保输出目录存在 (基于 args.output_path 的目录部分)
        output_dir = os.path.dirname(args.output_path)
        if not output_dir:
             output_dir = "." # Use current directory if output_path is just a filename
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建输出文件名
        base_name = os.path.splitext(os.path.basename(args.output_path))[0]
        if not base_name:
             base_name = os.path.splitext(os.path.basename(args.before_path))[0] + "_result" # Fallback name
             
        # --- 保存二值掩码 GeoTIFF --- 
        mask_output_path = os.path.join(output_dir, f"{base_name}_mask.tif")
        if args.save_binary_mask:
            save_geotiff_result(pred_mask, mask_output_path, geo_transform, projection, src_ds)
            results["output_path"] = mask_output_path # 主输出是掩码

        # --- 可视化结果保存 --- 
        viz_output_path = None
        if args.save_visualization:
            viz_output_path = os.path.join(output_dir, f"{base_name}_quadview.png")
            try:
                # 需要原始图像用于可视化
                pre_img, _, _, _ = read_geotiff(args.before_path)
                post_img, _, _, _ = read_geotiff(args.after_path)
                visualize_results(pre_img, post_img, pred_mask, viz_output_path)
                results["quad_view_path"] = viz_output_path
            except Exception as viz_e:
                 pass # Visualization failure shouldn't stop the process

        # --- 矢量导出 --- 
        vector_dir = os.path.join(output_dir, "vectors")
        os.makedirs(vector_dir, exist_ok=True)
        vector_files = []
        
        if args.export_shapefile or args.export_geojson:
            try:
                # 将掩码转换为多边形
                polygons, areas = mask_to_lines(
                    pred_mask, 
                    min_area=args.min_polygon_area,
                    simplify=True,
                    simplify_tolerance=args.simplify_tolerance
                )
                
                if polygons:
                    attributes = None
                    if args.attribute_change_type or args.calculate_area:
                        attributes = []
                        for i, area in enumerate(areas):
                            attr = {}
                            if args.attribute_change_type:
                                attr['change_type'] = 'change' # Or derive from model if multi-class
                            if args.calculate_area:
                                attr['area_m2'] = area # Assuming area is in square meters
                            attributes.append(attr)
                    
                    # 导出 Shapefile
                    if args.export_shapefile:
                        shp_path = os.path.join(vector_dir, f"{base_name}_changes.shp")
                        export_vector(polygons, areas, shp_path, geo_transform, projection, attributes, export_format='shp')
                        vector_files.append(shp_path)
                        
                    # 导出 GeoJSON
                    if args.export_geojson:
                        geojson_path = os.path.join(vector_dir, f"{base_name}_changes.geojson")
                        export_vector(polygons, areas, geojson_path, geo_transform, projection, attributes, export_format='geojson')
                        vector_files.append(geojson_path)
                        
                    results["vector_files"] = vector_files
                else:
                     pass
            except Exception as vec_e:
                 pass # Vectorization failure shouldn't stop the process

        results["status"] = "success"
        results["message"] = "栅格模型处理完成"
        
    except Exception as post_e:
        results["status"] = "failed"
        results["message"] = f"后处理或保存失败: {str(post_e)}"
        # Attempt to keep partial results if available
        if "output_path" not in results and mask_output_path and os.path.exists(mask_output_path):
            results["output_path"] = mask_output_path
        if "quad_view_path" not in results and viz_output_path and os.path.exists(viz_output_path):
            results["quad_view_path"] = viz_output_path
            
    post_time = time.time() - post_start_time
    results["processing_time"]["post"] = post_time
    results["processing_time"]["total"] = time.time() - start_total_time

    return results

def clear_model_cache():
    """清空模型缓存"""
    global MODEL_CACHE
    MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """主函数，用于命令行测试 (移除打印)"""
    args = get_args()
    
    results = process_and_save(args)
    
    return results

if __name__ == '__main__':
    main()
