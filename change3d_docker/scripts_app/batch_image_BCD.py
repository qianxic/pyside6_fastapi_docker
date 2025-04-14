#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 批量处理普通图像的变化检测推理
# 基于inference_large_image_BCD.py，增加了批量处理功能

# 批量图像批量推理示例:
'''
python scripts_app/batch_image_BCD.py `
  --before_path "dataes/test/t1" `
  --after_path "dataes/test/t2" `
  --output_path "results/image_batch_results"
'''

import os
import sys
import argparse
import numpy as np
from skimage import io
import time
import glob
from pathlib import Path
import math
import warnings
import torch
import torch.nn.functional as F
import cv2

# 修改导入方式，使用sys.path添加父目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.trainer import Trainer
from data.transforms import BCDTransforms

# 添加全局模型缓存字典
MODEL_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='批量图像变化检测推理')
    parser.add_argument('--before_path', type=str, required=True, 
                        help='前时相图像目录')
    parser.add_argument('--after_path', type=str, required=True,
                        help='后时相图像目录')
    parser.add_argument('--output_path', type=str, default='results/batch_results',
                        help='输出主目录')
    
    # 以下参数设为不可见的默认值
    args = parser.parse_args()
    
    # 添加默认参数
    args.checkpoint = "checkpoint/checkpoint.pth.tar"
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.file_ext = '.png,.jpg,.jpeg,.tif,.tiff'
    args.in_height = 256
    args.in_width = 256
    args.num_perception_frame = 1
    args.num_class = 1
    args.save_result = True
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
    
    # 为了向后兼容，保留原始参数名，但使用新的参数名
    args.pre_dir = args.before_path
    args.post_dir = args.after_path
    args.output_dir = args.output_path
    
    return args

def main():
    # 开始总计时
    start_time_total = time.time()
    
    # 解析参数
    args = get_args()
    
    # 设置输出目录
    if not os.path.isabs(args.output_dir):
        # 使用相对于当前目录的输出路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        output_abs_dir = os.path.join(project_root, args.output_dir)
        args.output_dir = os.path.normpath(output_abs_dir)
    
    # 创建输出目录结构
    main_output_dir = args.output_dir
    os.makedirs(main_output_dir, exist_ok=True)
    
    # 创建result目录用于保存可视化结果
    if args.save_result:
        result_dir = os.path.join(main_output_dir, 'result')
        os.makedirs(result_dir, exist_ok=True)
    
    # 如果需要保存掩码，创建masks目录
    if args.save_binary_mask:
        mask_dir = os.path.join(main_output_dir, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
    
    # 获取所有图像对
    image_pairs = get_image_pairs(args)
    print(f"找到 {len(image_pairs)} 对图像")
    
    if len(image_pairs) == 0:
        print("未找到匹配的图像对，请确认目录结构和文件扩展名是否正确")
        return
        
    # 限制处理的图像数量
    if args.max_images > 0 and args.max_images < len(image_pairs):
        print(f"根据设置，只处理前 {args.max_images} 对影像")
        image_pairs = image_pairs[:args.max_images]
    
    # 设置设备
    if torch.cuda.is_available() and 'cuda' in args.device:
        args.device = torch.device(args.device)
        print(f"使用GPU设备: {args.device}")
    else:
        args.device = torch.device('cpu')
        print(f"GPU不可用或指定使用CPU，使用CPU推理")
    
    # 开始加载模型计时
    start_time_model = time.time()
    
    # 加载模型
    try:
        model = load_model(args)
        model_load_time = time.time() - start_time_model
        print(f"模型加载成功，耗时: {model_load_time:.2f}秒")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return
    
    # 获取转换函数
    transform = get_transform(args)
    
    # 批量处理图像
    total_time = 0
    success_count = 0
    failed_count = 0
    
    # 计算10%的进度间隔
    total_images = len(image_pairs)
    last_progress_printed = -1  # 跟踪上次打印的进度百分比
    
    # 开始批量处理
    print("开始批量处理...")
    for idx, (pre_img_path, post_img_path, filename) in enumerate(image_pairs):
        # 计算当前进度百分比和10%的进度点
        current_progress = (idx + 1) / total_images * 100
        current_progress_point = int(current_progress / 10)  # 0, 1, 2, ..., 10
        
        # 只有当达到新的10%整数点时才打印进度
        if current_progress_point > last_progress_printed:
            last_progress_printed = current_progress_point
            print(f"处理进度: {current_progress_point * 10}% - 已处理 {idx+1}/{total_images} 个图像")
            
        try:
            # 设置输出路径
            result_path = None
            if args.save_result:
                result_dir = os.path.join(main_output_dir, 'result')
                os.makedirs(result_dir, exist_ok=True)  # 确保目录存在
                result_path = os.path.join(result_dir, f"{filename}_result.png")
                result_path = os.path.normpath(result_path)
            
            # 设置掩码路径
            mask_path = None
            if args.save_binary_mask:
                mask_dir = os.path.join(main_output_dir, 'masks')
                os.makedirs(mask_dir, exist_ok=True)  # 确保目录存在
                mask_path = os.path.join(mask_dir, f"{filename}_mask.png")
                mask_path = os.path.normpath(mask_path)
            
            # 记录处理时间
            start_time_process = time.time()
            
            # 处理图像
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred_mask, pre_img, post_img = process_large_image(
                    pre_img_path, post_img_path, model, transform, args
                )
            
            # 检查结果是否有效
            if pred_mask is None or pred_mask.size == 0 or pre_img is None or post_img is None:
                print(f"警告: 处理图像 '{filename}' 返回了无效结果")
                failed_count += 1
                continue
            
            process_time = time.time() - start_time_process
            
            # 开始保存结果
            start_time_save = time.time()
            
            # 保存可视化结果
            if args.save_result and result_path:
                try:
                    visualize_results(pre_img, post_img, pred_mask, result_path, args.raw_output)
                except Exception as e:
                    print(f"保存可视化结果失败: {str(e)}")
            
            # 保存二值掩码
            if args.save_binary_mask and mask_path:
                try:
                    # 规范化路径，确保使用正确的路径分隔符
                    normalized_mask_path = os.path.normpath(mask_path)
                    
                    # 直接使用PIL保存掩码
                    from PIL import Image
                    if args.raw_output:
                        # 对于概率图，乘以255转换为灰度图
                        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(normalized_mask_path)
                    else:
                        # 对于二值掩码，直接保存
                        Image.fromarray(pred_mask).save(normalized_mask_path)
                except Exception as e:
                    print(f"保存掩码失败: {str(e)}")
            
            # 计算总保存时间
            save_time = time.time() - start_time_save
            
            # 总耗时
            total_process_time = time.time() - start_time_process
            total_time += total_process_time
            success_count += 1
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"处理图像 '{filename}' 时出错: {str(e)}")
            failed_count += 1
    
    # 打印统计信息
    print("="*50)
    if success_count > 0:
        avg_time = total_time / success_count
        print(f"批量处理完成！成功处理 {success_count}/{total_images} 个图像，平均耗时: {avg_time:.2f}秒/图像")
        
        if failed_count > 0:
            print(f"处理失败: {failed_count} 个图像")
    else:
        print("处理失败！没有成功处理任何图像")
    
    # 总耗时
    total_elapsed = time.time() - start_time_total
    minutes, seconds = divmod(total_elapsed, 60)
    print(f"总耗时: {int(minutes)}分{seconds:.2f}秒")

def get_image_pairs(args):
    """获取前后时相图像对"""
    pre_dir = os.path.normpath(args.before_path)
    post_dir = os.path.normpath(args.after_path)
    
    if not os.path.exists(pre_dir) or not os.path.exists(post_dir):
        raise ValueError(f"前后时相目录必须存在: {pre_dir}, {post_dir}")
    
    # 支持多种文件扩展名
    file_extensions = args.file_ext.split(',')
    
    # 获取所有前时相图像文件
    pre_files = []
    for ext in file_extensions:
        pattern = os.path.join(pre_dir, f'*{ext}')
        # 规范化glob模式
        pattern = os.path.normpath(pattern)
        pre_files.extend(glob.glob(pattern))
    pre_files = sorted(pre_files)
    
    # 创建图像对
    image_pairs = []
    no_match_count = 0
    for pre_file in pre_files:
        # 提取文件名（不含路径和扩展名）
        filename = os.path.basename(pre_file)
        filename_without_ext, file_ext = os.path.splitext(filename)
        
        # 构造对应的后时相文件路径（使用相同的扩展名）
        post_file = os.path.join(post_dir, f"{filename_without_ext}{file_ext}")
        post_file = os.path.normpath(post_file)
        
        # 检查后时相文件是否存在
        if os.path.exists(post_file):
            image_pairs.append((pre_file, post_file, filename_without_ext))
        else:
            no_match_count += 1
    
    # 如果有未匹配的图像，仅打印一次汇总信息
    if no_match_count > 0:
        print(f"共有 {no_match_count} 个前时相图像没有找到对应的后时相图像")
    
    return image_pairs

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
            
            # 创建新的状态字典，仅包含匹配的参数
            new_state_dict = {}
            for k, v in checkpoint_state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    new_state_dict[k] = v
                elif k in model_state_dict:
                    # 忽略尺寸不匹配的参数
                    pass
            
            # 加载匹配的参数
            model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)
    
    # 设置模型为评估模式
    model.eval()
    
    # 将模型保存到缓存
    MODEL_CACHE[cache_key] = model
    
    return model

def get_transform(args):
    """获取验证/测试用的变换"""
    # 保存原始patch_size，以便后续恢复
    original_patch_size = args.patch_size
    
    # 创建一个临时参数对象，用于获取变换
    temp_args = argparse.Namespace()
    temp_args.dataset = args.dataset
    temp_args.in_height = args.in_height if args.in_height > 0 else 256
    temp_args.in_width = args.in_width if args.in_width > 0 else 256
    
    # 获取变换
    _, val_transform = BCDTransforms.get_transform_pipelines(temp_args)
    
    # 恢复原始patch_size
    args.patch_size = original_patch_size
    
    return val_transform

def create_sliding_windows(image_shape, patch_size, stride):
    """创建滑动窗口坐标列表"""
    h, w = image_shape[:2]
    windows = []
    
    # 确保最后一个窗口能覆盖到图像边缘
    for y in range(0, h - patch_size + 1, stride):
        # 调整最后一个窗口以确保覆盖图像边缘
        if y + stride > h - patch_size and y < h - patch_size:
            y = h - patch_size
            
        for x in range(0, w - patch_size + 1, stride):
            # 调整最后一个窗口以确保覆盖图像边缘
            if x + stride > w - patch_size and x < w - patch_size:
                x = w - patch_size
                
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
    h, w = image_shape[:2]
    
    # 如果用户指定了块大小，则使用用户设置
    if args.patch_size > 0:
        patch_size = args.patch_size
        return patch_size
    
    # 根据图像尺寸自动计算合适的块大小
    # 目标是将图像分成大约 auto_patch_divisor 个块
    h_patch = max(min(h // args.auto_patch_divisor, args.max_patch_size), args.min_patch_size)
    w_patch = max(min(w // args.auto_patch_divisor, args.max_patch_size), args.min_patch_size)
    
    # 确保块大小是模型输入的倍数(通常是8或16，这里取16)
    model_factor = 16
    h_patch = (h_patch // model_factor) * model_factor
    w_patch = (w_patch // model_factor) * model_factor
    
    # 为简单起见，使用正方形块
    patch_size = min(h_patch, w_patch)
    
    # 确保块大小至少为最小值
    patch_size = max(patch_size, args.min_patch_size)
    
    return patch_size

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
        reserved_memory = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
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

def preprocess_image(image_path):
    """读取和预处理图像"""
    try:
        img = io.imread(image_path)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] > 3:  # 处理多波段图像
            img = img[:, :, :3]
        return img
    except Exception as e:
        raise IOError(f"无法读取图像: {image_path}, 错误: {str(e)}")

def process_large_image(pre_img_path, post_img_path, model, transform, args):
    """处理大型图像，使用滑动窗口方法"""
    # 读取原始图像
    try:
        pre_img = preprocess_image(pre_img_path)
    except Exception as e:
        raise IOError(f"无法读取前时相图像: {pre_img_path}")
    
    try:
        post_img = preprocess_image(post_img_path)
    except Exception as e:
        raise IOError(f"无法读取后时相图像: {post_img_path}")
    
    # 确保两个图像尺寸一致
    if pre_img.shape[:2] != post_img.shape[:2]:
        # 不打印警告，直接调整
        post_img = cv2.resize(post_img, (pre_img.shape[1], pre_img.shape[0]))
    
    # 获取图像尺寸
    h, w = pre_img.shape[:2]
    
    # 自动确定最优滑动窗口大小
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
    
    # 开始处理
    model.eval()
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # 获取当前批次的窗口
            batch_windows = windows[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # 准备批次数据
            batch_pre = []
            batch_post = []
            
            for x1, y1, x2, y2 in batch_windows:
                # 提取图像块
                pre_patch = pre_img[y1:y2, x1:x2]
                post_patch = post_img[y1:y2, x1:x2]
                
                # 应用变换
                image = np.concatenate([pre_patch, post_patch], axis=2)
                mask = np.zeros((patch_size, patch_size), dtype=np.float32)
                image_t, _ = transform(image, mask)
                
                # 分离前后时相
                pre_t = image_t[0:3].unsqueeze(0)
                post_t = image_t[3:6].unsqueeze(0)
                
                batch_pre.append(pre_t)
                batch_post.append(post_t)
            
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
                for idx, (x1, y1, x2, y2) in enumerate(batch_windows):
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
                
                # 清理内存
                del batch_pre, batch_post, outputs, preds
                # 只在使用GPU时清理CUDA缓存
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
    
    # 对结果应用高斯滤波，使边界更平滑
    if not args.raw_output:
        # 对二值化结果应用高斯滤波
        result_map = cv2.GaussianBlur(result_map, (5, 5), 0)
        # 重新二值化
        result_map = (result_map > 127).astype(np.uint8) * 255
    else:
        # 对概率图应用轻微平滑
        result_map = cv2.GaussianBlur(result_map, (3, 3), 0)
    
    return result_map, pre_img, post_img

def visualize_results(pre_img, post_img, pred_mask, save_path, is_raw_output=False):
    """可视化结果 - 将前后时相图像、变化检测结果和二值掩码共四张图横排显示"""
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
        # 红色表示变化区域
        pred_colored[..., 0] = np.where(pred_mask > 127, 255, post_img[..., 0])
        pred_colored[..., 1] = np.where(pred_mask > 127, 0, post_img[..., 1])
        pred_colored[..., 2] = np.where(pred_mask > 127, 0, post_img[..., 2])
    
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
    
    # 规范化路径，确保使用正确的路径分隔符
    normalized_save_path = os.path.normpath(save_path)
    
    try:
        # 直接使用PIL保存图像，跳过OpenCV的imwrite
        from PIL import Image
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(normalized_save_path)
    except Exception as e:
        # 出错重新抛出，让上层函数处理
        raise

def clear_model_cache():
    """清除模型缓存"""
    global MODEL_CACHE
    MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_and_save(args):
    """对外暴露的批量处理函数，用于API调用
    
    Args:
        args: 参数对象，包含输入输出路径等信息
        
    Returns:
        Dict: 处理结果
    """
    # 开始总计时
    start_time_total = time.time()
    
    # 规范化路径
    args.before_path = args.before_path.replace('\\', '/')
    args.after_path = args.after_path.replace('\\', '/')
    args.output_path = args.output_path.replace('\\', '/')
    
    # 设置输出目录
    if not os.path.isabs(args.output_path):
        # 使用相对于当前目录的输出路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        output_abs_dir = os.path.join(project_root, args.output_path)
        args.output_path = os.path.normpath(output_abs_dir)
    
    # 创建输出目录结构
    main_output_dir = args.output_path
    os.makedirs(main_output_dir, exist_ok=True)
    
    # 创建result目录用于保存可视化结果
    result_dir = os.path.join(main_output_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    
    # 创建masks目录用于保存二值掩码
    mask_dir = os.path.join(main_output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    
    # 添加兼容性设置
    args.pre_dir = args.before_path
    args.post_dir = args.after_path
    args.output_dir = args.output_path
    args.save_result = True
    args.save_binary_mask = True
    
    # 获取所有图像对
    image_pairs = get_image_pairs(args)
    
    if len(image_pairs) == 0:
        return {
            "status": "failed",
            "message": "未找到匹配的图像对，请确认目录结构和文件扩展名是否正确",
            "output_path": args.output_path,
            "processing_time": {
                "total": 0
            },
            "details": []
        }
    
    # 限制处理的图像数量
    if args.max_images > 0 and args.max_images < len(image_pairs):
        image_pairs = image_pairs[:args.max_images]
    
    # 设置设备
    if torch.cuda.is_available() and 'cuda' in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device('cpu')
    
    # 开始加载模型计时
    start_time_model = time.time()
    
    # 加载模型
    try:
        model = load_model(args)
        model_load_time = time.time() - start_time_model
    except Exception as e:
        return {
            "status": "failed",
            "message": f"模型加载失败: {str(e)}",
            "output_path": args.output_path,
            "processing_time": {
                "total": time.time() - start_time_total,
                "model_load": time.time() - start_time_model
            },
            "details": []
        }
    
    # 获取转换函数
    transform = get_transform(args)
    
    # 批量处理图像
    total_process_time = 0
    success_count = 0
    failed_count = 0
    details = []
    
    # 开始批量处理
    for idx, (pre_img_path, post_img_path, filename) in enumerate(image_pairs):
        try:
            # 设置输出路径
            result_path = os.path.join(result_dir, f"{filename}_result.png")
            result_path = os.path.normpath(result_path)
            
            # 设置掩码路径
            mask_path = os.path.join(mask_dir, f"{filename}_mask.png")
            mask_path = os.path.normpath(mask_path)
            
            # 记录处理时间
            start_time_process = time.time()
            
            # 处理图像
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred_mask, pre_img, post_img = process_large_image(
                    pre_img_path, post_img_path, model, transform, args
                )
            
            # 检查结果是否有效
            if pred_mask is None or pred_mask.size == 0 or pre_img is None or post_img is None:
                failed_count += 1
                details.append({
                    "pre_img_path": pre_img_path,
                    "post_img_path": post_img_path,
                    "status": "failed",
                    "message": "处理返回了无效结果",
                    "result_path": None,
                    "mask_path": None,
                    "processing_time": time.time() - start_time_process
                })
                continue
            
            process_time = time.time() - start_time_process
            
            # 开始保存结果
            start_time_save = time.time()
            
            # 保存可视化结果
            try:
                visualize_success = visualize_results(pre_img, post_img, pred_mask, result_path, args.raw_output)
            except Exception as e:
                visualize_success = False
            
            # 保存二值掩码
            try:
                from PIL import Image
                if args.raw_output:
                    # 对于概率图，乘以255转换为灰度图
                    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_path)
                else:
                    # 对于二值掩码，直接保存
                    Image.fromarray(pred_mask).save(mask_path)
                mask_success = True
            except Exception as e:
                mask_success = False
            
            # 计算总保存时间
            save_time = time.time() - start_time_save
            
            # 总耗时
            total_item_time = time.time() - start_time_process
            total_process_time += total_item_time
            success_count += 1
            
            # 添加到处理明细
            details.append({
                "pre_img_path": pre_img_path,
                "post_img_path": post_img_path,
                "status": "success",
                "result_path": result_path if visualize_success else None,
                "mask_path": mask_path if mask_success else None,
                "processing_time": total_item_time
            })
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            failed_count += 1
            details.append({
                "pre_img_path": pre_img_path,
                "post_img_path": post_img_path,
                "status": "failed",
                "message": f"处理出错: {str(e)}",
                "result_path": None,
                "mask_path": None,
                "processing_time": time.time() - start_time_process if 'start_time_process' in locals() else 0
            })
    
    # 计算总耗时
    total_elapsed = time.time() - start_time_total
    
    # 构建返回结果
    result = {
        "status": "success" if success_count > 0 else "failed",
        "message": f"批量处理完成! 成功: {success_count}, 失败: {failed_count}, 总数: {len(image_pairs)}",
        "output_path": args.output_path,
        "result_dir": result_dir,
        "mask_dir": mask_dir,
        "processing_time": {
            "total": round(total_elapsed, 2),
            "model_load": round(model_load_time, 2),
            "processing": round(total_process_time, 2),
            "avg_per_image": round(total_process_time / max(success_count, 1), 2)
        },
        "details": details
    }
    
    return result

if __name__ == "__main__":
    main() 