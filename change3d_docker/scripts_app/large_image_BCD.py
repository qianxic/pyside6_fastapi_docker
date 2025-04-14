#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 基于训练好的模型对大型图像进行变化检测推理
# 使用滑动窗口方法，将大图像分割成小块进行推理，然后融合结果

# 单图像推理示例:
'''
python scripts_app/large_image_BCD.py `
  --before_path "dataes/test/t1/test_3.png" `
  --after_path "dataes/test/t2/test_3.png" `
  --output_path "output.png"
'''

import os
import sys
import argparse
import numpy as np
from skimage import io
from tqdm import tqdm
import time
import math

import torch
import torch.nn.functional as F
import cv2

# 修改导入方式，使用sys.path添加父目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.trainer import Trainer
from data.transforms import BCDTransforms

# 添加全局模型缓存字典
MODEL_CACHE = {}

def get_args(before_path=None, after_path=None, output_path=None, model_path=None, **kwargs):
    """获取命令行参数
    
    Args:
        before_path: 前时相图像路径
        after_path: 后时相图像路径
        output_path: 输出路径
        
    Returns:
        Namespace: 参数命名空间对象
    """
    parser = argparse.ArgumentParser(description='图像变化检测')
    
    # 输入输出路径
    parser.add_argument('--before_path', type=str, default=before_path,
                        help='前时相图像路径')
    parser.add_argument('--after_path', type=str, default=after_path,
                        help='后时相图像路径')
    parser.add_argument('--output_path', type=str, default=output_path,
                        help='输出路径')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default=model_path ,
                        help='模型路径')
    parser.add_argument('--model_arch', type=str, default='siam_unet',
                        help='模型架构')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='设备 (例如 cuda:0 或 cpu)')
    
    # 图像处理参数
    parser.add_argument('--patch_size', type=int, default=256,
                        help='图像块大小')
    parser.add_argument('--block_size', type=int, default=512,
                        help='处理块大小')
    parser.add_argument('--stride_ratio', type=float, default=0.5,
                        help='滑动步长比例 (相对于块大小)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批处理大小')
    parser.add_argument('--overlap_weights', action='store_true',
                        help='使用加权融合处理重叠区域')
    
    # 输出选项
    parser.add_argument('--raw_output', action='store_true',
                        help='输出原始预测 (浮点数) 而不是二值化结果')
    parser.add_argument('--save_binary_mask', action='store_true', default=True,
                        help='保存二值掩码 (或原始输出)')
    parser.add_argument('--save_visualization', action='store_true', default=True,
                        help='保存四连图可视化结果')
    
    # Internal/Advanced parameters (usually set by change_detection_model.py)
    parser.add_argument('--in_height', type=int, default=256)
    parser.add_argument('--in_width', type=int, default=256)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--num_perception_frame', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='CD')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--auto_memory', action='store_true', default=False)
    parser.add_argument('--quiet', action='store_true', default=True)
    
    # Handle parsing logic
    if len(sys.argv) <= 1 and (before_path or after_path or output_path or model_path or kwargs):
        args = parser.parse_args([])
        if before_path: args.before_path = before_path
        if after_path: args.after_path = after_path
        if output_path: args.output_path = output_path
        if model_path: args.checkpoint = model_path
        for key, value in kwargs.items():
            if hasattr(args, key):
                setattr(args, key, value)
    else:
        args = parser.parse_args()

    # Ensure patch_size is used for in_height/in_width if not set otherwise
    if not hasattr(args, 'in_height') or args.in_height != args.patch_size:
        args.in_height = args.patch_size
    if not hasattr(args, 'in_width') or args.in_width != args.patch_size:
        args.in_width = args.patch_size
    
    return args

def load_model(args):
    """加载模型"""
    cache_key = f"{args.checkpoint}_{args.device}"
    
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"模型权重文件不存在: {args.checkpoint}")
    
    if not hasattr(args, 'pretrained'): args.pretrained = None
    
    try:
        model = Trainer(args).to(args.device)
        
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        MODEL_CACHE[cache_key] = model
        
        return model
    except Exception as e:
        raise e

def get_transform(args):
    """获取变换"""
    args.in_height = args.patch_size
    args.in_width = args.patch_size
    _, val_transform = BCDTransforms.get_transform_pipelines(args)
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
    """
    根据图像尺寸自动确定最优滑动窗口大小
    """
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
    """确定最优的滑动步长"""
    return patch_size // 2  # 默认步长为块大小的一半，确保有50%的重叠

def estimate_memory_usage(patch_size, batch_size, device):
    """
    估计每批次的内存占用
    """
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
    """
    根据滑动窗口大小和可用内存调整批处理大小
    """
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
        return args.batch_size

def process_block(pre_block, post_block, model, transform, args):
    """处理单个图像块"""
    if pre_block is None or post_block is None or pre_block.size == 0 or post_block.size == 0:
        h, w = (args.patch_size, args.patch_size)
        dtype = np.float32 if args.raw_output else np.uint8
        return np.zeros((h, w), dtype=dtype)
    
    try:
        if len(pre_block.shape) == 2: pre_block = cv2.cvtColor(pre_block, cv2.COLOR_GRAY2RGB)
        elif pre_block.shape[2] == 4: pre_block = pre_block[:, :, :3]
        else: pre_block = cv2.cvtColor(pre_block, cv2.COLOR_BGR2RGB)
            
        if len(post_block.shape) == 2: post_block = cv2.cvtColor(post_block, cv2.COLOR_GRAY2RGB)
        elif post_block.shape[2] == 4: post_block = post_block[:, :, :3]
        else: post_block = cv2.cvtColor(post_block, cv2.COLOR_BGR2RGB)
    except Exception as e:
        h, w = pre_block.shape[:2]
        dtype = np.float32 if args.raw_output else np.uint8
        return np.zeros((h, w), dtype=dtype)
    
    try:
        image = np.concatenate([pre_block, post_block], axis=2)
        mask = np.zeros(pre_block.shape[:2], dtype=np.float32)
        image_t, _ = transform(image, mask)
        pre_t = image_t[0:3].unsqueeze(0).to(args.device)
        post_t = image_t[3:6].unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            outputs = model.update_bcd(pre_t, post_t)
        
        pred = outputs.squeeze(1).cpu().numpy()
        pred = pred.squeeze(0)
        
        if not args.raw_output:
            pred = (pred > 0.5).astype(np.uint8) * 255
        
        del pre_t, post_t, outputs
        if torch.cuda.is_available() and str(args.device) != 'cpu':
            torch.cuda.empty_cache()
        
        return pred
    
    except Exception as e:
        h, w = pre_block.shape[:2]
        dtype = np.float32 if args.raw_output else np.uint8
        return np.zeros((h, w), dtype=dtype)

def process_large_image(before_path, after_path, model, transform, args):
    """处理大图像"""
    before_path = before_path.replace('\\', '/')
    after_path = after_path.replace('\\', '/')
    
    try:
        pre_img = cv2.imread(before_path)
        post_img = cv2.imread(after_path)
        
        if pre_img is None or post_img is None:
            raise ValueError(f"无法读取图像: {before_path} 或 {after_path}")
        
        if pre_img.shape != post_img.shape:
            min_height = min(pre_img.shape[0], post_img.shape[0])
            min_width = min(pre_img.shape[1], post_img.shape[1])
            
            pre_img = cv2.resize(pre_img, (min_width, min_height))
            post_img = cv2.resize(post_img, (min_width, min_height))
        
        height, width = pre_img.shape[:2]
        block_size = args.block_size
        patch_size = args.patch_size
        stride = int(patch_size * args.stride_ratio)
        if stride == 0: stride = patch_size

        pred_mask_accum = np.zeros((height, width), dtype=np.float32)
        weight_accum = np.zeros((height, width), dtype=np.float32)
        
        weight_map = create_weight_map(patch_size, stride) if args.overlap_weights else np.ones((patch_size, patch_size), dtype=np.float32)

        num_blocks_y = math.ceil(height / block_size)
        num_blocks_x = math.ceil(width / block_size)
        total_processed_patches = 0

        for by in range(num_blocks_y):
            for bx in range(num_blocks_x):
                start_y = by * block_size
                start_x = bx * block_size
                end_y = min(start_y + block_size, height)
                end_x = min(start_x + block_size, width)

                pre_block = pre_img[start_y:end_y, start_x:end_x]
                post_block = post_img[start_y:end_y, start_x:end_x]
                block_h, block_w = pre_block.shape[:2]

                current_patch_size_h = min(patch_size, block_h)
                current_patch_size_w = min(patch_size, block_w)
                current_stride_h = min(stride, current_patch_size_h)
                current_stride_w = min(stride, current_patch_size_w)
                if current_stride_h == 0: current_stride_h = 1
                if current_stride_w == 0: current_stride_w = 1

                patch_coords = []
                for py in range(0, block_h - current_patch_size_h + 1, current_stride_h):
                    actual_py = py if py + current_patch_size_h <= block_h else block_h - current_patch_size_h
                    for px in range(0, block_w - current_patch_size_w + 1, current_stride_w):
                        actual_px = px if px + current_patch_size_w <= block_w else block_w - current_patch_size_w
                        patch_coords.append((actual_px, actual_py, actual_px + current_patch_size_w, actual_py + current_patch_size_h))
                patch_coords = sorted(list(set(patch_coords)))

                for i in range(0, len(patch_coords), args.batch_size):
                    batch_coords = patch_coords[i:i+args.batch_size]
                    batch_pre = []
                    batch_post = []
                    
                    for x1, y1, x2, y2 in batch_coords:
                        patch_pre = pre_block[y1:y2, x1:x2]
                        patch_post = post_block[y1:y2, x1:x2]
                        
                        if patch_pre is not None and patch_post is not None and patch_pre.size > 0 and patch_post.size > 0:
                            try:
                                if len(patch_pre.shape) == 2: patch_pre = cv2.cvtColor(patch_pre, cv2.COLOR_GRAY2RGB)
                                elif patch_pre.shape[2] == 4: patch_pre = patch_pre[:,:,:3]
                                else: patch_pre = cv2.cvtColor(patch_pre, cv2.COLOR_BGR2RGB)

                                if len(patch_post.shape) == 2: patch_post = cv2.cvtColor(patch_post, cv2.COLOR_GRAY2RGB)
                                elif patch_post.shape[2] == 4: patch_post = patch_post[:,:,:3]
                                else: patch_post = cv2.cvtColor(patch_post, cv2.COLOR_BGR2RGB)

                                image = np.concatenate([patch_pre, patch_post], axis=2)
                                mask = np.zeros(patch_pre.shape[:2], dtype=np.float32)
                                image_t, _ = transform(image, mask)
                                batch_pre.append(image_t[0:3])
                                batch_post.append(image_t[3:6])
                            except Exception:
                                continue

                    if not batch_pre or not batch_post: continue

                    batch_pre_t = torch.stack(batch_pre).to(args.device)
                    batch_post_t = torch.stack(batch_post).to(args.device)

                    with torch.no_grad():
                        outputs = model.update_bcd(batch_pre_t, batch_post_t)

                    preds = outputs.squeeze(1).cpu().numpy()
                    
                    for j, (x1, y1, x2, y2) in enumerate(batch_coords):
                        pred_patch = preds[j]
                        global_y1, global_y2 = start_y + y1, start_y + y2
                        global_x1, global_x2 = start_x + x1, start_x + x2
                        
                        current_weight_map = cv2.resize(weight_map, (pred_patch.shape[1], pred_patch.shape[0])) if pred_patch.shape != weight_map.shape[:2] else weight_map
                        
                        pred_mask_accum[global_y1:global_y2, global_x1:global_x2] += pred_patch * current_weight_map
                        weight_accum[global_y1:global_y2, global_x1:global_x2] += current_weight_map
                        total_processed_patches += 1

                    del batch_pre_t, batch_post_t, outputs, preds
                    if torch.cuda.is_available() and str(args.device) != 'cpu':
                        torch.cuda.empty_cache()

        pred_mask = np.divide(pred_mask_accum, weight_accum, out=np.zeros_like(pred_mask_accum), where=weight_accum!=0)

        if not args.raw_output:
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        return pred_mask, pre_img, post_img
        
    except Exception as e:
        raise

def visualize_results(pre_img, post_img, pred_mask, save_path, is_raw_output=False):
    """可视化结果"""
    if pred_mask is None or pred_mask.size == 0:
        return False
        
    if len(pred_mask.shape) > 2: pred_mask = pred_mask[:,:,0]
    if pred_mask.shape[:2] != pre_img.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (pre_img.shape[1], pre_img.shape[0]), 
                              interpolation=cv2.INTER_NEAREST if not is_raw_output else cv2.INTER_LINEAR)
    
    max_size = 1200
    h, w = pre_img.shape[:2]
    if h > max_size or w > max_size:
        scale = min(max_size / h, max_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pre_img = cv2.resize(pre_img, (new_w, new_h))
        post_img = cv2.resize(post_img, (new_w, new_h))
        pred_mask = cv2.resize(pred_mask, (new_w, new_h), 
                              interpolation=cv2.INTER_NEAREST if not is_raw_output else cv2.INTER_LINEAR)
    
    h, w = pre_img.shape[:2]
    binary_mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
    if is_raw_output:
        binary = (pred_mask > 0.5).astype(np.uint8) * 255
        binary_mask_colored[..., 1] = binary
    else:
        binary_mask_colored[..., 1] = pred_mask
    
    if is_raw_output:
        heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        pred_colored = cv2.addWeighted(post_img, 0.7, heatmap, 0.3, 0)
    else:
        pred_colored = np.copy(post_img)
        mask_condition = pred_mask > 127
        pred_colored[..., 0] = np.where(mask_condition, 0, post_img[..., 0])
        pred_colored[..., 1] = np.where(mask_condition, 0, post_img[..., 1])
        pred_colored[..., 2] = np.where(mask_condition, 255, post_img[..., 2])
    
    gap = 5
    canvas_width = w * 4 + gap * 3
    canvas_height = h
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    canvas[0:h, 0:w] = pre_img
    canvas[0:h, w+gap:w*2+gap] = post_img
    canvas[0:h, w*2+gap*2:w*3+gap*2] = pred_colored
    canvas[0:h, w*3+gap*3:w*4+gap*3] = binary_mask_colored
    
    save_path = save_path.replace('\\', '/')
    save_dir = os.path.dirname(save_path)
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        return False
    
    _, file_ext = os.path.splitext(save_path)
    if not file_ext or file_ext.lower() not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        save_path = save_path + '.png' if not file_ext else save_path.rsplit('.', 1)[0] + '.png'
    
    success = False
    try:
        from PIL import Image
        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_canvas)
        pil_img.save(save_path)
        success = True
    except Exception as e:
        try:
            result = cv2.imwrite(save_path, canvas)
            if result:
                success = True
        except Exception as e2:
            pass
    
    return success

def process_and_save(args):
    """对外暴露的处理函数"""
    start_time = time.time()
    
    if 'cuda' in str(args.device): args.device = torch.device(args.device)
    else: args.device = torch.device('cpu')
    
    args.before_path = args.before_path.replace('\\', '/')
    args.after_path = args.after_path.replace('\\', '/')
    args.output_path = args.output_path.replace('\\', '/')
    
    output_dir = os.path.dirname(args.output_path)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        return {"status": "error", "message": f"无法创建输出目录: {str(e)}"}
    
    if not os.path.exists(args.before_path):
        return {"status": "error", "message": f"前时相图像不存在: {args.before_path}"}
    if not os.path.exists(args.after_path):
        return {"status": "error", "message": f"后时相图像不存在: {args.after_path}"}
    
    if not os.path.splitext(args.output_path)[1]:
        args.output_path = args.output_path + ".png"
    
    model_load_start = time.time()
    try:
        model = load_model(args)
        model_load_time = time.time() - model_load_start
    except Exception as e:
        return {"status": "error", "message": f"模型加载失败: {str(e)}"}

    transform = get_transform(args)
    
    process_start = time.time()
    pred_mask, pre_img, post_img = None, None, None
    try:
        pred_mask, pre_img, post_img = process_large_image(
            args.before_path, args.after_path, model, transform, args
        )
        process_time = time.time() - process_start
    except Exception as e:
        return {"status": "error", "message": f"图像处理失败: {str(e)}"}

    post_start = time.time()
    mask_save_success = False
    if args.save_binary_mask and pred_mask is not None:
        try:
            from PIL import Image
            save_data = (pred_mask * 255).astype(np.uint8) if args.raw_output else pred_mask
            Image.fromarray(save_data).save(args.output_path)
            mask_save_success = True
        except Exception as e1:
            try:
                cv2.imwrite(args.output_path, pred_mask)
                mask_save_success = True
            except Exception as e2:
                pass
    elif not args.save_binary_mask:
        mask_save_success = True

    quad_view_path = None
    visualize_success = False
    if args.save_visualization and pre_img is not None and post_img is not None and pred_mask is not None:
        quad_view_path = os.path.splitext(args.output_path)[0] + "_quadview.png"
        visualize_success = visualize_results(pre_img, post_img, pred_mask, quad_view_path, is_raw_output=args.raw_output)
    elif not args.save_visualization:
        visualize_success = True

    post_time = time.time() - post_start
    total_time = time.time() - start_time

    final_status = "success" if mask_save_success and visualize_success else "error"
    final_message = "图像模型处理完成"
    if not mask_save_success: final_message += " (保存掩码失败)"
    if not visualize_success and args.save_visualization: final_message += " (保存可视化失败)"
    
    result = {
        "status": final_status,
        "message": final_message,
        "output_path": args.output_path if mask_save_success else None,
        "quad_view_path": quad_view_path if visualize_success and args.save_visualization else None,
        "processing_time": {
            "total": total_time,
            "model_load": model_load_time,
            "process": process_time,
            "post": post_time
        }
    }
    
    return result

def clear_model_cache():
    """清除模型缓存"""
    global MODEL_CACHE
    MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """主函数，用于命令行调用"""
    args = get_args()
    
    args.before_path = args.before_path.replace('\\', '/')
    args.after_path = args.after_path.replace('\\', '/')
    args.output_path = args.output_path.replace('\\', '/')
    
    process_and_save(args)

if __name__ == "__main__":
    main() 