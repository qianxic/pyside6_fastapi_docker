# Copyright (c) Duowang Zhu.
# All rights reserved.

'''
测试脚本 - 加载训练好的模型进行推理测试
python scripts_train/__pycache__/test_BCD.py `
    --file_root "G:\1代码\开发\RSIIS\遥感影像变化检测系统V1.1\change3d_docker\dataes_test" `
    --model_path "G:\1代码\开发\RSIIS\遥感影像变化检测系统V1.1\change3d_docker\checkpoint\best_model.pth" `
    --save_dir "./test_results" `
    --batch_size 1 `
    --num_workers 0
'''

import os
import sys
import time
import numpy as np
from os.path import join as osp
from argparse import ArgumentParser
from tqdm import tqdm

# GPU设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.backends.cudnn as cudnn

# 可视化相关导入
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from PIL import Image
import cv2

# 导入模块
sys.path.insert(0, '.')
import data.dataset as RSDataset
import data.transforms as RSTransforms
from utils.metric_tool import ConfuseMatrixMeter
from model.trainer import Trainer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_batch_results(pre_img, post_img, target, pred, save_path, batch_idx, sample_idx):
    """可视化单个样本的检测结果"""
    # 转换为numpy数组
    pre_img_np = pre_img.cpu().numpy().transpose(1, 2, 0)
    post_img_np = post_img.cpu().numpy().transpose(1, 2, 0)
    target_np = target.cpu().numpy().squeeze()
    pred_np = pred.cpu().numpy().squeeze()
    
    # 归一化图像到0-1范围
    pre_img_np = (pre_img_np - pre_img_np.min()) / (pre_img_np.max() - pre_img_np.min())
    post_img_np = (post_img_np - post_img_np.min()) / (post_img_np.max() - post_img_np.min())
    
    # 创建图像对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'变化检测结果 - 批次{batch_idx} 样本{sample_idx}', fontsize=16)
    
    # 时相1图像
    axes[0, 0].imshow(pre_img_np)
    axes[0, 0].set_title('时相1图像 (T1)', fontsize=12)
    axes[0, 0].axis('off')
    
    # 时相2图像
    axes[0, 1].imshow(post_img_np)
    axes[0, 1].set_title('时相2图像 (T2)', fontsize=12)
    axes[0, 1].axis('off')
    
    # 差异图
    diff_img = np.abs(post_img_np - pre_img_np)
    axes[0, 2].imshow(diff_img)
    axes[0, 2].set_title('图像差异', fontsize=12)
    axes[0, 2].axis('off')
    
    # 真实标签
    axes[1, 0].imshow(target_np, cmap='gray')
    axes[1, 0].set_title('真实标签 (Ground Truth)', fontsize=12)
    axes[1, 0].axis('off')
    
    # 预测结果
    axes[1, 1].imshow(pred_np, cmap='gray')
    axes[1, 1].set_title('预测结果 (Prediction)', fontsize=12)
    axes[1, 1].axis('off')
    
    # 预测概率图
    pred_prob = pred.cpu().numpy().squeeze()
    im = axes[1, 2].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title('预测概率图', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # 保存图像
    save_file = osp(save_path, f'result_batch{batch_idx}_sample{sample_idx}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_file

def create_confusion_matrix_visualization(cm, save_path):
    """创建混淆矩阵可视化"""
    # 计算混淆矩阵
    tn, fp, fn, tp = cm.ravel()
    
    # 创建混淆矩阵图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 数值混淆矩阵
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm_matrix, annot=True, fmt='.0f', cmap='Blues', 
                xticklabels=['预测无变化', '预测有变化'],
                yticklabels=['实际无变化', '实际有变化'],
                ax=ax1)
    ax1.set_title('混淆矩阵 (数值)', fontsize=14)
    ax1.set_xlabel('预测标签', fontsize=12)
    ax1.set_ylabel('真实标签', fontsize=12)
    
    # 百分比混淆矩阵
    cm_percent = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['预测无变化', '预测有变化'],
                yticklabels=['实际无变化', '实际有变化'],
                ax=ax2)
    ax2.set_title('混淆矩阵 (百分比)', fontsize=14)
    ax2.set_xlabel('预测标签', fontsize=12)
    ax2.set_ylabel('真实标签', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图像
    save_file = osp(save_path, 'confusion_matrix.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_file

def create_metrics_visualization(scores, save_path):
    """创建评估指标可视化"""
    # 提取指标 - 注意键名大小写匹配
    metrics = ['IoU', 'F1', 'OA', 'precision', 'recall', 'Kappa']
    values = [scores[metric] for metric in metrics]
    
    # 创建柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 主要指标柱状图
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#6B4E71']
    # 创建显示标签（首字母大写）
    display_labels = ['IoU', 'F1', 'OA', 'Precision', 'Recall', 'Kappa']
    bars = ax1.bar(display_labels, values, color=colors, alpha=0.8)
    ax1.set_title('评估指标对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('分数', fontsize=12)
    ax1.set_ylim(0, 1)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # 闭合图形
    angles += angles[:1]
    
    # 创建显示标签（首字母大写）
    display_labels = ['IoU', 'F1', 'OA', 'Precision', 'Recall', 'Kappa']
    
    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
    ax2.fill(angles, values, alpha=0.25, color='#2E86AB')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(display_labels)
    ax2.set_ylim(0, 1)
    ax2.set_title('评估指标雷达图', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存图像
    save_file = osp(save_path, 'metrics_visualization.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_file

def create_test_loader(args, test_transform):
    """创建测试数据加载器"""
    # 测试数据
    try:
        test_data = RSDataset.BCDDataset(
            file_root=args.file_root,
            split="test",
            transform=test_transform
        )
        print(f"使用测试集，样本数: {len(test_data)}")
    except:
        # 如果没有测试集，使用验证集
        test_data = RSDataset.BCDDataset(
            file_root=args.file_root,
            split="val",
            transform=test_transform
        )
        print(f"使用验证集作为测试集，样本数: {len(test_data)}")
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return test_loader, len(test_loader)

@torch.no_grad()
def test(args, test_loader, model, save_path):
    """测试函数"""
    model.eval()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    test_loss = []
    total_time = 0
    
    # 创建可视化保存目录
    vis_dir = osp(save_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    print("开始测试...")
    pbar = tqdm(test_loader, desc='Testing', leave=True)
    
    # 记录可视化样本数量
    vis_count = 0
    max_vis_samples = min(args.max_vis_samples, len(test_loader) * args.batch_size)
    
    for batch_idx, batched_inputs in enumerate(pbar):
        img, target = batched_inputs
        
        # 数据准备
        pre_img = img[:, 0:3].cuda().float()
        post_img = img[:, 3:6].cuda().float()
        target = target.cuda().float()

        # 计时
        start_time = time.time()
        
        # 前向传播
        output = model.update_bcd(pre_img, post_img)
        
        # 计算损失
        from model.utils import BCEDiceLoss
        loss = BCEDiceLoss(output, target)

        # 二值化预测
        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        # 记录时间和损失
        batch_time = time.time() - start_time
        total_time += batch_time
        test_loss.append(loss.data.item())
        
        # 更新评估指标
        f1 = eval_meter.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())
        
        # 可视化部分样本
        if vis_count < max_vis_samples:
            for sample_idx in range(min(args.batch_size, max_vis_samples - vis_count)):
                if vis_count >= max_vis_samples:
                    break
                    
                # 可视化单个样本
                vis_file = visualize_batch_results(
                    pre_img[sample_idx], post_img[sample_idx], 
                    target[sample_idx], output[sample_idx],
                    vis_dir, batch_idx, sample_idx
                )
                vis_count += 1
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.data.item():.4f}', 
            'F1': f'{f1:.4f}',
            'Time': f'{batch_time:.3f}s',
            'Vis': f'{vis_count}/{max_vis_samples}'
        })

    # 计算平均损失和总时间
    avg_loss = sum(test_loss) / len(test_loss)
    avg_time = total_time / len(test_loader)
    
    print(f"\n测试完成!")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"平均推理时间: {avg_time:.3f}s/批次")
    print(f"总推理时间: {total_time:.2f}s")
    print(f"可视化样本数: {vis_count}")
    
    return avg_loss, eval_meter.get_scores(), eval_meter.sum

def save_test_results(save_path, scores, confusion_matrix, args):
    """保存测试结果"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 保存详细结果到文件
    result_file = osp(save_path, 'test_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("遥感影像变化检测测试结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"数据集路径: {args.file_root}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"图像尺寸: {args.in_height}x{args.in_width}\n")
        f.write("-" * 60 + "\n")
        f.write("评估指标:\n")
        f.write(f"Kappa系数: {scores['Kappa']:.4f}\n")
        f.write(f"IoU (交并比): {scores['IoU']:.4f}\n")
        f.write(f"F1分数: {scores['F1']:.4f}\n")
        f.write(f"总体精度 (OA): {scores['OA']:.4f}\n")
        f.write(f"召回率 (Recall): {scores['recall']:.4f}\n")
        f.write(f"精确率 (Precision): {scores['precision']:.4f}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n测试结果已保存到: {result_file}")
    
    # 创建可视化
    print("\n正在生成可视化结果...")
    
    # 混淆矩阵可视化
    cm_file = create_confusion_matrix_visualization(confusion_matrix, save_path)
    print(f"混淆矩阵已保存到: {cm_file}")
    
    # 评估指标可视化
    metrics_file = create_metrics_visualization(scores, save_path)
    print(f"评估指标图已保存到: {metrics_file}")
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要:")
    print("=" * 60)
    print(f"Kappa系数: {scores['Kappa']:.4f}")
    print(f"IoU (交并比): {scores['IoU']:.4f}")
    print(f"F1分数: {scores['F1']:.4f}")
    print(f"总体精度 (OA): {scores['OA']:.4f}")
    print(f"召回率 (Recall): {scores['recall']:.4f}")
    print(f"精确率 (Precision): {scores['precision']:.4f}")
    print("=" * 60)
    print(f"\n可视化结果保存在: {osp(save_path, 'visualizations')}")

def test_model(args):
    """主测试函数"""
    # 环境设置
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(16)
    torch.cuda.manual_seed(16)

    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    print(f"加载模型: {args.model_path}")
    
    # 模型初始化
    model = Trainer(args).cuda().float()
    
    # 加载预训练权重
    try:
        state_dict = torch.load(args.model_path, map_location='cuda')
        model.load_state_dict(state_dict)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # 创建保存目录
    save_path = osp(args.save_dir, f"test_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_path, exist_ok=True)

    # 数据变换
    _, test_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    
    # 创建测试数据加载器
    test_loader, num_batches = create_test_loader(args, test_transform)
    
    print(f"测试配置:")
    print(f"  数据集: {args.file_root}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  总批次数: {num_batches}")
    print(f"  图像尺寸: {args.in_height}x{args.in_width}")
    print(f"  保存目录: {save_path}")
    print(f"  最大可视化样本数: {args.max_vis_samples}")
    print("-" * 60)

    # 开始测试
    torch.cuda.empty_cache()
    test_loss, test_scores, confusion_matrix = test(args, test_loader, model, save_path)
    
    # 保存测试结果
    save_test_results(save_path, test_scores, confusion_matrix, args)

if __name__ == '__main__':
    parser = ArgumentParser()
    
    # 数据集参数
    parser.add_argument('--dataset', default="LEVIR-CD", help='数据集名称')
    parser.add_argument('--file_root', default="F:\\deeplearning\\cd\\LEVIR-CD", help='数据集路径')
    
    # 模型参数
    parser.add_argument('--model_path', 
                       default="G:\\1代码\\开发\\RSIIS\\遥感影像变化检测系统V1.1\\change3d_docker\\checkpoint\\best_model.pth",
                       help='训练好的模型路径')
    parser.add_argument('--in_height', type=int, default=256, help='图像高度')
    parser.add_argument('--in_width', type=int, default=256, help='图像宽度')
    parser.add_argument('--num_perception_frame', type=int, default=1, help='感知帧数')
    parser.add_argument('--num_class', type=int, default=1, help='类别数量')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--save_dir', default='./test_results', help='结果保存目录')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID')
    parser.add_argument('--max_vis_samples', type=int, default=20, help='最大可视化样本数量')
    
    # 预训练权重路径（测试时不需要，但模型初始化需要）
    parser.add_argument('--pretrained', default='X3D_L.pyth', help='预训练权重路径')

    args = parser.parse_args()
    test_model(args)
