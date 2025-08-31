# Copyright (c) Duowang Zhu.
# All rights reserved.

'''
python scripts_train/__pycache__/train_BCD.py \
    --file_root "F:\\deeplearning\\cd\\LEVIR-CD" \
    --pretrained "X3D_L.pyth" \
    --save_dir "./exp_levir" \
    --max_steps 10000 \
    --batch_size 8 \
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

# 导入模块
sys.path.insert(0, '.')
import data.dataset as RSDataset
import data.transforms as RSTransforms
from utils.metric_tool import ConfuseMatrixMeter
from model.trainer import Trainer
from model.utils import (
    adjust_learning_rate,
    BCEDiceLoss,
    load_checkpoint,
    setup_logger
)

def create_data_loaders(args, train_transform, val_transform):
    """创建数据加载器"""
    # 训练数据
    train_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="train",
        transform=train_transform
    )
    
    # 验证数据
    val_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="val",
        transform=val_transform
    )
    
    # 测试数据
    try:
        test_data = RSDataset.BCDDataset(
            file_root=args.file_root,
            split="test",
            transform=val_transform
        )
    except:
        test_data = val_data  # 如果没有测试集，使用验证集
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, len(train_loader)

@torch.no_grad()
def val(args, val_loader, model, epoch):
    """验证函数"""
    model.eval()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    
    pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}', leave=False)
    
    for batched_inputs in pbar:
        img, target = batched_inputs
        
        # 数据准备
        pre_img = img[:, 0:3].cuda().float()
        post_img = img[:, 3:6].cuda().float()
        target = target.cuda().float()

        # 前向传播
        output = model.update_bcd(pre_img, post_img)
        loss = BCEDiceLoss(output, target)

        # 二值化预测
        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        # 记录损失和更新指标
        epoch_loss.append(loss.data.item())
        f1 = eval_meter.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'Loss': f'{loss.data.item():.4f}', 'F1': f'{f1:.4f}'})

    return sum(epoch_loss) / len(epoch_loss), eval_meter.get_scores()

def train(args, train_loader, model, optimizer, epoch, max_batches, cur_iter=0):
    """训练函数"""
    model.train()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}', leave=False)
    
    for iter_idx, batched_inputs in enumerate(pbar):
        img, target = batched_inputs

        # 数据准备
        pre_img = img[:, 0:3].cuda().float()
        post_img = img[:, 3:6].cuda().float()
        target = target.cuda().float()

        # 学习率调整
        lr = adjust_learning_rate(args, optimizer, epoch, iter_idx + cur_iter, max_batches)

        # 前向传播
        output = model.update_bcd(pre_img, post_img)
        loss = BCEDiceLoss(output, target)

        # 二值化预测
        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失和更新指标
        epoch_loss.append(loss.data.item())
        f1 = eval_meter.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.data.item():.4f}',
            'F1': f'{f1:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    return sum(epoch_loss) / len(epoch_loss), eval_meter.get_scores(), lr

def trainValidate(args):
    """主训练函数"""
    # 环境设置
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(16)
    torch.cuda.manual_seed(16)

    # 模型初始化
    model = Trainer(args).cuda().float()

    # 创建保存目录
    save_path = osp(args.save_dir, f"{args.dataset}_iter_{args.max_steps}_lr_{args.lr}")
    os.makedirs(save_path, exist_ok=True)

    # 数据变换和数据加载器
    train_transform, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    train_loader, val_loader, test_loader, max_batches = create_data_loaders(args, train_transform, val_transform)

    # 计算最大epoch数
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    
    # 加载检查点
    start_epoch, cur_iter = load_checkpoint(args, model, save_path, max_batches)
    
    # 设置日志记录器和优化器
    logger = setup_logger(args, save_path)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)
    
    # 跟踪最佳F1分数
    max_F1_val = 0

    print(f"开始训练: {args.max_epochs} epochs, {max_batches} batches/epoch")
    
    # 主训练循环
    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()
        
        # 训练一个epoch
        loss_train, score_tr, lr = train(args, train_loader, model, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(train_loader)
        
        # 验证
        torch.cuda.empty_cache()
        loss_val, score_val = val(args, test_loader, model, epoch)
        
        # 记录日志
        logger.write(f"{epoch}\t{score_val['Kappa']:.4f}\t{score_val['IoU']:.4f}\t{score_val['F1']:.4f}\t{score_val['recall']:.4f}\t{score_val['precision']:.4f}")
        logger.flush()

        # 保存检查点
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_val': loss_val,
            'F_train': score_tr['F1'],
            'F_val': score_val['F1'],
            'lr': lr
        }, osp(save_path, 'checkpoint.pth.tar'))

        # 保存最佳模型
        if max_F1_val <= score_val['F1']:
            max_F1_val = score_val['F1']
            torch.save(model.state_dict(), osp(save_path, 'best_model.pth'))

        # 打印epoch总结
        print(f"Epoch {epoch}: Train Loss={loss_train:.4f}, Val Loss={loss_val:.4f}, Val F1={score_val['F1']:.4f}")
    
    # 最终测试
    print("最终测试...")
    state_dict = torch.load(osp(save_path, 'best_model.pth'))
    model.load_state_dict(state_dict)
    loss_test, score_test = val(args, test_loader, model, 0)
    
    print(f"最终结果: Kappa={score_test['Kappa']:.4f}, IoU={score_test['IoU']:.4f}, F1={score_test['F1']:.4f}")
    
    logger.write(f"Test\t{score_test['Kappa']:.4f}\t{score_test['IoU']:.4f}\t{score_test['F1']:.4f}\t{score_test['recall']:.4f}\t{score_test['precision']:.4f}")
    logger.flush()
    logger.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    
    # 数据集参数
    parser.add_argument('--dataset', default="LEVIR-CD", help='数据集名称')
    parser.add_argument('--file_root', default="F:\\deeplearning\\cd\\LEVIR-CD", help='数据集路径')
    
    # 模型参数
    parser.add_argument('--in_height', type=int, default=256, help='图像高度')
    parser.add_argument('--in_width', type=int, default=256, help='图像宽度')
    parser.add_argument('--num_perception_frame', type=int, default=1, help='感知帧数')
    parser.add_argument('--num_class', type=int, default=1, help='类别数量')
    
    # 训练参数
    parser.add_argument('--max_steps', type=int, default=80000, help='最大迭代次数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--lr_mode', default='poly', help='学习率策略')
    parser.add_argument('--step_loss', type=int, default=100, help='学习率衰减周期')
    parser.add_argument('--pretrained', default='X3D_L.pyth', help='预训练权重路径')
    parser.add_argument('--save_dir', default='./exp', help='保存目录')
    parser.add_argument('--resume', default=None, help='恢复训练检查点')
    parser.add_argument('--log_file', default='train_val_log.txt', help='日志文件名')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID')

    args = parser.parse_args()
    trainValidate(args)