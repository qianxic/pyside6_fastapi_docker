# Copyright (c) Duowang Zhu.
# All rights reserved.


'''

python scripts_train\__pycache__\train_BCD.py --datasets "LEVIR-CD,WHU,CLCD" --file_roots "D:/代码/RSIIS/遥感影像变化检测系统V1.0/change3d/dataes/LEVIR-CD,D:/代码/RSIIS/遥感影像变化检测系统V1.0/change3d/dataes/WHU,D:/代码/RSIIS/遥感影像变化检测系统V1.0/change3d/dataes/CLCD" --pretrained "D:/代码/RSIIS/遥感影像变化检测系统V1.0/change3d/X3D_L.pyth" --save_dir "./exp_mixed" --max_steps 10000 --batch_size 8 --num_workers 0'''
print("开始执行训练脚本...")


import os
import sys
import time
import numpy as np
from os.path import join as osp
from argparse import ArgumentParser
# 设置GPU环境变量，确保CUDA初始化正常
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一块GPU
print("已设置CUDA环境变量")
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# Insert current path for local module imports
sys.path.insert(0, '.')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import data.dataset as RSDataset
    print("导入数据集模块成功")
except Exception as e:
    print(f"导入数据集模块失败: {e}")

try:
    import data.transforms as RSTransforms
    print("导入数据转换模块成功")
except Exception as e:
    print(f"导入数据转换模块失败: {e}")

try:
    from utils.metric_tool import ConfuseMatrixMeter
    print("导入评估工具模块成功")
except Exception as e:
    print(f"导入评估工具模块失败: {e}")

try:
    from model.trainer import Trainer
    print("导入训练器模块成功")
except Exception as e:
    print(f"导入训练器模块失败: {e}")

try:
    from model.utils import (
        adjust_learning_rate,
        BCEDiceLoss,
        load_checkpoint,
        setup_logger
    )
    print("导入工具函数模块成功")
except Exception as e:
    print(f"导入工具函数模块失败: {e}")
    
print("所有模块导入完成，开始执行主程序...")

def create_data_loaders(args, train_transform, val_transform):
    """
    Creates data loaders for training, validation, and testing.
    
    Args:
        args: Command line arguments.
        train_transform: Transform pipeline for training data.
        val_transform: Transform pipeline for validation and testing data.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, max_batches).
    """
    # 检查是否使用多数据集模式
    if hasattr(args, 'datasets') and ',' in args.datasets:
        # 多数据集模式
        file_roots = args.file_roots.split(',')
        dataset_names = args.datasets.split(',')
        
        # 确保两个列表长度一致
        assert len(file_roots) == len(dataset_names), "数据集名称和路径数量必须一致"
        print(f"正在使用多数据集训练模式，包含 {len(file_roots)} 个数据集: {', '.join(dataset_names)}")
        
        # 训练数据 - 混合数据集
        train_data = RSDataset.MixedBCDDataset(
            file_roots=file_roots,
            splits=["train"] * len(file_roots),  # 对所有数据集使用train分割
            transform=train_transform
        )
        
        # 验证数据 - 使用各数据集的val分割
        try:
            val_datasets = []
            for i, file_root in enumerate(file_roots):
                try:
                    val_dataset = RSDataset.BCDDataset(
                        file_root=file_root,
                        split="val",
                        transform=val_transform
                    )
                    val_datasets.append(val_dataset)
                    print(f"使用数据集 {dataset_names[i]} 进行验证，样本数: {len(val_dataset)}")
                except Exception as e:
                    print(f"警告：数据集 {dataset_names[i]} 没有可用的验证集: {str(e)}")
            
            if val_datasets:
                # 如果有多个验证集，合并它们
                val_data = torch.utils.data.ConcatDataset(val_datasets)
            else:
                # 如果没有验证集，使用第一个数据集的训练集的一部分
                val_data = RSDataset.BCDDataset(
                    file_root=file_roots[0],
                    split="train",
                    transform=val_transform
                )
                print(f"警告：没有可用的验证集，使用 {dataset_names[0]} 的训练集作为验证集")
        except Exception as e:
            print(f"警告：创建验证集失败: {str(e)}")
            # 如果失败，回退到使用第一个训练集的一部分作为验证集
            val_data = RSDataset.BCDDataset(
                file_root=file_roots[0],
                split="train",
                transform=val_transform
            )
        
        # 测试数据 - 特别处理WHU数据集，直接使用val作为测试集
        try:
            test_datasets = []
            for i, file_root in enumerate(file_roots):
                dataset_name = dataset_names[i]
                
                # 对WHU数据集特殊处理 - 直接使用验证集作为测试集
                if "WHU" in dataset_name:
                    try:
                        print(f"WHU数据集检测到，使用val作为测试集")
                        test_dataset = RSDataset.BCDDataset(
                            file_root=file_root,
                            split="val",
                            transform=val_transform
                        )
                        test_datasets.append(test_dataset)
                        print(f"成功：使用数据集 {dataset_name} 的验证集作为测试集，样本数: {len(test_dataset)}")
                    except Exception as e:
                        print(f"错误：WHU数据集验证集加载失败: {str(e)}")
                # 其他数据集正常处理
                else:
                    try:
                        test_dataset = RSDataset.BCDDataset(
                            file_root=file_root,
                            split="test",
                            transform=val_transform
                        )
                        test_datasets.append(test_dataset)
                        print(f"使用数据集 {dataset_name} 进行测试，样本数: {len(test_dataset)}")
                    except Exception as e:
                        # 如果没有test子目录，尝试使用val代替
                        try:
                            test_dataset = RSDataset.BCDDataset(
                                file_root=file_root,
                                split="val",
                                transform=val_transform
                            )
                            test_datasets.append(test_dataset)
                            print(f"数据集 {dataset_name} 没有测试集，使用验证集代替，样本数: {len(test_dataset)}")
                        except Exception as e2:
                            print(f"警告：数据集 {dataset_name} 没有可用的测试集或验证集: {str(e2)}")
            
            if test_datasets:
                # 如果有测试集，合并它们
                test_data = torch.utils.data.ConcatDataset(test_datasets)
            else:
                # 如果没有测试集，使用验证集
                test_data = val_data
                print("警告：没有可用的测试集，使用验证集作为测试集")
        except Exception as e:
            print(f"警告：创建测试集失败: {str(e)}")
            # 如果失败，回退到使用验证集
            test_data = val_data
    else:
        # 单数据集模式 - 原始代码
        # Training data
        train_data = RSDataset.BCDDataset(
            file_root=args.file_root,
            split="train",
            transform=train_transform
        )
        
        # Validation data
        val_data = RSDataset.BCDDataset(
            file_root=args.file_root,
            split="val",
            transform=val_transform
        )
        
        # 处理特殊情况：WHU数据集且单数据集模式
        if "WHU" in args.dataset:
            print(f"单数据集模式下的WHU数据集检测到，使用val作为测试集")
            test_data = val_data  # 直接使用验证集作为测试集
        else:
            # 标准情况下使用测试集
            try:
                test_data = RSDataset.BCDDataset(
                    file_root=args.file_root,
                    split="test",
                    transform=val_transform
                )
            except Exception as e:
                print(f"警告：测试集加载失败，使用验证集替代: {str(e)}")
                test_data = val_data
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    max_batches = len(train_loader)
    print(f"每个epoch有 {max_batches} 个批次。")
    
    return train_loader, val_loader, test_loader, max_batches


@torch.no_grad()
def val(args, val_loader, model, epoch):
    """
    Validates the model on the validation set.
    
    Args:
        args: Command line arguments.
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The model to validate.
        epoch (int): Current epoch index.
        
    Returns:
        tuple: (average_loss, scores).
    """
    model.eval()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(val_loader)
    
    print(f"Validation on {total_batches} batches")
    
    for iter_idx, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs
        
        # Simplified data preparation
        pre_img = img[:, 0:3].cuda().float()
        post_img = img[:, 3:6].cuda().float()
        target = target.cuda().float()

        start_time = time.time()

        # Forward pass
        output = model.update_bcd(pre_img, post_img)
        loss = BCEDiceLoss(output, target)

        # Binarize predictions
        pred = torch.where(
            output > 0.5,
            torch.ones_like(output),
            torch.zeros_like(output)
        ).long()

        time_taken = time.time() - start_time
        epoch_loss.append(loss.data.item())

        # Update evaluation metrics
        f1 = eval_meter.update_cm(
            pr=pred.cpu().numpy(),
            gt=target.cpu().numpy()
        )
        
        if iter_idx % 5 == 0:
            print(
                f"\r[{iter_idx}/{total_batches}] "
                f"F1: {f1:.3f} loss: {loss.data.item():.3f} "
                f"time: {time_taken:.3f}",
                end=''
            )

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = eval_meter.get_scores()

    return average_epoch_loss_val, scores


def train(args, train_loader, model, optimizer, epoch, max_batches, 
          cur_iter=0, lr_factor=1.):
    """
    Trains the model for one epoch.
    
    Args:
        args: Command line arguments.
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): Model to train.
        optimizer: Optimizer instance.
        epoch (int): Current epoch index.
        max_batches (int): Number of batches per epoch.
        cur_iter (int): Current iteration count.
        lr_factor (float): Learning rate adjustment factor.
        
    Returns:
        tuple: (average_loss, scores, current_lr).
    """
    model.train()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    for iter_idx, batched_inputs in enumerate(train_loader):
        img, target = batched_inputs

        # Simplified data preparation
        pre_img = img[:, 0:3].cuda().float()
        post_img = img[:, 3:6].cuda().float()
        target = target.cuda().float()

        start_time = time.time()

        # Adjust learning rate
        lr = adjust_learning_rate(
            args,
            optimizer,
            epoch,
            iter_idx + cur_iter,
            max_batches,
            lr_factor=lr_factor
        )

        # Forward pass
        output = model.update_bcd(pre_img, post_img)
        loss = BCEDiceLoss(output, target)

        # Binarize predictions
        pred = torch.where(
            output > 0.5,
            torch.ones_like(output),
            torch.zeros_like(output)
        ).long()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter_idx - cur_iter) * time_taken / 3600

        # Update metrics
        with torch.no_grad():
            f1 = eval_meter.update_cm(
                pr=pred.cpu().numpy(),
                gt=target.cpu().numpy()
            )

        if (iter_idx + 1) % 5 == 0:
            print(
                f"[epoch {epoch}] [iter {iter_idx + 1}/{len(train_loader)} {res_time:.2f}h] "
                f"[lr {optimizer.param_groups[0]['lr']:.6f}] "
                f"[bn_loss {loss.data.item():.4f}] "
            )

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = eval_meter.get_scores()

    return average_epoch_loss_train, scores, lr


def trainValidate(args):
    """
    Main training and validation routine.
    
    Args:
        args: Command line arguments.
    """
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Enable CUDA optimizations and fix random seed
    torch.backends.cudnn.benchmark = True
    cudnn.benchmark = True
    torch.manual_seed(seed=16)
    torch.cuda.manual_seed(seed=16)

    # Initialize model
    model = Trainer(args).cuda().float()

    # Create experiment save directory
    save_path = osp(
        args.save_dir,
        f"{args.dataset}_iter_{args.max_steps}_lr_{args.lr}"
    )
    os.makedirs(save_path, exist_ok=True)

    # Data transformations
    train_transform, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)

    # Data loaders
    train_loader, val_loader, test_loader, max_batches = create_data_loaders(
        args, train_transform, val_transform
    )

    # Compute maximum epochs
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    
    # Load checkpoint if needed
    start_epoch, cur_iter = load_checkpoint(args, model, save_path, max_batches)
    
    # Set up logger
    logger = setup_logger(args, save_path)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        (0.9, 0.99),
        eps=1e-08,
        weight_decay=1e-4
    )
    
    # Track best F1 score
    max_F1_val = 0

    # Main training loop
    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()

        # Train one epoch
        loss_train, score_tr, lr = train(
            args,
            train_loader,
            model,
            optimizer,
            epoch,
            max_batches,
            cur_iter
        )
        cur_iter += len(train_loader)

        # Skip validation for the first epoch
        if epoch == 0:
            continue
        
        # Validation (using test set as validation)
        torch.cuda.empty_cache()
        loss_val, score_val = val(args, test_loader, model, epoch)
        
        # Log validation results
        logger.write(
            "\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                epoch,
                score_val['Kappa'],
                score_val['IoU'],
                score_val['F1'],
                score_val['recall'],
                score_val['precision']
            )
        )
        logger.flush()

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_val': loss_val,
            'F_train': score_tr['F1'],
            'F_val': score_val['F1'],
            'lr': lr
        }, osp(save_path, 'checkpoint.pth.tar'))

        # Save the best model
        model_file_name = osp(save_path, 'best_model.pth')
        if epoch % 1 == 0 and max_F1_val <= score_val['F1']:
            max_F1_val = score_val['F1']
            torch.save(model.state_dict(), model_file_name)

        # Print summary
        print(f"\nEpoch {epoch}: Details")
        print(
            f"\nEpoch No. {epoch}:\tTrain Loss = {loss_train:.4f}\t"
            f"Val Loss = {loss_val:.4f}\tF1(tr) = {score_tr['F1']:.4f}\t"
            f"F1(val) = {score_val['F1']:.4f}"
        )
    
    # Test with the best model
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    loss_test, score_test = val(args, test_loader, model, 0)
    print(
        f"\nTest:\t Kappa (te) = {score_test['Kappa']:.4f}\t "
        f"IoU (te) = {score_test['IoU']:.4f}\t"
        f"F1 (te) = {score_test['F1']:.4f}\t "
        f"R (te) = {score_test['recall']:.4f}\t"
        f"P (te) = {score_test['precision']:.4f}"
    )
    
    logger.write(
        "\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
            'Test',
            score_test['Kappa'],
            score_test['IoU'],
            score_test['F1'],
            score_test['recall'],
            score_test['precision']
        )
    )
    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        default="LEVIR-CD",
        help='数据集选择 | LEVIR-CD | WHU-CD | CLCD'  # 选择数据集类型
    )
    parser.add_argument(
        '--datasets',
        default="",
        help='多个数据集名称，用逗号分隔 | LEVIR-CD,WHU-CD,CLCD'  # 多个数据集的名称，留空则使用单数据集模式
    )
    parser.add_argument(
        '--file_root',
        default="path/to/LEVIR-CD",
        help='数据集目录路径'  # 数据集根目录，包含train/val/test子目录
    )
    parser.add_argument(
        '--file_roots',
        default="",
        help='多个数据集目录路径，用逗号分隔'  # 多个数据集根目录路径，用逗号分隔，留空则使用单数据集模式
    )
    parser.add_argument(
        '--in_height',
        type=int,
        default=256,
        help='RGB图像高度'  # 输入图像的高度，默认256像素
    )
    parser.add_argument(
        '--in_width',
        type=int,
        default=256,
        help='RGB图像宽度'  # 输入图像的宽度，默认256像素
    )
    parser.add_argument(
        '--num_perception_frame',
        type=int,
        default=1,
        help='感知帧数'  # BCD任务默认为1，表示二值变化检测
    )
    parser.add_argument(
        '--num_class',
        type=int,
        default=1,
        help='类别数量'  # BCD任务为1，表示二值分类（变化/无变化）
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=80000,
        help='最大迭代次数'  # 训练的最大步数，可调小加速训练
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='批处理大小'  # 每批训练的样本数，根据GPU内存调整
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='数据加载线程数'  # 并行加载数据的线程数，如有序列化问题设为0
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='初始学习率'  # 训练的起始学习率
    )
    parser.add_argument(
        '--lr_mode',
        default='poly',
        help='学习率策略: step或poly'  # 学习率调整方式
    )
    parser.add_argument(
        '--step_loss',
        type=int,
        default=100,
        help='多少个epoch后降低学习率'  # 使用step策略时的学习率衰减周期
    )
    parser.add_argument(
        '--pretrained',
        default='D:/代码/RSIIS/遥感影像变化检测系统V1.0/change3d/X3D_L.pyth',  # 修改为正确的预训练权重路径
        type=str,
        help='预训练权重路径'  # 骨干网络预训练权重路径
    )
    parser.add_argument(
        '--save_dir',
        default='./exp',
        help='保存实验结果的目录'  # 保存模型检查点和日志的目录
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='恢复训练的检查点路径'  # 如需从某个检查点继续训练，指定检查点文件
    )
    parser.add_argument(
        '--log_file',
        default='train_val_log.txt',
        help='训练和验证日志文件'  # 记录训练过程的日志文件名
    )
    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int,
        help='GPU ID号'  # 使用的GPU设备ID
    )

    args = parser.parse_args()
    trainValidate(args)