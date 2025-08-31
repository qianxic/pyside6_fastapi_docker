# Copyright (c) Duowang Zhu.
# All rights reserved.

import os
import sys
from os.path import join as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy import stats
from typing import Union, Type, List




def weight_init(module):
    """
    Initialize weights for neural network modules using best practices.
    
    This function recursively initializes weights in a module:
    - Conv2D layers: Kaiming normal initialization (fan_in, relu)
    - BatchNorm and GroupNorm: Weights set to 1, biases to 0
    - Linear layers: Kaiming normal initialization (fan_in, relu)
    - Sequential containers: Each component initialized individually
    - Pooling, ModuleList, Loss functions: Skipped (no initialization needed)
    
    Args:
        module: PyTorch module whose weights will be initialized
    """
    # Process all named children in the module
    for name, child_module in module.named_children():
        # Skip modules that don't need initialization
        if isinstance(child_module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, 
                                     nn.ModuleList, nn.BCELoss)):
            continue
            
        # Initialize convolutional layers
        elif isinstance(child_module, nn.Conv2d):
            nn.init.kaiming_normal_(child_module.weight, mode='fan_in', nonlinearity='relu')
            if child_module.bias is not None:
                nn.init.zeros_(child_module.bias)
                
        # Initialize normalization layers
        elif isinstance(child_module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(child_module.weight)
            if child_module.bias is not None:
                nn.init.zeros_(child_module.bias)
                
        # Initialize linear layers
        elif isinstance(child_module, nn.Linear):
            nn.init.kaiming_normal_(child_module.weight, mode='fan_in', nonlinearity='relu')
            if child_module.bias is not None:
                nn.init.zeros_(child_module.bias)

        # Handle Sequential containers
        elif isinstance(child_module, nn.Sequential):
            for seq_name, seq_module in child_module.named_children():
                if isinstance(seq_module, nn.Conv2d):
                    nn.init.kaiming_normal_(seq_module.weight, mode='fan_in', nonlinearity='relu')
                    if seq_module.bias is not None:
                        nn.init.zeros_(seq_module.bias)

                elif isinstance(seq_module, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(seq_module.weight)
                    if seq_module.bias is not None:
                        nn.init.zeros_(seq_module.bias)

                elif isinstance(seq_module, nn.Linear):
                    nn.init.kaiming_normal_(seq_module.weight, mode='fan_in', nonlinearity='relu')
                    if seq_module.bias is not None:
                        nn.init.zeros_(seq_module.bias)
                else:
                    # Recursively initialize other modules in sequential container
                    weight_init(seq_module)

        # Recursively handle other module types
        elif len(list(child_module.children())) > 0:
            weight_init(child_module)

def adjust_learning_rate(args, optimizer, epoch=None, iter=None, max_batches=None, 
                         lr_factor=1.0, shrink_factor=None, verbose=True):
    """
    Adjust learning rate based on scheduler type, epoch, iteration, or explicit shrinking.
    
    This function supports multiple learning rate adjustment strategies:
    1. Step decay: Reduces LR at fixed intervals
    2. Polynomial decay: Smoothly reduces LR according to a polynomial function
    3. Manual shrinking: Explicitly shrinks LR by a specified factor
    4. Warm-up phase: Gradually increases LR at the beginning of training
    
    Args:
        args: Command line arguments containing lr_mode, lr, step_loss, max_epochs
        optimizer: Optimizer instance whose learning rate will be adjusted
        epoch: Current epoch (required for step and poly modes)
        iter: Current iteration (required for poly mode and warm-up)
        max_batches: Total batches per epoch (required for poly mode)
        lr_factor: Additional scaling factor for the learning rate (default: 1.0)
        shrink_factor: If provided, explicitly shrink LR by this factor (0-1)
        verbose: Whether to print the learning rate change (default: True)
        
    Returns:
        float: Current learning rate after adjustment
    """
    if shrink_factor is not None:
        # Manual shrinking mode (from the second implementation)
        if not 0 < shrink_factor < 1:
            raise ValueError(f"Shrink factor must be between 0 and 1, got {shrink_factor}")
            
        if verbose:
            print("\nDECAYING learning rate.")
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * shrink_factor
            
        if verbose:
            print(f"The new learning rate is {optimizer.param_groups[0]['lr']:.6f}\n")
            
        return optimizer.param_groups[0]['lr']
    
    # Scheduler-based learning rate adjustment
    if args.lr_mode == 'step':
        if epoch is None:
            raise ValueError("Epoch must be provided for step lr_mode")
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
        
    elif args.lr_mode == 'poly':
        if any(param is None for param in [epoch, iter, max_batches]):
            raise ValueError("Epoch, iter, and max_batches must be provided for poly lr_mode")
            
        cur_iter = iter
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
        
    else:
        raise ValueError(f'Unknown lr mode {args.lr_mode}')
    
    # Apply warm-up phase if we're in the first epoch
    if epoch == 0 and iter is not None and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr
    
    # Apply additional lr factor
    lr *= lr_factor
    
    # Update learning rate for all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def BCEDiceLoss(inputs, targets):
    """
    Combined BCE and Dice loss for binary segmentation.
    
    Args:
        inputs: Model predictions after sigmoid
        targets: Ground truth binary masks
        
    Returns:
        Combined loss value
    """
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice

def EdgeAwareBCEDiceLoss(inputs, targets, edge_weight=2.0):
    """
    增强版边缘感知的BCE-Dice损失函数，使用改进的边缘检测和注意力机制
    
    Args:
        inputs: 模型预测 (已经过sigmoid)
        targets: 真实标签
        edge_weight: 边缘权重系数
        
    Returns:
        final_loss: 总损失
        loss_details: 包含各部分损失的字典
    """
    # 确保输入维度正确
    if inputs.dim() == 5:  # [B, 1, 1, H, W]
        inputs = inputs.squeeze(2)
    if targets.dim() == 5:  # [B, 1, 1, H, W]
        targets = targets.squeeze(2)
    
    # 确保输入是4D张量 [B, C, H, W]
    if inputs.dim() == 3:  # [B, H, W]
        inputs = inputs.unsqueeze(1)
    if targets.dim() == 3:  # [B, H, W]
        targets = targets.unsqueeze(1)
    
    # 计算边缘
    targets_batch = targets  # 已经是 [B, 1, H, W]
    
    # 定义改进的Sobel算子（8方向）
    sobel_kernels = [
        # 水平方向
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
        # 垂直方向
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
        # 对角线方向
        torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32).reshape(1, 1, 3, 3),
        # 反对角线方向
        torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32).reshape(1, 1, 3, 3)
    ]
    
    # 将所有kernel移到正确的设备
    sobel_kernels = [k.to(targets.device) for k in sobel_kernels]
    
    # 计算多方向边缘
    edge_maps = []
    for kernel in sobel_kernels:
        edge = F.conv2d(targets_batch, kernel, padding=1)
        edge_maps.append(edge)
    
    # 合并边缘图
    edge_maps = torch.stack(edge_maps, dim=1)  # [B, 4, 1, H, W]
    edge_magnitude = torch.sqrt(torch.sum(edge_maps**2, dim=1)).squeeze(1)  # [B, H, W]
    
    # 自适应阈值二值化（使用OTSU方法的思想）
    edge_mean = edge_magnitude.mean()
    edge_std = edge_magnitude.std()
    threshold = edge_mean + 0.8 * edge_std  # 提高阈值以获取更清晰的边缘
    edge_targets = (edge_magnitude > threshold).float()
    
    # 使用改进的高斯核进行边缘平滑
    gaussian_kernel = torch.tensor([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=torch.float32).reshape(1, 1, 5, 5).to(targets.device) / 256.0
    
    # 多次平滑以获得更连续的边缘
    edge_targets = F.conv2d(edge_targets.unsqueeze(1), gaussian_kernel, padding=2)
    edge_targets = F.conv2d(edge_targets, gaussian_kernel, padding=2)
    edge_targets = edge_targets.squeeze(1)
    
    # 多尺度边缘注意力
    attention_scales = [1.0, 0.75, 0.5]  # 多尺度权重
    edge_attention = torch.zeros_like(edge_targets)
    
    for scale in attention_scales:
        scaled_kernel = gaussian_kernel * scale
        attention = F.conv2d(edge_targets.unsqueeze(1), scaled_kernel, padding=2)
        edge_attention += attention.squeeze(1)
    
    edge_attention = edge_attention / len(attention_scales)
    edge_attention = torch.sigmoid(edge_attention * edge_weight)
    
    # 在边缘区域的损失加权
    edge_area = edge_targets > 0.15  # 略微提高阈值以获取更精确的边缘
    weight_map = torch.ones_like(targets.squeeze(1))
    weight_map[edge_area] = edge_weight
    
    # 计算加权BCE损失
    weighted_bce = F.binary_cross_entropy(inputs, targets, weight=weight_map.unsqueeze(1))
    
    # 计算边缘区域的BCE损失
    edge_bce = F.binary_cross_entropy(inputs, targets, weight=edge_attention.unsqueeze(1))
    
    # 计算Dice损失
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    dice_loss = 1 - dice
    
    # 计算边缘连续性损失
    edge_continuity = F.conv2d(edge_targets.unsqueeze(1), 
                             torch.ones(1, 1, 3, 3).to(targets.device) / 9.0,
                             padding=1)
    edge_continuity = torch.abs(edge_continuity.squeeze(1) - edge_targets)
    continuity_loss = edge_continuity.mean()
    
    # 最终损失
    final_loss = weighted_bce + dice_loss + 0.1 * continuity_loss
    
    # 返回损失详情
    loss_details = {
        'total_loss': final_loss.item(),
        'weighted_bce': weighted_bce.item(),
        'dice_loss': dice_loss.item(),
        'edge_bce': edge_bce.item(),
        'continuity_loss': continuity_loss.item(),
        'edge_ratio': edge_area.float().mean().item(),
        'edge_attention_mean': edge_attention.mean().item()
    }
    
    return final_loss, loss_details

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
    
class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        
    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x1 = torch.reshape(x1,[b*h*w,c])
        x2 = torch.reshape(x2,[b*h*w,c])
        
        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target,[b*h*w])
        
        loss = self.loss_f(x1, x2, target)
        return loss

def load_checkpoint(args, model, save_path, max_batches):
    """
    Load checkpoint if resume is specified.
    
    Args:
        args: Command line arguments
        model: Model instance
        save_path: Path to save directory
        
    Returns:
        start_epoch, cur_iter
    """
    start_epoch = 0
    cur_iter = 0
    
    if args.resume is not None:
        checkpoint_path = osp(save_path, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            print(f"=> loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
            start_epoch = checkpoint['epoch']
            cur_iter = start_epoch * max_batches
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{checkpoint_path}'")
    
    return start_epoch, cur_iter


def setup_logger(args, save_path):
    """设置训练日志记录器"""
    log_file = osp(save_path, args.log_file)
    
    # 使用UTF-8编码打开文件
    logger = open(log_file, 'w', encoding='utf-8')
    
    # 写入配置信息
    logger.write("Model Configurations:\n")
    for arg in vars(args):
        logger.write(f"{arg}: {getattr(args, arg)}\n")
    
    logger.write("\n------------------------------------------------------------\n")
    logger.write("Epoch\tKappa (val)\tIoU (val)\tF1 (val)\tR (val)\tP (val)")
    logger.flush()
    
    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def accuracy(pred, label, ignore_zero=False):
    valid = (label >= 0)
    if ignore_zero: valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum
    
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist

def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def SCDD_eval_all(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)
    
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    
    pixel_sum = hist.sum()
    change_pred_sum  = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum/pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP/change_pred_sum
    SC_Recall = SC_TP/change_label_sum
    Fscd = stats.hmean([SC_Precision, SC_Recall])
    return Fscd, IoU_mean, Sek

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-7)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return Pre

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return Rec

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return F1


    def calculate_per_class_metrics(self):
        # Adjustments to exclude class 0 in calculations
        TPs = np.diag(self.confusion_matrix)[1:]  # Start from index 1 to exclude class 0
        FNs = np.sum(self.confusion_matrix, axis=1)[1:] - TPs
        FPs = np.sum(self.confusion_matrix, axis=0)[1:] - TPs
        return TPs, FNs, FPs
    
    def Damage_F1_socore(self):
        TPs, FNs, FPs = self.calculate_per_class_metrics()
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        return f1_scores
    
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix) + 1e-7)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = self.confusion_matrix[1, 1] / (
                self.confusion_matrix[0, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return IoU

    def Kappa_coefficient(self):
        # Number of observations (total number of classifications)
        # num_total = np.array(0, dtype=np.long)
        # row_sums = np.array([0, 0], dtype=np.long)
        # col_sums = np.array([0, 0], dtype=np.long)
        # total += np.sum(self.confusion_matrix)
        # # Observed agreement (i.e., sum of diagonal elements)
        # observed_agreement = np.sum(np.diag(self.confusion_matrix))
        # # Compute expected agreement
        # row_sums += np.sum(self.confusion_matrix, axis=0)
        # col_sums += np.sum(self.confusion_matrix, axis=1)
        # expected_agreement = np.sum((row_sums * col_sums) / total)
        num_total = np.sum(self.confusion_matrix)
        observed_accuracy = np.trace(self.confusion_matrix) / num_total
        expected_accuracy = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total * np.sum(self.confusion_matrix, axis=1) / num_total)

        # Calculate Cohen's kappa
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def caption_accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def eval_caption_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict

class EdgeLoss(nn.Module):
    def __init__(self, loss_type='l1', edge_influence=0.5):
        super(EdgeLoss, self).__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")
            
        # Sobel kernel (3x3)
        self.sobel_x = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        
        self.sobel_y = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        
        # 边缘影响系数 - 控制边缘损失对主预测的引导强度
        self.edge_influence = edge_influence

    def sobel_edge(self, img):
        # 确保输入是4D张量 [B, C, H, W]
        if img.dim() == 3:
            img = img.unsqueeze(1)
            
        # 将kernel移到正确的设备
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)
        
        # 计算梯度
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        
        # 计算梯度幅值
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return grad

    def forward(self, pred_mask, gt_mask):
        # 确保输入是4D张量 [B, C, H, W]
        if pred_mask.dim() == 3:
            pred_mask = pred_mask.unsqueeze(1)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
            
        # 计算真实标签的边缘
        gt_edge = self.sobel_edge(gt_mask)
        
        # 对预测进行sigmoid激活后再提取边缘
        # 这确保了我们从预测的主变化区域中提取边缘
        pred_sigmoid = torch.sigmoid(pred_mask)
        pred_edge = self.sobel_edge(pred_sigmoid)
        
        # 1. 计算边缘直接损失
        direct_edge_loss = self.loss(pred_edge, gt_edge)
        
        # 2. 计算边缘注意力图 - 标识边缘区域重要性
        # 根据真实边缘强度创建注意力权重
        edge_attention = gt_edge / (gt_edge.max() + 1e-6)
        
        # 3. 计算加权边缘损失 - 确保边缘区域的主预测更接近标签
        # 在边缘区域，对主预测增加额外监督
        edge_guided_pred_loss = F.binary_cross_entropy_with_logits(
            pred_mask,
            gt_mask,
            weight=1.0 + edge_attention * self.edge_influence,  # 边缘区域权重更高
            reduction='mean'
        )
        
        # 返回边缘直接损失和引导损失
        return direct_edge_loss, edge_guided_pred_loss

class TrueEdgeAwareLoss(nn.Module):
    def __init__(self, edge_weight=2.0, loss_type='l1', edge_influence=0.5):
        super().__init__()
        self.edge_weight = edge_weight
        self.edge_loss = EdgeLoss(loss_type=loss_type, edge_influence=edge_influence)
        
    def forward(self, pred, target):
        # 1. 计算边缘损失
        direct_edge_loss, edge_guided_pred_loss = self.edge_loss(pred, target)
        edge_loss = direct_edge_loss * self.edge_weight
        
        # 2. 计算未经边缘引导的原始BCE损失
        original_bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # 3. 计算Dice损失
        dice_loss = self.dice_loss(pred, target)
        
        # 总损失
        total_loss = edge_guided_pred_loss + dice_loss + edge_loss
        
        return total_loss, {
            'original_bce_loss': original_bce_loss.item(),  # 未经边缘引导的原始BCE损失
            'edge_guided_loss': edge_guided_pred_loss.item(),  # 边缘引导后的分割损失
            'dice_loss': dice_loss.item(),
            'direct_edge_loss': direct_edge_loss.item(),  # 直接计算的边缘损失
            'edge_loss': edge_loss.item(),  # 加权后的边缘损失
            'total_loss': total_loss.item()
        }
        
    def dice_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        return 1 - (2. * intersection + 1) / (union + 1)