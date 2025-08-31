# Copyright (c) Duowang Zhu.
# All rights reserved.

# 导入必要的库
from functools import partial # 用于创建偏函数
import math # 数学运算
import logging # 日志记录
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable, Any # 类型注解

import torch
import torch.nn as nn
import torch.utils.checkpoint # 用于梯度检查点，节省显存
from einops import rearrange, repeat # 强大的张量操作库
import os

# 从当前目录导入其他模块
from .x3d import create_x3d # 导入 X3D 模型创建函数
from .change_decoder import ChangeDecoder # 导入变化检测解码器
from .utils import weight_init # 导入权重初始化函数


class Encoder(nn.Module):
    """
    编码器模块，基于 X3D 架构，并具备特征增强能力。
    
    该编码器使用 X3D 处理视频帧（或构建的伪视频帧），
    并使用可学习的令牌和时间差信息来增强中间特征。
    """
    
    def __init__(self, args: Any, embed_dims: List[int]) -> None:
        """
        初始化编码器。
        
        Args:
            args: 包含配置参数的对象 (例如命令行参数)。
            embed_dims: 一个列表，包含 X3D 各个阶段输出特征的维度。
        """
        super().__init__()
        self.args = args # 保存配置参数

        # 初始化 X3D 骨干网络
        # create_x3d 根据参数构建 X3D 模型实例
        # input_clip_length=3 表示期望的输入时间长度 (对应 pre, perception, post)
        # depth_factor=5.0 控制模型的深度 (对应 X3D-L)
        self.x3d = create_x3d(input_clip_length=3, depth_factor=5.0)

        # 加载预训练权重
        try:
            # 检查预训练权重路径是否存在且有效
            if args.pretrained and os.path.exists(args.pretrained):
                # 加载权重文件中的 'model_state' 字典
                state_dict = torch.load(args.pretrained, map_location='cpu')['model_state']
                # 加载权重到 self.x3d。strict=True 表示模型结构必须完全匹配，False 允许部分匹配
                msg = self.x3d.load_state_dict(state_dict, strict=True)
                print(f'成功加载预训练权重: {args.pretrained}, 加载信息: {msg}.')
            else:
                # 如果未提供或找不到预训练权重，则使用随机初始化
                pass
        except Exception as e:
            # 处理加载过程中可能出现的异常
            print(f"加载预训练权重失败: {e}")

        # 可学习的感知帧 (Learnable perception frames)，用于引导模型关注变化
        # 这是一个 nn.Parameter，意味着它的值会在训练过程中被优化
        # 形状: [1, 3 (RGB), num_perception_frame, H, W]
        self.perception_frames = nn.Parameter(
            torch.randn(1, 3, args.num_perception_frame, args.in_height, args.in_width), 
            requires_grad=True # 确保这个参数参与梯度计算和更新
        )

        # 用于特征增强的卷积层列表 (每个 stage 一个)
        # 通常用于 enhance 方法中处理时间差特征
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ) for dim in embed_dims # 根据每个 stage 的输出维度创建对应的卷积层
        ])

    def enhance(self, x: torch.Tensor, fc: nn.Module) -> torch.Tensor:
        """
        使用来自 pre/post 帧的时间信息来增强中间的变化帧特征。
        
        Args:
            x: 输入张量，形状为 [B, C, T, H, W]，T 是时间维度。
            fc: 用于处理时间差特征的特征增强模块 (通常是一个卷积层)。

        Returns:
            torch.Tensor: 增强后的张量，形状与输入相同，只有中间的帧被增强。

        Note:
            增强操作只应用于中间的 perception frame(s)，其他帧（原始 pre/post）保持不变。
            这里假设 T = 1 (pre) + num_perception_frame + 1 (post)。
        """
        # 获取中间 perception frame 的起始索引 (假设从 1 开始)
        middle_idx = 1 # (旧实现是 x.shape[2] // 2，这里根据 base_forward 逻辑调整为固定或基于 num_perception_frame)
        # 注意：原始代码中 enhance 和 base_forward 关于 middle_idx 的处理可能需要根据具体 num_perception_frame 调整统一
        
        # 获取 pre 帧 (时间索引 0)
        pre_frame = x[:, :, 0]
        # 获取 post 帧 (时间索引 num_perception_frame + 1)
        post_frame = x[:, :, self.args.num_perception_frame+1]
        
        # 计算时间差特征 (pre 和 post 帧的绝对差)
        temporal_diff = torch.abs(pre_frame - post_frame)
        
        # 使用传入的 fc 模块处理时间差，提取用于增强的特征
        enhancement_features = fc(temporal_diff)
        
        # 创建输出张量的副本
        enhanced_x = x.clone()
        # 将增强特征加到对应的 perception frame(s) 上 (残差连接)
        # 这里假设只增强第一个 perception frame (索引为 1)
        # 如果 num_perception_frame > 1，可能需要调整逻辑
        if self.args.num_perception_frame == 1:
             middle_frame = x[:, :, middle_idx]
             enhanced_middle_frame = middle_frame + enhancement_features
             enhanced_x[:, :, middle_idx] = enhanced_middle_frame
        else:
            # 如果有多个 perception frame，可能需要更复杂的增强逻辑
            # 例如，将增强特征广播或分别应用到每个 perception frame
            for idx_p in range(self.args.num_perception_frame):
                current_middle_idx = middle_idx + idx_p
                middle_frame = x[:, :, current_middle_idx]
                enhanced_middle_frame = middle_frame + enhancement_features # 简单地将同一个增强特征加到所有 perception frame
                enhanced_x[:, :, current_middle_idx] = enhanced_middle_frame

        # 返回增强后的完整序列张量
        return enhanced_x

    def base_forward(self, x: torch.Tensor, output_final: bool=False) -> List[torch.Tensor]:
        """
        执行通过 X3D 模块和特征增强的前向传播。
        
        Args:
            x: 输入张量，形状为 [B, C, T, H, W]。
            output_final (bool): 如果为 True，则只返回最后一个 Stage 的输出（通常用于分类任务）。
                                如果为 False (默认)，则返回每个 Stage 的中间特征列表。
            
        Returns:
            List[List[torch.Tensor]]: 如果 output_final=False, 返回一个列表，包含 4 个子列表，
                                       每个子列表代表一个 Stage 输出的 perception frame 特征。
            torch.Tensor: 如果 output_final=True, 返回最后一个 block 输出的特定 perception frame 特征。
        """
        # 如果只需要最终输出 (例如用于 Change Captioning)
        if output_final:
            # 通过 X3D 的所有 5 个 block (4个 stage + head block?)
            for i in range(5):
                x = self.x3d.blocks[i](x)
            # 返回最后一个 block 输出中，对应于 perception frame 的那个时间步的特征
            # 注意：这里的索引 self.args.num_perception_frame 似乎不正确，应该是指 perception frame 在 T 维度中的索引，如 1
            # 需要根据实际情况确认正确的索引
            return x[:, :, 1] # 假设我们总是需要第一个 perception frame
        else:
            # 如果需要所有中间阶段的特征 (用于变化检测解码器)
            out = [] # 初始化用于存储各阶段特征的列表
            # 循环处理 X3D 的前 4 个主要 block/stage (对应 c1, c2, c3, c4)
            for i in range(4):
                # 通过 X3D 的第 i 个 block 处理特征
                x = self.x3d.blocks[i](x)

                # 使用对应的全连接层(fc)和 enhance 方法增强特征
                # enhance 方法通常利用 pre/post 帧信息增强中间的 perception frame 特征
                x = self.enhance(x, self.fc[i])

                # 从增强后的特征 x (形状 B, C, T, H, W) 中提取与 perception frame 对应的特征图
                # T 通常等于 1 (pre) + num_perception_frame + 1 (post)
                layer_feature = [] # 存储当前 stage 的所有 perception frame 特征
                # 假设 perception frame 在时间维度上的索引是从 1 开始的
                for idx in range(self.args.num_perception_frame):
                    # 提取第 idx 个 perception frame 对应的特征图 (索引为 idx+1)
                    # 例如，如果 num_perception_frame=1, 提取 x[:, :, 1]
                    # 如果 num_perception_frame=3, 提取 x[:, :, 1], x[:, :, 2], x[:, :, 3]
                    layer_feature.append(x[:, :, idx+1])
                # 将当前 stage 提取出的 perception frame 特征图列表添加到最终输出列表中
                # out 的最终结构是 [[c1_p1, c1_p2,...], [c2_p1, c2_p2,...], [c3_p1,...], [c4_p1,...]]
                # 其中 cN_pM 表示第 N 个 stage 的第 M 个 perception frame 特征
                out.append(layer_feature)

            # 返回包含 4 个 stage 特征的列表
            return out

    def forward(self, x: torch.Tensor, y: torch.Tensor, output_final: bool=False) -> List[torch.Tensor]:
        """
        编码器的主要前向传播函数，处理输入的 pre/post 图像对。
        
        Args:
            x: 变化前图像张量，形状 [B, C, H, W]。
            y: 变化后图像张量，形状 [B, C, H, W]。
            output_final (bool): 是否只输出最终特征 (传递给 base_forward)。
            
        Returns:
            List[List[torch.Tensor]] or torch.Tensor: 根据 output_final 返回中间特征列表或最终特征。
        """        
        # 将可学习的 perception_frames 扩展到当前批次大小 B
        expand_percep_frames = repeat(self.perception_frames, '1 c t h w -> b c t h w', b=x.shape[0])

        # --- 构建伪视频序列 --- 
        # 将 pre 图像、perception_frames、post 图像在时间维度 (dim=2) 上拼接
        frames = torch.cat([
            x.unsqueeze(2), # 为 pre 图像增加时间维度 (T=1)
            expand_percep_frames, # 中间是可学习的 perception frames (T=num_perception_frame)
            y.unsqueeze(2)  # 为 post 图像增加时间维度 (T=1)
        ], dim=2) # 沿着时间维度拼接
        # --- 伪视频序列构建完成 --- 
        # frames 的形状为 [B, C, T, H, W], 其中 T = 1 + num_perception_frame + 1
        
        # 将构建好的伪视频序列传递给 base_forward 进行处理
        features = self.base_forward(frames, output_final)

        # 返回提取的特征
        return features


class Trainer(nn.Module):
    """
    完整的训练模型，包含编码器和针对不同任务的解码器。
    """
    
    def __init__(self, args: Any) -> None:
        """
        初始化 Trainer，包含编码器和解码器。
        
        Args:
            args: 包含配置参数的对象。
        """
        super().__init__()
        self.args = args # 保存配置参数
    
        # 定义 X3D 各个阶段的输出维度 (与 Encoder 中一致)
        self.embed_dims = [24, 24, 48, 96]
        
        # 初始化编码器 Encoder
        self.encoder = Encoder(args, self.embed_dims)
        
        # --- 根据任务类型选择和初始化解码器 --- 
        
        self.decoder = ChangeDecoder(args, in_dim=self.embed_dims, has_sigmoid=True)
        # 初始化解码器的权重
        weight_init(self.decoder)


        # --- 解码器初始化结束 --- 

    def update_bcd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        用于二元变化检测任务 (BCD) 的前向传播。

        Args:
            x: 变化前图像张量 [B, C, H, W]。
            y: 变化后图像张量 [B, C, H, W]。

        Returns:
            torch.Tensor: 预测的二元变化图 [B, 1, H, W]。
        """
        # 1. 使用编码器提取特征
        # features 的结构是 [[c1_p1], [c2_p1], [c3_p1], [c4_p1]] (因为 num_perception_frame=1)
        features = self.encoder(x, y)

        # 2. 提取每个 stage 的第一个 (也是唯一的) perception frame 特征
        # perception_change_feat 的结构是 [c1_p1, c2_p1, c3_p1, c4_p1]
        perception_change_feat = list(map(lambda feat_list: feat_list[0], features))

        # 3. 使用 BCD 解码器生成预测
        prediction = self.decoder(perception_change_feat)

        # 返回最终预测结果
        return prediction