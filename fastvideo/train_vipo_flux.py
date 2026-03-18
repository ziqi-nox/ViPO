# Copyright (c) 2025 FastVideo Team
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# Copyright (c) 2026 ViPO Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is derived from FastVideo / DanceGRPO and has been modified by the ViPO authors in 2026.
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/XueZeyue/DanceGRPO/blob/main/LICENSE].
#
# This modified file is released under the same license.

import argparse
import math
import os
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.utils.data.distributed import DistributedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List
from PIL import Image
from diffusers import FluxTransformer2DModel, AutoencoderKL

from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

def compute_dinov2_feature_map(
    video_frames, 
    target_size=(64, 64), 
    target_time=1,  # Flux通常是单帧
    device=None,
    pca_method="weighted",
    smooth_method="gaussian_strong",
    sigma=1.0
):
    """
    使用DINOv2提取图像的语义特征权重图，适配Flux的单帧格式
    
    Args:
        video_frames: torch.Tensor, shape (C, H, W) 或 (C, T, H, W), 范围 [0, 1] 或 [-1, 1]
        target_size: tuple, 目标空间尺寸 (height, width)
        target_time: int, 目标时间维度（Flux通常为1）
        device: torch.device, 计算设备
        pca_method: str, PCA降维方法 ("weighted", "average", "first_pc")
        smooth_method: str, 平滑方法，默认使用 "gaussian_strong"
        sigma: float, 高斯平滑的标准差
        
    Returns:
        feature_map: torch.Tensor, shape (T_target, H, W), 平滑后的语义特征权重图，范围[0, 1]
    """
    # 处理输入维度 - Flux可能是3D或4D
    if video_frames.dim() == 3:  # (C, H, W) -> (C, 1, H, W)
        video_frames = video_frames.unsqueeze(1)
    
    # 确保输入在 [0, 1] 范围内
    if video_frames.min() < 0:
        video_frames = (video_frames + 1.0) / 2.0
    video_frames = torch.clamp(video_frames, 0, 1).float()
    
    if device is None:
        device = video_frames.device
    
    # 初始化DINOv2模型（只在第一次调用时加载）
    if not hasattr(compute_dinov2_feature_map, 'dinov2_model'):
        print("Loading DINOv2 model for feature extraction...")
        from transformers import AutoImageProcessor, AutoModel, AutoConfig
        
        model_path = './dinov2-large'
        config = AutoConfig.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
        dinov2_model = AutoModel.from_pretrained(model_path).to(device)
        dinov2_model.eval()
        
        compute_dinov2_feature_map.dinov2_model = dinov2_model
        compute_dinov2_feature_map.dinov2_processor = processor
        print("✓ DINOv2 model loaded and cached")
    
    dinov2_model = compute_dinov2_feature_map.dinov2_model
    dinov2_processor = compute_dinov2_feature_map.dinov2_processor
    
    C, T, H, W = video_frames.shape
    frame_feature_maps = []
    
    for t in range(T):
        frame = video_frames[:, t, :, :]  # (C, H, W)
        
        try:
            with torch.no_grad():
                # 转换为PIL图像进行DINOv2处理
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
                frame_np = (frame_np * 255).astype(np.uint8)
                frame_pil = Image.fromarray(frame_np)
                
                original_size = frame_pil.size  # (width, height)
                
                # DINOv2处理
                try:
                    dinov2_inputs = dinov2_processor(
                        images=frame_pil, 
                        return_tensors="pt", 
                        size={"height": original_size[1], "width": original_size[0]},
                        do_center_crop=False,
                        do_resize=True
                    ).to(device)
                except Exception as e:
                    try:
                        min_edge = min(original_size)
                        target_short_edge = min(min_edge, 518) if min_edge >= 224 else 224
                        dinov2_inputs = dinov2_processor(
                            images=frame_pil, 
                            return_tensors="pt",
                            size={"shortest_edge": target_short_edge},
                            do_center_crop=False,
                            do_resize=True
                        ).to(device)
                    except Exception as e2:
                        dinov2_inputs = dinov2_processor(
                            images=frame_pil, 
                            return_tensors="pt"
                        ).to(device)
                
                # 获取实际处理后图像的尺寸
                processed_img_tensor = dinov2_inputs['pixel_values'][0]
                processed_height, processed_width = processed_img_tensor.shape[1], processed_img_tensor.shape[2]
                
                # 提取DINOv2特征
                dinov2_outputs = dinov2_model(**dinov2_inputs)
                features = dinov2_outputs.last_hidden_state[0, 1:, :].cpu().numpy()  # 排除CLS token
                
                # 计算patch网格尺寸
                num_patches = features.shape[0]
                patch_size = 14
                grid_h = processed_height // patch_size
                grid_w = processed_width // patch_size
                
                # 处理patch数量不匹配
                if grid_h * grid_w != num_patches:
                    possible_factors = []
                    for h in range(1, int(np.sqrt(num_patches)) + 10):
                        if num_patches % h == 0:
                            w = num_patches // h
                            possible_factors.append((h, w, abs(h - w)))
                    target_ratio = processed_height / processed_width
                    best_grid = min(possible_factors, key=lambda x: abs((x[0]/x[1]) - target_ratio))
                    grid_h, grid_w = best_grid[0], best_grid[1]
                
                # PCA降维并计算语义权重
                pca = PCA(n_components=3)
                pca_features = pca.fit_transform(features)
                explained_variance = pca.explained_variance_ratio_
                
                # 根据指定方法计算权重
                if pca_method == "weighted":
                    semantic_weights = np.sum(pca_features * explained_variance.reshape(1, -1), axis=1)
                elif pca_method == "average":
                    semantic_weights = np.mean(pca_features, axis=1)
                elif pca_method == "first_pc":
                    semantic_weights = pca_features[:, 0]
                else:
                    raise ValueError(f"Unknown pca_method: {pca_method}")
                
                # 归一化权重到 [0, 1]
                semantic_weights = (semantic_weights - semantic_weights.min()) / (semantic_weights.max() - semantic_weights.min() + 1e-8)
                
                # 重塑权重图到patch网格
                if len(semantic_weights) == grid_h * grid_w:
                    weight_map = semantic_weights.reshape(grid_h, grid_w)
                else:
                    from scipy.ndimage import zoom
                    temp_grid = int(np.sqrt(len(semantic_weights)))
                    temp_map = semantic_weights.reshape(temp_grid, temp_grid)
                    scale_h = grid_h / temp_grid
                    scale_w = grid_w / temp_grid
                    weight_map = zoom(temp_map, (scale_h, scale_w), order=1)
                
                # 应用高斯平滑
                smoothed_weight_map = gaussian_filter(weight_map, sigma=sigma)
                smoothed_weight_map = (smoothed_weight_map - smoothed_weight_map.min()) / (smoothed_weight_map.max() - smoothed_weight_map.min() + 1e-8)
                
                # 直接上采样到目标latent尺寸
                import torch.nn.functional as F
                weight_tensor = torch.from_numpy(smoothed_weight_map).float().to(device)
                weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(0)
                
                target_h, target_w = target_size
                upsampled_tensor = F.interpolate(
                    weight_tensor,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )
                
                final_feature_map = upsampled_tensor.squeeze(0).squeeze(0)
                final_feature_map = (final_feature_map - final_feature_map.min()) / (final_feature_map.max() - final_feature_map.min() + 1e-8)
                
                frame_feature_map = final_feature_map
                
        except Exception as e:
            print(f"Error extracting DINOv2 features for frame {t}: {e}")
            frame_feature_map = torch.ones(target_size, dtype=torch.float32, device=device)
        
        frame_feature_maps.append(frame_feature_map)
    
    # 堆叠所有帧的特征图
    feature_maps = torch.stack(frame_feature_maps, dim=0)  # (T, H, W)
    
    # 时间维度重采样（对于Flux单帧，通常不需要）
    if T != target_time:
        if target_time == 1:
            # 对于单帧，取第一帧或平均
            feature_maps = feature_maps[0:1]  # (1, H, W)
        else:
            feature_maps_resampled = F.interpolate(
                feature_maps.unsqueeze(0).unsqueeze(0),
                size=(target_time, feature_maps.shape[1], feature_maps.shape[2]),
                mode='trilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)
            feature_maps = feature_maps_resampled
    
    return feature_maps

def compute_dinov2_feature_map_with_visualization(
    video_frames, 
    target_size=(64, 64), 
    target_time=1,  # Flux通常是单帧
    device=None,
    pca_method="weighted",
    smooth_method="gaussian_strong",
    sigma=1.0,
    save_visualization=False,  # 【新增】是否保存可视化
    viz_save_path="./dinov2_viz",  # 【新增】可视化保存路径
    step=None  # 【新增】训练步数，用于文件命名
):
    if video_frames.dim() == 3:  # (C, H, W) -> (C, 1, H, W)
        video_frames = video_frames.unsqueeze(1)

    if video_frames.min() < 0:
        video_frames = (video_frames + 1.0) / 2.0
    video_frames = torch.clamp(video_frames, 0, 1).float()
    
    if device is None:
        device = video_frames.device
    
    # 初始化DINOv2模型（只在第一次调用时加载）
    if not hasattr(compute_dinov2_feature_map_with_visualization, 'dinov2_model'):
        print("Loading DINOv2 model for feature extraction...")
        from transformers import AutoImageProcessor, AutoModel, AutoConfig
        
        model_path = './dinov2-large'
        config = AutoConfig.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
        dinov2_model = AutoModel.from_pretrained(model_path).to(device)
        dinov2_model.eval()
        
        compute_dinov2_feature_map_with_visualization.dinov2_model = dinov2_model
        compute_dinov2_feature_map_with_visualization.dinov2_processor = processor
        print("✓ DINOv2 model loaded and cached")
    
    dinov2_model = compute_dinov2_feature_map_with_visualization.dinov2_model
    dinov2_processor = compute_dinov2_feature_map_with_visualization.dinov2_processor
    
    C, T, H, W = video_frames.shape
    frame_feature_maps = []
    
    viz_data = {
        'original_frames': [],
        'raw_feature_maps': [],
        'pca_components_raw': [],  # 【新增】存储原始PCA组件（保持正负值）
        'pca_components_remapped': [],  # 【新增】存储重映射后的PCA组件
        'smoothed_feature_maps': [],
        'final_feature_maps': [],
        'pca_explained_variance': [],
        'grid_sizes': []
    } if save_visualization else None

    for t in range(T):
        frame = video_frames[:, t, :, :]  # (C, H, W)
        
        try:
            with torch.no_grad():
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
                frame_np = (frame_np * 255).astype(np.uint8)
                frame_pil = Image.fromarray(frame_np)
                
                if save_visualization:
                    viz_data['original_frames'].append(frame_pil.copy())
                
                original_size = frame_pil.size  # (width, height)
                
                # DINOv2处理
                try:
                    dinov2_inputs = dinov2_processor(
                        images=frame_pil, 
                        return_tensors="pt", 
                        size={"height": original_size[1], "width": original_size[0]},
                        do_center_crop=False,
                        do_resize=True
                    ).to(device)
                except Exception as e:
                    try:
                        min_edge = min(original_size)
                        target_short_edge = min(min_edge, 518) if min_edge >= 224 else 224
                        dinov2_inputs = dinov2_processor(
                            images=frame_pil, 
                            return_tensors="pt",
                            size={"shortest_edge": target_short_edge},
                            do_center_crop=False,
                            do_resize=True
                        ).to(device)
                    except Exception as e2:
                        dinov2_inputs = dinov2_processor(
                            images=frame_pil, 
                            return_tensors="pt"
                        ).to(device)
                
                # 获取实际处理后图像的尺寸
                processed_img_tensor = dinov2_inputs['pixel_values'][0]
                processed_height, processed_width = processed_img_tensor.shape[1], processed_img_tensor.shape[2]
                
                # 提取DINOv2特征
                dinov2_outputs = dinov2_model(**dinov2_inputs)
                features = dinov2_outputs.last_hidden_state[0, 1:, :].cpu().numpy()  # 排除CLS token
                
                # 计算patch网格尺寸
                num_patches = features.shape[0]
                patch_size = 14
                grid_h = processed_height // patch_size
                grid_w = processed_width // patch_size
                
                # 处理patch数量不匹配
                if grid_h * grid_w != num_patches:
                    possible_factors = []
                    for h in range(1, int(np.sqrt(num_patches)) + 10):
                        if num_patches % h == 0:
                            w = num_patches // h
                            possible_factors.append((h, w, abs(h - w)))
                    target_ratio = processed_height / processed_width
                    best_grid = min(possible_factors, key=lambda x: abs((x[0]/x[1]) - target_ratio))
                    grid_h, grid_w = best_grid[0], best_grid[1]
                
                # 【可视化】保存网格尺寸
                if save_visualization:
                    viz_data['grid_sizes'].append((grid_h, grid_w))
                
                # PCA降维并计算语义权重
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                pca_features = pca.fit_transform(features)
                explained_variance = pca.explained_variance_ratio_
                
                # 【可视化】保存PCA解释方差和原始组件
                if save_visualization:
                    viz_data['pca_explained_variance'].append(explained_variance.copy())
                    
                    # 【修改】保存原始PCA组件（保持正负值）和重映射后的组件
                    pca_components_raw = []
                    pca_components_remapped = []
                    
                    for comp_idx in range(3):
                        # 获取当前主成分的权重（保持原始正负值）
                        component_weights_raw = pca_features[:, comp_idx]
                        
                        # 重塑到patch网格
                        if len(component_weights_raw) == grid_h * grid_w:
                            component_map_raw = component_weights_raw.reshape(grid_h, grid_w)
                        else:
                            from scipy.ndimage import zoom
                            temp_grid = int(np.sqrt(len(component_weights_raw)))
                            temp_map = component_weights_raw.reshape(temp_grid, temp_grid)
                            scale_h = grid_h / temp_grid
                            scale_w = grid_w / temp_grid
                            component_map_raw = zoom(temp_map, (scale_h, scale_w), order=1)
                        
                        # 【关键修改】重新映射：越接近-1的值越重要
                        # 使用反向映射：weight = (max_val - original_val) / (max_val - min_val)
                        min_val = component_map_raw.min()
                        max_val = component_map_raw.max()
                        
                        if max_val != min_val:
                            component_map_remapped = (max_val - component_map_raw) / (max_val - min_val)
                        else:
                            component_map_remapped = np.ones_like(component_map_raw) * 0.5
                        
                        pca_components_raw.append(component_map_raw.copy())
                        pca_components_remapped.append(component_map_remapped.copy())
                    
                    viz_data['pca_components_raw'].append(pca_components_raw)
                    viz_data['pca_components_remapped'].append(pca_components_remapped)
                
                if pca_method == "weighted":
                    semantic_weights = np.zeros(pca_features.shape[0])
                    for comp_idx in range(3):
                        comp_weights = pca_features[:, comp_idx]
                        min_val = comp_weights.min()
                        max_val = comp_weights.max()
                        if max_val != min_val:
                            comp_weights_remapped = (max_val - comp_weights) / (max_val - min_val)
                        else:
                            comp_weights_remapped = np.ones_like(comp_weights) * 0.5
                        
                        semantic_weights += explained_variance[comp_idx] * comp_weights_remapped
                        
                elif pca_method == "average":
                    avg_features = np.mean(pca_features, axis=1)
                    min_val = avg_features.min()
                    max_val = avg_features.max()
                    if max_val != min_val:
                        semantic_weights = (max_val - avg_features) / (max_val - min_val)
                    else:
                        semantic_weights = np.ones_like(avg_features) * 0.5
                        
                elif pca_method == "first_pc":
                    first_pc = pca_features[:, 0]
                    min_val = first_pc.min()
                    max_val = first_pc.max()
                    if max_val != min_val:
                        semantic_weights = (max_val - first_pc) / (max_val - min_val)
                    else:
                        semantic_weights = np.ones_like(first_pc) * 0.5
                else:
                    raise ValueError(f"Unknown pca_method: {pca_method}")
                
                # 确保权重在 [0, 1] 范围内
                semantic_weights = np.clip(semantic_weights, 0, 1)
                
                # 重塑权重图到patch网格
                if len(semantic_weights) == grid_h * grid_w:
                    weight_map = semantic_weights.reshape(grid_h, grid_w)
                else:
                    from scipy.ndimage import zoom
                    temp_grid = int(np.sqrt(len(semantic_weights)))
                    temp_map = semantic_weights.reshape(temp_grid, temp_grid)
                    scale_h = grid_h / temp_grid
                    scale_w = grid_w / temp_grid
                    weight_map = zoom(temp_map, (scale_h, scale_w), order=1)
                
                # 【可视化】保存原始特征图
                if save_visualization:
                    viz_data['raw_feature_maps'].append(weight_map.copy())
                
                # 应用高斯平滑
                from scipy.ndimage import gaussian_filter
                smoothed_weight_map = gaussian_filter(weight_map, sigma=sigma)
                # 重新归一化平滑后的权重图
                smoothed_weight_map = (smoothed_weight_map - smoothed_weight_map.min()) / (smoothed_weight_map.max() - smoothed_weight_map.min() + 1e-8)
                
                # 【可视化】保存平滑后特征图
                if save_visualization:
                    viz_data['smoothed_feature_maps'].append(smoothed_weight_map.copy())
                
                # 直接上采样到目标latent尺寸
                import torch.nn.functional as F
                weight_tensor = torch.from_numpy(smoothed_weight_map).float().to(device)
                weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(0)
                
                target_h, target_w = target_size
                upsampled_tensor = F.interpolate(
                    weight_tensor,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )
                
                final_feature_map = upsampled_tensor.squeeze(0).squeeze(0)
                # 确保最终特征图在 [0, 1] 范围内，1代表最重要的区域
                final_feature_map = (final_feature_map - final_feature_map.min()) / (final_feature_map.max() - final_feature_map.min() + 1e-8)
                
                # 【可视化】保存最终特征图
                if save_visualization:
                    viz_data['final_feature_maps'].append(final_feature_map.cpu().numpy().copy())
                
                frame_feature_map = final_feature_map
                
        except Exception as e:
            print(f"Error extracting DINOv2 features for frame {t}: {e}")
            frame_feature_map = torch.ones(target_size, dtype=torch.float32, device=device)
            
            if save_visualization:
                viz_data['raw_feature_maps'].append(np.ones((8, 8)))
                viz_data['smoothed_feature_maps'].append(np.ones((8, 8)))
                viz_data['final_feature_maps'].append(np.ones(target_size))
                viz_data['pca_explained_variance'].append(np.array([0.33, 0.33, 0.33]))
                viz_data['pca_components_raw'].append([np.ones((8, 8)), np.ones((8, 8)), np.ones((8, 8))])
                viz_data['pca_components_remapped'].append([np.ones((8, 8)), np.ones((8, 8)), np.ones((8, 8))])
                viz_data['grid_sizes'].append((8, 8))
        
        frame_feature_maps.append(frame_feature_map)
    
    # 堆叠所有帧的特征图
    feature_maps = torch.stack(frame_feature_maps, dim=0)  # (T, H, W)
    
    # 时间维度重采样（对于Flux单帧，通常不需要）
    if T != target_time:
        if target_time == 1:
            # 对于单帧，取第一帧
            feature_maps = feature_maps[0:1]  # (1, H, W)
        else:
            feature_maps_resampled = F.interpolate(
                feature_maps.unsqueeze(0).unsqueeze(0),
                size=(target_time, feature_maps.shape[1], feature_maps.shape[2]),
                mode='trilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)
            feature_maps = feature_maps_resampled
    
    # 【新增】生成可视化，现在包含原始和重映射的PCA组件对比
    if save_visualization and viz_data:
        create_flux_dinov2_visualization_with_remapping(viz_data, feature_maps, viz_save_path, step, pca_method)
    
    return feature_maps

def create_flux_dinov2_visualization_with_remapping(viz_data, final_feature_maps, save_path, step=None, pca_method="weighted"):
    import numpy as np
    """
    创建包含重映射对比的DINOv2特征可视化
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import os
    import time
    
    os.makedirs(save_path, exist_ok=True)
    
    # Flux通常是单帧，但保持处理多帧的能力
    num_frames = len(viz_data['original_frames'])
    key_frame_indices = [0] if num_frames == 1 else [0, num_frames//2, num_frames-1]
    
    # 添加时间戳和rank信息避免覆盖
    timestamp = int(time.time())
    rank = int(os.environ.get("RANK", 0))
    
    for frame_idx in key_frame_indices:
        if frame_idx >= num_frames:
            continue
            
        # 【修改】创建更大的网格来显示原始和重映射的对比 (5x4)
        fig, axes = plt.subplots(5, 4, figsize=(20, 20))
        step_str = f"_step{step}" if step is not None else ""
        fig.suptitle(f'Flux DINOv2 Feature Remapping Analysis - Frame {frame_idx}{step_str}\nPCA Method: {pca_method}', fontsize=18)
        
        # 第一行：原始图像、原始PCA组件1-3
        # 原始图像
        axes[0, 0].imshow(viz_data['original_frames'][frame_idx])
        axes[0, 0].set_title('Original Generated Image')
        axes[0, 0].axis('off')
        
        # 原始PCA组件（保持正负值）
        pca_components_raw = viz_data['pca_components_raw'][frame_idx]
        explained_var = viz_data['pca_explained_variance'][frame_idx]
        
        for comp_idx in range(3):
            ax = axes[0, comp_idx + 1]
            component_map = pca_components_raw[comp_idx]
            
            # 使用对称颜色映射显示原始正负值
            vmax = max(abs(component_map.min()), abs(component_map.max()))
            vmin = -vmax
            
            im = ax.imshow(component_map, cmap='RdBu_r', interpolation='nearest', 
                          vmin=vmin, vmax=vmax)
            ax.set_title(f'Raw PC{comp_idx + 1}\n(Var: {explained_var[comp_idx]:.3f})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # 第二行：重映射后的PCA组件1-3，RGB合成
        pca_components_remapped = viz_data['pca_components_remapped'][frame_idx]
        
        for comp_idx in range(3):
            ax = axes[1, comp_idx]
            component_map = pca_components_remapped[comp_idx]
            
            # 重映射后的组件使用热力图显示
            im = ax.imshow(component_map, cmap='hot', interpolation='nearest', 
                          vmin=0, vmax=1)
            ax.set_title(f'Remapped PC{comp_idx + 1}\n(原值越小 → 权重越大)')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # RGB合成（重映射后的组件）
        rgb_composite = np.zeros((*pca_components_remapped[0].shape, 3))
        for i in range(3):
            rgb_composite[:, :, i] = pca_components_remapped[i]
        
        axes[1, 3].imshow(rgb_composite)
        axes[1, 3].set_title('Remapped RGB Composite\n(R=PC1, G=PC2, B=PC3)')
        axes[1, 3].axis('off')
        
        # 第三行：组合特征图、解释方差、权重分布对比
        # 组合特征图（最终权重）
        raw_map = viz_data['raw_feature_maps'][frame_idx]
        grid_size = viz_data['grid_sizes'][frame_idx]
        im1 = axes[2, 0].imshow(raw_map, cmap='hot', interpolation='nearest')
        axes[2, 0].set_title(f'Combined Feature Map\nGrid: {grid_size[0]}×{grid_size[1]}')
        axes[2, 0].axis('off')
        plt.colorbar(im1, ax=axes[2, 0], shrink=0.6)
        
        # PCA解释方差条形图
        axes[2, 1].bar(range(len(explained_var)), explained_var, 
                      color=['red', 'green', 'blue'][:len(explained_var)])
        axes[2, 1].set_title('PCA Explained Variance Ratio')
        axes[2, 1].set_xlabel('Principal Component')
        axes[2, 1].set_ylabel('Explained Variance')
        axes[2, 1].set_xticks(range(len(explained_var)))
        axes[2, 1].set_xticklabels([f'PC{i+1}' for i in range(len(explained_var))])
        for i, v in enumerate(explained_var):
            axes[2, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 原始vs重映射权重分布对比
        axes[2, 2].hist(pca_components_raw[0].flatten(), bins=20, alpha=0.5, 
                       label='Raw PC1', color='blue', density=True)
        axes[2, 2].hist(pca_components_remapped[0].flatten(), bins=20, alpha=0.5, 
                       label='Remapped PC1', color='red', density=True)
        axes[2, 2].set_title('PC1: Raw vs Remapped Distribution')
        axes[2, 2].set_xlabel('Value')
        axes[2, 2].set_ylabel('Density')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        # 重映射说明
        remap_text = f"""Remapping Strategy:
        weight = (max_val - original_val) / (max_val - min_val)
        """
        
        axes[2, 3].text(0.05, 0.95, remap_text, transform=axes[2, 3].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        axes[2, 3].set_title('Remapping Explanation')
        axes[2, 3].axis('off')
        
        # 第四行：平滑后特征图、最终特征图、统计信息、叠加显示
        # 平滑后特征图
        smoothed_map = viz_data['smoothed_feature_maps'][frame_idx]
        im2 = axes[3, 0].imshow(smoothed_map, cmap='hot', interpolation='nearest')
        axes[3, 0].set_title(f'Smoothed Feature Map\n(σ={1.0})')
        axes[3, 0].axis('off')
        plt.colorbar(im2, ax=axes[3, 0], shrink=0.6)
        
        # 最终特征图
        final_map = viz_data['final_feature_maps'][frame_idx]
        im3 = axes[3, 1].imshow(final_map, cmap='hot', interpolation='nearest')
        axes[3, 1].set_title(f'Final Feature Map\n{final_map.shape[0]}×{final_map.shape[1]} (Latent Size)')
        axes[3, 1].axis('off')
        plt.colorbar(im3, ax=axes[3, 1], shrink=0.6)
        
        # 统计信息对比
        stats_text = f"""Remapping Statistics:

        Raw PCA Components:
        PC1 range: [{pca_components_raw[0].min():.3f}, {pca_components_raw[0].max():.3f}]
        PC2 range: [{pca_components_raw[1].min():.3f}, {pca_components_raw[1].max():.3f}]
        PC3 range: [{pca_components_raw[2].min():.3f}, {pca_components_raw[2].max():.3f}]

        Remapped Components:
        PC1: [{pca_components_remapped[0].min():.3f}, {pca_components_remapped[0].max():.3f}] (0=不重要, 1=重要)
        PC2: [{pca_components_remapped[1].min():.3f}, {pca_components_remapped[1].max():.3f}]
        PC3: [{pca_components_remapped[2].min():.3f}, {pca_components_remapped[2].max():.3f}]

        Final Feature Map:
        Mean: {final_map.mean():.4f}
        Std: {final_map.std():.4f}
        Range: [{final_map.min():.4f}, {final_map.max():.4f}]

        高权重区域 (>0.7): {(final_map > 0.7).sum()}/{final_map.size} pixels
        ({(final_map > 0.7).sum()/final_map.size*100:.1f}%)"""
        
        axes[3, 2].text(0.05, 0.95, stats_text, transform=axes[3, 2].transAxes, 
                        fontsize=7, verticalalignment='top', fontfamily='monospace')
        axes[3, 2].set_title('Remapping Statistics')
        axes[3, 2].axis('off')
        
        # 叠加显示（原图+最终特征图）
        original_resized = np.array(viz_data['original_frames'][frame_idx].resize((final_map.shape[1], final_map.shape[0])))
        axes[3, 3].imshow(original_resized)
        axes[3, 3].imshow(final_map, alpha=0.6, cmap='hot')
        axes[3, 3].set_title('Final Feature Overlay\n(亮区域 = 人类关注区域)')
        axes[3, 3].axis('off')
        
        # 第五行：重映射前后的对比分析
        # 原始PC1 vs 重映射PC1的散点图
        axes[4, 0].scatter(pca_components_raw[0].flatten(), pca_components_remapped[0].flatten(), 
                          alpha=0.6, s=10, c='blue')
        axes[4, 0].set_xlabel('Original PC1 Value')
        axes[4, 0].set_ylabel('Remapped PC1 Value')
        axes[4, 0].set_title('PC1: Original vs Remapped')
        axes[4, 0].grid(True, alpha=0.3)
        axes[4, 0].axvline(0, color='red', linestyle='--', alpha=0.5, label='原值=0')
        axes[4, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='映射=0.5')
        axes[4, 0].legend(fontsize=8)
        
        # 权重分布的累积分布函数
        import numpy as np
        final_sorted = np.sort(final_map.flatten())
        cumulative = np.arange(1, len(final_sorted) + 1) / len(final_sorted)
        axes[4, 1].plot(final_sorted, cumulative, 'b-', linewidth=2)
        axes[4, 1].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='中等重要')
        axes[4, 1].axvline(0.7, color='orange', linestyle='--', alpha=0.7, label='高重要')
        axes[4, 1].set_xlabel('Feature Weight')
        axes[4, 1].set_ylabel('Cumulative Probability')
        axes[4, 1].set_title('Final Feature Weight CDF')
        axes[4, 1].grid(True, alpha=0.3)
        axes[4, 1].legend(fontsize=8)
        
        # 空间热图 - 显示最重要的区域
        high_attention_mask = final_map > 0.7
        axes[4, 2].imshow(high_attention_mask, cmap='Reds', interpolation='nearest')
        axes[4, 2].set_title(f'High Attention Regions\n({high_attention_mask.sum()} pixels > 0.7)')
        axes[4, 2].axis('off')
        
        # 方法总结
        method_summary = f"""Method: {pca_method}"""
        
        axes[4, 3].text(0.05, 0.95, method_summary, transform=axes[4, 3].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        axes[4, 3].set_title(f'Method Summary: {pca_method}')
        axes[4, 3].axis('off')
        
        plt.tight_layout()
        
        # 保存图片时添加更多唯一标识符
        filename = f'flux_dinov2_remapping_analysis_rank{rank}_frame{frame_idx}{step_str}_ts{timestamp}.png'
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Flux DINOv2 Remapping visualization saved: {filepath}")
        print(f"✓ Remapping Analysis - High attention pixels: {(final_map > 0.7).sum()}/{final_map.size}")

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def flux_step(
    model_output: torch.Tensor, # 模型预测输出（噪声/速度场）
    latents: torch.Tensor, # 当前时间步的潜在表示
    eta: float, #控制随机性强度（SDE公式里的随机性，越强探索度越大）
    sigmas: torch.Tensor, # 完整的噪声调度序列
    index: int, # 当前时间步索引
    prev_sample: torch.Tensor, # 前一步的样本（用于GRPO重计算）
    grpo: bool, #True时会得到logprob
    sde_solver: bool,
    return_unpacked_logprob: bool = False,  # 【新增】
    original_height: int = None,  # 【新增】
    original_width: int = None,   # 【新增】
    ):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma # 噪声差分
    prev_sample_mean = latents + dsigma * model_output # 确定性更新部分

    pred_original_sample = latents - sigma * model_output # 预测的原始样本

    delta_t = sigma - sigmas[index + 1] # 噪声差分
    std_dev_t = eta * math.sqrt(delta_t) # 随机噪声的std

    if sde_solver: #使用SDE求解器
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2 # 估计的得分
        log_term = -0.5 * eta**2 * score_estimate # 对数项修正
        prev_sample_mean = prev_sample_mean + log_term * dsigma #prev的样本mean

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t # prev样本
        

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = (
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))) # log_prob shape: (1,1024,64)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample



def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

# 采样过程，包括计算logprob
def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        image_ids, 
        grpo_sample,
    ):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:  # Add progress bar
            B = encoder_hidden_states.shape[0] # 批次大小（文本的批次）
            sigma = sigma_schedule[i] #当前timestep的噪声水平
            timestep_value = int(sigma * 1000) # 转换为模型期望的时间步格式
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long) #为批次中每个样本创建相同的时间步张量
            transformer.eval()
            with torch.autocast("cuda", torch.bfloat16):
                pred= transformer(
                    hidden_states=z, # 当前的潜在状态
                    encoder_hidden_states=encoder_hidden_states, # 文本条件
                    timestep=timesteps/1000, # 标准化的时间步 [0,1]
                    guidance=torch.tensor( # CFG引导强度
                        [3.5],
                        device=z.device,
                        dtype=torch.bfloat16
                    ),
                    txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L # 重复文本ID以匹配序列长度
                    pooled_projections=pooled_prompt_embeds, # 池化的文本嵌入
                    img_ids=image_ids, # 图像位置编码
                    joint_attention_kwargs=None, # 联合注意力参数（未使用）
                    return_dict=False, # 返回张量而非字典
                )[0] #1, 1024, 64

            z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True,
                                                    return_unpacked_logprob=True,  # 【关键】启用unpack
                                                    original_height=args.h,  # 【关键】传递原始高度
                                                    original_width=args.w)   # 【关键】传递原始宽度
            
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1) #在dim=1拼接是了保存不同时间步的latent
        all_log_probs = torch.stack(all_log_probs, dim=1)
        return z, latents, all_latents, all_log_probs

        
def grpo_one_step(
            args,
            latents,
            pre_latents,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            image_ids,
            transformer,
            timesteps,
            i,
            sigma_schedule,
    ):
    B = encoder_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred= transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps/1000,
            guidance=torch.tensor(
                [3.5],
                device=latents.device,
                dtype=torch.bfloat16
            ),
            txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
            pooled_projections=pooled_prompt_embeds,
            img_ids=image_ids.squeeze(0),
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True,
                                            return_unpacked_logprob=True,  # 【关键】启用unpack
                                            original_height=args.h,  # 【关键】传递原始高度
                                            original_width=args.w)   # 【关键】传递原始宽度
    return log_prob

def sample_reference_model_with_pixel_reward(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
    step=None,
):
    w, h, t = args.w, args.h, 1  # Flux是单帧，t=1
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []  
    all_image_ids = []
    all_dinov2_feature_maps = []  # 【新增】存储DINOv2特征图

    if args.init_same_noise:
        input_latents = torch.randn(
            (1, IN_CHANNELS, latent_h, latent_w),
            device=device,
            dtype=torch.bfloat16,
        )

    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_text_ids = text_ids[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        
        if not args.init_same_noise:
            input_latents = torch.randn(
                (len(batch_idx), IN_CHANNELS, latent_h, latent_w),
                device=device,
                dtype=torch.bfloat16,
            )
        # 先创建噪声，然后把噪声打成patch
        input_latents_new = pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w)
        image_ids = prepare_latent_image_ids(len(batch_idx), latent_h // 2, latent_w // 2, device, torch.bfloat16)
        
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        
        with torch.no_grad():
            # 运行采样过程得到latent路径, log_probs
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents_new,
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_pooled_prompt_embeds,
                batch_text_ids,
                image_ids,
                grpo_sample=True,
            )
        
        all_image_ids.append(image_ids)
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        
        vae.enable_tiling()
        image_processor = VaeImageProcessor(16)
        rank = int(os.environ["RANK"])

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                print("latents shape before decode:", latents.shape)
                latents_for_decode = unpack_latents(latents, h, w, 8)
                print("latents shape after decode:", latents_for_decode.shape)
                latents_for_decode = (latents_for_decode / 0.3611) + 0.1159
                image = vae.decode(latents_for_decode, return_dict=False)[0]
                decoded_image = image_processor.postprocess(image)
        
        decoded_image[0].save(f"./images/flux_{rank}_{index}.png")

        # 将生成的图像转换为合适的格式
        generated_image_tensor = image  # (B, C, H, W)
        rank = int(os.environ.get("RANK", 0))
        viz_save_dir = os.path.join(args.output_dir, "dinov2_viz", f"rank_{rank}")
        if step is not None:
            viz_save_dir = os.path.join(viz_save_dir, f"step_{step}")
        batch_dinov2_features = []
        for img_idx in range(generated_image_tensor.shape[0]):
            single_image = generated_image_tensor[img_idx]  # (C, H, W)
            target_h, target_w = latent_h, latent_w

                
            dinov2_feature_map = compute_dinov2_feature_map_with_visualization(
                single_image,  # (C, H, W)
                target_size=(target_h, target_w),
                target_time=1,  # 单帧
                device=device,
                pca_method=getattr(args, 'dinov2_pca_method', 'weighted'),
                smooth_method="gaussian_strong",
                sigma=getattr(args, 'dinov2_sigma', 1.0),
                save_visualization=False,  # 【新增】
                viz_save_path=viz_save_dir,
                step=step  # 【新增】
            )  # 返回 (1, H, W)
            print("DINOv2 feature map shape:", dinov2_feature_map.shape)  # 应该是 (1, latent_h, latent_w)
                
            batch_dinov2_features.append(dinov2_feature_map.squeeze(0))  # (H, W)
            
        batch_dinov2_features = torch.stack(batch_dinov2_features, dim=0)  # (B, H, W)
        all_dinov2_feature_maps.append(batch_dinov2_features)

        # 计算标准奖励（HPS或其他）
        if args.use_hpsv2:
            with torch.no_grad():
                image_path = decoded_image[0]
                image_for_reward = preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                text = tokenizer([batch_caption[0]]).to(device=device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = reward_model(image_for_reward, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                    all_rewards.append(hps_score)
        else:
            empty_reward = 1
            all_rewards.append(empty_reward)

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_rewards = torch.cat(all_rewards, dim=0)
    all_image_ids = torch.stack(all_image_ids, dim=0)
    all_dinov2_feature_maps = torch.cat(all_dinov2_feature_maps, dim=0)  # 【修正】现在是正确的尺寸
    print("All DINOv2 feature maps shape:", all_dinov2_feature_maps.shape)  # (B, latent_h, latent_w
    print("All rewards shape:", all_rewards.shape)
    print("All latents shape:", all_latents.shape)
    print("All log_probs shape:", all_log_probs.shape)
    return all_rewards, all_latents, all_log_probs, sigma_schedule, all_image_ids, all_dinov2_feature_maps

def train_one_step_pixel_wise(
    args,
    device,
    transformer,
    vae,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    max_grad_norm,
    preprocess_val,
    step=None,
):
    total_loss = 0.0
    optimizer.zero_grad()
    
    (encoder_hidden_states, pooled_prompt_embeds, text_ids, caption) = next(loader)
    
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        pooled_prompt_embeds = repeat_tensor(pooled_prompt_embeds)
        text_ids = repeat_tensor(text_ids)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    # 采样参考模型（包含DINOv2特征图）
    reward, all_latents, all_log_probs, sigma_schedule, all_image_ids, dinov2_feature_maps = sample_reference_model_with_pixel_reward(
        args, device, transformer, vae, encoder_hidden_states, pooled_prompt_embeds, 
        text_ids, reward_model, tokenizer, caption, preprocess_val, step
    )
    
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps = torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[:, :-1][:, :-1],  # 采样轨迹中的latent状态
        "next_latents": all_latents[:, 1:][:, :-1],  # 下一步的latent状态
        "log_probs": all_log_probs[:, :-1],
        "rewards": reward.to(torch.float32),
        "dinov2_feature_maps": dinov2_feature_maps,
        "image_ids": all_image_ids,
        "text_ids": text_ids,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }
    
    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank() == 0:
        print("gathered_reward", gathered_reward)
        with open('./pixel_flux_reward_logprob_dinozhuti_firstpc.txt', 'a') as f: 
            f.write(f"{gathered_reward.mean().item()}\n")

    # 【关键修改】计算pixel-wise advantages
    if args.use_group:
        n = len(samples["rewards"]) // args.num_generations
        
        # 计算DINOv2加权的advantages
        pixel_advantages = torch.zeros_like(samples["dinov2_feature_maps"])  # (num_gen, H, W)
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            
            # 计算标量优势
            group_rewards = samples["rewards"][start_idx:end_idx]  # (num_gen,) 标量
            group_mean = group_rewards.mean()  # 标量
            group_std = group_rewards.std() + 1e-8  # 标量
            scalar_advantages = (group_rewards - group_mean) / group_std  # (num_gen,) 标量优势
            
            # 【关键】应用DINOv2加权：标量优势 × DINOv2特征图
            for j in range(args.num_generations):
                sample_idx = start_idx + j
                scalar_adv = scalar_advantages[j].item()  # 标量
                feature_map = samples["dinov2_feature_maps"][sample_idx]  # (H, W)
                pixel_advantages[sample_idx] = scalar_adv * feature_map  # 广播乘法：(H, W)
        
        samples["pixel_advantages"] = pixel_advantages  # (num_gen, H, W)
    else:
        # 全局advantages
        global_advantages = (samples["rewards"] - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)
        # 扩展到pixel-wise
        pixel_advantages = global_advantages.unsqueeze(-1).unsqueeze(-1) * samples["dinov2_feature_maps"]
        samples["pixel_advantages"] = pixel_advantages

    # 随机打乱时间步
    perms = torch.stack([
        torch.randperm(len(samples["timesteps"][0]))
        for _ in range(batch_size)
    ]).to(device) 
    
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device)[:, None],
            perms,
        ]
    
    samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    
    train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)
    grad_norm = None
    
    for i, sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = getattr(args, 'clip_range', 1e-4)
            adv_clip_max = getattr(args, 'adv_clip_max', 5.0)
            
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:, _],
                sample["next_latents"][:, _],
                sample["encoder_hidden_states"],
                sample["pooled_prompt_embeds"],
                sample["text_ids"],
                sample["image_ids"],
                transformer,
                sample["timesteps"][:, _],
                perms[i][_],
                sigma_schedule,
            )
            print("new_log_probs shape:", new_log_probs.shape)  # (B, H, W)
            pixel_advantages = torch.clamp(
                sample["pixel_advantages"],
                -adv_clip_max,
                adv_clip_max,
            )
            print("pixel_advantages shape:", pixel_advantages.shape)  # (B, H, W)
            ratio = torch.exp(new_log_probs - sample["log_probs"][:, _])
            print("ratio shape:", ratio.shape)  # (B, H, W)
            pixel_unclipped_loss = -pixel_advantages * ratio.unsqueeze(-1).unsqueeze(-1)  # 广播ratio到spatial维度
            pixel_clipped_loss = -pixel_advantages * torch.clamp(
                ratio.unsqueeze(-1).unsqueeze(-1),
                1.0 - clip_range,
                1.0 + clip_range,
            )
            
            pixel_loss = torch.mean(torch.maximum(pixel_unclipped_loss, pixel_clipped_loss))
            loss = pixel_loss / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
            
        if (i + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        if dist.get_rank() % 8 == 0:
            print("reward", sample["rewards"].item())
            print("ratio", ratio.mean().item())
            print("pixel advantage (mean/std):", pixel_advantages.mean().item(), pixel_advantages.std().item())
            print("final loss", loss.item())
        dist.barrier()
        
    return total_loss, grad_norm.item()

def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    if args.use_hpsv2:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from typing import Union
        import huggingface_hub
        from hpsv2.utils import root_path, hps_version_map
        def initialize_model():
            model_dict = {}
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                './hps_ckpt/open_clip_pytorch_model.bin',
                precision='amp',
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict
        model_dict = initialize_model()
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        #cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
        cp = "./hps_ckpt/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(cp, map_location=f'cuda:{device}')
        model.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(device)
        reward_model.eval()

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    
    # 加载FLUX的DiT
    transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32
    )

     # 添加参数量统计
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    main_print(f"=== FLUX Transformer Model Statistics ===")
    main_print(f"  Total parameters: {total_params / 1e9:.2f} B")
    main_print(f"  Trainable parameters: {trainable_params / 1e9:.2f} B")
    main_print(f"  Non-trainable parameters: {non_trainable_params / 1e9:.2f} B")
    main_print(f"  Trainable ratio: {trainable_params / total_params * 100:.1f}%")

    # fsdp并行
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    
    transformer = FSDP(transformer, **fsdp_kwargs,)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )
    
    # 加载VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    # 返回text_emb, pooled_prompt_embeds, text_ids, caption
    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    vae.enable_tiling()

    if rank <= 0:
        project = "flux"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)


    for epoch in range(1000000):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch
        for step in range(init_steps+1, args.max_train_steps+1):
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                save_checkpoint(transformer, rank, args.output_dir,step,epoch)

                dist.barrier()
            loss, grad_norm = train_one_step_pixel_wise(
                args,
                device, 
                transformer,
                vae,
                reward_model,
                processor,
                optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                args.max_grad_norm,
                preprocess_val,
            )
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )



    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader

    parser.add_argument(
        "--use_pixel_wise_grpo",
        action="store_true",
        default=False,
        help="whether to use pixel-wise GRPO with DINOv2 features",
    )
    parser.add_argument(
        "--dinov2_pca_method",
        type=str,
        default="weighted",
        choices=["weighted", "average", "first_pc"],
        help="PCA method for DINOv2 feature extraction"
    )
    parser.add_argument(
        "--dinov2_sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma for DINOv2 features"
    )

    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--use_simple_color_reward",
        action="store_true",
        default=False,
        help="whether use simple color reward instead of complex color diversity reward",
    )




    args = parser.parse_args()
    main(args)
