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

from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler

from accelerate.utils import set_seed
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
import torch.distributed as dist

import numpy as np

import torch.distributed as dist
from torch.nn import functional as F
from PIL import Image
import cv2
import logging
import json
import os

from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel  
from wan.modules.vae import WanVAE

import numpy as np
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from fastvideo.reward_tracker import RewardTracker
# 加入hps
from HPSv2.hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from rewardmodel.video_align.inference import VideoVLMRewardInference
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
check_min_version("0.31.0")

def compute_dinov2_feature_map_reverse(
    video_frames, 
    target_size=(64, 64), 
    target_time=7,
    device=None,
    pca_method="weighted",  # "weighted", "average", "first_pc"
    smooth_method="gaussian_strong",  # 新增平滑方法参数
    sigma=1.0  # gaussian_strong 对应的 sigma 值
):
    """
    使用DINOv2提取视频帧的语义特征权重图，应用gaussian_strong平滑，直接处理到latent维度
    
    Args:
        video_frames: torch.Tensor, shape (C, T, H, W), 范围 [0, 1] 或 [-1, 1]
        target_size: tuple, 目标空间尺寸 (height, width)
        target_time: int, 目标时间维度
        device: torch.device, 计算设备
        pca_method: str, PCA降维方法 ("weighted", "average", "first_pc")
        smooth_method: str, 平滑方法，默认使用 "gaussian_strong"
        sigma: float, 高斯平滑的标准差
        
    Returns:
        feature_map: torch.Tensor, shape (T_target, H, W), 平滑后的语义特征权重图，范围[0, 1]
    """
    # 确保输入在 [0, 1] 范围内
    if video_frames.min() < 0:
        video_frames = (video_frames + 1.0) / 2.0
    video_frames = torch.clamp(video_frames, 0, 1).float()
    
    if device is None:
        device = video_frames.device
    
    # 初始化DINOv2模型（只在第一次调用时加载）
    if not hasattr(compute_dinov2_feature_map_reverse, 'dinov2_model'):
        from transformers import AutoImageProcessor, AutoModel, AutoConfig
        
        model_path = './dinov2-large'
        config = AutoConfig.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
        dinov2_model = AutoModel.from_pretrained(model_path).to(device)
        dinov2_model.eval()
        
        # 将模型和处理器保存为函数属性
        compute_dinov2_feature_map_reverse.dinov2_model = dinov2_model
        compute_dinov2_feature_map_reverse.dinov2_processor = processor
        print("✓ DINOv2 model loaded and cached")
    
    dinov2_model = compute_dinov2_feature_map_reverse.dinov2_model
    dinov2_processor = compute_dinov2_feature_map_reverse.dinov2_processor
    
    C, T, H, W = video_frames.shape
    frame_feature_maps = []
    
    # print(f"Extracting DINOv2 features from {T} frames with gaussian_strong smoothing...")

    for t in range(T):
        frame = video_frames[:, t, :, :]  # (C, H, W)
        
        try:
            with torch.no_grad():
                # 转换为PIL图像进行DINOv2处理
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
                frame_np = (frame_np * 255).astype(np.uint8)
                frame_pil = Image.fromarray(frame_np)
                
                original_size = frame_pil.size  # (width, height)
                
                # DINOv2处理，尽量保持原始尺寸
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
                        # 保持宽高比，设置最短边
                        min_edge = min(original_size)
                        if min_edge >= 224:
                            target_short_edge = min(min_edge, 518)
                        else:
                            target_short_edge = 224
                        
                        dinov2_inputs = dinov2_processor(
                            images=frame_pil, 
                            return_tensors="pt",
                            size={"shortest_edge": target_short_edge},
                            do_center_crop=False,
                            do_resize=True
                        ).to(device)
                        
                    except Exception as e2:
                        # 使用默认处理
                        dinov2_inputs = dinov2_processor(
                            images=frame_pil, 
                            return_tensors="pt"
                        ).to(device)
                
                # 获取实际处理后图像的尺寸
                processed_img_tensor = dinov2_inputs['pixel_values'][0]  # (3, H_proc, W_proc)
                processed_height, processed_width = processed_img_tensor.shape[1], processed_img_tensor.shape[2]
                
                # 提取DINOv2特征
                dinov2_outputs = dinov2_model(**dinov2_inputs)
                features = dinov2_outputs.last_hidden_state[0, 1:, :].cpu().numpy()  # 排除CLS token
                
                # 计算patch网格尺寸
                num_patches = features.shape[0]
                patch_size = 14  # DINOv2标准patch size
                
                grid_h = processed_height // patch_size
                grid_w = processed_width // patch_size
                
                # 处理patch数量不匹配的情况
                if grid_h * grid_w != num_patches:
                    # 寻找最佳的矩形网格
                    possible_factors = []
                    for h in range(1, int(np.sqrt(num_patches)) + 10):
                        if num_patches % h == 0:
                            w = num_patches // h
                            possible_factors.append((h, w, abs(h - w)))
                    
                    target_ratio = processed_height / processed_width
                    best_grid = min(possible_factors, key=lambda x: abs((x[0]/x[1]) - target_ratio))
                    grid_h, grid_w = best_grid[0], best_grid[1]
                
                # PCA降维并计算语义权重
                from sklearn.decomposition import PCA
                
                pca = PCA(n_components=3)
                pca_features = pca.fit_transform(features)
                explained_variance = pca.explained_variance_ratio_
                
                # 【关键修改】根据指定方法计算权重，使用反向映射
                if pca_method == "weighted":
                    # 对每个主成分应用反向映射，然后加权组合
                    semantic_weights = np.zeros(pca_features.shape[0])
                    for comp_idx in range(3):
                        comp_weights = pca_features[:, comp_idx]
                        # 反向映射：原值越小，权重越大（越负值代表越是语义主体）
                        min_val = comp_weights.min()
                        max_val = comp_weights.max()
                        if max_val != min_val:
                            comp_weights_remapped = (max_val - comp_weights) / (max_val - min_val)
                        else:
                            comp_weights_remapped = np.ones_like(comp_weights) * 0.5
                        
                        # 用解释方差加权
                        semantic_weights += explained_variance[comp_idx] * comp_weights_remapped
                elif pca_method == "average":
                    # 对所有主成分取平均，然后反向映射
                    avg_features = np.mean(pca_features, axis=1)
                    min_val = avg_features.min()
                    max_val = avg_features.max()
                    if max_val != min_val:
                        semantic_weights = (max_val - avg_features) / (max_val - min_val)
                    else:
                        semantic_weights = np.ones_like(avg_features) * 0.5
                elif pca_method == "first_pc":
                    # 只使用第一主成分
                    first_pc = pca_features[:, 0]
                    min_val = first_pc.min()
                    max_val = first_pc.max()
                    if max_val != min_val:
                        semantic_weights = (max_val - first_pc) / (max_val - min_val)
                    else:
                        semantic_weights = np.ones_like(first_pc) * 0.5
                else:
                    raise ValueError(f"Unknown pca_method: {pca_method}")
                
                semantic_weights = np.clip(semantic_weights, 0, 1)
                
                # 重塑权重图到patch网格
                if len(semantic_weights) == grid_h * grid_w:
                    weight_map = semantic_weights.reshape(grid_h, grid_w)
                else:
                    # 如果尺寸不匹配，使用插值
                    from scipy.ndimage import zoom
                    temp_grid = int(np.sqrt(len(semantic_weights)))
                    temp_map = semantic_weights.reshape(temp_grid, temp_grid)
                    scale_h = grid_h / temp_grid
                    scale_w = grid_w / temp_grid
                    weight_map = zoom(temp_map, (scale_h, scale_w), order=1)
                
                from scipy.ndimage import gaussian_filter
                
                # 应用高斯平滑：空间平滑滤波
                smoothed_weight_map = gaussian_filter(weight_map, sigma=sigma)
                
                # 重新归一化到[0, 1]
                smoothed_weight_map = (smoothed_weight_map - smoothed_weight_map.min()) / (smoothed_weight_map.max() - smoothed_weight_map.min() + 1e-8)
                
                import torch.nn.functional as F
                
                # 转换为tensor进行上采样
                weight_tensor = torch.from_numpy(smoothed_weight_map).float().to(device)
                weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_h, grid_w)
                
                # 直接双线性插值到目标latent尺寸
                target_h, target_w = target_size
                upsampled_tensor = F.interpolate(
                    weight_tensor,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True 
                )
                
                # 压缩回2D并转换为最终特征图
                final_feature_map = upsampled_tensor.squeeze(0).squeeze(0)  # (target_h, target_w)
                
                # 最终归一化确保范围在[0, 1]
                final_feature_map = (final_feature_map - final_feature_map.min()) / (final_feature_map.max() - final_feature_map.min() + 1e-8)
                
                frame_feature_map = final_feature_map
                
        except Exception as e:
            print(f"Error extracting DINOv2 features for frame {t}: {e}")
            # 如果特征提取失败，使用均匀权重
            frame_feature_map = torch.ones(target_size, dtype=torch.float32, device=device)
        frame_feature_maps.append(frame_feature_map)
    
    # 堆叠所有帧的特征图
    feature_maps = torch.stack(frame_feature_maps, dim=0)  # (T, H, W)
    
    # 【修复】时间维度重采样 - 移除antialias参数，因为trilinear不支持
    if T != target_time:
        print(f"Resampling temporal dimension from {T} to {target_time}")
        feature_maps_resampled = F.interpolate(
            feature_maps.unsqueeze(0).unsqueeze(0),  # (1, 1, T, H, W)
            size=(target_time, feature_maps.shape[1], feature_maps.shape[2]),
            mode='trilinear',
            align_corners=False,
            # 【修复】移除antialias参数，trilinear模式不支持
        ).squeeze(0).squeeze(0)  # (T_target, H, W)
        feature_maps = feature_maps_resampled
    
    # 【可选】轻度时间平滑处理
    if T > 1:
        # 对时间维度也进行轻度高斯平滑
        feature_maps_np = feature_maps.cpu().numpy()
        for h in range(feature_maps.shape[1]):
            for w in range(feature_maps.shape[2]):
                # 对每个像素位置的时间序列进行1D高斯平滑
                time_series = feature_maps_np[:, h, w]
                smoothed_time_series = gaussian_filter(time_series, sigma=0.3)  # 轻度时间平滑
                feature_maps_np[:, h, w] = smoothed_time_series
        
        feature_maps = torch.from_numpy(feature_maps_np).to(device, dtype=feature_maps.dtype)
        
        # 重新归一化
        feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min() + 1e-8)

    return feature_maps

def save_checkpoint_simple(model, optimizer, lr_scheduler, step, epoch, rank, output_dir, args):
    """
    简单的checkpoint保存功能 - 只保存，不加载，不清理
    Args:
        model: FSDP包装的模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        step: 当前训练步数
        epoch: 当前epoch
        rank: 当前进程rank
        output_dir: 输出目录
        args: 训练参数
    """
    if not output_dir:
        print("Warning: output_dir is None, skipping checkpoint save")
        return
    
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-step-{step}")
    
    # 只有rank 0创建目录和保存文件
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"💾 Saving checkpoint at step {step} to {checkpoint_dir}")
    
    # 同步所有进程
    if dist.is_initialized():
        dist.barrier()
    try:
        # 使用FSDP的state_dict保存
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=False, rank0_only=True)):
            model_state_dict = model.state_dict()
        if rank == 0:
            # 【修改】导入safetensors并使用safetensor格式保存模型权重
            from safetensors.torch import save_file
            # 保存模型权重为safetensor格式
            model_file = os.path.join(checkpoint_dir, 'model.safetensors')
            save_file(model_state_dict, model_file)

            save_opt_sche = True # 这里设置要不要保存optimizer和scheduler
            if save_opt_sche:
                # 2. 保存优化器状态
                optimizer_file = os.path.join(checkpoint_dir, 'optimizer.pt')
                torch.save(optimizer.state_dict(), optimizer_file)
                
                # 3. 保存学习率调度器状态
                scheduler_file = os.path.join(checkpoint_dir, 'scheduler.pt')
                torch.save(lr_scheduler.state_dict(), scheduler_file)

                # 4. 保存训练状态和配置
                training_state = {
                    'step': step,
                    'epoch': epoch,
                    'args': vars(args),  # 保存所有训练参数
                    'random_state': {
                        'python': random.getstate(),
                        'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    }
                }
                
                training_state_file = os.path.join(checkpoint_dir, 'training_state.pt')
                torch.save(training_state, training_state_file)
            
            latest_file = os.path.join(output_dir, 'latest_checkpoint.txt')
            with open(latest_file, 'w') as f:
                f.write(f"checkpoint-step-{step}")
    
    except Exception as e:
        if rank == 0:
            print(f"❌ Error saving checkpoint at step {step}: {e}")
    
    # 最终同步
    if dist.is_initialized():
        dist.barrier()

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

def wan_step(
    model_output: torch.Tensor,  # 模型预测的flow
    latents: torch.Tensor,       # 当前时间步的潜在表示 (16, 7, 64, 64)
    eta: float,                  # 控制随机性强度
    sigmas: torch.Tensor,        # sigma调度序列 (类似FLUX)
    index: int,                  # 当前时间步索引  
    prev_sample: torch.Tensor,   # 前一步的样本（用于GRPO重计算）
    grpo: bool,                  # True时会得到logprob
    sde_solver: bool,            # 使用SDE求解器
):
    """WAN的Flow Matching采样步骤，转换为SDE求解器支持GRPO"""
    
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma  # sigma差分
    
    # 确定性更新部分
    prev_sample_mean = latents + dsigma * model_output
    
    # 预测的原始样本
    pred_original_sample = latents - sigma * model_output
    
    delta_t = sigma - sigmas[index + 1]  # 时间差分
    # std_dev_t = eta * math.sqrt(abs(delta_t))  # 随机噪声的std
    std_dev_t = eta * torch.sqrt(delta_t)  # 根据hunyuan改的
    
    if sde_solver:  # 使用SDE求解器（和FLUX相同）
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / (sigma**2)  # 估计的得分
        log_term = -0.5 * eta**2 * score_estimate  # 对数项修正
        prev_sample_mean = prev_sample_mean + log_term * dsigma  # 修正的均值
    
    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if grpo:
        # 计算log概率
        log_prob = (-((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))) - math.log(std_dev_t + 1e-8) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        log_prob = log_prob.sum(dim=0)
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample

class WanPreprocessedDataset(torch.utils.data.Dataset):
    """加载预处理context的WAN数据集"""
    
    def __init__(self, processed_json_path):
        # 加载处理后的数据
        with open(processed_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} preprocessed items")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载numpy文件并转换为tensor
        context_numpy = np.load(item['context_path'])
        context = torch.from_numpy(context_numpy)  # shape: (L, C)
        
        return {
            'context': context,           # 单个tensor，shape (L, C)
            'caption': item['caption']    # 原始文本（用于调试）
        }

def wan_preprocessed_collate_function(batch):
    """修复的collate函数 - 返回List[Tensor]格式"""
    # 将每个context tensor放入列表中，这样就是List[Tensor]格式
    contexts = [item['context'] for item in batch]  # List[Tensor]，每个tensor shape (L, C)
    captions = [item['caption'] for item in batch]
    
    return {
        'contexts': contexts,  # List[Tensor]，符合WAN模型期望
        'captions': captions   # list of strings
    }

class WanDataset(torch.utils.data.Dataset):
    """WAN训练数据集"""
    
    def __init__(self, data_json_path, text_len=512):
        self.text_len = text_len
        
        # 加载数据
        with open(data_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 如果数据是字典格式，转换为列表
        if isinstance(self.data, dict):
            self.data = list(self.data.values())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 提取caption
        if isinstance(item, dict):
            caption = item.get('caption', item.get('text', 'A video'))
        else:
            caption = str(item)
        
        return {
            'caption': caption
        }

def wan_collate_function(batch):
    """WAN数据集的collate函数"""
    captions = [item['caption'] for item in batch]
    return captions

def save_video_and_prompt(video_frames, caption, rank, index, args, step=None):
    """
    保存视频文件和对应的prompt文本
    Args:
        video_frames: torch.Tensor, shape (C, T, H, W), 范围 [0, 1]
        caption: str, 对应的文本prompt
        rank: int, 当前进程的rank
        index: int, 当前batch的索引
        args: 配置参数
    """
    from datetime import datetime
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保video_frames是正确的格式 (C, T, H, W)
    if video_frames.dim() == 4:
        C, T, H, W = video_frames.shape
        
        # 转换为numpy格式 (T, H, W, C)
        video_np = video_frames.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
        video_np = (video_np * 255).astype(np.uint8)
        
        # 如果是单通道，扩展为3通道
        if C == 1:
            video_np = np.repeat(video_np, 3, axis=-1)
        
        # 1. 保存第一帧图像
        first_frame = video_np[0]  # (H, W, C)
        
        # 保存第一帧为PNG图像
        if C >= 3:
            first_frame_pil = Image.fromarray(first_frame)
        else:
            first_frame_pil = Image.fromarray(first_frame[:,:,0], mode='L')

            # 【修改】文件名包含step信息
        if step is not None:
            image_filename = f"wan_frame_step{step}_rank{rank}_batch{index}.png"
            video_filename = f"wan_video_step{step}_rank{rank}_batch{index}.mp4"
        else:
            image_filename = f"wan_frame_rank{rank}_batch{index}_{timestamp}.png"
            video_filename = f"wan_video_rank{rank}_batch{index}_{timestamp}.mp4"
        
        image_path = os.path.join("./images_mqvq_pix_14b", image_filename)
        video_path = os.path.join("./videos_mqvq_pix_14b", video_filename)

        first_frame_pil.save(image_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = args.video_fps if hasattr(args, 'video_fps') else 25  # 默认8fps
            
        out = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
            
        for t in range(T):
            frame = video_np[t]  # (H, W, C)
            # OpenCV使用BGR格式
            if C == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
            
        out.release()
        print(f"Video saved: {video_path}")

    else:
        print(f"Unexpected video_frames shape: {video_frames.shape}")

def create_video_grid(video_list, captions_list, output_path, fps=25):
    """
    创建视频网格，将多个视频排列在一起
    
    Args:
        video_list: List[torch.Tensor], 每个tensor shape (C, T, H, W)
        captions_list: List[str], 对应的caption列表
        output_path: str, 输出视频路径
        fps: int, 视频帧率
    """
    if not video_list:
        return
    
    # 确定网格大小
    num_videos = len(video_list)
    grid_size = int(np.ceil(np.sqrt(num_videos)))
    
    # 获取视频尺寸
    C, T, H, W = video_list[0].shape
    
    # 创建网格画布
    grid_H = H * grid_size
    grid_W = W * grid_size
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (grid_W, grid_H))
        
        for t in range(T):
            # 创建当前帧的网格
            grid_frame = np.zeros((grid_H, grid_W, 3), dtype=np.uint8)
            
            for i, video in enumerate(video_list):
                row = i // grid_size
                col = i % grid_size
                
                # 获取当前视频的当前帧
                frame = video[:, t, :, :].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                frame = (frame * 255).astype(np.uint8)
                
                if C == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                
                # 将帧放置到网格中
                start_h = row * H
                end_h = start_h + H
                start_w = col * W
                end_w = start_w + W
                
                grid_frame[start_h:end_h, start_w:end_w] = frame
            
            # 转换为BGR格式并写入视频
            grid_frame_bgr = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
            out.write(grid_frame_bgr)
        
        out.release()
        print(f"Video grid saved: {output_path}")
        
        # 保存对应的captions
        caption_path = output_path.replace('.mp4', '_captions.txt')
        with open(caption_path, 'w', encoding='utf-8') as f:
            for i, caption in enumerate(captions_list):
                f.write(f"Video {i+1}: {caption}\n")
        
    except Exception as e:
        print(f"Error creating video grid: {e}")

def run_wan_sample_step(
    args,
    latents,  # [(16, 7, 64, 64)]
    progress_bar, 
    sigma_schedule,  # 添加sigma_schedule
    transformer,
    context,
    context_null,
    seq_len,
    grpo_sample,
    guide_scale=5.0,
):
    """WAN采样步骤，修正CFG实现避免通道数错误"""
    if grpo_sample:
        all_latents = [latents[0]]  # 存储初始latent (16, 7, 64, 64)
        all_log_probs = []
        for i in progress_bar:
            B = len(context) if isinstance(context, list) else context.shape[0]
            # 确保设备一致
            device = latents[0].device
            
            # 使用sigma值计算timestep
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timestep = torch.full([B], timestep_value, device=device, dtype=torch.long)
            
            transformer.eval()
            
            # 【修正】序列化CFG，避免在latents维度合并
            with torch.autocast("cuda", torch.bfloat16 if args.use_bf16 else torch.float32):
                with torch.no_grad():
                    # 方法1：完全序列化CFG，分别计算条件和无条件
                    if args.use_sequential_cfg:
                        # 先计算条件预测
                        pred_cond = transformer(
                            x=latents,  # 保持原始格式 [(16, 7, 64, 64)]
                            t=timestep,
                            context=context,  # List[Tensor]
                            seq_len=seq_len,
                        )
                        
                        if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
                            model_output_cond = pred_cond['rgb'][0]
                        elif isinstance(pred_cond, list):
                            model_output_cond = pred_cond[0]
                        else:
                            model_output_cond = pred_cond
                        
                        # 立即清理显存
                        del pred_cond
                        torch.cuda.empty_cache()
                        
                        # 再计算无条件预测
                        pred_uncond = transformer(
                            x=latents,  # 保持原始格式 [(16, 7, 64, 64)]
                            t=timestep,
                            context=context_null,  # List[Tensor]
                            seq_len=seq_len
                        )
                        
                        if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                            model_output_uncond = pred_uncond['rgb'][0]
                        elif isinstance(pred_uncond, list):
                            model_output_uncond = pred_uncond[0]
                        else:
                            model_output_uncond = pred_uncond
                        
                        del pred_uncond
                        torch.cuda.empty_cache()
                    else:   
                        # 为条件和无条件分别创建timestep
                        timestep_cond = timestep
                        timestep_uncond = timestep
                        
                        # 为条件预测准备输入
                        pred_cond = transformer(
                            x=latents,  # [(16, 7, 64, 64)]
                            t=timestep_cond,
                            context=context,
                            seq_len=seq_len
                        )
                        
                        if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
                            model_output_cond = pred_cond['rgb'][0]
                        elif isinstance(pred_cond, list):
                            model_output_cond = pred_cond[0]
                        else:
                            model_output_cond = pred_cond
                        
                        # 为无条件预测准备输入
                        pred_uncond = transformer(
                            x=latents,  # [(16, 7, 64, 64)]
                            t=timestep_uncond,
                            context=context_null,
                            seq_len=seq_len
                        )
                        
                        if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                            model_output_uncond = pred_uncond['rgb'][0]
                        elif isinstance(pred_uncond, list):
                            model_output_uncond = pred_uncond[0]
                        else:
                            model_output_uncond = pred_uncond
                        
                        del pred_cond, pred_uncond
                
                # CFG组合
                model_output = model_output_uncond + guide_scale * (model_output_cond - model_output_uncond)
                del model_output_cond, model_output_uncond
                torch.cuda.empty_cache()

            # WAN的SDE采样步骤
            next_latents, pred_original, log_prob = wan_step(
                model_output, 
                latents[0].to(torch.float32),  # (16, 7, 64, 64)
                args.eta, 
                sigma_schedule,  # 传入sigma_schedule
                i, 
                prev_sample=None, 
                grpo=True, 
                sde_solver=True  # 启用SDE求解器
            )
            
            latents = [next_latents.to(torch.float32)]
            all_latents.append(latents[0])
            all_log_probs.append(log_prob) 
        
        final_latents = pred_original
        all_latents = torch.stack(all_latents, dim=0)  # (21, 16, 7, 64, 64)  timestep,C,T,H,W
        all_log_probs = torch.stack(all_log_probs, dim=0)  # (20, 7, 64, 64)  timestep,T,H,W
        return latents, final_latents, all_latents, all_log_probs


def save_video_file_for_videoalign(video_frames, video_path, args):
    """
    专门为VideoAlign保存视频文件
    Args:
        video_frames: torch.Tensor, shape (C, T, H, W), 范围 [0, 1]
        video_path: str, 视频保存路径
        args: 训练参数
    """
    if video_frames.dim() == 4:
        C, T, H, W = video_frames.shape
        
        # 转换为numpy格式 (T, H, W, C)
        video_np = video_frames.permute(1, 2, 3, 0).cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        # 如果是单通道，扩展为3通道
        if C == 1:
            video_np = np.repeat(video_np, 3, axis=-1)
        
        try:
            # 使用OpenCV保存视频
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = getattr(args, 'video_fps', 25)  # VideoAlign可能对帧率有要求
            
            out = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
            
            for t in range(T):
                frame = video_np[t]  # (H, W, C)
                # OpenCV使用BGR格式
                if C == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)
            
            out.release()
            print(f"VideoAlign video saved: {video_path}")
            
        except Exception as e:
            print(f"❌ Error saving VideoAlign video {video_path}: {e}")
            raise e

def sample_wan_reference_model(
    args,
    device, 
    transformer,
    vae,
    contexts,
    captions,
    context_null_single,
    # 新增hpsv2参数
    reward_model=None,
    tokenizer=None,
    preprocess_val=None,
    videoalign_inferencer=None,
    step=None
):
    """WAN参考模型采样，支持FP16优化"""
    # 视频参数
    frame_num = args.t
    size = (args.w, args.h)
    
    # 创建sigma调度（类似FLUX）
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, sample_steps + 1).to(device)
    
    # 应用时间偏移（如果需要）
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
    
    B = len(captions)
    batch_size = 1
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []
    all_dinov2_feature_maps = []

    # 【新增】分别存储VQ和MQ奖励
    all_vq_rewards = []
    all_mq_rewards = []

    # VAE参数
    vae_stride = [4, 8, 8]
    patch_size = [1, 2, 2]
    
    # 根据是否使用FP16选择数据类型
    latent_dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    if args.init_same_noise:
        latent_shape = (
            16,
            (frame_num - 1) // vae_stride[0] + 1,
            size[1] // vae_stride[1],
            size[0] // vae_stride[2]
        )
        input_latents = torch.randn(latent_shape, device=device, dtype=latent_dtype)

    for index, batch_idx in enumerate(batch_indices):
        batch_captions = [captions[i] for i in batch_idx]
        batch_contexts = [contexts[i].to(device) for i in batch_idx]
        # 【优化】直接复制预计算的context_null，以匹配当前批次大小
        batch_context_null = [context_null_single[0] for _ in batch_idx]
        
        if not args.init_same_noise:
            latent_shape = (
                16,
                (frame_num - 1) // vae_stride[0] + 1,
                size[1] // vae_stride[1],
                size[0] // vae_stride[2]
            )
            input_latents = torch.randn(latent_shape, device=device, dtype=latent_dtype)

        seq_len = math.ceil(
            (latent_shape[2] * latent_shape[3]) / (patch_size[1] * patch_size[2]) * latent_shape[1]
        )

        grpo_sample = True
        progress_bar = tqdm(range(0, args.sampling_steps), desc="WAN Sampling Progress")
        
        # 获取CFG引导强度
        guide_scale = getattr(args, 'guide_scale', 5.0)

        # 使用梯度检查点减少内存使用
        with torch.no_grad():      
            sampling_result = run_wan_sample_step(
                args,
                [input_latents],
                progress_bar,
                sigma_schedule,
                transformer,
                batch_contexts,
                batch_context_null,
                seq_len,
                grpo_sample,
                guide_scale,
            )
            _, final_latents, batch_latents, batch_log_probs = sampling_result

        batch_latents = batch_latents.unsqueeze(0)
        batch_log_probs = batch_log_probs.unsqueeze(0)
        
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)

        # VAE解码
        rank = int(os.environ.get("RANK", 0))
        
        with torch.inference_mode():
            # 使用适当的数据类型进行autocast
            autocast_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
            with torch.autocast("cuda", dtype=autocast_dtype):
                # 确保final_latents的数据类型正确
                final_latents_vae = final_latents.to(dtype=autocast_dtype)
                decoded_videos = vae.decode([final_latents_vae])
                video_frames = decoded_videos[0]
                
                # 后处理
                video_frames = (video_frames + 1.0) / 2.0
                video_frames = torch.clamp(video_frames, 0, 1)
                
                # 创建输出目录
                os.makedirs("./videos_mqvq_pix_14b", exist_ok=True)
                os.makedirs("./images_mqvq_pix_14b", exist_ok=True)
                # 【修改】保存视频文件，VideoAlign需要从文件路径读取
                if args.use_videoalign:
                    # VideoAlign需要从文件读取，所以必须保存
                    video_filename = f"wan_video_step{step}_rank{rank}_batch{index}.mp4"
                    video_path = os.path.join("./videos_mqvq_pix_14b", video_filename)
                    save_video_file_for_videoalign(video_frames, video_path, args)
                    
                    # 同时保存图片用于调试
                    save_video_and_prompt(video_frames, batch_captions[0], rank, index, args, step=step)
                else:
                    # 不使用VideoAlign时正常保存
                    save_video_and_prompt(video_frames, batch_captions[0], rank, index, args, step=step)

        # 计算像素级奖励
        latent_t, latent_h, latent_w = latent_shape[1],latent_shape[2], latent_shape[3]  # latent的空间尺寸
        target_size = (latent_h, latent_w)
        # 步骤1: 提取DINOv2特征图（保持原始视频尺寸处理）
        dinov2_feature_map = compute_dinov2_feature_map_reverse(
            video_frames,                                    # 原始尺寸视频帧
            target_size=target_size,                         # latent尺寸
            target_time=latent_t,                           # latent时间维度
            device=device,
            pca_method=getattr(args, 'dinov2_pca_method', 'weighted'),
            smooth_method="gaussian_strong",  # 使用强高斯平滑
            sigma=1.0
        )  # 返回 (T, H, W)，已经下采样到latent尺寸
        all_dinov2_feature_maps.append(dinov2_feature_map)
        # 【新增】VideoAlign奖励计算
        if args.use_videoalign and videoalign_inferencer is not None:
            print(f"Computing VideoAlign reward for video: {video_path}")
            try:
                with torch.no_grad():
                    # 获取视频文件的绝对路径
                    absolute_video_path = os.path.abspath(video_path)
                    
                    # 调用VideoAlign推理器
                    reward_result = videoalign_inferencer.reward(
                        [absolute_video_path],           # 视频路径列表
                        [batch_captions[0]],            # 对应的caption列表
                        use_norm=True,                  # 使用归一化
                    )
                    
                    # 提取VQ和MQ奖励
                    vq_reward = torch.tensor(reward_result[0]['VQ']).to(device)
                    mq_reward = torch.tensor(reward_result[0]['MQ']).to(device)
                    # ta_reward = torch.tensor(reward_result[0]['TA']).to(device)

                    all_vq_rewards.append(vq_reward.unsqueeze(0))   
                    all_mq_rewards.append(mq_reward.unsqueeze(0))

                    combined_reward = args.vq_coef * vq_reward + args.mq_coef * mq_reward
                    video_reward = combined_reward.item()
                    all_rewards.append(video_reward)

            except Exception as e:
                print(f"❌ VideoAlign reward computation failed: {e}")
                all_rewards.append(1)
        elif args.use_hpsv2 and reward_model is not None and tokenizer is not None and preprocess_val is not None:
            # 对每一帧都进行HPS v2评分
            with torch.no_grad():
                frame_rewards = []
                C, T, H, W = video_frames.shape  # (C, T, H, W)
                
                for t in range(T):  # 遍历每一帧
                    frame = video_frames[:, t, :, :]  # (C, H, W)
                    
                    # 转换为PIL图像
                    frame_np = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                    frame_np = (frame_np * 255).astype(np.uint8)
                    frame_pil = Image.fromarray(frame_np)
                    
                    # 预处理图像
                    image = preprocess_val(frame_pil).unsqueeze(0).to(device=device, non_blocking=True)
                    # 处理文本prompt
                    text = tokenizer([batch_captions[0]]).to(device=device, non_blocking=True)
                    
                   # 计算HPS分数
                    outputs = reward_model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                    frame_rewards.append(hps_score)
                 # 将所有帧的奖励汇总
                frame_rewards = torch.stack(frame_rewards, dim=0)  # (T, 1)
                # 计算视频整体奖励的策略选择
                if args.hps_aggregation == "mean":
                    video_reward = frame_rewards.mean(dim=0)  # 平均值
                elif args.hps_aggregation == "max":
                    video_reward = frame_rewards.max(dim=0)[0]  # 最大值
                elif args.hps_aggregation == "min":
                    video_reward = frame_rewards.min(dim=0)[0]  # 最小值
                elif args.hps_aggregation == "weighted":
                    # 时间加权：后面的帧权重更高
                    weights = torch.linspace(0.5, 1.0, T, device='cpu').unsqueeze(1)  # (T, 1)
                    video_reward = (frame_rewards * weights).sum(dim=0) / weights.sum()
                else:  # 默认使用平均值
                    video_reward = frame_rewards.mean(dim=0)
                
                # 确保video_reward在正确的设备上且为正确的数据类型
                video_reward = video_reward.to(device, dtype=torch.float32)
                print(f"Video reward shape: {video_reward.shape}, dtype: {video_reward.dtype}")  # 应该是 (1,)
                print(f"Dinov2 feature map shape: {dinov2_feature_map.shape}, dtype: {dinov2_feature_map.dtype}")  # (T, H, W)
                video_reward = video_reward * dinov2_feature_map
                all_rewards.append(video_reward)
        else:
            # replace with rule-based reward
            all_rewards.append(1) 

        del final_latents, video_frames
        torch.cuda.empty_cache()

    if len(all_latents) > 1:
        all_latents = torch.cat(all_latents, dim=0)
        all_log_probs = torch.cat(all_log_probs, dim=0)
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)  # (B,)
        all_dinov2_feature_maps = torch.stack(all_dinov2_feature_maps, dim=0)  # (B, T, H, W)
        if args.use_videoalign and len(all_vq_rewards) > 0 and len(all_mq_rewards) > 0:
            all_vq_rewards = torch.cat(all_vq_rewards, dim=0)
            all_mq_rewards = torch.cat(all_mq_rewards, dim=0)
        else:
            all_vq_rewards = None
            all_mq_rewards = None
        # print('log_prob:',all_log_probs.shape, 'latent:',all_latents.shape, 'reward:',all_rewards.shape,"all_dinov2_feature_maps",all_dinov2_feature_maps.shape)
    else:
        all_latents = all_latents[0]
        all_log_probs = all_log_probs[0]
        all_rewards = all_rewards[0]
        all_dinov2_feature_maps = all_dinov2_feature_maps[0].unsqueeze(0)
    
    return all_rewards, all_latents, all_log_probs, sigma_schedule, all_dinov2_feature_maps, all_vq_rewards, all_mq_rewards


def grpo_wan_one_step(
    args,
    latents,
    pre_latents,
    context,
    context_null,
    seq_len,
    transformer,
    timesteps,
    i,
    sigma_schedule,
    guide_scale=5.0,
):
    """GRPO的单步训练，修正CFG实现"""
    B = len(context) if isinstance(context, list) else context.shape[0]
    
    # 确保latents维度正确：(16, 7, 64, 64)
    if latents.dim() == 5:
        latents = latents.squeeze(0)
    if pre_latents.dim() == 5:
        pre_latents = pre_latents.squeeze(0)
    
    if latents.shape[0] != 16:
        raise ValueError(f"Expected 16 channels, got {latents.shape[0]} channels")
    
    # 使用适当的数据类型进行autocast
    autocast_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    with torch.autocast("cuda", dtype=autocast_dtype):
        # 【关键修改】先计算无条件预测，并冻结梯度
        with torch.no_grad():  # 冻结无条件分支
            pred_uncond = transformer(
                x=[latents],
                t=timesteps,
                context=context_null,  # 无条件
                seq_len=seq_len
            )
            
            # 处理无条件预测输出
            if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                model_output_uncond = pred_uncond['rgb'][0].detach()  # 确保detach
            elif isinstance(pred_uncond, list):
                model_output_uncond = pred_uncond[0].detach()
            else:
                model_output_uncond = pred_uncond.detach()
                
            del pred_uncond
            torch.cuda.empty_cache()
        
        transformer.train()
        # 计算条件预测
        pred_cond = transformer(
            x=[latents],  # 保持List[Tensor]格式
            t=timesteps,
            context=context,  # List[Tensor]
            seq_len=seq_len
        )
            
        # 处理条件预测输出
        if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
            model_output_cond = pred_cond['rgb'][0]
        elif isinstance(pred_cond, list):
            model_output_cond = pred_cond[0]
        else:
            model_output_cond = pred_cond
            
        # 立即清理
        del pred_cond
        torch.cuda.empty_cache()
        
        # CFG组合
        model_output = model_output_uncond + guide_scale * (model_output_cond - model_output_uncond)
        del model_output_cond, model_output_uncond

    # 确保数据类型一致性
    computation_dtype = torch.float32
    _, _, log_prob = wan_step(
        model_output.to(computation_dtype), 
        latents.to(computation_dtype), 
        args.eta, 
        sigma_schedule,
        i, 
        prev_sample=pre_latents.to(computation_dtype), 
        grpo=True, 
        sde_solver=True
    )
    
    return log_prob

def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_wan_one_step(
    args,
    device,
    transformer,
    vae,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
    context_null_single,
    reward_model=None,
    tokenizer=None,
    preprocess_val=None,
    videoalign_inferencer=None,
    step=None,
):
    """WAN的一步训练，处理(B, T, C, F, H, W)维度"""
    total_loss = 0.0
    optimizer.zero_grad()
    
    batch = next(loader)
    contexts = batch['contexts']  # List[Tensor]格式
    captions = batch['captions']  # 原始文本
    
    if args.use_group:
        # 扩展contexts和captions
        expanded_contexts = []
        expanded_captions = []
        for context, caption in zip(contexts, captions):
            for _ in range(args.num_generations):
                expanded_contexts.append(context)
                expanded_captions.append(caption)
        contexts = expanded_contexts
        captions = expanded_captions

    # 在某些步骤中启用注意力可视化
    step = getattr(args, '_current_step', 0)  # 需要从外部传入当前步数

    # 正常采样，不提取注意力权重
    reward, all_latents, all_log_probs, sigma_schedule, dinov2_feature_maps, vq_rewards, mq_rewards = sample_wan_reference_model(
        args, device, transformer, vae, contexts, captions, context_null_single,
        reward_model, tokenizer, preprocess_val, videoalign_inferencer, step=step,
    )
    
    batch_size = all_latents.shape[0]

    context_null = [context_null_single[0] for _ in range(batch_size)]
    
    # 【关键修正】参考Hunyuan的timesteps构建方式
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    timesteps_tensor = torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
    
    # 【关键修正】参考Hunyuan构建samples，注意维度处理
    samples = {
        "timesteps": timesteps_tensor.detach().clone()[:, :-1],           # (B, steps)
        "latents": all_latents[:, :-1][:,:-1],                  # (B, steps, C, T, H, W) - 当前状态  
        "next_latents": all_latents[:, 1:][:,:-1],              # (B, steps, C, T, H, W) - 下一状态
        "log_probs": all_log_probs[:,:-1],                      # (B, steps, T, H, W)
        "rewards": reward.to(torch.float32),             # (B, T, H, W) - 奖励
        "dinov2_feature_maps": dinov2_feature_maps,      # (B, T, H, W) - DINOv2特征图
        "contexts": contexts,                            # List[Tensor]
        "context_null": context_null,                    # List[Tensor]
        "sigma_schedule": sigma_schedule,
    }
    if vq_rewards is not None and mq_rewards is not None:
        samples["vq_rewards"] = vq_rewards.to(torch.float32)
        samples["mq_rewards"] = mq_rewards.to(torch.float32)

    gathered_reward = gather_tensor(samples["rewards"]) # (B_total[8*4=32], T, H, W)

    if vq_rewards is not None and mq_rewards is not None:
        gathered_vq_reward = gather_tensor(samples["vq_rewards"])
        gathered_mq_reward = gather_tensor(samples["mq_rewards"])
        if dist.get_rank() == 0:
            if args.use_videoalign:
                print("gathered_vq_reward", gathered_vq_reward.mean().item())
                print("gathered_mq_reward", gathered_mq_reward.mean().item())
                with open('./wan_videoalign_vq_reward.txt', 'a') as f: 
                    f.write(f"{gathered_vq_reward.mean().item()}\n")
                with open('./wan_videoalign_mq_reward.txt', 'a') as f: 
                    f.write(f"{gathered_mq_reward.mean().item()}\n")
    else:
        if dist.get_rank() == 0:
            if args.use_videoalign:
                print("gathered_videoalign_reward", gathered_reward.mean().item())
                with open('./wan_videoalign_reward.txt', 'a') as f: 
                    f.write(f"{gathered_reward.mean().item()}\n")
            elif args.use_hpsv2:
                with open('./wan_hps_reward.txt', 'a') as f: 
                    f.write(f"{gathered_reward.mean().item()}\n")
            else:
                print("gathered_color_reward", gathered_reward)
                with open('./wan_color_reward.txt', 'a') as f: 
                    f.write(f"{gathered_reward.mean().item()}\n")

    if args.use_group:
        n = len(samples["rewards"]) // args.num_generations
        advantages = torch.zeros_like(samples["dinov2_feature_maps"])

        if vq_rewards is not None and mq_rewards is not None:
            vq_advantages = torch.zeros_like(samples["dinov2_feature_maps"])
            mq_advantages = torch.zeros_like(samples["dinov2_feature_maps"])
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations

            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8

            scalar_advantages = (group_rewards - group_mean) / group_std

            if vq_rewards is not None and mq_rewards is not None:
                group_vq_rewards = samples["vq_rewards"][start_idx:end_idx]
                group_vq_mean = group_vq_rewards.mean()
                group_vq_std = group_vq_rewards.std() + 1e-8
                vq_scalar_advantages = (group_vq_rewards - group_vq_mean) / group_vq_std

                group_mq_rewards = samples["mq_rewards"][start_idx:end_idx]
                group_mq_mean = group_mq_rewards.mean()
                group_mq_std = group_mq_rewards.std() + 1e-8
                mq_scalar_advantages = (group_mq_rewards - group_mq_mean) / group_mq_std
            
            for j in range(args.num_generations):
                sample_idx = start_idx + j
                scalar_adv = scalar_advantages[j]
                feature_map = samples["dinov2_feature_maps"][sample_idx]
                weighted_advantage = scalar_adv * feature_map
                advantages[sample_idx] = weighted_advantage

                if vq_rewards is not None and mq_rewards is not None:
                    vq_scalar_adv = vq_scalar_advantages[j]
                    mq_scalar_adv = mq_scalar_advantages[j]
                    vq_weighted_advantage = vq_scalar_adv * feature_map
                    mq_weighted_advantage = mq_scalar_adv * feature_map
                    vq_advantages[sample_idx] = vq_weighted_advantage
                    mq_advantages[sample_idx] = mq_weighted_advantage

        samples["advantages"] = advantages
        if vq_rewards is not None and mq_rewards is not None:
            samples["vq_advantages"] = vq_advantages
            samples["mq_advantages"] = mq_advantages
    else:
        # 全局标准化
        scalar_advantages = (samples["rewards"] - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)
        advantages = torch.zeros_like(samples["dinov2_feature_maps"])
        
        if vq_rewards is not None and mq_rewards is not None:
            vq_scalar_advantages = (samples["vq_rewards"] - gathered_vq_reward.mean()) / (gathered_vq_reward.std() + 1e-8)
            mq_scalar_advantages = (samples["mq_rewards"] - gathered_mq_reward.mean()) / (gathered_mq_reward.std() + 1e-8)
            vq_advantages = torch.zeros_like(samples["dinov2_feature_maps"])
            mq_advantages = torch.zeros_like(samples["dinov2_feature_maps"])
        
        for i in range(batch_size):
            scalar_adv = scalar_advantages[i]
            feature_map = samples["dinov2_feature_maps"][i]
            weighted_advantage = scalar_adv * feature_map
            advantages[i] = weighted_advantage
            
            if vq_rewards is not None and mq_rewards is not None:
                vq_scalar_adv = vq_scalar_advantages[i]
                mq_scalar_adv = mq_scalar_advantages[i]
                vq_weighted_advantage = vq_scalar_adv * feature_map
                mq_weighted_advantage = mq_scalar_adv * feature_map
                vq_advantages[i] = vq_weighted_advantage
                mq_advantages[i] = mq_weighted_advantage
        
        samples["advantages"] = advantages
        if vq_rewards is not None and mq_rewards is not None:
            samples["vq_advantages"] = vq_advantages
            samples["mq_advantages"] = mq_advantages

    if args.bestofn > 0 and args.num_generations != args.bestofn:
        print("use best of n !!!!!!!!!!")
        total_scores = samples["rewards"]
        sorted_indices = torch.argsort(total_scores)
        top_indices = sorted_indices[-args.bestofn//2:]     
        bottom_indices = sorted_indices[:args.bestofn//2]     
        selected_indices = torch.cat([top_indices, bottom_indices])
        shuffled_order = torch.randperm(len(selected_indices), device=selected_indices.device)
        selected_indices = selected_indices[shuffled_order]
        
        # 选择样本
        for key in ["timesteps", "latents", "next_latents", "log_probs", "rewards", "advantages"]:
            samples[key] = samples[key][selected_indices]
        
        # 更新contexts
        new_contexts = [contexts[i] for i in selected_indices.cpu().numpy()]
        new_context_null = [context_null[i] for i in selected_indices.cpu().numpy()]
        samples["contexts"] = new_contexts
        samples["context_null"] = new_context_null
        batch_size = len(selected_indices)
    perms = torch.stack([
        torch.randperm(len(samples["timesteps"][0])) 
        for _ in range(batch_size)
    ]).to(device)
    
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device)[:, None],
            perms,
        ]
    
    samples_batched = {
        k: v.unsqueeze(1) if k in ["timesteps", "latents", "next_latents", "log_probs"] else v
        for k, v in samples.items()
    }
    
    samples_batched_list = []
    for i in range(batch_size):
        sample_dict = {
            "timesteps": samples_batched["timesteps"][i],      # (1, steps)
            "latents": samples_batched["latents"][i],          # (1, steps, C, T, H, W)
            "next_latents": samples_batched["next_latents"][i], # (1, steps, C, T, H, W)
            "log_probs": samples_batched["log_probs"][i],      # (1, steps, T, H, W)
            "rewards": samples["rewards"][i],                   # (T, H, W)
            "advantages": samples["advantages"][i],             # (T, H, W)
            "contexts": [samples["contexts"][i]],               # List[Tensor]
            "context_null": [samples["context_null"][i]],       # List[Tensor]
            "sigma_schedule": sigma_schedule,
        }
        
        if vq_rewards is not None and mq_rewards is not None:
            sample_dict.update({
                "vq_rewards": samples["vq_rewards"][i],
                "mq_rewards": samples["mq_rewards"][i],
                "vq_advantages": samples["vq_advantages"][i],
                "mq_advantages": samples["mq_advantages"][i],
            })
        samples_batched_list.append(sample_dict)

    train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)
    grad_norm = None
    print(f"DEBUG: Processing {len(samples_batched_list)} samples")

    for i, sample in enumerate(samples_batched_list):
        for step_idx in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            
            current_latents = sample["latents"][0, step_idx]      # (C, T, H, W)
            next_latents = sample["next_latents"][0, step_idx]    # (C, T, H, W)
            current_timesteps = sample["timesteps"][0, step_idx]  # scalar
            current_log_probs = sample["log_probs"][0, step_idx]  # scalar
            
            # 确保latents维度正确
            if current_latents.shape[0] != 16:
                raise ValueError(f"Expected 16 channels, got {current_latents.shape[0]} channels")

            # 计算序列长度
            latent_shape = current_latents.shape  # (C, T, H, W)
            seq_len = math.ceil(
                (latent_shape[2] * latent_shape[3]) / (2 * 2) * latent_shape[1]
            )
            
            # GRPO单步
            new_log_probs = grpo_wan_one_step(
                args,
                current_latents,           # (C, T, H, W)
                next_latents,             # (C, T, H, W)
                sample["contexts"],       # List[Tensor]
                sample["context_null"],   # List[Tensor]
                seq_len,
                transformer,
                current_timesteps.unsqueeze(0),  # (1,)
                perms[i][step_idx],
                sigma_schedule, 
                getattr(args, 'guide_scale', 5.0),
            )

            ratio = torch.exp(new_log_probs - current_log_probs)

            if vq_rewards is not None and mq_rewards is not None:
                # VQ损失
                vq_advantages = torch.clamp(sample["vq_advantages"], -adv_clip_max, adv_clip_max)
                vq_unclipped_loss = -vq_advantages * ratio
                vq_clipped_loss = -vq_advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                vq_loss = torch.mean(torch.maximum(vq_unclipped_loss, vq_clipped_loss))
                
                # MQ损失
                mq_advantages = torch.clamp(sample["mq_advantages"], -adv_clip_max, adv_clip_max)
                mq_unclipped_loss = -mq_advantages * ratio
                mq_clipped_loss = -mq_advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                mq_loss = torch.mean(torch.maximum(mq_unclipped_loss, mq_clipped_loss))
                
                # 综合损失
                loss = (args.vq_coef * vq_loss + args.mq_coef * mq_loss) / (args.gradient_accumulation_steps * train_timesteps)
                
                # 调试信息
                if dist.get_rank() % 8 == 0:
                    print(f"VQ loss: {vq_loss.item():.6f}, MQ loss: {mq_loss.item():.6f}")
            else:
                advantages = torch.clamp(sample["advantages"], -adv_clip_max, adv_clip_max)
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
            
        if (i + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        if dist.get_rank() % 8 == 0:
            print("ratio", ratio.mean().item())
            print("final loss", loss.item())
        
        dist.barrier()
    
    # 【修改】返回值，包含VQ和MQ奖励
    return_data = {
        'total_loss': total_loss,
        'grad_norm': grad_norm.item() if grad_norm is not None else 0.0,
        'rewards': samples["rewards"]
    }
    
    if vq_rewards is not None and mq_rewards is not None:
        return_data.update({
            'vq_rewards': samples["vq_rewards"],
            'mq_rewards': samples["mq_rewards"]
        })
    
    return return_data

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed + rank)

    if args.use_bf16:
        model_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
        precision_name = "BF16"
    else:
        model_dtype = torch.float32
        autocast_dtype = torch.float32
        precision_name = "FP32"

    # 创建输出目录
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 【新增】VideoAlign奖励模型初始化
    videoalign_inferencer = None
    if args.use_videoalign:
        print("Initializing VideoAlign reward model...")
        from rewardmodel.video_align.inference import VideoVLMRewardInference
        load_from_pretrained = os.getenv("VIDEOALIGN_CHECKPOINT", "checkpoints/videoalign")
        dtype = torch.bfloat16 if args.use_bf16 else torch.float32
        videoalign_inferencer = VideoVLMRewardInference(
            load_from_pretrained, 
            device=f'cuda:{device}', 
            dtype=dtype
        )
        print(f"✓ VideoAlign reward model loaded successfully")
        print(f"  - Device: cuda:{device}")
        print(f"  - Dtype: {dtype}")
    
    # 初始化reward跟踪器（只在rank 0上）
    reward_tracker = None
    if rank == 0:
        reward_tracker = RewardTracker(
            save_dir=os.path.join(args.output_dir, "realtime_plots"),
            save_every=1,  # 每步都保存
            max_points=10000  # 最多保存10000个数据点
        )
        print(f"Real-time reward tracker initialized, plots will be saved to: {reward_tracker.save_dir}")

    print(f"--> loading WAN model from {args.pretrained_model_name_or_path}")
    
    # 加载WAN配置
    from wan.configs import t2v_1_3B
    config = t2v_1_3B
    
    # 加载WAN模型 - 使用内存优化配置
    transformer = WanModel.from_pretrained(args.pretrained_model_name_or_path)
    
    # 根据args.use_bf16决定数据类型
    model_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    transformer = transformer.to(dtype=model_dtype)
    transformer = transformer.to(device)
    
    # 添加参数量统计
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    
    print(f"=== WAN Transformer Model Statistics ===")
    print(f"  Total parameters: {total_params / 1e9:.2f} B")
    print(f"  Trainable parameters: {trainable_params / 1e9:.2f} B")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.1f}%")
    print(f"  Model dtype: {model_dtype}")
    print(f"  Precision: {precision_name}")

    # 使用FSDP包装模型 - 优化配置
    from torch.distributed.fsdp.api import CPUOffload, MixedPrecision, BackwardPrefetch, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy, transformer_auto_wrap_policy
    from torch.distributed._composable.fsdp import fully_shard
    
    # 配置CPU offload（如果启用）
    cpu_offload = None
    if args.cpu_offload:
        cpu_offload = CPUOffload(offload_params=True)
        print("CPU offload enabled for parameters")
    
    # 配置混合精度
    if args.use_bf16:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,      # 参数使用BF16
            reduce_dtype=torch.bfloat16,      # 梯度聚合使用FP32（更稳定）
            buffer_dtype=torch.bfloat16,     # buffer使用BF16
            cast_forward_inputs=True,        # 自动转换输入类型
        )
        print("Using BF16 mixed precision with FP32 gradient reduction")
    else:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        print("Using FP32 precision")
    
    # 【关键修改】完全分片的包装策略 - 包装所有模块
    def wan_full_shard_wrap_policy(module, recurse, nonwrapped_numel):
        """
        WAN模型完全分片策略 - 包装所有可能的模块以实现参数完全平分
        """
        # 1. 降低参数阈值，包装更多模块
        if nonwrapped_numel >= 1e3:  # 10K参数就包装，而不是100K
            return True
        # 2. 包装所有WAN的主要组件
        module_name = str(type(module).__name__)
        # 包装所有attention相关模块
        attention_modules = [
            'WanAttentionBlock', 'WanSelfAttention', 'WanT2VCrossAttention', 
            'WanI2VCrossAttention', 'attention', 'Attention'
        ]
        if any(name in module_name for name in attention_modules):
            return True
        # 包装所有Linear层（如果参数足够多）
        if module_name == 'Linear' and nonwrapped_numel >= 1e3:  # 5K参数以上的Linear层
            return True
        # 包装所有卷积层
        conv_modules = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']
        if any(name in module_name for name in conv_modules):
            return True
        # 包装embedding和projection层
        embed_modules = ['Embedding', 'MLPProj', 'Head']
        if any(name in module_name for name in embed_modules):
            return True
        # 包装normalization层（如果参数足够多）
        norm_modules = ['LayerNorm', 'WanLayerNorm', 'WanRMSNorm', 'RMSNorm', 'GroupNorm', 'BatchNorm']
        if any(name in module_name for name in norm_modules) and nonwrapped_numel >= 1e3:
            return True
        # 包装FFN/MLP相关
        ffn_modules = ['Sequential', 'GELU', 'SiLU', 'ReLU']
        if any(name in module_name for name in ffn_modules) and nonwrapped_numel >= 1e3:
            return True
        # 如果模块有超过1000个参数，就包装
        if nonwrapped_numel >= 1e3:
            return True
        return False

    # 【强制】使用FULL_SHARD策略
    if args.no_sharding:
        sharding_strategy = ShardingStrategy.NO_SHARD
        print("FSDP is running in NO_SHARD (DDP-like) mode. Sampling will be fast.")
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
        print(f"Using ZeRO-3 (FULL_SHARD) - each GPU will store ~1/{world_size} of parameters. Sampling may be slow.")
    
    # 清理显存
    torch.cuda.empty_cache()

    # 【关键】导入梯度检查点包装
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        apply_activation_checkpointing,
    )
    from wan.modules.model import WanAttentionBlock # 确保导入了核心模块
    check_fn = lambda m: isinstance(m, WanAttentionBlock)
    if args.enable_gradient_checkpointing:
        print("Enabling gradient checkpointing for WanAttentionBlock...")
        apply_activation_checkpointing(
            transformer,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn
        )

    print("Applying FSDP with optimized configuration...")
    transformer = FSDP(
        transformer,
        auto_wrap_policy=wan_full_shard_wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=local_rank,
        cpu_offload=cpu_offload,
        sharding_strategy=sharding_strategy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=False,
        sync_module_states=True,
        # forward_prefetch=True if args.enable_sequence_parallel else False,
        ignored_modules=None,
    )
    print("✓ FSDP successfully applied to transformer")
    # 【验证】检查分片效果
    if rank == 0:
        # 统计实际的参数分布
        local_params = sum(p.numel() for p in transformer.parameters() if p.is_meta == False)
            
        print(f"=== Full Shard Verification ===")
        print(f"  Original total parameters: {total_params / 1e9:.2f}B")
        print(f"  Local parameters per GPU: {local_params / 1e9:.2f}B")
        print(f"  Target per GPU: {total_params / world_size / 1e9:.2f}B")
        print(f"  Sharding efficiency: {(local_params / (total_params / world_size)) * 100:.1f}%")
    
    # 清理显存
    torch.cuda.empty_cache()
    
    # 加载VAE - 如果启用fp16也使用fp16
    vae_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    vae = WanVAE(
        vae_pth=os.path.join(args.pretrained_model_name_or_path, config.vae_checkpoint),
        device=device
    )
    vae.model = vae.model.to(dtype=vae_dtype)
    print(f"VAE dtype: {vae_dtype}")

    print(f"--> WAN model loaded with memory optimizations")

    # 【优化】预先计算全局不变的 context_null，然后删除T5编码器
    print("Pre-calculating global context_null...")
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device('cpu'),  # 始终在CPU上加载
        checkpoint_path=os.path.join(args.pretrained_model_name_or_path, config.t5_checkpoint),
        tokenizer_path=os.path.join(args.pretrained_model_name_or_path, config.t5_tokenizer),
    )
    
    neg_prompt = getattr(args, 'neg_prompt', "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    
    # 只计算一次，batch size为1
    if args.use_bf16:
        text_encoder.model.to(device) # 临时移到GPU计算
        context_null_single = text_encoder([neg_prompt], device)
        text_encoder.model.cpu() # 移回CPU
    else:
        context_null_single = text_encoder([neg_prompt], torch.device('cpu'))
    
    # 将其移动到目标设备上，以便后续高效复制
    context_null_single = [t.to(device) for t in context_null_single]
    
    # 删除T5编码器以释放CPU内存
    del text_encoder
    torch.cuda.empty_cache() # 清理因T5移动产生的缓存
    print("✓ Global context_null pre-calculated and T5 encoder has been deleted.")

    reward_model = None
    tokenizer = None
    preprocess_val = None

    # 添加 HPSv2 初始化
    if args.use_hpsv2:
        if create_model_and_transforms is None or get_tokenizer is None:
            raise ImportError("HPSv2 modules not found. Please install HPSv2.")
            
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
        cp = "./hps_ckpt/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(cp, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        tokenizer = get_tokenizer('ViT-H-14')
        reward_model = model.to(device)
        reward_model.eval()

        print(f"HPS aggregation method: {args.hps_aggregation}")

    transformer.train()

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    print("Using standard AdamW optimizer (no CPU offload to avoid gradient issues)")
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # 学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=-1,
    )

    # 使用预处理数据集
    train_dataset = WanPreprocessedDataset(args.data_json_path)
    sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=wan_preprocessed_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # 训练信息
    total_batch_size = args.train_batch_size * world_size * args.gradient_accumulation_steps
    print("***** Running WAN GRPO training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Dataloader size = {len(train_dataloader)}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps per epoch = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, 100000),
        initial=0,
        desc="Steps",
        disable=local_rank > 0,
    )

    # 创建数据加载器迭代器
    def get_loader_iterator():
        while True:
            for batch in train_dataloader:
                yield batch

    loader = get_loader_iterator()

    for epoch in range(1000000):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for step in range(1, args.max_train_steps + 1):
            # 在第1步和每5步保存checkpoint
            args._current_step = step
            if step % 10 == 0:
                save_checkpoint_simple(
                    model=transformer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step=step,
                    epoch=epoch,
                    rank=rank,
                    output_dir=args.output_dir,
                    args=args
                )
            train_result = train_wan_one_step(
                args,
                device, 
                transformer,
                vae,
                optimizer,
                lr_scheduler,
                loader,
                args.max_grad_norm,
                context_null_single,
                reward_model,
                tokenizer,
                preprocess_val,
                videoalign_inferencer,
                step=step,
            )
            # 【修改】提取训练结果
            loss = train_result['total_loss']
            grad_norm = train_result['grad_norm']
            batch_rewards = train_result['rewards']
            vq_rewards = train_result.get('vq_rewards', None)
            mq_rewards = train_result.get('mq_rewards', None)

            # 实时更新图表（只在rank 0上）
            if rank == 0 and reward_tracker is not None:
                n = len(batch_rewards) // args.num_generations
                group_means = []
                group_stds = []
                advantages = []
                for i in range(n):
                    start_idx = i * args.num_generations
                    end_idx = (i + 1) * args.num_generations
                    group_rewards = batch_rewards[start_idx:end_idx]
                    group_mean = group_rewards.mean().item()
                    group_std = group_rewards.std().item()
                    group_means.append(group_mean)
                    group_stds.append(group_std)
                    advantages.extend(((group_rewards - group_mean) / (group_std + 1e-8)).cpu().numpy())
                # 这里只记录最后一组，也可以记录均值
                group_mean = np.mean(group_means)
                group_std = np.mean(group_stds)
                advantage = np.mean(advantages)

                reward_tracker.add_data(
                    step=step,
                    loss=loss,
                    reward_tensor=batch_rewards,
                    grad_norm=grad_norm,
                    advantage=advantage,
                    vq_reward_tensor=vq_rewards,
                    mq_reward_tensor=mq_rewards,
                )
                # 显示保存信息
                print(f"Step {step}: Plots saved to {reward_tracker.save_dir}")

            # 更新进度条
            if rank == 0:
                progress_bar.update(1)
    
    # 清理分布式进程组
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 数据集和数据加载器
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    
    # 模型路径
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # 验证和日志
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # 优化器和训练
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # 学习率调度器
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    # GRPO训练参数
    parser.add_argument("--h", type=int, default=512, help="video height")
    parser.add_argument("--w", type=int, default=512, help="video width")
    parser.add_argument("--t", type=int, default=25, help="video length")
    parser.add_argument("--sampling_steps", type=int, default=20, help="sampling steps")
    parser.add_argument("--eta", type=float, default=0.3, help="noise eta")
    parser.add_argument("--sampler_seed", type=int, default=42, help="seed of sampler")
    parser.add_argument("--use_group", action="store_true", default=False)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--init_same_noise", action="store_true", default=False)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--timestep_fraction", type=float, default=1.0)
    parser.add_argument("--clip_range", type=float, default=1e-4)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--bestofn", type=int, default=8)

    # 内存优化参数
    parser.add_argument("--use_bf16", action="store_true", default=False, 
                       help="Use BF16 mixed precision training")
    parser.add_argument("--cpu_offload", action="store_true", default=False,
                       help="Offload model parameters to CPU")
    parser.add_argument("--use_zero2", action="store_true", default=False,
                       help="Use ZeRO-2 instead of ZeRO-3")
    
    # 视频保存相关参数
    parser.add_argument("--video_fps", type=int, default=25, help="保存视频的帧率")
    parser.add_argument("--save_video_every_steps", type=int, default=100, help="每隔多少步保存一次视频")

    parser.add_argument("--save_plot_every_steps", type=int, default=50, help="每隔多少步保存一次reward图表")
    
    parser.add_argument("--guide_scale", type=float, default=5.0, 
                   help="Classifier-free guidance scale")
    parser.add_argument("--neg_prompt", type=str, 
                   default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                   help="Negative prompt for CFG")
    # 添加 HPSv2 相关参数
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model for video generation",
    )
    
    parser.add_argument(
        "--hps_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "min", "weighted"],
        help="How to aggregate HPS scores across video frames: mean, max, min, or weighted (later frames have higher weight)",
    )
    parser.add_argument(
        "--use_dbcnn_quality",
        action="store_true",
        default=False,
        help="使用 pyiqa DBCNN 作为视频画质 reward (单标量, 越高越好)",
    )
    parser.add_argument(
        "--dbcnn_pool",
        type=str,
        default="max",
        choices=["mean", "max", "median"],
        help="DBCNN 帧分数聚合方式",
    )
    # 序列并行和内存优化参数
    parser.add_argument("--use_sequential_cfg", action="store_true", default=False,
                       help="Use sequential CFG to save memory")
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", default=True,
                       help="Enable gradient checkpointing")
    # 【新增参数】
    parser.add_argument("--no_sharding", action="store_true", default=False,
                       help="Run FSDP in DDP-like mode (NO_SHARD) for fast sampling, disables parameter sharding.")

    # 【新增】VideoAlign相关参数
    parser.add_argument(
        "--use_videoalign",
        action="store_true",
        default=False,
        help="Use VideoAlign reward model for video generation",
    )
    
    parser.add_argument(
        "--vq_coef",
        type=float,
        default=1.0,
        help="Coefficient for VQ (Video Quality) reward from VideoAlign",
    )
    
    parser.add_argument(
        "--mq_coef",
        type=float,
        default=1.0,
        help="Coefficient for MQ (Motion Quality) reward from VideoAlign",
    )
    
    args = parser.parse_args()
    main(args)
