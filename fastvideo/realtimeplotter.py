import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
from collections import deque
import os
from pathlib import Path

class RealTimePlotter:
    """实时绘制训练指标的类"""
    
    def __init__(self, save_dir: str, max_points: int = 10000):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_points = max_points
        
        # 存储数据
        self.steps = deque(maxlen=max_points)
        self.losses = deque(maxlen=max_points)
        self.rewards = deque(maxlen=max_points)
        self.grad_norms = deque(maxlen=max_points)
        self.step_times = deque(maxlen=max_points)
        
        # 文件路径
        self.loss_reward_path = self.save_dir / "loss_reward_curve.png"
        self.grad_norm_path = self.save_dir / "grad_norm_curve.png"
        self.step_time_path = self.save_dir / "step_time_curve.png"
        self.data_path = self.save_dir / "training_data.txt"
        
        print(f"RealTimePlotter initialized, plots will be saved to: {self.save_dir}")
    
    def add_data(self, step: int, loss: float, reward: float, grad_norm: float, step_time: float):
        """添加新的数据点"""
        self.steps.append(step)
        self.losses.append(loss)
        self.rewards.append(reward)
        self.grad_norms.append(grad_norm)
        self.step_times.append(step_time)
    
    def update_plots(self):
        """更新所有图表"""
        if len(self.steps) < 2:
            return
        
        try:
            # 转换为numpy数组
            steps = np.array(list(self.steps))
            losses = np.array(list(self.losses))
            rewards = np.array(list(self.rewards))
            grad_norms = np.array(list(self.grad_norms))
            step_times = np.array(list(self.step_times))
            
            # 1. 绘制Loss和Reward曲线
            self._plot_loss_reward(steps, losses, rewards)
            
            # 2. 绘制梯度范数曲线
            self._plot_grad_norm(steps, grad_norms)
            
            # 3. 绘制步骤时间曲线
            self._plot_step_time(steps, step_times)
            
            # 4. 保存数据到文本文件
            self._save_data_to_file(steps, losses, rewards, grad_norms, step_times)
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def _plot_loss_reward(self, steps, losses, rewards):
        """绘制Loss和Reward双轴图"""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Loss曲线（左轴）
        color1 = 'tab:red'
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss', color=color1)
        line1 = ax1.plot(steps, losses, color=color1, linewidth=1.5, alpha=0.8, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # 添加Loss的移动平均线
        if len(losses) > 10:
            window_size = min(50, len(losses) // 10)
            loss_ma = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            loss_ma_steps = steps[window_size-1:]
            ax1.plot(loss_ma_steps, loss_ma, '--', color='darkred', linewidth=2, alpha=0.7, label=f'Loss MA({window_size})')
        
        # Reward曲线（右轴）
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Reward', color=color2)
        line2 = ax2.plot(steps, rewards, color=color2, linewidth=1.5, alpha=0.8, label='Reward')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 添加Reward的移动平均线
        if len(rewards) > 10:
            window_size = min(50, len(rewards) // 10)
            reward_ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            reward_ma_steps = steps[window_size-1:]
            ax2.plot(reward_ma_steps, reward_ma, '--', color='darkblue', linewidth=2, alpha=0.7, label=f'Reward MA({window_size})')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f'WAN GRPO Training Progress (Step {steps[-1]})')
        plt.tight_layout()
        plt.savefig(self.loss_reward_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_grad_norm(self, steps, grad_norms):
        """绘制梯度范数曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(steps, grad_norms, color='green', linewidth=1, alpha=0.7, label='Gradient Norm')
        
        # 添加移动平均线
        if len(grad_norms) > 10:
            window_size = min(50, len(grad_norms) // 10)
            grad_ma = np.convolve(grad_norms, np.ones(window_size)/window_size, mode='valid')
            grad_ma_steps = steps[window_size-1:]
            plt.plot(grad_ma_steps, grad_ma, '--', color='darkgreen', linewidth=2, label=f'Grad Norm MA({window_size})')
        
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title(f'Gradient Norm Over Time (Step {steps[-1]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.grad_norm_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_step_time(self, steps, step_times):
        """绘制步骤时间曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(steps, step_times, color='orange', linewidth=1, alpha=0.7, label='Step Time')
        
        # 添加移动平均线
        if len(step_times) > 10:
            window_size = min(50, len(step_times) // 10)
            time_ma = np.convolve(step_times, np.ones(window_size)/window_size, mode='valid')
            time_ma_steps = steps[window_size-1:]
            plt.plot(time_ma_steps, time_ma, '--', color='darkorange', linewidth=2, label=f'Step Time MA({window_size})')
        
        plt.xlabel('Training Step')
        plt.ylabel('Step Time (seconds)')
        plt.title(f'Training Step Time Over Time (Step {steps[-1]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.step_time_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _save_data_to_file(self, steps, losses, rewards, grad_norms, step_times):
        """保存数据到文本文件"""
        with open(self.data_path, 'w') as f:
            f.write("Step,Loss,Reward,GradNorm,StepTime\n")
            for i in range(len(steps)):
                f.write(f"{steps[i]},{losses[i]:.6f},{rewards[i]:.6f},{grad_norms[i]:.6f},{step_times[i]:.3f}\n")
    
    def create_summary_plot(self):
        """创建训练总结图"""
        if len(self.steps) < 2:
            return
        
        steps = np.array(list(self.steps))
        losses = np.array(list(self.losses))
        rewards = np.array(list(self.rewards))
        grad_norms = np.array(list(self.grad_norms))
        step_times = np.array(list(self.step_times))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(steps, losses, color='red', linewidth=1, alpha=0.7)
        if len(losses) > 10:
            window_size = min(100, len(losses) // 5)
            loss_ma = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            loss_ma_steps = steps[window_size-1:]
            ax1.plot(loss_ma_steps, loss_ma, '--', color='darkred', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Reward
        ax2.plot(steps, rewards, color='blue', linewidth=1, alpha=0.7)
        if len(rewards) > 10:
            window_size = min(100, len(rewards) // 5)
            reward_ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            reward_ma_steps = steps[window_size-1:]
            ax2.plot(reward_ma_steps, reward_ma, '--', color='darkblue', linewidth=2)
        ax2.set_title('Training Reward')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        
        # Gradient Norm
        ax3.plot(steps, grad_norms, color='green', linewidth=1, alpha=0.7)
        if len(grad_norms) > 10:
            window_size = min(100, len(grad_norms) // 5)
            grad_ma = np.convolve(grad_norms, np.ones(window_size)/window_size, mode='valid')
            grad_ma_steps = steps[window_size-1:]
            ax3.plot(grad_ma_steps, grad_ma, '--', color='darkgreen', linewidth=2)
        ax3.set_title('Gradient Norm')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Grad Norm')
        ax3.grid(True, alpha=0.3)
        
        # Step Time
        ax4.plot(steps, step_times, color='orange', linewidth=1, alpha=0.7)
        if len(step_times) > 10:
            window_size = min(100, len(step_times) // 5)
            time_ma = np.convolve(step_times, np.ones(window_size)/window_size, mode='valid')
            time_ma_steps = steps[window_size-1:]
            ax4.plot(time_ma_steps, time_ma, '--', color='darkorange', linewidth=2)
        ax4.set_title('Step Time')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Time (s)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'WAN GRPO Training Summary (Total Steps: {steps[-1]})', fontsize=16)
        plt.tight_layout()
        
        summary_path = self.save_dir / "training_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training summary saved to: {summary_path}")