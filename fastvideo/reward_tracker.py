import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from collections import deque
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境

class RewardTracker:
    """跟踪和可视化reward和loss变化的类 - 实时更新版本"""
    
    def __init__(self, save_dir="./plots", save_every=1, max_points=10000):
        self.save_dir = save_dir
        self.save_every = save_every
        self.max_points = max_points
        
        # 使用deque来限制内存使用
        self.rewards = deque(maxlen=max_points)
        self.losses = deque(maxlen=max_points)
        self.grad_norms = deque(maxlen=max_points)
        self.steps = deque(maxlen=max_points)
        self.mean_rewards = deque(maxlen=max_points)
        self.std_rewards = deque(maxlen=max_points)
        self.advantages = deque(maxlen=max_points)
        self.group_reward_means = deque(maxlen=max_points)
        self.group_reward_stds = deque(maxlen=max_points)
        # 【新增】VQ和MQ分离的奖励存储
        self.vq_rewards = deque(maxlen=max_points)
        self.mq_rewards = deque(maxlen=max_points)
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"RewardTracker initialized, saving to: {self.save_dir}")
    
    def add_data(self, step, loss=None, advantage=None ,reward_tensor=None, grad_norm=None,
                 vq_reward_tensor=None, mq_reward_tensor=None):
        """
        添加一个step的训练数据（loss, reward, grad_norm）
        
        Args:
            step: int, 当前训练步数
            loss: float, 当前步的loss值
            reward_tensor: torch.Tensor, shape (B,), 当前batch的rewards
            grad_norm: float, 梯度范数
        """
        self.steps.append(step)
        
        # 添加loss
        if loss is not None:
            self.losses.append(loss)
        else:
            self.losses.append(0.0)
        
        # 添加grad_norm
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        else:
            self.grad_norms.append(0.0)
        
        # 处理reward
        if reward_tensor is not None:
            # 转换为numpy并计算统计信息
            if hasattr(reward_tensor, 'cpu'):
                rewards_np = reward_tensor.cpu().numpy()
            elif hasattr(reward_tensor, 'numpy'):
                rewards_np = reward_tensor.numpy()
            else:
                rewards_np = np.array([reward_tensor]) if not isinstance(reward_tensor, (list, np.ndarray)) else np.array(reward_tensor)
            
            mean_reward = np.mean(rewards_np)
            std_reward = np.std(rewards_np)
            
            self.mean_rewards.append(mean_reward)
            self.std_rewards.append(std_reward)
            
            # 保存所有individual rewards（可选，用于更详细的分析）
            for reward in rewards_np:
                self.rewards.append(reward)
        else:
            self.mean_rewards.append(0.0)
            self.std_rewards.append(0.0)
        # 添加 advantage
        if advantage is not None:
            if hasattr(advantage, 'cpu'):
                adv_np = advantage.cpu().numpy()
            elif hasattr(advantage, 'numpy'):
                adv_np = advantage.numpy()
            else:
                adv_np = np.array([advantage]) if not isinstance(advantage, (list, np.ndarray)) else np.array(advantage)
            self.advantages.append(np.mean(adv_np))
        else:
            self.advantages.append(0.0)
        
         # 【新增】处理VQ奖励
        if vq_reward_tensor is not None:
            if hasattr(vq_reward_tensor, 'cpu'):
                vq_rewards_np = vq_reward_tensor.cpu().numpy()
            else:
                vq_rewards_np = np.array([vq_reward_tensor]) if not isinstance(vq_reward_tensor, (list, np.ndarray)) else np.array(vq_reward_tensor)
            self.vq_rewards.append(np.mean(vq_rewards_np))
        else:
            self.vq_rewards.append(0.0)
        
        # 【新增】处理MQ奖励
        if mq_reward_tensor is not None:
            if hasattr(mq_reward_tensor, 'cpu'):
                mq_rewards_np = mq_reward_tensor.cpu().numpy()
            else:
                mq_rewards_np = np.array([mq_reward_tensor]) if not isinstance(mq_reward_tensor, (list, np.ndarray)) else np.array(mq_reward_tensor)
            self.mq_rewards.append(np.mean(mq_rewards_np))
        else:
            self.mq_rewards.append(0.0)


        # 实时保存图表
        self.save_realtime_plots(step)
    
    def add_reward(self, step, reward_tensor):
        """兼容旧接口"""
        self.add_data(step, reward_tensor=reward_tensor)
    
    def save_realtime_plots(self, step):
        """实时保存图表 - 每次调用都更新"""
        if len(self.steps) < 1:
            return
        
        try:
            # 转换为numpy数组以便绘图
            steps_arr = np.array(list(self.steps))
            losses_arr = np.array(list(self.losses))
            mean_rewards_arr = np.array(list(self.mean_rewards))
            std_rewards_arr = np.array(list(self.std_rewards))
            grad_norms_arr = np.array(list(self.grad_norms))
            advantages_arr = np.array(list(self.advantages))
            group_means_arr = np.array(list(self.group_reward_means))
            group_stds_arr = np.array(list(self.group_reward_stds))

            vq_rewards_arr = np.array(list(self.vq_rewards))
            mq_rewards_arr = np.array(list(self.mq_rewards))
            
            # 1. 原有的Loss和Reward图表
            self._create_loss_reward_plot(steps_arr, losses_arr, mean_rewards_arr, std_rewards_arr, step)
            
            # 2. 【新增】VQ和MQ分离图表
            if np.any(vq_rewards_arr != 0) or np.any(mq_rewards_arr != 0):
                self._create_vq_mq_plots(steps_arr, vq_rewards_arr, mq_rewards_arr, step)
            # 1. 创建Loss和Reward的双轴图（主图）
            # self._create_loss_reward_plot(steps_arr, losses_arr, mean_rewards_arr, std_rewards_arr, step)
            
            # 2. 创建详细的四合一图表
            if len(self.steps) >= 2:
                self._create_detailed_plots(steps_arr, losses_arr, mean_rewards_arr, std_rewards_arr, grad_norms_arr, step)
            
            # 3. 保存数据到文件
            self._save_data_to_file(steps_arr, losses_arr, mean_rewards_arr, std_rewards_arr, grad_norms_arr)
            # 新增：画 advantages 曲线
            self._create_advantages_plot(steps_arr, advantages_arr, step)

        except Exception as e:
            print(f"Error saving realtime plots: {e}")
            
    def _create_vq_mq_plots(self, steps, vq_rewards, mq_rewards, current_step):
        """创建VQ和MQ分离的奖励图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # VQ奖励图
        ax1.plot(steps, vq_rewards, 'purple', linewidth=2, alpha=0.8, label='VQ Reward')
        if len(vq_rewards) >= 10:
            window_size = min(20, len(vq_rewards) // 5)
            if window_size > 1:
                vq_ma = np.convolve(vq_rewards, np.ones(window_size)/window_size, mode='valid')
                vq_ma_steps = steps[window_size-1:]
                ax1.plot(vq_ma_steps, vq_ma, '--', color='darkviolet', linewidth=2, alpha=0.9, label=f'VQ MA({window_size})')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('VQ Reward Value')
        ax1.set_title(f'Video Quality (VQ) Reward - Step {current_step}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # MQ奖励图
        ax2.plot(steps, mq_rewards, 'orange', linewidth=2, alpha=0.8, label='MQ Reward')
        if len(mq_rewards) >= 10:
            window_size = min(20, len(mq_rewards) // 5)
            if window_size > 1:
                mq_ma = np.convolve(mq_rewards, np.ones(window_size)/window_size, mode='valid')
                mq_ma_steps = steps[window_size-1:]
                ax2.plot(mq_ma_steps, mq_ma, '--', color='darkorange', linewidth=2, alpha=0.9, label=f'MQ MA({window_size})')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('MQ Reward Value')
        ax2.set_title(f'Motion Quality (MQ) Reward - Step {current_step}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存VQ和MQ图表
        vq_mq_plot_path = os.path.join(self.save_dir, 'realtime_vq_mq_rewards.png')
        plt.savefig(vq_mq_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 同时创建对比图
        self._create_vq_mq_comparison(steps, vq_rewards, mq_rewards, current_step)

    def _create_vq_mq_comparison(self, steps, vq_rewards, mq_rewards, current_step):
        """创建VQ和MQ对比图"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(steps, vq_rewards, 'purple', linewidth=2, alpha=0.8, label='VQ Reward')
        plt.plot(steps, mq_rewards, 'orange', linewidth=2, alpha=0.8, label='MQ Reward')
        
        # 添加移动平均线
        if len(vq_rewards) >= 10:
            window_size = min(20, len(vq_rewards) // 5)
            if window_size > 1:
                vq_ma = np.convolve(vq_rewards, np.ones(window_size)/window_size, mode='valid')
                mq_ma = np.convolve(mq_rewards, np.ones(window_size)/window_size, mode='valid')
                ma_steps = steps[window_size-1:]
                plt.plot(ma_steps, vq_ma, '--', color='darkviolet', linewidth=1.5, alpha=0.7, label=f'VQ MA({window_size})')
                plt.plot(ma_steps, mq_ma, '--', color='darkorange', linewidth=1.5, alpha=0.7, label=f'MQ MA({window_size})')
        
        plt.xlabel('Training Step')
        plt.ylabel('Reward Value')
        plt.title(f'VQ vs MQ Rewards Comparison - Step {current_step}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 显示当前值
        if len(vq_rewards) > 0 and len(mq_rewards) > 0:
            plt.text(0.02, 0.98, f'Current VQ: {vq_rewards[-1]:.4f}\nCurrent MQ: {mq_rewards[-1]:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        comparison_plot_path = os.path.join(self.save_dir, 'realtime_vq_mq_comparison.png')
        plt.savefig(comparison_plot_path, dpi=100, bbox_inches='tight')
        plt.close()

    def _create_advantages_plot(self, steps, advantages, current_step):
        plt.figure(figsize=(10, 5))
        plt.plot(steps, advantages, label='Advantage', color='purple')
        plt.xlabel('Training Step')
        plt.ylabel('Advantage')
        plt.title(f'Advantage Progress (Step {current_step})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'realtime_advantages.png'), dpi=100, bbox_inches='tight')
        plt.close()

    def _create_loss_reward_plot(self, steps, losses, mean_rewards, std_rewards, current_step):
        """创建Loss和Reward的双轴实时图表"""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Loss曲线（左轴）
        color1 = 'tab:red'
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss', color=color1)
        line1 = ax1.plot(steps, losses, color=color1, linewidth=1.5, alpha=0.8, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # 添加Loss的移动平均线
        if len(losses) > 5:
            window_size = min(20, len(losses) // 3)
            if window_size > 1:
                loss_ma = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                loss_ma_steps = steps[window_size-1:]
                ax1.plot(loss_ma_steps, loss_ma, '--', color='darkred', linewidth=2, alpha=0.9, label=f'Loss MA({window_size})')
        
        # Reward曲线（右轴）
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Reward', color=color2)
        line2 = ax2.plot(steps, mean_rewards, color=color2, linewidth=1.5, alpha=0.8, label='Reward')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 添加Reward的移动平均线和置信区间
        if len(mean_rewards) > 5:
            window_size = min(20, len(mean_rewards) // 3)
            if window_size > 1:
                reward_ma = np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid')
                reward_ma_steps = steps[window_size-1:]
                ax2.plot(reward_ma_steps, reward_ma, '--', color='darkblue', linewidth=2, alpha=0.9, label=f'Reward MA({window_size})')
        
        # 添加置信区间
        if len(std_rewards) > 0 and np.any(std_rewards > 0):
            ax2.fill_between(steps, 
                           mean_rewards - std_rewards,
                           mean_rewards + std_rewards,
                           alpha=0.2, color=color2, label='±1 Std')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 添加当前值显示
        if len(losses) > 0 and len(mean_rewards) > 0:
            plt.title(f'WAN GRPO Training Progress (Step {current_step})\n'
                     f'Current Loss: {losses[-1]:.4f}, Current Reward: {mean_rewards[-1]:.4f}')
        else:
            plt.title(f'WAN GRPO Training Progress (Step {current_step})')
        
        plt.tight_layout()
        
        # 保存主图表
        main_plot_path = os.path.join(self.save_dir, 'realtime_loss_reward.png')
        plt.savefig(main_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_plots(self, steps, losses, mean_rewards, std_rewards, grad_norms, current_step):
        """创建详细的四合一图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'WAN GRPO Training Details - Step {current_step}', fontsize=16)
        
        # 1. Loss曲线
        axes[0, 0].plot(steps, losses, 'r-', linewidth=1.5, alpha=0.8, label='Loss')
        if len(losses) > 10:
            window_size = min(20, len(losses) // 5)
            if window_size > 1:
                loss_ma = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                loss_ma_steps = steps[window_size-1:]
                axes[0, 0].plot(loss_ma_steps, loss_ma, '--', color='darkred', linewidth=2, label=f'MA({window_size})')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Reward曲线
        axes[0, 1].plot(steps, mean_rewards, 'b-', linewidth=1.5, alpha=0.8, label='Mean Reward')
        if len(std_rewards) > 0 and np.any(std_rewards > 0):
            axes[0, 1].fill_between(steps, 
                                   mean_rewards - std_rewards,
                                   mean_rewards + std_rewards,
                                   alpha=0.3, color='blue', label='±1 Std')
        if len(mean_rewards) > 10:
            window_size = min(20, len(mean_rewards) // 5)
            if window_size > 1:
                reward_ma = np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid')
                reward_ma_steps = steps[window_size-1:]
                axes[0, 1].plot(reward_ma_steps, reward_ma, '--', color='darkblue', linewidth=2, label=f'MA({window_size})')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Training Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 梯度范数
        axes[1, 0].plot(steps, grad_norms, 'g-', linewidth=1.5, alpha=0.8, label='Gradient Norm')
        if len(grad_norms) > 10:
            window_size = min(20, len(grad_norms) // 5)
            if window_size > 1:
                grad_ma = np.convolve(grad_norms, np.ones(window_size)/window_size, mode='valid')
                grad_ma_steps = steps[window_size-1:]
                axes[1, 0].plot(grad_ma_steps, grad_ma, '--', color='darkgreen', linewidth=2, label=f'MA({window_size})')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 训练统计
        axes[1, 1].axis('off')
        if len(losses) > 0:
            stats_text = f"""Training Statistics (Step {current_step}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Loss: {losses[-1]:.6f}
Loss - Min: {np.min(losses):.6f}  Max: {np.max(losses):.6f}
Loss - Mean: {np.mean(losses):.6f}  Std: {np.std(losses):.6f}

Current Reward: {mean_rewards[-1]:.6f}
Reward - Min: {np.min(mean_rewards):.6f}  Max: {np.max(mean_rewards):.6f}
Reward - Mean: {np.mean(mean_rewards):.6f}  Std: {np.std(mean_rewards):.6f}

Current Grad Norm: {grad_norms[-1]:.6f}
Grad - Min: {np.min(grad_norms):.6f}  Max: {np.max(grad_norms):.6f}

Total Steps: {len(steps)}
Data Points: {len(self.rewards)} individual rewards
            """
            axes[1, 1].text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
                            fontfamily='monospace', 
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存详细图表
        detailed_plot_path = os.path.join(self.save_dir, 'realtime_detailed.png')
        plt.savefig(detailed_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _save_data_to_file(self, steps, losses, mean_rewards, std_rewards, grad_norms):
        """保存数据到CSV文件"""
        try:
            import pandas as pd
            
            # 【修改】保存主要训练数据（包含VQ和MQ）
            vq_rewards_arr = np.array(list(self.vq_rewards))
            mq_rewards_arr = np.array(list(self.mq_rewards))
            
            df = pd.DataFrame({
                'step': steps,
                'loss': losses,
                'mean_reward': mean_rewards,  # 组合奖励
                'std_reward': std_rewards,
                'grad_norm': grad_norms,
                'vq_reward': vq_rewards_arr,  # 【新增】VQ奖励
                'mq_reward': mq_rewards_arr,  # 【新增】MQ奖励
            })
            
            # 1. 保存完整训练数据（包含所有字段）
            complete_csv = os.path.join(self.save_dir, 'realtime_training_data.csv')
            df.to_csv(complete_csv, index=False)
            
            # 2. 【新增】分别保存VQ奖励CSV
            if np.any(vq_rewards_arr != 0):
                vq_df = pd.DataFrame({
                    'step': steps,
                    'vq_reward': vq_rewards_arr
                })
                vq_csv = os.path.join(self.save_dir, 'vq_rewards.csv')
                vq_df.to_csv(vq_csv, index=False)
            
            # 3. 【新增】分别保存MQ奖励CSV
            if np.any(mq_rewards_arr != 0):
                mq_df = pd.DataFrame({
                    'step': steps,
                    'mq_reward': mq_rewards_arr
                })
                mq_csv = os.path.join(self.save_dir, 'mq_rewards.csv')
                mq_df.to_csv(mq_csv, index=False)
            
            # 4. 【新增】保存组合奖励CSV（用于对比）
            combined_df = pd.DataFrame({
                'step': steps,
                'combined_reward': mean_rewards
            })
            combined_csv = os.path.join(self.save_dir, 'combined_rewards.csv')
            combined_df.to_csv(combined_csv, index=False)
            
        except Exception as e:
            # 如果pandas不可用，使用原生方法保存
            try:
                # 完整数据
                complete_txt = os.path.join(self.save_dir, 'realtime_training_data.txt')
                with open(complete_txt, 'w') as f:
                    f.write("step,loss,mean_reward,std_reward,grad_norm,vq_reward,mq_reward\n")
                    for i in range(len(steps)):
                        vq_val = list(self.vq_rewards)[i] if i < len(self.vq_rewards) else 0.0
                        mq_val = list(self.mq_rewards)[i] if i < len(self.mq_rewards) else 0.0
                        f.write(f"{steps[i]},{losses[i]:.6f},{mean_rewards[i]:.6f},{std_rewards[i]:.6f},{grad_norms[i]:.6f},{vq_val:.6f},{mq_val:.6f}\n")
                
                # VQ数据
                vq_txt = os.path.join(self.save_dir, 'vq_rewards.txt')
                with open(vq_txt, 'w') as f:
                    f.write("step,vq_reward\n")
                    for i, vq_val in enumerate(list(self.vq_rewards)):
                        if i < len(steps):
                            f.write(f"{steps[i]},{vq_val:.6f}\n")
                
                # MQ数据
                mq_txt = os.path.join(self.save_dir, 'mq_rewards.txt')
                with open(mq_txt, 'w') as f:
                    f.write("step,mq_reward\n")
                    for i, mq_val in enumerate(list(self.mq_rewards)):
                        if i < len(steps):
                            f.write(f"{steps[i]},{mq_val:.6f}\n")
                            
            except Exception as e2:
                print(f"Error saving data: {e2}")
    
    def save_summary_plot(self, final_step):
        """保存训练结束时的总结图表"""
        if len(self.steps) < 2:
            return
        
        try:
            steps = np.array(list(self.steps))
            losses = np.array(list(self.losses))
            mean_rewards = np.array(list(self.mean_rewards))
            std_rewards = np.array(list(self.std_rewards))
            grad_norms = np.array(list(self.grad_norms))
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'WAN GRPO Training Summary - Final Step {final_step}', fontsize=16)
            
            # 1. 完整的loss曲线
            axes[0, 0].plot(steps, losses, 'r-', linewidth=1.5, alpha=0.8)
            if len(losses) > 20:
                window = min(50, len(losses) // 10)
                loss_ma = np.convolve(losses, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(steps[window-1:], loss_ma, '--', color='darkred', linewidth=2)
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Complete Loss Progress')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 完整的reward曲线
            axes[0, 1].plot(steps, mean_rewards, 'b-', linewidth=1.5, alpha=0.8)
            axes[0, 1].fill_between(steps, 
                                   mean_rewards - std_rewards,
                                   mean_rewards + std_rewards,
                                   alpha=0.3, color='blue')
            if len(mean_rewards) > 20:
                window = min(50, len(mean_rewards) // 10)
                reward_ma = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(steps[window-1:], reward_ma, '--', color='darkblue', linewidth=2)
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Mean Reward')
            axes[0, 1].set_title('Complete Reward Progress')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 梯度范数
            axes[0, 2].plot(steps, grad_norms, 'g-', linewidth=1.5, alpha=0.8)
            if len(grad_norms) > 20:
                window = min(50, len(grad_norms) // 10)
                grad_ma = np.convolve(grad_norms, np.ones(window)/window, mode='valid')
                axes[0, 2].plot(steps[window-1:], grad_ma, '--', color='darkgreen', linewidth=2)
            axes[0, 2].set_xlabel('Training Step')
            axes[0, 2].set_ylabel('Gradient Norm')
            axes[0, 2].set_title('Gradient Norm Progress')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Loss和Reward的改善率
            if len(losses) >= 2:
                loss_changes = np.diff(losses)
                reward_changes = np.diff(mean_rewards)
                axes[1, 0].plot(steps[1:], loss_changes, 'r-', linewidth=1, alpha=0.7, label='Loss Change')
                axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                axes[1, 0].set_xlabel('Training Step')
                axes[1, 0].set_ylabel('Change per Step')
                axes[1, 0].set_title('Loss Improvement Rate')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                axes[1, 1].plot(steps[1:], reward_changes, 'b-', linewidth=1, alpha=0.7, label='Reward Change')
                axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                axes[1, 1].set_xlabel('Training Step')
                axes[1, 1].set_ylabel('Change per Step')
                axes[1, 1].set_title('Reward Improvement Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # 5. 训练统计摘要
            axes[1, 2].axis('off')
            stats_text = f"""Final Training Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Steps: {len(steps)}

Loss Statistics:
  Initial: {losses[0]:.6f}
  Final: {losses[-1]:.6f}
  Min: {np.min(losses):.6f}
  Max: {np.max(losses):.6f}
  Mean: {np.mean(losses):.6f}
  Improvement: {losses[0] - losses[-1]:.6f}

Reward Statistics:
  Initial: {mean_rewards[0]:.6f}
  Final: {mean_rewards[-1]:.6f}  
  Min: {np.min(mean_rewards):.6f}
  Max: {np.max(mean_rewards):.6f}
  Mean: {np.mean(mean_rewards):.6f}
  Improvement: {mean_rewards[-1] - mean_rewards[0]:.6f}

Gradient Statistics:
  Final: {grad_norms[-1]:.6f}
  Mean: {np.mean(grad_norms):.6f}
  Max: {np.max(grad_norms):.6f}
            """
            axes[1, 2].text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
                            fontfamily='monospace', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            
            summary_path = os.path.join(self.save_dir, 'final_training_summary.png')
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Final training summary saved: {summary_path}")
            
        except Exception as e:
            print(f"Error creating summary plot: {e}")
    
    def save_plots(self, step):
        """兼容旧接口 - 现在实时更新已经在add_data中处理"""
        pass
    
    def should_save(self, step):
        """兼容旧接口 - 现在每步都保存"""
        return True
    
    def save_data(self, step):
        """兼容旧接口 - 数据已在实时保存"""
        if step == "final":
            self.save_summary_plot(step)

def create_reward_summary_plot(reward_tracker, save_path):
    """兼容旧接口"""
    reward_tracker.save_summary_plot("final")